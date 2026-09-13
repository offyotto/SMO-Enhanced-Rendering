#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <objc/runtime.h>
#import <os/lock.h>
#include <stdatomic.h>

typedef void (*FrameProcessor)(id<MTLCommandBuffer>, id<MTLTexture>, id<MTLTexture>);
static FrameProcessor processor;
static id<CAMetalDrawable> currentDrawable;
static id<MTLTexture> currentDepth;
static uint64_t depthSerial, frameSerial, commits, presents, draws;
static NSUInteger depthArea;
static os_unfair_lock stateLock = OS_UNFAIR_LOCK_INIT;
static FILE *logFile;
static IMP originalNext, originalRender, originalCommit;
static NSMutableSet<NSString *> *seenDepths, *seenColors, *seenLabels;
static _Thread_local BOOL insideProcessor;

static void Log(NSString *message) {
    if (logFile) { fprintf(logFile, "%s\n", message.UTF8String); fflush(logFile); }
}

static IMP Replace(Class cls, SEL sel, IMP replacement) {
    Method method = class_getInstanceMethod(cls, sel);
    if (!method) return NULL;
    IMP previous = method_getImplementation(method);
    if (!class_addMethod(cls, sel, replacement, method_getTypeEncoding(method))) {
        method_setImplementation(class_getInstanceMethod(cls, sel), replacement);
    }
    return previous;
}

static id<CAMetalDrawable> NextDrawable(CAMetalLayer *layer, SEL sel) {
    if (processor && layer.framebufferOnly && layer.drawableSize.width >= 640) layer.framebufferOnly = NO;
    id<CAMetalDrawable> drawable = ((id (*)(id, SEL))originalNext)(layer, sel);
    if (drawable && drawable.texture.width >= 640) {
        os_unfair_lock_lock(&stateLock);
        currentDrawable = drawable;
        os_unfair_lock_unlock(&stateLock);
    }
    return drawable;
}

static id<MTLRenderCommandEncoder> Render(id<MTLCommandBuffer> cb, SEL sel, MTLRenderPassDescriptor *desc) {
    if (!insideProcessor) {
        os_unfair_lock_lock(&stateLock);

        // Record unique color-attachment signatures from the guest renderer.
        // A native motion-vector/velocity target, if Odyssey emits one, must
        // eventually appear here because MoltenVK maps guest render targets to
        // Metal render-pass color attachments. We deliberately do not retain,
        // modify, or force-store anything yet; this first stage is observation.
        for (NSUInteger i=0; i<8; ++i) {
            MTLRenderPassColorAttachmentDescriptor *att=desc.colorAttachments[i];
            id<MTLTexture> tex=att.texture;
            if (!tex || tex.width < 160 || tex.height < 90) continue;
            id<MTLTexture> resolve=att.resolveTexture;
            NSString *key=[NSString stringWithFormat:
                @"slot=%lu %lux%lu fmt=%lu type=%lu samples=%lu usage=%lu storage=%lu load=%lu store=%lu resolve=%lux%lu rfmt=%lu label=%@",
                (unsigned long)i,
                (unsigned long)tex.width,(unsigned long)tex.height,
                (unsigned long)tex.pixelFormat,(unsigned long)tex.textureType,
                (unsigned long)tex.sampleCount,(unsigned long)tex.usage,
                (unsigned long)tex.storageMode,(unsigned long)att.loadAction,
                (unsigned long)att.storeAction,
                (unsigned long)resolve.width,(unsigned long)resolve.height,
                (unsigned long)resolve.pixelFormat,tex.label ?: @"<none>"];
            if (seenColors.count < 240 && ![seenColors containsObject:key]) {
                [seenColors addObject:key];
                Log([@"color " stringByAppendingString:key]);
            }
        }

        id<MTLTexture> tex = desc.depthAttachment.texture;
        if (tex && tex.width >= 320 && tex.height >= 180) {
            draws++;
            NSString *key = [NSString stringWithFormat:@"%p %lux%lu fmt=%lu type=%lu samples=%lu usage=%lu storage=%lu load=%lu store=%lu clear=%.9g label=%@", (__bridge void *)tex, tex.width, tex.height, tex.pixelFormat, tex.textureType, tex.sampleCount, tex.usage, tex.storageMode, desc.depthAttachment.loadAction, desc.depthAttachment.storeAction, desc.depthAttachment.clearDepth, tex.label];
            if (seenDepths.count < 120 && ![seenDepths containsObject:key]) { [seenDepths addObject:key]; Log([@"depth " stringByAppendingString:key]); }
            float aspect = (float)tex.width / tex.height;
            if (tex.sampleCount == 1 && aspect > 1.70 && aspect < 1.85 && tex.width >= 640 && tex.width*tex.height >= depthArea) {
                currentDepth = tex;
                depthSerial = frameSerial;
                depthArea = tex.width*tex.height;
            }
        }
        os_unfair_lock_unlock(&stateLock);
    }
    return ((id (*)(id, SEL, id))originalRender)(cb, sel, desc);
}

static void Commit(id<MTLCommandBuffer> cb, SEL sel) {
    NSString *label = cb.label ?: @"<no label>";
    id<CAMetalDrawable> drawable = nil;
    id<MTLTexture> depth = nil;
    FrameProcessor effect = NULL;
    os_unfair_lock_lock(&stateLock);
    commits++;
    if (seenLabels.count < 30 && ![seenLabels containsObject:label]) { [seenLabels addObject:label]; Log([@"commit " stringByAppendingString:label]); }
    if ([label containsString:@"vkQueuePresent"] || [label containsString:@"Present"]) {
        presents++;
        drawable = currentDrawable;
        depth = depthSerial == frameSerial ? currentDepth : nil;
        effect = processor;
        frameSerial++;
        depthArea=0;
        if (presents <= 3 || presents % 1800 == 0) Log([NSString stringWithFormat:@"present %llu drawable=%p %lux%lu fmt=%lu framebufferOnly=%d depth=%p", presents, (__bridge void *)drawable, drawable.texture.width, drawable.texture.height, drawable.texture.pixelFormat, drawable.texture.framebufferOnly, (__bridge void *)depth]);
    }
    os_unfair_lock_unlock(&stateLock);
    if (effect && drawable && !drawable.texture.framebufferOnly) {
        insideProcessor = YES;
        effect(cb, drawable.texture, depth);
        insideProcessor = NO;
    }
    ((void (*)(id, SEL))originalCommit)(cb, sel);
}

__attribute__((visibility("default"))) void SMOSetFrameProcessor(FrameProcessor callback) {
    os_unfair_lock_lock(&stateLock);
    processor = callback;
    os_unfair_lock_unlock(&stateLock);
    Log(callback ? @"frame processor enabled" : @"frame processor disabled");
}

__attribute__((visibility("default"))) const char *SMOBridgeStatus(void) {
    static char status[512];
    os_unfair_lock_lock(&stateLock);
    snprintf(status, sizeof(status), "commits=%llu presents=%llu depthPasses=%llu drawable=%lux%lu depth=%lux%lu effect=%s colorSignatures=%lu", commits, presents, draws, currentDrawable.texture.width, currentDrawable.texture.height, currentDepth.width, currentDepth.height, processor ? "on" : "off", (unsigned long)seenColors.count);
    os_unfair_lock_unlock(&stateLock);
    return status;
}

__attribute__((constructor)) static void Install(void) {
    @autoreleasepool {
        NSString *directory = [NSHomeDirectory() stringByAppendingPathComponent:@"Documents/SMOShaders"];
        [[NSFileManager defaultManager] createDirectoryAtPath:directory withIntermediateDirectories:YES attributes:nil error:nil];
        logFile = fopen([[directory stringByAppendingPathComponent:@"bridge.log"] fileSystemRepresentation], "a");
        seenDepths = [NSMutableSet new]; seenColors = [NSMutableSet new]; seenLabels = [NSMutableSet new];
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        Class cls = object_getClass(cb);
        Log([NSString stringWithFormat:@"install %@ %@", device.name, NSStringFromClass(cls)]);
        originalNext = Replace(CAMetalLayer.class, @selector(nextDrawable), (IMP)NextDrawable);
        originalRender = Replace(cls, @selector(renderCommandEncoderWithDescriptor:), (IMP)Render);
        originalCommit = Replace(cls, @selector(commit), (IMP)Commit);
        Log([NSString stringWithFormat:@"hooks next=%p render=%p commit=%p", originalNext, originalRender, originalCommit]);
    }
}

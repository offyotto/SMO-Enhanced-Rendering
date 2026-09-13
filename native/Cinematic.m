#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <AppKit/AppKit.h>
#import <simd/simd.h>
#include <dlfcn.h>
#include <math.h>

typedef struct {
    simd_float2 texel, depthTexel;
    float aspect, tanHalfFov, reflections, occlusion, bloom, exposure;
    float saturation, contrast, debugView, depthAvailable;
    simd_float2 direction;
} ShaderSettings;
typedef void (*FrameProcessor)(id<MTLCommandBuffer>, id<MTLTexture>, id<MTLTexture>);

@interface ShaderFrame : NSObject
@property id<MTLTexture> color, effects, bloomA, bloomB;
@property BOOL busy;
@end
@implementation ShaderFrame
@end

@interface ShaderProgram : NSObject
@property id<MTLDevice> device;
@property id<MTLLibrary> library;
@property id<MTLRenderPipelineState> surface, extract, blur, composite;
@property NSMutableArray<ShaderFrame *> *frames;
@property ShaderSettings settings;
@property NSUInteger effectWidth;
@property BOOL enabled;
@property BOOL failed;
@end
@implementation ShaderProgram
@end

static NSLock *lock;
static ShaderProgram *program;
static NSString *directory;
static dispatch_source_t timer;
static id<MTLTexture> lastDepth;
static uint64_t frameCount, errorCount, skippedFrames, missingDepthFrames;
static double gpuMilliseconds;
static NSString *lastError = @"none";
static FILE *logFile;
static BOOL gameAllowed=YES;

static void Log(NSString *text) {
    if (logFile) { fprintf(logFile,"%s\n",text.UTF8String); fflush(logFile); }
}
static float Number(NSDictionary *d, NSString *key, float fallback, float lo, float hi) {
    id value=d[key];
    return [value isKindOfClass:NSNumber.class] ? fminf(hi,fmaxf(lo,[value floatValue])) : fallback;
}
static void UpdateSettings(ShaderProgram *p, NSDictionary *d) {
    ShaderSettings s={0};
    s.reflections=Number(d,@"reflections",1.65,0,3);
    s.occlusion=Number(d,@"occlusion",1.55,0,3);
    s.bloom=Number(d,@"bloom",.30,0,2);
    s.exposure=Number(d,@"exposure",1.10,.5,2);
    s.saturation=Number(d,@"saturation",1.12,0,2);
    s.contrast=Number(d,@"contrast",1.055,.5,1.5);
    s.tanHalfFov=tanf(Number(d,@"verticalFov",50,30,100)*M_PI/360);
    s.debugView=Number(d,@"debugView",0,0,3);
    p.settings=s;
    p.effectWidth=(NSUInteger)Number(d,@"effectWidth",960,480,1920);
    p.enabled=d[@"enabled"] ? [d[@"enabled"] boolValue] : YES;
}
static NSDictionary *ReadSettings(void) {
    NSData *data=[NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"preset.json"]];
    id value=data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    return [value isKindOfClass:NSDictionary.class] ? value : @{};
}
static id<MTLRenderPipelineState> Pipeline(ShaderProgram *p, NSString *fragment, NSError **error) {
    MTLRenderPipelineDescriptor *desc=[MTLRenderPipelineDescriptor new];
    desc.label=[@"SMO Cinematic / " stringByAppendingString:fragment];
    desc.vertexFunction=[p.library newFunctionWithName:@"fullScreen"];
    desc.fragmentFunction=[p.library newFunctionWithName:fragment];
    desc.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA16Float;
    return [p.device newRenderPipelineStateWithDescriptor:desc error:error];
}
static BOOL LoadProgram(void) {
    NSError *error=nil;
    NSString *source=[NSString stringWithContentsOfFile:[directory stringByAppendingPathComponent:@"Cinematic.metal"] encoding:NSUTF8StringEncoding error:&error];
    ShaderProgram *p=[ShaderProgram new];
    p.device=MTLCreateSystemDefaultDevice();
    if (source) {
        MTLCompileOptions *options=[MTLCompileOptions new];
        p.library=[p.device newLibraryWithSource:source options:options error:&error];
    }
    if (p.library) {
        p.surface=Pipeline(p,@"surfaceEffects",&error);
        p.extract=Pipeline(p,@"bloomExtract",&error);
        p.blur=Pipeline(p,@"bloomBlur",&error);
        p.composite=Pipeline(p,@"composite",&error);
    }
    if (!p.surface || !p.extract || !p.blur || !p.composite) {
        lastError=error.description ?: @"Shader pipeline creation failed";
        Log(lastError);
        return NO;
    }
    p.frames=[NSMutableArray new];
    UpdateSettings(p,ReadSettings());
    [lock lock]; program=p; [lock unlock];
    Log(@"Metal shader compilation passed. Four render pipelines are ready.");
    return YES;
}
static id<MTLTexture> Texture(id<MTLDevice> device, NSUInteger w, NSUInteger h, NSString *name) {
    MTLTextureDescriptor *d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:w height:h mipmapped:NO];
    d.storageMode=MTLStorageModePrivate;
    d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;
    id<MTLTexture> t=[device newTextureWithDescriptor:d];
    t.label=name;
    return t;
}
static void Draw(id<MTLCommandBuffer> cb, id<MTLTexture> target, id<MTLRenderPipelineState> pipeline,
                 NSArray<id<MTLTexture>> *textures, ShaderSettings settings) {
    MTLRenderPassDescriptor *desc=[MTLRenderPassDescriptor renderPassDescriptor];
    desc.colorAttachments[0].texture=target;
    desc.colorAttachments[0].loadAction=MTLLoadActionDontCare;
    desc.colorAttachments[0].storeAction=MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc=[cb renderCommandEncoderWithDescriptor:desc];
    enc.label=@"SMO Cinematic";
    [enc setRenderPipelineState:pipeline];
    for (NSUInteger i=0;i<textures.count;i++) [enc setFragmentTexture:textures[i] atIndex:i];
    [enc setFragmentBytes:&settings length:sizeof(settings) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
}
static void ProcessFrame(id<MTLCommandBuffer> cb, id<MTLTexture> color, id<MTLTexture> depth) {
    @autoreleasepool {
        if (color.pixelFormat != MTLPixelFormatRGBA16Float || color.width < 640) return;
        [lock lock];
        ShaderProgram *p=program;
        if (!p || !p.enabled || p.failed || !gameAllowed) { [lock unlock]; return; }

        // Do not drive SSR/AO from old geometry. The previous implementation
        // reused the last depth texture for up to seven missing presents, which
        // pairs current color with stale geometry and looks exactly like motion
        // ghosting during camera movement. We still bind the last texture so the
        // Metal pipeline remains valid, but depthAvailable=0 disables all
        // depth-driven effects on frames where the bridge did not provide fresh
        // depth. Bloom and color grading can continue normally.
        BOOL freshDepth=(depth != nil);
        if (freshDepth) {
            lastDepth=depth;
        } else {
            missingDepthFrames++;
            depth=lastDepth;
        }
        if (!depth) { skippedFrames++; [lock unlock]; return; }

        ShaderSettings settings=p.settings;
        settings.texel=(simd_float2){1.f/color.width,1.f/color.height};
        settings.depthTexel=(simd_float2){1.f/depth.width,1.f/depth.height};
        settings.aspect=(float)depth.width/depth.height;
        settings.depthAvailable=freshDepth ? 1.f : 0.f;
        NSUInteger ew=MIN(p.effectWidth,depth.width), eh=MAX(1,(NSUInteger)round(ew/settings.aspect));
        NSUInteger bw=MAX(1,ew/2), bh=MAX(1,eh/2);
        ShaderFrame *frame=nil;
        for (ShaderFrame *candidate in p.frames) if (!candidate.busy) { frame=candidate; break; }
        if (!frame && p.frames.count<4) { frame=[ShaderFrame new]; [p.frames addObject:frame]; }
        if (!frame) { skippedFrames++; [lock unlock]; return; }
        frame.busy=YES;
        [lock unlock];
        if (frame.color.width!=color.width || frame.color.height!=color.height || frame.effects.width!=ew || frame.effects.height!=eh) {
            frame.color=Texture(p.device,color.width,color.height,@"SMO frame copy");
            frame.effects=Texture(p.device,ew,eh,@"SMO reflections and contact shadows");
            frame.bloomA=Texture(p.device,bw,bh,@"SMO bloom A");
            frame.bloomB=Texture(p.device,bw,bh,@"SMO bloom B");
        }
        if (!frame.color || !frame.effects || !frame.bloomA || !frame.bloomB) {
            [lock lock]; frame.busy=NO; p.failed=YES; lastError=@"Metal texture allocation failed"; [lock unlock];
            Log(lastError); return;
        }
        id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
        [blit copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(color.width,color.height,1) toTexture:frame.color destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
        [blit endEncoding];
        Draw(cb,frame.effects,p.surface,@[frame.color,depth],settings);
        Draw(cb,frame.bloomA,p.extract,@[frame.color],settings);
        settings.direction=(simd_float2){2.f/bw,0};
        Draw(cb,frame.bloomB,p.blur,@[frame.bloomA],settings);
        settings.direction=(simd_float2){0,2.f/bh};
        Draw(cb,frame.bloomA,p.blur,@[frame.bloomB],settings);
        Draw(cb,color,p.composite,@[frame.color,frame.effects,frame.bloomA],settings);
        [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
            [lock lock];
            frame.busy=NO;
            frameCount++;
            double ms=(done.GPUEndTime-done.GPUStartTime)*1000;
            gpuMilliseconds=frameCount==1 ? ms : gpuMilliseconds*.95+ms*.05;
            if (done.error) { errorCount++; p.failed=YES; lastError=done.error.description; Log(lastError); }
            if (frameCount==1 || frameCount%1800==0) Log([NSString stringWithFormat:@"frames=%llu gpuMs=%.3f errors=%llu skipped=%llu missingDepth=%llu",frameCount,gpuMilliseconds,errorCount,skippedFrames,missingDepthFrames]);
            [lock unlock];
        }];
    }
}

__attribute__((visibility("default"))) const char *SMOShadersStatus(void) {
    static char status[2048];
    [lock lock];
    snprintf(status,sizeof(status),"enabled=%d frames=%llu gpuMs=%.3f errors=%llu skipped=%llu missingDepth=%llu lastError=%s",program.enabled && !program.failed && gameAllowed,frameCount,gpuMilliseconds,errorCount,skippedFrames,missingDepthFrames,lastError.UTF8String);
    [lock unlock];
    return status;
}
__attribute__((visibility("default"))) void SMOShadersReload(void) {
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{ LoadProgram(); });
}

#ifdef SMO_COMPILE_ONLY
int main(int argc, const char **argv) {
    @autoreleasepool {
        directory=argc>1 ? @(argv[1]) : @".";
        lock=[NSLock new]; logFile=stdout;
        if (!LoadProgram()) return 1;
        if (argc<3) return 0;
        NSString *captureDir=@(argv[2]);
        NSDictionary *info=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[captureDir stringByAppendingPathComponent:@"capture.json"]] options:0 error:nil];
        NSUInteger cw=[info[@"colorWidth"] unsignedIntegerValue], ch=[info[@"colorHeight"] unsignedIntegerValue];
        NSUInteger dw=[info[@"depthWidth"] unsignedIntegerValue], dh=[info[@"depthHeight"] unsignedIntegerValue];
        NSUInteger cs=[info[@"colorStride"] unsignedIntegerValue], ds=[info[@"depthStride"] unsignedIntegerValue];
        NSData *colorData=[NSData dataWithContentsOfFile:[captureDir stringByAppendingPathComponent:@"color.rgba16f"]];
        NSData *depthData=[NSData dataWithContentsOfFile:[captureDir stringByAppendingPathComponent:@"depth.r32f"]];
        if (!cw || !dw || colorData.length!=cs*ch || depthData.length!=ds*dh) return 2;
        id<MTLDevice> device=program.device;
        id<MTLCommandQueue> queue=[device newCommandQueue];
        id<MTLBuffer> colorBuffer=[device newBufferWithBytes:colorData.bytes length:colorData.length options:MTLResourceStorageModeShared];
        id<MTLBuffer> depthBuffer=[device newBufferWithBytes:depthData.bytes length:depthData.length options:MTLResourceStorageModeShared];
        id<MTLTexture> color=Texture(device,cw,ch,@"Test color");
        MTLTextureDescriptor *dd=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:dw height:dh mipmapped:NO];
        dd.storageMode=MTLStorageModePrivate; dd.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;
        id<MTLTexture> depth=[device newTextureWithDescriptor:dd];
        id<MTLCommandBuffer> cb=[queue commandBuffer];
        id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
        [blit copyFromBuffer:colorBuffer sourceOffset:0 sourceBytesPerRow:cs sourceBytesPerImage:cs*ch sourceSize:MTLSizeMake(cw,ch,1) toTexture:color destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
        [blit copyFromBuffer:depthBuffer sourceOffset:0 sourceBytesPerRow:ds sourceBytesPerImage:ds*dh sourceSize:MTLSizeMake(dw,dh,1) toTexture:depth destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
        [blit endEncoding];
        ProcessFrame(cb,color,depth);
        blit=[cb blitCommandEncoder];
        [blit copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(cw,ch,1) toBuffer:colorBuffer destinationOffset:0 destinationBytesPerRow:cs destinationBytesPerImage:cs*ch];
        [blit endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        printf("%s\n",SMOShadersStatus());
        if (cb.error) return 3;
        if (argc>3) [[NSData dataWithBytes:colorBuffer.contents length:colorBuffer.length] writeToFile:@(argv[3]) atomically:YES];
        return 0;
    }
}
#else
__attribute__((constructor)) static void Install(void) {
    @autoreleasepool {
        Dl_info info;
        dladdr((void *)Install,&info);
        directory=[@(info.dli_fname) stringByDeletingLastPathComponent];
        lock=[NSLock new];
        logFile=fopen([[directory stringByAppendingPathComponent:@"cinematic.log"] fileSystemRepresentation],"a");
        void (*setProcessor)(FrameProcessor)=dlsym(RTLD_DEFAULT,"SMOSetFrameProcessor");
        if (!setProcessor) { Log(@"Metal bridge is not loaded."); return; }
        if (!LoadProgram()) return;
        setProcessor(ProcessFrame);
        timer=dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER,0,0,dispatch_get_global_queue(QOS_CLASS_UTILITY,0));
        dispatch_source_set_timer(timer,dispatch_time(DISPATCH_TIME_NOW,NSEC_PER_SEC),NSEC_PER_SEC,NSEC_PER_SEC/4);
        dispatch_source_set_event_handler(timer,^{
            NSDictionary *settings=ReadSettings();
            [lock lock]; UpdateSettings(program,settings); [lock unlock];
            dispatch_async(dispatch_get_main_queue(),^{
                BOOL isOdyssey=NO;
                for (NSWindow *window in NSApp.windows) {
                    if ([window.title rangeOfString:@"SUPER MARIO ODYSSEY" options:NSCaseInsensitiveSearch].location != NSNotFound) { isOdyssey=YES; break; }
                }
                [lock lock]; gameAllowed=isOdyssey; [lock unlock];
            });
        });
        dispatch_resume(timer);
    }
}
#endif

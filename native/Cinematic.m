#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <AppKit/AppKit.h>
#import <simd/simd.h>
#include <dlfcn.h>
#include <math.h>
#include <stddef.h>

typedef struct {
    simd_float2 texel, depthTexel;
    float aspect, tanHalfFov, reflections, occlusion, bloom, exposure;
    float saturation, contrast, debugView, depthAvailable;
    simd_float2 direction;
    float debugMip, hierarchyLevels;
} ShaderSettings;
_Static_assert(sizeof(ShaderSettings)==72 && offsetof(ShaderSettings,debugMip)==64,
               "ShaderSettings must match the Metal constant buffer.");
typedef void (*FrameProcessor)(id<MTLCommandBuffer>, id<MTLTexture>, id<MTLTexture>);

@interface ShaderFrame : NSObject
@property id<MTLTexture> color, effects, reflections, bloomA, bloomB, depthHierarchy;
@property NSArray<id<MTLTexture>> *hierarchyMips;
@property BOOL busy;
@end
@implementation ShaderFrame
@end

@interface ShaderProgram : NSObject
@property id<MTLDevice> device;
@property id<MTLLibrary> library;
@property id<MTLRenderPipelineState> surface, extract, blur, composite;
@property id<MTLComputePipelineState> hierarchyCopy, hierarchyReduce;
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
static double presentCBMilliseconds;
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
    s.debugView=Number(d,@"debugView",0,0,9);
    s.debugMip=Number(d,@"debugMip",0,0,6);
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
    if ([fragment isEqualToString:@"surfaceEffects"]) desc.colorAttachments[1].pixelFormat=MTLPixelFormatRGBA16Float;
    return [p.device newRenderPipelineStateWithDescriptor:desc error:error];
}
static id<MTLComputePipelineState> ComputePipeline(ShaderProgram *p, NSString *kernel, NSError **error) {
    id<MTLFunction> function=[p.library newFunctionWithName:kernel];
    if (!function) {
        if (error) *error=[NSError errorWithDomain:@"SMOCinematic" code:1 userInfo:
            @{NSLocalizedDescriptionKey:[@"Missing Metal kernel: " stringByAppendingString:kernel]}];
        return nil;
    }
    MTLComputePipelineDescriptor *desc=[MTLComputePipelineDescriptor new];
    desc.label=[@"SMO Cinematic / " stringByAppendingString:kernel];
    desc.computeFunction=function;
    return [p.device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:error];
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
        if (p.surface) p.extract=Pipeline(p,@"bloomExtract",&error);
        if (p.extract) p.blur=Pipeline(p,@"bloomBlur",&error);
        if (p.blur) p.composite=Pipeline(p,@"composite",&error);
        if (p.composite) p.hierarchyCopy=ComputePipeline(p,@"depthHierarchyCopy",&error);
        if (p.hierarchyCopy) p.hierarchyReduce=ComputePipeline(p,@"depthHierarchyReduce",&error);
    }
    if (!p.surface || !p.extract || !p.blur || !p.composite || !p.hierarchyCopy || !p.hierarchyReduce) {
        [lock lock];
        lastError=error.description ?: @"Shader pipeline creation failed";
        Log(lastError);
        [lock unlock];
        return NO;
    }
    p.frames=[NSMutableArray new];
    UpdateSettings(p,ReadSettings());
    [lock lock]; program=p; [lock unlock];
    Log(@"Metal shader compilation passed. Four render pipelines and two compute pipelines are ready.");
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
static NSUInteger HierarchyLevels(NSUInteger width, NSUInteger height) {
    NSUInteger levels=1;
    for (NSUInteger dimension=MAX(width,height); dimension>1 && levels<7; dimension>>=1) levels++;
    return levels;
}
static BOOL AllocateHierarchy(ShaderFrame *frame, id<MTLDevice> device, NSUInteger width, NSUInteger height) {
    MTLTextureDescriptor *desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
        width:width height:height mipmapped:YES];
    desc.mipmapLevelCount=HierarchyLevels(width,height);
    desc.storageMode=MTLStorageModePrivate;
    desc.hazardTrackingMode=MTLHazardTrackingModeTracked;
    desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    id<MTLTexture> hierarchy=[device newTextureWithDescriptor:desc];
    if (!hierarchy) return NO;
    hierarchy.label=@"SMO closest-depth hierarchy";
    NSMutableArray<id<MTLTexture>> *views=[NSMutableArray arrayWithCapacity:desc.mipmapLevelCount];
    for (NSUInteger level=0; level<desc.mipmapLevelCount; level++) {
        id<MTLTexture> view=[hierarchy newTextureViewWithPixelFormat:MTLPixelFormatR32Float textureType:MTLTextureType2D
            levels:NSMakeRange(level,1) slices:NSMakeRange(0,1)];
        if (!view) return NO;
        view.label=[NSString stringWithFormat:@"SMO closest depth / mip %lu",(unsigned long)level];
        [views addObject:view];
    }
    // Each frame slot owns its hierarchy. Other slots can remain in use by the GPU.
    frame.depthHierarchy=hierarchy;
    frame.hierarchyMips=views;
    Log([NSString stringWithFormat:@"Hi-Z allocation: %lux%lu R32Float, %lu mip levels, private storage",
        (unsigned long)width,(unsigned long)height,(unsigned long)desc.mipmapLevelCount]);
    return YES;
}
static BOOL BuildHierarchy(id<MTLCommandBuffer> cb, ShaderProgram *p, ShaderFrame *frame, id<MTLTexture> depth) {
    for (NSUInteger level=0; level<frame.hierarchyMips.count; level++) {
        id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        if (!enc) return NO;
        id<MTLComputePipelineState> pipeline=level==0 ? p.hierarchyCopy : p.hierarchyReduce;
        id<MTLTexture> output=frame.hierarchyMips[level];
        enc.label=output.label;
        [enc setComputePipelineState:pipeline];
        [enc setTexture:level==0 ? depth : frame.hierarchyMips[level-1] atIndex:0];
        [enc setTexture:output atIndex:1];
        [enc dispatchThreads:MTLSizeMake(output.width,output.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
        // Serial encoders and tracked views order each reduction after its source mip.
        [enc endEncoding];
    }
    return YES;
}
static BOOL UsableDepth(id<MTLTexture> depth, id<MTLDevice> device) {
    if (!depth || !depth.width || !depth.height || depth.textureType!=MTLTextureType2D || depth.sampleCount!=1 ||
        depth.device.registryID!=device.registryID || depth.storageMode==MTLStorageModeMemoryless) return NO;
    if (depth.usage!=MTLTextureUsageUnknown && !(depth.usage&MTLTextureUsageShaderRead)) return NO;
    switch (depth.pixelFormat) {
        case MTLPixelFormatDepth16Unorm:
        case MTLPixelFormatDepth32Float:
        case MTLPixelFormatDepth24Unorm_Stencil8:
        case MTLPixelFormatDepth32Float_Stencil8:
            return YES;
        default:
            return NO;
    }
}
static void EncodingFailed(ShaderProgram *p, NSString *message) {
    [lock lock]; p.failed=YES; errorCount++; lastError=message; Log(message); [lock unlock];
}
static BOOL Draw(id<MTLCommandBuffer> cb, id<MTLTexture> target, id<MTLTexture> reflectionTarget, id<MTLRenderPipelineState> pipeline,
                 NSArray<id<MTLTexture>> *textures, ShaderSettings settings) {
    MTLRenderPassDescriptor *desc=[MTLRenderPassDescriptor renderPassDescriptor];
    desc.colorAttachments[0].texture=target;
    desc.colorAttachments[0].loadAction=MTLLoadActionDontCare;
    desc.colorAttachments[0].storeAction=MTLStoreActionStore;
    if (reflectionTarget) {
        desc.colorAttachments[1].texture=reflectionTarget;
        desc.colorAttachments[1].loadAction=MTLLoadActionDontCare;
        desc.colorAttachments[1].storeAction=MTLStoreActionStore;
    }
    id<MTLRenderCommandEncoder> enc=[cb renderCommandEncoderWithDescriptor:desc];
    if (!enc) return NO;
    enc.label=pipeline.label;
    [enc setRenderPipelineState:pipeline];
    for (NSUInteger i=0;i<textures.count;i++) [enc setFragmentTexture:textures[i] atIndex:i];
    [enc setFragmentBytes:&settings length:sizeof(settings) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    return YES;
}
static void ProcessFrame(id<MTLCommandBuffer> cb, id<MTLTexture> color, id<MTLTexture> depth) {
    @autoreleasepool {
        if (!cb || color.pixelFormat!=MTLPixelFormatRGBA16Float || color.width<640 || !color.height ||
            color.textureType!=MTLTextureType2D || color.sampleCount!=1 || color.framebufferOnly) return;
        [lock lock];
        ShaderProgram *p=program;
        if (!p || !p.enabled || p.failed || !gameAllowed) { [lock unlock]; return; }
        if (color.device.registryID!=p.device.registryID || cb.device.registryID!=p.device.registryID) {
            skippedFrames++; [lock unlock]; return;
        }

        // Do not drive SSR/AO from old geometry. The previous implementation
        // reused the last depth texture for up to seven missing presents, which
        // pairs current color with stale geometry and looks exactly like motion
        // ghosting during camera movement. We still bind the last texture so the
        // Metal pipeline remains valid, but depthAvailable=0 disables all
        // depth-driven effects on frames where the bridge did not provide fresh
        // depth. Bloom and color grading can continue normally.
        BOOL freshDepth=UsableDepth(depth,p.device);
        if (freshDepth) {
            lastDepth=depth;
        } else {
            missingDepthFrames++;
            depth=lastDepth;
        }
        if (!UsableDepth(depth,p.device)) { skippedFrames++; [lock unlock]; return; }

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
            frame.effects=Texture(p.device,ew,eh,@"SMO bounce and contact shadows");
            frame.reflections=Texture(p.device,ew,eh,@"SMO reflection radiance and confidence");
            frame.bloomA=Texture(p.device,bw,bh,@"SMO bloom A");
            frame.bloomB=Texture(p.device,bw,bh,@"SMO bloom B");
        }
        // Depth can resize without a change to output size or the effect-width limit.
        BOOL hierarchyReady=YES;
        if (frame.depthHierarchy.width!=depth.width || frame.depthHierarchy.height!=depth.height) {
            hierarchyReady=AllocateHierarchy(frame,p.device,depth.width,depth.height);
        }
        if (!frame.color || !frame.effects || !frame.reflections || !frame.bloomA || !frame.bloomB ||
            !hierarchyReady || !frame.depthHierarchy || frame.hierarchyMips.count!=frame.depthHierarchy.mipmapLevelCount) {
            [lock lock]; frame.busy=NO; p.failed=YES; lastError=@"Metal texture allocation failed"; [lock unlock];
            Log(lastError); return;
        }
        settings.hierarchyLevels=freshDepth ? (float)frame.depthHierarchy.mipmapLevelCount : 0.f;
        // Keep the slot reserved even if an encoder fails after commands were encoded.
        [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
            [lock lock];
            frame.busy=NO;
            frameCount++;
            double ms=(done.GPUEndTime-done.GPUStartTime)*1000;
            presentCBMilliseconds=frameCount==1 ? ms : presentCBMilliseconds*.95+ms*.05;
            if (done.error) { errorCount++; p.failed=YES; lastError=done.error.description; Log(lastError); }
            // This duration covers the complete presentation command buffer, including guest work.
            if (frameCount==1 || frameCount%1800==0) Log([NSString stringWithFormat:@"frames=%llu presentCBms=%.3f errors=%llu skipped=%llu missingDepth=%llu",frameCount,presentCBMilliseconds,errorCount,skippedFrames,missingDepthFrames]);
            [lock unlock];
        }];
        id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
        if (!blit) { EncodingFailed(p,@"Metal color-copy encoder creation failed"); return; }
        blit.label=@"SMO HDR scene copy";
        [blit copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(color.width,color.height,1) toTexture:frame.color destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
        [blit endEncoding];
        if (freshDepth && !BuildHierarchy(cb,p,frame,depth)) {
            EncodingFailed(p,@"Metal depth-hierarchy encoder creation failed"); return;
        }
        if (!Draw(cb,frame.effects,frame.reflections,p.surface,@[frame.color,depth,frame.depthHierarchy],settings) ||
            !Draw(cb,frame.bloomA,nil,p.extract,@[frame.color],settings)) {
            EncodingFailed(p,@"Metal surface or bloom encoder creation failed"); return;
        }
        settings.direction=(simd_float2){2.f/bw,0};
        if (!Draw(cb,frame.bloomB,nil,p.blur,@[frame.bloomA],settings)) {
            EncodingFailed(p,@"Metal horizontal bloom encoder creation failed"); return;
        }
        settings.direction=(simd_float2){0,2.f/bh};
        if (!Draw(cb,frame.bloomA,nil,p.blur,@[frame.bloomB],settings) ||
            !Draw(cb,color,nil,p.composite,@[frame.color,frame.effects,frame.bloomA,frame.reflections],settings)) {
            EncodingFailed(p,@"Metal vertical bloom or composite encoder creation failed"); return;
        }
    }
}

__attribute__((visibility("default"))) const char *SMOShadersStatus(void) {
    static char status[2048];
    [lock lock];
    snprintf(status,sizeof(status),"enabled=%d frames=%llu presentCBms=%.3f errors=%llu skipped=%llu missingDepth=%llu lastError=%s",program.enabled && !program.failed && gameAllowed,frameCount,presentCBMilliseconds,errorCount,skippedFrames,missingDepthFrames,lastError.UTF8String);
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

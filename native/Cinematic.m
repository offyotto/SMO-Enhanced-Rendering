#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <AppKit/AppKit.h>
#import <simd/simd.h>
#include <dlfcn.h>
#include <math.h>
#include <stddef.h>
#include <sys/stat.h>

typedef struct {
    simd_float2 texel, depthTexel;
    float aspect, tanHalfFov, reflections, occlusion, bloom, exposure;
    float saturation, contrast, debugView, depthAvailable;
    simd_float2 direction;
    float debugMip, hierarchyLevels, surfaceMode, reserved;
} ShaderSettings;
// Cinematic.metal carries the same layout number.
#define SETTINGS_LAYOUT @"// settings-layout: 3"
_Static_assert(sizeof(ShaderSettings)==80 && offsetof(ShaderSettings,debugMip)==64,
               "ShaderSettings must match the Metal constant buffer.");
typedef void (*FrameProcessor)(id<MTLCommandBuffer>, id<MTLTexture>, id<MTLTexture>);

@interface ShaderFrame : NSObject
@property id<MTLTexture> low, surface, effects, bloomA, bloomB, depthHierarchy;
@property NSArray<id<MTLTexture>> *hierarchyMips;
@property BOOL busy;
@end
@implementation ShaderFrame
@end

@interface ShaderProgram : NSObject
@property id<MTLDevice> device;
@property id<MTLLibrary> library;
@property id<MTLRenderPipelineState> surface, resolve, extract, blur, composite;
@property id<MTLComputePipelineState> hierarchyCopy, hierarchyShrink, hierarchyReduce;
@property NSMutableArray<ShaderFrame *> *frames;
@property ShaderSettings settings;
@property NSUInteger effectWidth, depthDelay;
@property BOOL enabled, adaptivePower, halfResolution;
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
// The slot whose hierarchy the previous present built, for depthDelay.
static ShaderFrame *heldFrame;
static BOOL captureRequested;
static double presentCBMilliseconds;
static NSString *lastError = @"none";
static FILE *logFile;
static BOOL gameAllowed=YES, reducedPower;
static struct timespec settingsTime;
static off_t settingsSize=-1;
static ino_t settingsFile;

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
    p.depthDelay=(NSUInteger)Number(d,@"depthDelay",1,0,1);
    p.halfResolution=d[@"halfResolution"] ? [d[@"halfResolution"] boolValue] : YES;
    p.adaptivePower=d[@"adaptivePower"] ? [d[@"adaptivePower"] boolValue] : YES;
    p.enabled=d[@"enabled"] ? [d[@"enabled"] boolValue] : YES;
}
static NSDictionary *ReadSettings(void) {
    NSData *data=[NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"preset.json"]];
    id value=data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    return [value isKindOfClass:NSDictionary.class] ? value : @{};
}
// The preset is polled once per second. Parse it only after it changes.
static BOOL SettingsChanged(void) {
    struct stat info;
    if (stat([[directory stringByAppendingPathComponent:@"preset.json"] fileSystemRepresentation],&info)) {
        // A removed preset restores the defaults once.
        BOOL existed=settingsSize!=-1;
        settingsSize=-1;
        return existed;
    }
    if (info.st_size==settingsSize && info.st_ino==settingsFile && info.st_mtimespec.tv_sec==settingsTime.tv_sec &&
        info.st_mtimespec.tv_nsec==settingsTime.tv_nsec) return NO;
    settingsSize=info.st_size; settingsFile=info.st_ino; settingsTime=info.st_mtimespec;
    return YES;
}
static id<MTLRenderPipelineState> Pipeline(ShaderProgram *p, NSString *fragment, NSError **error) {
    MTLRenderPipelineDescriptor *desc=[MTLRenderPipelineDescriptor new];
    desc.label=[@"SMO Cinematic / " stringByAppendingString:fragment];
    desc.vertexFunction=[p.library newFunctionWithName:@"fullScreen"];
    desc.fragmentFunction=[p.library newFunctionWithName:fragment];
    desc.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA16Float;
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
    if (source && ![source containsString:SETTINGS_LAYOUT]) {
        [lock lock];
        lastError=@"Cinematic.metal does not match this libSMOCinematic.dylib. Install both files from the same version.";
        Log(lastError);
        [lock unlock];
        return NO;
    }
    if (source) {
        MTLCompileOptions *options=[MTLCompileOptions new];
        p.library=[p.device newLibraryWithSource:source options:options error:&error];
    }
    if (p.library) {
        p.surface=Pipeline(p,@"surfaceEffects",&error);
        if (p.surface) p.resolve=Pipeline(p,@"resolveEffects",&error);
        if (p.resolve) p.extract=Pipeline(p,@"bloomExtract",&error);
        if (p.extract) p.blur=Pipeline(p,@"bloomBlur",&error);
        if (p.blur) p.composite=Pipeline(p,@"composite",&error);
        if (p.composite) p.hierarchyCopy=ComputePipeline(p,@"depthHierarchyCopy",&error);
        if (p.hierarchyCopy) p.hierarchyShrink=ComputePipeline(p,@"depthHierarchyShrink",&error);
        if (p.hierarchyShrink) p.hierarchyReduce=ComputePipeline(p,@"depthHierarchyReduce",&error);
    }
    if (!p.surface || !p.resolve || !p.extract || !p.blur || !p.composite || !p.hierarchyCopy || !p.hierarchyShrink || !p.hierarchyReduce) {
        [lock lock];
        lastError=error.description ?: @"Shader pipeline creation failed";
        Log(lastError);
        [lock unlock];
        return NO;
    }
    p.frames=[NSMutableArray new];
    UpdateSettings(p,ReadSettings());
    [lock lock]; program=p; [lock unlock];
    Log(@"Metal shader compilation passed. Five render pipelines and three compute pipelines are ready.");
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
        id<MTLTexture> output=frame.hierarchyMips[level];
        id<MTLComputePipelineState> pipeline=level>0 ? p.hierarchyReduce :
            (output.width==depth.width && output.height==depth.height ? p.hierarchyCopy : p.hierarchyShrink);
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
static BOOL Draw(id<MTLCommandBuffer> cb, id<MTLTexture> target, MTLLoadAction load, id<MTLRenderPipelineState> pipeline,
                 id<MTLTexture> first, id<MTLTexture> second, id<MTLTexture> third, const ShaderSettings *settings) {
    MTLRenderPassDescriptor *desc=[MTLRenderPassDescriptor renderPassDescriptor];
    desc.colorAttachments[0].texture=target;
    desc.colorAttachments[0].loadAction=load;
    desc.colorAttachments[0].storeAction=MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc=[cb renderCommandEncoderWithDescriptor:desc];
    if (!enc) return NO;
    enc.label=pipeline.label;
    [enc setRenderPipelineState:pipeline];
    if (first) [enc setFragmentTexture:first atIndex:0];
    if (second) [enc setFragmentTexture:second atIndex:1];
    if (third) [enc setFragmentTexture:third atIndex:2];
    [enc setFragmentBytes:settings length:sizeof(*settings) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    return YES;
}
// Writes the unprocessed color, this present's depth, and the previous
// present's depth to align-capture/. The files show which depth matches the
// presented color. Create the file capture-request beside the preset to arm it.
static void EncodeCapture(id<MTLCommandBuffer> cb, id<MTLTexture> color, id<MTLTexture> depth, id<MTLTexture> previous) {
    NSUInteger cs=(color.width*8+255)&~(NSUInteger)255, ds=(depth.width*4+255)&~(NSUInteger)255;
    id<MTLBuffer> a=[color.device newBufferWithLength:cs*color.height options:MTLResourceStorageModeShared];
    id<MTLBuffer> b=[color.device newBufferWithLength:ds*depth.height options:MTLResourceStorageModeShared];
    id<MTLBuffer> c=previous ? [color.device newBufferWithLength:ds*depth.height options:MTLResourceStorageModeShared] : nil;
    if (!a || !b) return;
    id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
    if (!blit) return;
    MTLOrigin origin=MTLOriginMake(0,0,0);
    MTLSize size=MTLSizeMake(depth.width,depth.height,1);
    [blit copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:origin sourceSize:MTLSizeMake(color.width,color.height,1) toBuffer:a destinationOffset:0 destinationBytesPerRow:cs destinationBytesPerImage:cs*color.height];
    [blit copyFromTexture:depth sourceSlice:0 sourceLevel:0 sourceOrigin:origin sourceSize:size toBuffer:b destinationOffset:0 destinationBytesPerRow:ds destinationBytesPerImage:ds*depth.height];
    if (c) [blit copyFromTexture:previous sourceSlice:0 sourceLevel:0 sourceOrigin:origin sourceSize:size toBuffer:c destinationOffset:0 destinationBytesPerRow:ds destinationBytesPerImage:ds*depth.height];
    [blit endEncoding];
    NSDictionary *info=@{@"colorWidth":@(color.width),@"colorHeight":@(color.height),@"colorStride":@(cs),
        @"depthWidth":@(depth.width),@"depthHeight":@(depth.height),@"depthStride":@(ds),@"previousDepth":@(c!=nil)};
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        if (done.status!=MTLCommandBufferStatusCompleted) { Log(@"Alignment capture failed; nothing written."); return; }
        NSString *folder=[directory stringByAppendingPathComponent:@"align-capture"];
        [[NSFileManager defaultManager] createDirectoryAtPath:folder withIntermediateDirectories:YES attributes:nil error:nil];
        [[NSData dataWithBytesNoCopy:a.contents length:a.length freeWhenDone:NO] writeToFile:[folder stringByAppendingPathComponent:@"color.rgba16f"] atomically:YES];
        [[NSData dataWithBytesNoCopy:b.contents length:b.length freeWhenDone:NO] writeToFile:[folder stringByAppendingPathComponent:@"depth.r32f"] atomically:YES];
        if (c) [[NSData dataWithBytesNoCopy:c.contents length:c.length freeWhenDone:NO] writeToFile:[folder stringByAppendingPathComponent:@"depth-previous.r32f"] atomically:YES];
        [[NSJSONSerialization dataWithJSONObject:info options:NSJSONWritingPrettyPrinted error:nil] writeToFile:[folder stringByAppendingPathComponent:@"capture.json"] atomically:YES];
        Log(@"Alignment capture written to align-capture.");
    }];
}
// Arms a capture when the request file exists, and removes the file.
static void CheckCaptureRequest(void) {
    NSString *request=[directory stringByAppendingPathComponent:@"capture-request"];
    NSFileManager *files=[NSFileManager defaultManager];
    BOOL folder=NO;
    if (![files fileExistsAtPath:request isDirectory:&folder] || folder || ![files removeItemAtPath:request error:nil]) return;
    [lock lock]; captureRequested=YES; [lock unlock];
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
        // Mip 0 of the hierarchy is the only depth the shaders read. Halve a
        // guest depth buffer wider than 1920, so the trace and the hierarchy
        // build cost the same at every guest resolution scale.
        NSUInteger hw=depth.width, hh=depth.height;
        while (hw>1920) { hw=(hw+1)/2; hh=(hh+1)/2; }
        settings.depthTexel=(simd_float2){1.f/hw,1.f/hh};
        settings.aspect=(float)depth.width/depth.height;
        settings.depthAvailable=freshDepth ? 1.f : 0.f;
        NSUInteger effectWidth=p.effectWidth;
        // Low Power Mode and thermal pressure shrink the effect buffers.
        if (reducedPower && p.adaptivePower) effectWidth=MAX(480,effectWidth*3/4);
        NSUInteger ew=MIN(effectWidth,depth.width), eh=MAX(1,(NSUInteger)round(ew/settings.aspect));
        NSUInteger bw=MAX(1,ew/2), bh=MAX(1,eh/2);
        ShaderFrame *frame=nil;
        ShaderFrame *previous=heldFrame;
        heldFrame=nil;
        BOOL capture=captureRequested && freshDepth;
        if (capture) captureRequested=NO;
        for (ShaderFrame *candidate in p.frames) if (!candidate.busy && candidate!=previous) { frame=candidate; break; }
        if (!frame && p.frames.count<5) { frame=[ShaderFrame new]; [p.frames addObject:frame]; }
        if (!frame) { skippedFrames++; [lock unlock]; return; }
        frame.busy=YES;
        [lock unlock];
        if (frame.effects.width!=ew || frame.effects.height!=eh) {
            frame.low=Texture(p.device,MAX(1,ew/2),MAX(1,eh/2),@"SMO half-size surface effects");
            frame.surface=Texture(p.device,ew,eh,@"SMO surface effects");
            frame.effects=Texture(p.device,ew,eh,@"SMO resolved effects");
            frame.bloomA=Texture(p.device,bw,bh,@"SMO bloom A");
            frame.bloomB=Texture(p.device,bw,bh,@"SMO bloom B");
        }
        // Depth can resize without a change to output size or the effect-width limit.
        BOOL hierarchyReady=YES;
        if (frame.depthHierarchy.width!=hw || frame.depthHierarchy.height!=hh) {
            hierarchyReady=AllocateHierarchy(frame,p.device,hw,hh);
        }
        if (!frame.low || !frame.surface || !frame.effects || !frame.bloomA || !frame.bloomB ||
            !hierarchyReady || !frame.depthHierarchy || frame.hierarchyMips.count!=frame.depthHierarchy.mipmapLevelCount) {
            [lock lock]; frame.busy=NO; p.failed=YES; lastError=@"Metal texture allocation failed"; [lock unlock];
            Log(lastError); return;
        }
        if (freshDepth) { [lock lock]; heldFrame=frame; [lock unlock]; }
        // Live captures show that the guest depth at a present is one present
        // ahead of the presented color. depthDelay 1 therefore reads the
        // hierarchy that the previous present built, which matches the color.
        // That command buffer precedes this one on the queue.
        BOOL previousUsable=previous.depthHierarchy.width==hw && previous.depthHierarchy.height==hh;
        BOOL delayed=p.depthDelay>0;
        id<MTLTexture> hierarchy=delayed && previousUsable ? previous.depthHierarchy : frame.depthHierarchy;
        BOOL usable=delayed ? previousUsable : freshDepth;
        settings.depthAvailable=usable ? 1.f : 0.f;
        settings.hierarchyLevels=usable ? (float)hierarchy.mipmapLevelCount : 0.f;
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
        // The passes sample the drawable directly, and the composite reads it
        // in place, so no copy of the frame exists.
        BOOL debug=settings.debugView>.5f;
        if (freshDepth && !BuildHierarchy(cb,p,frame,depth)) {
            EncodingFailed(p,@"Metal depth-hierarchy encoder creation failed"); return;
        }
        if (capture) EncodeCapture(cb,color,frame.hierarchyMips[0],previousUsable ? previous.hierarchyMips[0] : nil);
        if (usable || debug) {
            BOOL drawn;
            if (p.halfResolution && !debug) {
                // Occlusion and rough reflections at half size, mirror-like
                // reflections at full size. The resolve pass joins them.
                settings.surfaceMode=1;
                drawn=Draw(cb,frame.low,MTLLoadActionDontCare,p.surface,color,hierarchy,nil,&settings);
                settings.surfaceMode=2;
                drawn=drawn && Draw(cb,frame.surface,MTLLoadActionDontCare,p.surface,color,hierarchy,nil,&settings);
                settings.surfaceMode=1;
                settings.direction=(simd_float2){1.f/frame.low.width,1.f/frame.low.height};
                drawn=drawn && Draw(cb,frame.effects,MTLLoadActionDontCare,p.resolve,frame.low,hierarchy,frame.surface,&settings);
            } else {
                settings.surfaceMode=0;
                settings.direction=(simd_float2){1.f/ew,1.f/eh};
                drawn=Draw(cb,frame.surface,MTLLoadActionDontCare,p.surface,color,hierarchy,nil,&settings) &&
                    Draw(cb,frame.effects,MTLLoadActionDontCare,p.resolve,frame.surface,hierarchy,frame.surface,&settings);
            }
            if (!drawn) { EncodingFailed(p,@"Metal surface encoder creation failed"); return; }
        }
        if (settings.bloom>0 && !debug) {
            if (!Draw(cb,frame.bloomA,MTLLoadActionDontCare,p.extract,color,nil,nil,&settings)) {
                EncodingFailed(p,@"Metal bloom encoder creation failed"); return;
            }
            settings.direction=(simd_float2){2.f/bw,0};
            if (!Draw(cb,frame.bloomB,MTLLoadActionDontCare,p.blur,frame.bloomA,nil,nil,&settings)) {
                EncodingFailed(p,@"Metal horizontal bloom encoder creation failed"); return;
            }
            settings.direction=(simd_float2){0,2.f/bh};
            if (!Draw(cb,frame.bloomA,MTLLoadActionDontCare,p.blur,frame.bloomB,nil,nil,&settings)) {
                EncodingFailed(p,@"Metal vertical bloom encoder creation failed"); return;
            }
        }
        // The composite samples the effects only on a frame with usable depth.
        settings.depthAvailable=usable || debug ? 1.f : 0.f;
        if (!Draw(cb,color,MTLLoadActionLoad,p.composite,frame.effects,frame.bloomA,nil,&settings)) {
            EncodingFailed(p,@"Metal composite encoder creation failed"); return;
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
        // With depthDelay 1 the first present only builds the hierarchy, so
        // process the frame once and upload the color again.
        ProcessFrame(cb,color,depth);
        blit=[cb blitCommandEncoder];
        [blit copyFromBuffer:colorBuffer sourceOffset:0 sourceBytesPerRow:cs sourceBytesPerImage:cs*ch sourceSize:MTLSizeMake(cw,ch,1) toTexture:color destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
        [blit endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        cb=[queue commandBuffer];
        ProcessFrame(cb,color,depth);
        blit=[cb blitCommandEncoder];
        [blit copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(cw,ch,1) toBuffer:colorBuffer destinationOffset:0 destinationBytesPerRow:cs destinationBytesPerImage:cs*ch];
        [blit endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        printf("%s\n",SMOShadersStatus());
        if (cb.error) return 3;
        if (argc>3) [[NSData dataWithBytes:colorBuffer.contents length:colorBuffer.length] writeToFile:@(argv[3]) atomically:YES];
        // Optional fourth argument: frame count for a GPU time measurement.
        // Each frame restores the captured color in an untimed command buffer.
        int frames=argc>4 ? atoi(argv[4]) : 0;
        if (frames>0) {
            id<MTLBuffer> pristine=[device newBufferWithBytes:colorData.bytes length:colorData.length options:MTLResourceStorageModeShared];
            double total=0, best=1e9;
            for (int i=0;i<frames;i++) {
                if (i==frames/2) CheckCaptureRequest();
                cb=[queue commandBuffer];
                blit=[cb blitCommandEncoder];
                [blit copyFromBuffer:pristine sourceOffset:0 sourceBytesPerRow:cs sourceBytesPerImage:cs*ch sourceSize:MTLSizeMake(cw,ch,1) toTexture:color destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
                [blit endEncoding];
                [cb commit]; [cb waitUntilCompleted];
                cb=[queue commandBuffer];
                ProcessFrame(cb,color,depth);
                [cb commit]; [cb waitUntilCompleted];
                if (cb.error) return 3;
                double ms=(cb.GPUEndTime-cb.GPUStartTime)*1000;
                total+=ms; if (ms<best) best=ms;
            }
            printf("benchmark frames=%d averageGpuMs=%.3f bestGpuMs=%.3f\n",frames,total/frames,best);
        }
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
        SettingsChanged();
        setProcessor(ProcessFrame);
        timer=dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER,0,0,dispatch_get_global_queue(QOS_CLASS_UTILITY,0));
        dispatch_source_set_timer(timer,dispatch_time(DISPATCH_TIME_NOW,NSEC_PER_SEC),NSEC_PER_SEC,NSEC_PER_SEC/2);
        dispatch_source_set_event_handler(timer,^{
            NSProcessInfo *process=NSProcessInfo.processInfo;
            BOOL reduced=process.thermalState>=NSProcessInfoThermalStateSerious || process.lowPowerModeEnabled;
            NSDictionary *settings=SettingsChanged() ? ReadSettings() : nil;
            CheckCaptureRequest();
            [lock lock];
            reducedPower=reduced;
            if (settings) UpdateSettings(program,settings);
            [lock unlock];
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

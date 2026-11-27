#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <AppKit/AppKit.h>
#import <simd/simd.h>
#include <dlfcn.h>
#include <math.h>
#include <sys/stat.h>

// Must match Settings in Cinematic.metal and its "settings-layout" marker.
#define SETTINGS_LAYOUT @"// settings-layout: 2"
typedef struct {
    simd_float2 texel, effectTexel, sourceTexel;
    float aspect, tanHalfFov, reflections, occlusion, bloom, exposure;
    float saturation, contrast, debugView, effectsValid;
    float shadows, sunX, sunY, sunZ, taps, blend;
} ShaderSettings;
typedef void (*FrameProcessor)(id<MTLCommandBuffer>, id<MTLTexture>, id<MTLTexture>);

enum { PassDepth, PassScene, PassSurface, PassResolve, PassExtract, PassDown1, PassDown2, PassUp1, PassUp0, PassComposite, PassCount };

// One set of intermediate targets. Command buffers on a queue execute in
// order, so consecutive frames can share it.
@interface ShaderTargets : NSObject {
@public
    id<MTLTexture> depth, scene, surface, effects, bloom[3];
    MTLRenderPassDescriptor *pass[PassCount];
    NSUInteger outputWidth, outputHeight, effectWidth, effectHeight;
    BOOL effectsReady;
}
@end
@implementation ShaderTargets
@end

@interface ShaderProgram : NSObject
@property id<MTLDevice> device;
@property id<MTLLibrary> library;
@property id<MTLRenderPipelineState> depth, scene, surface, resolve, extract, down, up, composite;
@property ShaderTargets *targets;
@property ShaderSettings settings;
@property NSUInteger effectWidth, effectInterval;
@property BOOL enabled, adaptivePower, failed;
@end
@implementation ShaderProgram
@end

static NSLock *lock, *encodeLock;
static ShaderProgram *program;
static NSString *directory;
static dispatch_source_t timer;
static id<MTLTexture> lastDepth;
static NSUInteger depthAge;
static uint64_t frameCount, encodedFrames, errorCount;
static double gpuMilliseconds;
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
    s.bloom=Number(d,@"bloom",.26,0,2);
    s.exposure=Number(d,@"exposure",1.07,.5,2);
    s.saturation=Number(d,@"saturation",1.10,0,2);
    s.contrast=Number(d,@"contrast",1.055,.5,1.5);
    s.tanHalfFov=tanf(Number(d,@"verticalFov",50,30,100)*M_PI/360);
    s.debugView=Number(d,@"debugView",0,0,4);
    s.shadows=Number(d,@"shadows",.85,0,2);
    float azimuth=Number(d,@"sunAzimuth",90,-360,360)*M_PI/180, elevation=Number(d,@"sunElevation",55,0,90)*M_PI/180;
    s.sunX=cosf(elevation)*sinf(azimuth); s.sunY=sinf(elevation); s.sunZ=-cosf(elevation)*cosf(azimuth);
    p.settings=s;
    p.effectWidth=(NSUInteger)Number(d,@"effectWidth",1280,480,1920);
    p.effectInterval=(NSUInteger)Number(d,@"effectInterval",1,1,4);
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
    if (info.st_size==settingsSize && info.st_ino==settingsFile && info.st_mtimespec.tv_sec==settingsTime.tv_sec && info.st_mtimespec.tv_nsec==settingsTime.tv_nsec) return NO;
    settingsSize=info.st_size; settingsFile=info.st_ino; settingsTime=info.st_mtimespec;
    return YES;
}
static id<MTLRenderPipelineState> Pipeline(ShaderProgram *p, NSString *fragment, MTLPixelFormat format, BOOL blend, NSError **error) {
    MTLRenderPipelineDescriptor *desc=[MTLRenderPipelineDescriptor new];
    desc.label=[@"SMO Cinematic / " stringByAppendingString:fragment];
    desc.vertexFunction=[p.library newFunctionWithName:@"fullScreen"];
    desc.fragmentFunction=[p.library newFunctionWithName:fragment];
    desc.colorAttachments[0].pixelFormat=format;
    if (blend) {
        desc.colorAttachments[0].blendingEnabled=YES;
        desc.colorAttachments[0].sourceRGBBlendFactor=MTLBlendFactorSourceAlpha;
        desc.colorAttachments[0].destinationRGBBlendFactor=MTLBlendFactorOneMinusSourceAlpha;
        desc.colorAttachments[0].sourceAlphaBlendFactor=MTLBlendFactorZero;
        desc.colorAttachments[0].destinationAlphaBlendFactor=MTLBlendFactorOne;
    }
    return [p.device newRenderPipelineStateWithDescriptor:desc error:error];
}
static BOOL LoadProgram(void) {
    NSError *error=nil;
    NSString *source=[NSString stringWithContentsOfFile:[directory stringByAppendingPathComponent:@"Cinematic.metal"] encoding:NSUTF8StringEncoding error:&error];
    ShaderProgram *p=[ShaderProgram new];
    p.device=MTLCreateSystemDefaultDevice();
    if (source && ![source containsString:SETTINGS_LAYOUT]) {
        lastError=@"Cinematic.metal does not match this libSMOCinematic.dylib. Install both files from the same version.";
        Log(lastError);
        return NO;
    }
    if (source) p.library=[p.device newLibraryWithSource:source options:[MTLCompileOptions new] error:&error];
    if (p.library) {
        MTLPixelFormat color=MTLPixelFormatRGBA16Float;
        p.depth=Pipeline(p,@"linearDepth",MTLPixelFormatR32Float,NO,&error);
        p.scene=Pipeline(p,@"sceneDown",color,NO,&error);
        p.surface=Pipeline(p,@"surfaceEffects",color,NO,&error);
        p.resolve=Pipeline(p,@"resolveEffects",color,NO,&error);
        p.extract=Pipeline(p,@"bloomExtract",color,NO,&error);
        p.down=Pipeline(p,@"bloomDown",color,NO,&error);
        p.up=Pipeline(p,@"bloomUp",color,YES,&error);
        p.composite=Pipeline(p,@"composite",color,NO,&error);
    }
    if (!p.depth || !p.scene || !p.surface || !p.resolve || !p.extract || !p.down || !p.up || !p.composite) {
        lastError=error.description ?: @"Shader pipeline creation failed";
        Log(lastError);
        return NO;
    }
    UpdateSettings(p,ReadSettings());
    [lock lock]; program=p; [lock unlock];
    Log(@"Metal shader compilation passed. Eight render pipelines are ready.");
    return YES;
}
static id<MTLTexture> Texture(id<MTLDevice> device, MTLPixelFormat format, NSUInteger w, NSUInteger h, NSString *name) {
    MTLTextureDescriptor *d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:MAX(1,w) height:MAX(1,h) mipmapped:NO];
    d.storageMode=MTLStorageModePrivate;
    d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;
    id<MTLTexture> t=[device newTextureWithDescriptor:d];
    t.label=name;
    return t;
}
static MTLRenderPassDescriptor *Pass(id<MTLTexture> target, MTLLoadAction load) {
    MTLRenderPassDescriptor *desc=[MTLRenderPassDescriptor renderPassDescriptor];
    desc.colorAttachments[0].texture=target;
    desc.colorAttachments[0].loadAction=load;
    desc.colorAttachments[0].storeAction=MTLStoreActionStore;
    return desc;
}
static ShaderTargets *MakeTargets(id<MTLDevice> device, NSUInteger cw, NSUInteger ch, NSUInteger ew, NSUInteger eh) {
    ShaderTargets *t=[ShaderTargets new];
    MTLPixelFormat color=MTLPixelFormatRGBA16Float;
    t->outputWidth=cw; t->outputHeight=ch; t->effectWidth=ew; t->effectHeight=eh;
    t->depth=Texture(device,MTLPixelFormatR32Float,ew,eh,@"SMO linear depth");
    t->scene=Texture(device,color,ew/2,eh/2,@"SMO scene half");
    t->surface=Texture(device,color,ew,eh,@"SMO surface effects");
    t->effects=Texture(device,color,ew,eh,@"SMO resolved effects");
    for (NSUInteger i=0;i<3;i++) t->bloom[i]=Texture(device,color,ew>>(i+2),eh>>(i+2),@"SMO bloom");
    if (!t->depth || !t->scene || !t->surface || !t->effects || !t->bloom[0] || !t->bloom[1] || !t->bloom[2]) return nil;
    t->pass[PassDepth]=Pass(t->depth,MTLLoadActionDontCare);
    t->pass[PassScene]=Pass(t->scene,MTLLoadActionDontCare);
    t->pass[PassSurface]=Pass(t->surface,MTLLoadActionDontCare);
    t->pass[PassResolve]=Pass(t->effects,MTLLoadActionDontCare);
    t->pass[PassExtract]=Pass(t->bloom[0],MTLLoadActionDontCare);
    t->pass[PassDown1]=Pass(t->bloom[1],MTLLoadActionDontCare);
    t->pass[PassDown2]=Pass(t->bloom[2],MTLLoadActionDontCare);
    t->pass[PassUp1]=Pass(t->bloom[1],MTLLoadActionLoad);
    t->pass[PassUp0]=Pass(t->bloom[0],MTLLoadActionLoad);
    t->pass[PassComposite]=Pass(nil,MTLLoadActionLoad);
    return t;
}
static simd_float2 Texel(id<MTLTexture> t) { return (simd_float2){1.f/t.width,1.f/t.height}; }
static void Draw(id<MTLCommandBuffer> cb, MTLRenderPassDescriptor *desc, id<MTLRenderPipelineState> pipeline,
                 id<MTLTexture> a, id<MTLTexture> b, id<MTLTexture> c, const ShaderSettings *settings) {
    id<MTLRenderCommandEncoder> enc=[cb renderCommandEncoderWithDescriptor:desc];
    enc.label=@"SMO Cinematic";
    [enc setRenderPipelineState:pipeline];
    if (a) [enc setFragmentTexture:a atIndex:0];
    if (b) [enc setFragmentTexture:b atIndex:1];
    if (c) [enc setFragmentTexture:c atIndex:2];
    [enc setFragmentBytes:settings length:sizeof(*settings) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
}
static void ProcessFrame(id<MTLCommandBuffer> cb, id<MTLTexture> color, id<MTLTexture> depth) {
    @autoreleasepool {
        if (color.pixelFormat != MTLPixelFormatRGBA16Float || color.width < 640) return;
        [lock lock];
        ShaderProgram *p=program;
        if (!p || !p.enabled || p.failed || !gameAllowed) { [lock unlock]; return; }
        if (depth) { lastDepth=depth; depthAge=0; } else { depth=lastDepth; depthAge++; }
        if (!depth) { [lock unlock]; return; }
        BOOL depthAvailable=depthAge<8;
        ShaderSettings s=p.settings;
        NSUInteger interval=p.effectInterval, width=p.effectWidth;
        // Low Power Mode and thermal pressure shrink the effect buffers.
        if (reducedPower && p.adaptivePower) width=MAX(480,width*3/4);
        uint64_t index=encodedFrames++;
        [lock unlock];

        [encodeLock lock];
        s.aspect=(float)depth.width/depth.height;
        NSUInteger ew=MIN(width,depth.width)&~(NSUInteger)15, eh=MAX(16,(NSUInteger)round(ew/s.aspect));
        ShaderTargets *t=p.targets;
        if (!t || t->outputWidth!=color.width || t->outputHeight!=color.height || t->effectWidth!=ew || t->effectHeight!=eh) {
            t=MakeTargets(p.device,color.width,color.height,ew,eh);
            if (!t) {
                [encodeLock unlock];
                [lock lock]; p.failed=YES; lastError=@"Metal texture allocation failed"; [lock unlock];
                Log(lastError); return;
            }
            p.targets=t;
        }
        s.texel=Texel(color);
        s.effectTexel=Texel(t->surface);

        // Half-effect-resolution scene color: an exact box filter of the output.
        float ratio=(float)color.width/t->scene.width;
        s.taps=fminf(4,fmaxf(1,ceilf(ratio/2)));
        s.sourceTexel=s.texel*(ratio/s.taps);
        Draw(cb,t->pass[PassScene],p.scene,color,nil,nil,&s);

        // The surface pass is the expensive one. A larger effectInterval
        // reuses its result for the frames in between.
        BOOL refresh=depthAvailable && (!t->effectsReady || index%interval==0 || s.debugView>.5f);
        if (refresh) {
            Draw(cb,t->pass[PassDepth],p.depth,depth,nil,nil,&s);
            Draw(cb,t->pass[PassSurface],p.surface,color,t->depth,t->scene,&s);
            Draw(cb,t->pass[PassResolve],p.resolve,t->surface,t->depth,nil,&s);
            t->effectsReady=YES;
        } else if (!depthAvailable) t->effectsReady=NO;
        s.effectsValid=t->effectsReady ? 1:0;

        if (s.bloom>.001f && s.debugView<.5f) {
            s.sourceTexel=Texel(t->scene);
            Draw(cb,t->pass[PassExtract],p.extract,t->scene,nil,nil,&s);
            s.sourceTexel=Texel(t->bloom[0]);
            Draw(cb,t->pass[PassDown1],p.down,t->bloom[0],nil,nil,&s);
            s.sourceTexel=Texel(t->bloom[1]);
            Draw(cb,t->pass[PassDown2],p.down,t->bloom[1],nil,nil,&s);
            s.blend=.6f;
            s.sourceTexel=Texel(t->bloom[2]);
            Draw(cb,t->pass[PassUp1],p.up,t->bloom[2],nil,nil,&s);
            s.blend=.7f;
            s.sourceTexel=Texel(t->bloom[1]);
            Draw(cb,t->pass[PassUp0],p.up,t->bloom[1],nil,nil,&s);
        } else s.bloom=0;

        // The composite reads the drawable in place, so no frame copy exists.
        t->pass[PassComposite].colorAttachments[0].texture=color;
        Draw(cb,t->pass[PassComposite],p.composite,t->effects,t->bloom[0],nil,&s);
        t->pass[PassComposite].colorAttachments[0].texture=nil;
        [encodeLock unlock];
        [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
            [lock lock];
            frameCount++;
            double ms=(done.GPUEndTime-done.GPUStartTime)*1000;
            gpuMilliseconds=frameCount==1 ? ms : gpuMilliseconds*.95+ms*.05;
            if (done.error) { errorCount++; p.failed=YES; lastError=done.error.description; Log(lastError); }
            if (frameCount==1 || frameCount%1800==0) Log([NSString stringWithFormat:@"frames=%llu gpuMs=%.3f errors=%llu effect=%lux%lu reducedPower=%d",frameCount,gpuMilliseconds,errorCount,ew,eh,reducedPower]);
            [lock unlock];
        }];
    }
}

__attribute__((visibility("default"))) const char *SMOShadersStatus(void) {
    static char status[2048];
    [lock lock];
    snprintf(status,sizeof(status),"enabled=%d frames=%llu gpuMs=%.3f errors=%llu reducedPower=%d lastError=%s",program.enabled && !program.failed && gameAllowed,frameCount,gpuMilliseconds,errorCount,reducedPower,lastError.UTF8String);
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
        lock=[NSLock new]; encodeLock=[NSLock new]; logFile=stdout;
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
        id<MTLTexture> color=Texture(device,MTLPixelFormatRGBA16Float,cw,ch,@"Test color");
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
        // Optional fourth argument: frame count for a GPU time measurement.
        int frames=argc>4 ? atoi(argv[4]) : 0;
        double total=0;
        for (int i=0;i<frames;i++) {
            cb=[queue commandBuffer];
            ProcessFrame(cb,color,depth);
            [cb commit]; [cb waitUntilCompleted];
            if (cb.error) return 3;
            total+=(cb.GPUEndTime-cb.GPUStartTime)*1000;
        }
        if (frames>0) printf("average gpuMs=%.3f over %d frames\n",total/frames,frames);
        return 0;
    }
}
#else
__attribute__((constructor)) static void Install(void) {
    @autoreleasepool {
        Dl_info info;
        dladdr((void *)Install,&info);
        directory=[@(info.dli_fname) stringByDeletingLastPathComponent];
        lock=[NSLock new]; encodeLock=[NSLock new];
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
            BOOL reduced=process.thermalState>=NSProcessInfoThermalStateSerious;
            if (@available(macOS 12.0,*)) reduced=reduced || process.lowPowerModeEnabled;
            NSDictionary *settings=SettingsChanged() ? ReadSettings() : nil;
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

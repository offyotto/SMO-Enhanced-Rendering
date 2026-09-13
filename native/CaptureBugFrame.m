#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <stdatomic.h>
typedef void (*Processor)(id<MTLCommandBuffer>,id<MTLTexture>,id<MTLTexture>);
static Processor previous;
static atomic_bool requested=false;
static void Capture(id<MTLCommandBuffer> cb,id<MTLTexture> color,id<MTLTexture> depth) {
    if (!depth || !atomic_exchange(&requested,false)) { if (previous) previous(cb,color,depth); return; }
    NSUInteger cs=(color.width*8+255)&~255, ds=(depth.width*4+255)&~255;
    id<MTLBuffer> a=[color.device newBufferWithLength:cs*color.height options:MTLResourceStorageModeShared];
    id<MTLBuffer> b=[color.device newBufferWithLength:ds*depth.height options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> enc=[cb blitCommandEncoder];
    [enc copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(color.width,color.height,1) toBuffer:a destinationOffset:0 destinationBytesPerRow:cs destinationBytesPerImage:cs*color.height];
    [enc copyFromTexture:depth sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(depth.width,depth.height,1) toBuffer:b destinationOffset:0 destinationBytesPerRow:ds destinationBytesPerImage:ds*depth.height];
    [enc endEncoding];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        NSString *dir=[NSHomeDirectory() stringByAppendingPathComponent:@"Documents/SMOShaders/bug-capture"];
        [[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
        [[NSData dataWithBytes:a.contents length:a.length] writeToFile:[dir stringByAppendingPathComponent:@"color.rgba16f"] atomically:YES];
        [[NSData dataWithBytes:b.contents length:b.length] writeToFile:[dir stringByAppendingPathComponent:@"depth.r32f"] atomically:YES];
        NSDictionary *info=@{@"colorWidth":@(color.width),@"colorHeight":@(color.height),@"colorStride":@(cs),@"depthWidth":@(depth.width),@"depthHeight":@(depth.height),@"depthStride":@(ds),@"status":@(done.status),@"error":done.error.description ?: @"none"};
        [[NSJSONSerialization dataWithJSONObject:info options:NSJSONWritingPrettyPrinted error:nil] writeToFile:[dir stringByAppendingPathComponent:@"capture.json"] atomically:YES];
    }];
    void (*set)(Processor)=dlsym(RTLD_DEFAULT,"SMOSetFrameProcessor");
    if (set) set(previous);
    if (previous) previous(cb,color,depth);
}
__attribute__((visibility("default"))) void SMOCaptureBugFrame(Processor callback) {
    previous=callback; atomic_store(&requested,true);
    void (*set)(Processor)=dlsym(RTLD_DEFAULT,"SMOSetFrameProcessor");
    if (set) set(Capture);
}

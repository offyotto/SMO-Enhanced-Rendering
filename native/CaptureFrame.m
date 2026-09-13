#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <stdatomic.h>
static atomic_bool captured = false;
static void Capture(id<MTLCommandBuffer> cb, id<MTLTexture> color, id<MTLTexture> depth) {
    if (!depth || atomic_exchange(&captured, true)) return;
    NSUInteger colorStride = (color.width * 8 + 255) & ~255;
    NSUInteger depthStride = (depth.width * 4 + 255) & ~255;
    id<MTLBuffer> a = [color.device newBufferWithLength:colorStride * color.height options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [color.device newBufferWithLength:depthStride * depth.height options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    [enc copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(color.width,color.height,1) toBuffer:a destinationOffset:0 destinationBytesPerRow:colorStride destinationBytesPerImage:colorStride*color.height];
    [enc copyFromTexture:depth sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(depth.width,depth.height,1) toBuffer:b destinationOffset:0 destinationBytesPerRow:depthStride destinationBytesPerImage:depthStride*depth.height];
    [enc endEncoding];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        NSString *dir = [NSHomeDirectory() stringByAppendingPathComponent:@"Documents/SMOShaders"];
        [[NSData dataWithBytes:a.contents length:a.length] writeToFile:[dir stringByAppendingPathComponent:@"color.rgba16f"] atomically:YES];
        [[NSData dataWithBytes:b.contents length:b.length] writeToFile:[dir stringByAppendingPathComponent:@"depth.r32f"] atomically:YES];
        NSDictionary *info = @{@"colorWidth":@(color.width),@"colorHeight":@(color.height),@"colorStride":@(colorStride),@"depthWidth":@(depth.width),@"depthHeight":@(depth.height),@"depthStride":@(depthStride),@"status":@(done.status),@"error":done.error.description ?: @"none"};
        [[NSJSONSerialization dataWithJSONObject:info options:NSJSONWritingPrettyPrinted error:nil] writeToFile:[dir stringByAppendingPathComponent:@"capture.json"] atomically:YES];
    }];
}
__attribute__((constructor)) static void Install(void) {
    void (*setProcessor)(void *) = dlsym(RTLD_DEFAULT, "SMOSetFrameProcessor");
    if (setProcessor) setProcessor(Capture);
}

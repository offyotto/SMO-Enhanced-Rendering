#import <Foundation/Foundation.h>
#include <dlfcn.h>

// Astris loads this small library through an added LC_LOAD_DYLIB command.
// Load the effects after dyld finishes the application's initializers.
__attribute__((constructor)) static void StartShaders(void) {
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY,0),^{
        @autoreleasepool {
            NSBundle *app=NSBundle.mainBundle;
            if (![app.bundleIdentifier isEqualToString:@"V380-Ori.Astris"]) return;
            if (![[app objectForInfoDictionaryKey:@"CFBundleVersion"] isEqual:@"3814"]) return;
            NSString *directory=[NSHomeDirectory() stringByAppendingPathComponent:@"Documents/SMOShaders"];
            NSString *bridge=[directory stringByAppendingPathComponent:@"libSMOMetalBridge.dylib"];
            NSString *effect=[directory stringByAppendingPathComponent:@"libSMOCinematic.dylib"];
            if (![[NSFileManager defaultManager] fileExistsAtPath:bridge] || ![[NSFileManager defaultManager] fileExistsAtPath:effect]) return;
            if (!dlopen(bridge.fileSystemRepresentation,RTLD_NOW|RTLD_GLOBAL)) { NSLog(@"SMO shader bridge: %s",dlerror()); return; }
            if (!dlopen(effect.fileSystemRepresentation,RTLD_NOW|RTLD_GLOBAL)) NSLog(@"SMO cinematic shaders: %s",dlerror());
        }
    });
}

#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <string.h>
int main(void) {
    @autoreleasepool {
        NSDate *deadline=[NSDate dateWithTimeIntervalSinceNow:8];
        const char *(*status)(void)=NULL;
        BOOL ready=NO;
        do {
            [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:.025]];
            status=dlsym(RTLD_DEFAULT,"SMOShadersStatus");
            const char *(*bridgeStatus)(void)=dlsym(RTLD_DEFAULT,"SMOBridgeStatus");
            ready=status && bridgeStatus && strstr(bridgeStatus(),"effect=on");
        } while (!ready && deadline.timeIntervalSinceNow>0);
        if (!ready) { fprintf(stderr,"The startup loader did not load the effects.\n"); return 1; }
        printf("Startup loader passed. Container: %s\n%s\n",NSHomeDirectory().UTF8String,status());
        return 0;
    }
}

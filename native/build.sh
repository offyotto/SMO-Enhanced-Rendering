#!/bin/zsh
# Build the libraries that install_shaders.py expects, beside the sources.
set -e
cd "${0:A:h}"
flags=(-dynamiclib -fobjc-arc -O2 -arch arm64 -framework Foundation -framework Metal)
clang $flags -framework QuartzCore MetalBridge.m -o libSMOMetalBridge.dylib
clang $flags -framework AppKit Cinematic.m -o libSMOCinematic.dylib
clang $flags ShaderLoader.m -o libSMOShaderLoader.dylib
# Offline check: compiles the shaders and, with a capture, renders one frame.
#   ./cinematic-test . <capture directory> <output.rgba16f> <benchmark frames>
clang -fobjc-arc -O2 -arch arm64 -DSMO_COMPILE_ONLY -framework Foundation -framework Metal -framework AppKit Cinematic.m -o cinematic-test
./cinematic-test .

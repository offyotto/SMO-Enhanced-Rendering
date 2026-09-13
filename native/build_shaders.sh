#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

COMMON=(-dynamiclib -fobjc-arc -O2 -mmacosx-version-min=15.0)

clang "${COMMON[@]}" \
  -framework Foundation -framework Metal -framework AppKit -framework QuartzCore \
  Cinematic.m -o libSMOCinematic.dylib

clang "${COMMON[@]}" \
  -framework Foundation -framework Metal -framework QuartzCore \
  MetalBridge.m -o libSMOMetalBridge.dylib

clang "${COMMON[@]}" \
  -framework Foundation \
  ShaderLoader.m -o libSMOShaderLoader.dylib

echo "Built:"
ls -lh libSMOCinematic.dylib libSMOMetalBridge.dylib libSMOShaderLoader.dylib

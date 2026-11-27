# SMO Cinematic Shaders

This package adds Metal shaders to Super Mario Odyssey in Astris 1.0.18, build 3814.

- Sand and water receive screen-space reflections.
- Ambient occlusion and sun-facing contact shadows add depth near surfaces.
- Nearby colors add a small light contribution.
- Multi-scale bloom adds a wide, soft glow around bright areas.
- Color adjustments increase contrast and saturation.

The shaders use the color and depth textures from the current frame. They do not use hardware ray tracing.
Reflections can include only objects that appear in the frame. Reflections fade at screen edges.
Transparent surfaces and thin objects can show reflection errors. The shaders preserve fixed HUD regions.

The installed preset uses the current HDR output. Keep HDR enabled in Astris.
The reflections use a 1280-pixel texture. The final image retains the original output resolution.

## Frame pipeline

Each frame runs these passes. Only the last pass runs at the output resolution.

1. A box filter reduces the frame to half the effect width. Bloom, bounce light, and rough reflections read this texture.
2. The emulator depth buffer becomes a view-space depth texture at the effect width.
3. The surface pass computes occlusion, bounce light, reflections, and contact shadows. Each pixel uses its own sample offsets.
4. A depth-aware filter removes the sampling grain and keeps silhouettes sharp.
5. Bloom uses three levels, from 1/4 to 1/16 of the effect width.
6. The composite pass reads the drawable in place through framebuffer fetch. The frame is never copied.

Framebuffer fetch requires an Apple GPU. Astris runs only on Apple silicon.
The surface pass skips HUD regions, sky pixels, and rays that point back at the camera.

The shader loader starts with Astris. It applies the shaders when an Odyssey window is present.
The loader supports Astris build 3814. An Astris update can remove the loader.
The game executable and the existing gameplay patches remain separate from these native shader files.

Double-click `Disable Shaders.command` to disable the effects. The change takes about one second.
Double-click `Enable Shaders.command` to enable the effects.

The active preset is here:

`~/Library/Containers/V380-Ori.Astris/Data/Documents/SMOShaders/preset.json`

The preset supports `reflections`, `occlusion`, `bloom`, `exposure`, `saturation`, `contrast`, `verticalFov`, and `effectWidth`.
`shadows` sets the contact shadow strength. `sunAzimuth` and `sunElevation` set the sun direction in degrees, relative to the camera.
Set `debugView` to `0` for normal use. Values `1`, `2`, `3`, and `4` show depth, normals, reflection hits, and contact shadows.

## Power use

- `effectWidth` has the largest effect on GPU load. `960` uses about half the surface work of `1280`.
- `effectInterval` set to `2` computes the surface effects on every second frame. Fast camera motion can then show a one-frame lag in reflections and shadows.
- `adaptivePower` reduces the effect width by a quarter in Low Power Mode and under thermal pressure. Set it to `false` to keep the full width.
- Set `reflections`, `occlusion`, `shadows`, or `bloom` to `0` to skip that work.

`cinematic.log` records the average GPU time of the presented frame.

## Build and update

Run `./build.sh` in this folder on a Mac to build the three libraries and compile the shaders.
Run this command to replace the shader files of an existing installation, then restart Astris:

```sh
python3 install_shaders.py --update
```

Add `--preset` to replace the active preset as well. `Cinematic.metal` and `libSMOCinematic.dylib` must come from the same version.

Run this command from this folder to restore the original Astris application:

```sh
python3 install_shaders.py --restore
```

The installer records the original application backup in `installation.json` beside the active preset.
The restore command checks the installed file hashes before it changes Astris.

Performance depends on the scene and output resolution.

The direct Metal integration follows the presentation sequence in [MoltenVK](https://github.com/KhronosGroup/MoltenVK/blob/main/MoltenVK/MoltenVK/GPUObjects/MVKImage.mm).
The shaders and native integration in this package are custom source code.

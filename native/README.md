# SMO Cinematic Shaders

This package adds Metal shaders to Super Mario Odyssey in Astris 1.0.18, build 3814.

- Sand and water receive screen-space reflections.
- Contact shadows add depth near surfaces.
- Nearby colors add a small light contribution.
- Bloom adds light around bright areas.
- Color adjustments increase contrast and saturation.

The shaders use the color and depth textures from the current frame. They do not use hardware ray tracing.
Reflections can include only objects that appear in the frame. Reflections fade at screen edges.
Transparent surfaces and thin objects can show reflection errors. The shaders preserve fixed HUD regions.

The installed preset uses the current HDR output. Keep HDR enabled in Astris.
The reflections use a 1280-pixel texture. The final image retains the original output resolution.

The shader loader starts with Astris. It applies the shaders when an Odyssey window is present.
The loader supports Astris build 3814. An Astris update can remove the loader.
The game executable and the existing gameplay patches remain separate from these native shader files.

Double-click `Disable Shaders.command` to disable the effects. The change takes about one second.
Double-click `Enable Shaders.command` to enable the effects.

The active preset is here:

`~/Library/Containers/V380-Ori.Astris/Data/Documents/SMOShaders/preset.json`

The preset supports `reflections`, `occlusion`, `bloom`, `exposure`, `saturation`, `contrast`, `verticalFov`, and `effectWidth`.
Set `debugView` to `0` for normal use. Values `1`, `2`, and `3` show depth, normals, and reflection hits.

Run this command from this folder to restore the original Astris application:

```sh
python3 install_shaders.py --restore
```

The installer records the original application backup in `installation.json` beside the active preset.
The restore command checks the installed file hashes before it changes Astris.

The shader pipelines passed Metal API Validation with a captured Odyssey frame.
The live preset maintained 60 FPS during the observed gameplay. Performance depends on the scene and output resolution.

The direct Metal integration follows the presentation sequence in [MoltenVK](https://github.com/KhronosGroup/MoltenVK/blob/main/MoltenVK/MoltenVK/GPUObjects/MVKImage.mm).
The shaders and native integration in this package are custom source code.

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

The current SSR path builds a conservative R32Float closest-depth hierarchy with up to seven mip levels every fresh-depth frame, then traverses that hierarchy instead of using the older fixed/exponential full-resolution depth march. The hierarchy is private GPU storage and is reused per in-flight frame slot.

The installed preset uses the current HDR output. Keep HDR enabled in Astris.
The reflection/contact pass uses a 1280-pixel effect texture by default. The final image retains the original output resolution.

The shader loader starts with Astris. It applies the shaders when an Odyssey window is present.
The loader supports Astris build 3814. An Astris update can remove the loader.
The game executable and the existing gameplay patches remain separate from these native shader files.

Double-click `Disable Shaders.command` to disable the effects. The change takes about one second.
Double-click `Enable Shaders.command` to enable the effects.

The active preset is here:

`~/Library/Containers/V380-Ori.Astris/Data/Documents/SMOShaders/preset.json`

The preset supports `reflections`, `occlusion`, `bloom`, `exposure`, `saturation`, `contrast`, `verticalFov`, `effectWidth`, `debugView`, and `debugMip`.

Hi-Z debug modes:

- `debugView: 0` — normal composite.
- `debugView: 1` — reconstructed view-space depth.
- `debugView: 2` — reconstructed normals.
- `debugView: 3` — SSR hit confidence.
- `debugView: 4` — raw guest depth.
- `debugView: 5` — depth-hierarchy visualization. Select mip `0` through `6` with `debugMip`.
- `debugView: 6` — traversal cost: red = total hierarchy reads / 64, green = mip-0 reads / 64, blue = candidate hits / 8.
- `debugView: 7` — reflected radiance before final compositing.
- `debugView: 8` — trace termination/rejection reason.
- `debugView: 9` — material diagnostics: red = reflection weight, green = roughness, blue = surface trust.

`debugView: 8` colors are: green = accepted hit, red = rejected candidate, blue = screen edge, purple = depth/far-depth rejection, yellow = low material weight, magenta = traversal budget, brown = unstable geometry, cyan = HDR/transparent-color rejection, gray = fresh depth or hierarchy unavailable.

Run this command from this folder to restore the original Astris application:

```sh
python3 install_shaders.py --restore
```

The installer records the original application backup in `installation.json` beside the active preset.
The restore command checks the installed file hashes before it changes Astris.

The Hi-Z implementation has passed the repository's compile-only Metal pipeline check on Apple Silicon. Visual quality, traversal cost, and live 60 FPS behavior still require in-game validation because the compile-only path cannot reproduce every Astris/MoltenVK render-pass interaction.

The direct Metal integration follows the presentation sequence in [MoltenVK](https://github.com/KhronosGroup/MoltenVK/blob/main/MoltenVK/MoltenVK/GPUObjects/MVKImage.mm).
The shaders and native integration in this package are custom source code.

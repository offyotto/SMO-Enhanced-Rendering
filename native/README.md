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

## Frame pipeline

1. Compute kernels build the depth hierarchy. Mip 0 is at most 1920 texels wide, so a guest depth buffer at a higher resolution scale costs the same to trace. Every later depth read uses the hierarchy.
2. With `halfResolution`, the surface pass runs twice. The first run shades occlusion, bounce light, and rough reflections at half the effect size. The second run shades mirror-like reflections, such as water, at the full effect size and skips all other pixels.
3. The resolve pass joins the two results with a depth-aware filter. The filter removes sampling grain and keeps silhouettes sharp.
4. Bloom runs only when `bloom` is above `0`.
5. The composite pass reads the drawable in place through framebuffer fetch. The frame is never copied. Framebuffer fetch requires an Apple GPU.

The surface passes skip the HUD regions. Each pixel rotates its occlusion samples, which removes the fixed pattern of the earlier version.

## Depth alignment

At a present, the guest depth buffer is one present ahead of the color on screen. Live captures show the offset: depth silhouettes lead the image during motion, and occlusion and reflections appear beside a moving character.
`gameNormals` (default `true`) shades with the game's own G-buffer normals. The bridge picks the `RG16Snorm` color attachment drawn together with the scene depth; it holds world-space normals, octahedral-encoded. No camera matrix is available, so each present fits the world-to-view rotation by matching a 40x22 grid of those normals against normals derived from depth, and stores view-space normals next to the depth hierarchy. A fit that agrees less than 90% with the depth normals is rejected and that frame falls back to depth-derived normals. The log reports `gameNormals=` and the fit agreement.

`gameMaterials` (default `true`) takes reflectivity and roughness from the game instead of guessing them from color. The same pass has an `RGBA8Unorm` base-color target whose alpha packs two 4-bit fields: the high one is low on glass and polished metal and high on stone and cloth, and the low one is set on glass and metal. This reading comes from live captures, not from a format description. Where a flat stretch of depth faces away from the stored normal, something drawn after the G-buffer (water, for example) covers the surface, and that pixel keeps the depth normal and the color-based material. The one-shot capture also saves every color target of that pass as `gbuffer-N.bin`.

`depthDelay` (default `1`) pairs each frame with the depth hierarchy that the previous present built. Set it to `0` for the earlier behavior.

To check the alignment in a running game, create an empty file named `capture-request` beside the preset. The next frame writes the unprocessed color, the current depth, and the previous depth to `align-capture/`.

## Power use

GPU time of the shader passes on an M1 Pro, measured with `cinematic-test` on two captured frames at a 3566x2006 output:

| Capture | Before | After |
| --- | --- | --- |
| 1600x900 guest depth | 3.45 ms | 1.58 ms |
| 3200x1800 guest depth | 4.97 ms | 2.14 ms |

- `halfResolution` (default `true`): set it to `false` to shade every effect at the full effect size. That costs about 3 ms and 4 ms on the two captures.
- `effectWidth` scales the surface work. `960` uses about half the work of `1280`.
- `adaptivePower` (default `true`) reduces the effect width by a quarter in Low Power Mode and under thermal pressure.
- Set `reflections`, `occlusion`, or `bloom` to `0` to skip that work.

The Metal bridge formats a diagnostic string only for a render target that the log has not recorded. The preset file is parsed only after it changes.

The installed preset uses the current HDR output. Keep HDR enabled in Astris.
The reflection/contact pass uses a 1280-pixel effect texture by default. The final image retains the original output resolution.

The shader loader starts with Astris. It applies the shaders when an Odyssey window is present.
The loader supports Astris build 3814. An Astris update can remove the loader.
The game executable and the existing gameplay patches remain separate from these native shader files.

Double-click `Disable Shaders.command` to disable the effects. The change takes about one second.
Double-click `Enable Shaders.command` to enable the effects.

The active preset is here:

`~/Library/Containers/V380-Ori.Astris/Data/Documents/SMOShaders/preset.json`

The preset supports `reflections`, `occlusion`, `bloom`, `exposure`, `saturation`, `contrast`, `verticalFov`, `effectWidth`, `depthDelay`, `gameNormals`, `gameMaterials`, `halfResolution`, `adaptivePower`, `debugView`, and `debugMip`. Each debug view shades at the full effect size.

Hi-Z debug modes:

- `debugView: 0` — normal composite.
- `debugView: 1` — reconstructed view-space depth.
- `debugView: 2` — the normals in use: the game's where available, otherwise reconstructed from depth.
- `debugView: 3` — SSR hit confidence.
- `debugView: 4` — guest depth as the shaders read it: mip 0 of the hierarchy, with invalid values replaced by 1.
- `debugView: 5` — depth-hierarchy visualization. Select mip `0` through `6` with `debugMip`.
- `debugView: 6` — traversal cost: red = total hierarchy reads / 64, green = mip-0 reads / 64, blue = candidate hits / 8.
- `debugView: 7` — reflected radiance before final compositing.
- `debugView: 8` — trace termination/rejection reason.
- `debugView: 9` — material diagnostics: red = reflection weight, green = roughness, blue = surface trust.
- `debugView: 10` — normals reconstructed from depth, for comparison with view 2.

`debugView: 8` colors are: green = accepted hit, red = rejected candidate, blue = screen edge, purple = depth/far-depth rejection, yellow = low material weight, magenta = traversal budget, brown = unstable geometry, cyan = HDR/transparent-color rejection, gray = fresh depth or hierarchy unavailable.

Run this command from this folder to restore the original Astris application:

```sh
python3 install_shaders.py --restore
```

The installer records the original application backup in `installation.json` beside the active preset.
The restore command checks the installed file hashes before it changes Astris.

`build_shaders.sh` also builds `cinematic-test`. It compiles the shaders, renders a captured frame, and measures GPU time:

```sh
./cinematic-test . <capture directory> <output.rgba16f> <benchmark frames>
```

`Cinematic.metal` and `libSMOCinematic.dylib` must come from the same version. The loader rejects a shader file with another settings layout.

The pipeline passed Metal API Validation and shader validation with two captured Odyssey frames on Apple Silicon. Visual quality, traversal cost, and live 60 FPS behavior still require in-game validation because the compile-only path cannot reproduce every Astris/MoltenVK render-pass interaction.

The direct Metal integration follows the presentation sequence in [MoltenVK](https://github.com/KhronosGroup/MoltenVK/blob/main/MoltenVK/MoltenVK/GPUObjects/MVKImage.mm).
The shaders and native integration in this package are custom source code.

# Motion vector acquisition plan

This document tracks the path to temporal SSR in Astris without guessing motion from the final image.

## Priority order

1. Probe Metal render-pass color attachments and look for an existing game velocity buffer.
2. If a velocity buffer exists, preserve it at the render pass where it is produced and pass it to the cinematic post-process.
3. If no velocity buffer exists, acquire current/previous camera matrices and reconstruct camera motion from depth.
4. Reject or heavily downweight temporal history on moving characters unless per-object motion becomes available.
5. Optical-flow estimation is a last-resort fallback, not the preferred path.

## Why an existing velocity buffer may exist

Super Mario Odyssey's executable symbols expose a deferred G-buffer path and an anti-aliasing filter with projection jitter. That is strong evidence of temporal rendering machinery, but it does not prove that a dedicated velocity render target exists. The first step is therefore observation, not assumption.

## Metal probe

`MetalBridge.m` can see the `MTLRenderPassDescriptor` for every Metal render encoder created by MoltenVK. Enumerating all color attachments lets us record dimensions, pixel formats, sample counts, load/store actions, storage mode, and resolve targets. A motion-vector target is likely to be full or half resolution, generally two-channel or compact RGBA, and should change coherently with camera/object movement.

Once a candidate is identified, capture it at the end of the producing render pass before later passes overwrite or discard it. If the attachment uses `MTLStoreActionDontCare` or memoryless storage, preserving it may require a diagnostic-only store-action override.

## Camera-only reprojection fallback

With current and previous view-projection matrices, current depth can reconstruct static-world motion:

1. Reconstruct current view-space position from depth.
2. Transform to world space with inverse current view matrix.
3. Project that world position by the previous view-projection matrix.
4. Motion = current UV - previous UV.

This handles static level geometry well but not skinned/animated objects. For temporal SSR that is still valuable because reflections should primarily be received by floors/water; history around characters can be rejected using depth, normal, and color disagreement.

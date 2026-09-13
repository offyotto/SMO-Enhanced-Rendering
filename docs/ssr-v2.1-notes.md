# SSR v2.1 notes

Goals for this pass:

- Prevent stale depth from being reused for SSR/AO when the bridge misses a fresh depth attachment.
- Suppress SSR reception on characters and strongly curved/vertical geometry while still allowing characters to appear in floor/water reflections.
- Protect bright HDR scenes from post-process exposure/bloom washout.
- Add render-target diagnostics needed to find a native game motion-vector buffer.

The current post-process has no temporal history buffer, so persistent trails are not caused by temporal accumulation in this shader. The main local source of ghost-like SSR mismatch is pairing current color with stale depth; v2.1 disables depth-driven effects on frames without fresh depth.

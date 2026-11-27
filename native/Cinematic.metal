// settings-layout: 2
#include <metal_stdlib>
using namespace metal;

struct VertexOut { float4 position [[position]]; float2 uv; };
// Must match ShaderSettings in Cinematic.m.
struct Settings {
    float2 texel;        // 1 / output size
    float2 effectTexel;  // 1 / effect size
    float2 sourceTexel;  // per pass: tap spacing in the bound source
    float aspect;
    float tanHalfFov;
    float reflections;
    float occlusion;
    float bloom;
    float exposure;
    float saturation;
    float contrast;
    float debugView;
    float effectsValid;
    float shadows;
    float sunX;
    float sunY;
    float sunZ;
    float taps;
    float blend;
};
constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);
constexpr sampler pointSampler(coord::normalized, address::clamp_to_edge, filter::nearest);

constant float skyZ = 49999.0;

vertex VertexOut fullScreen(uint id [[vertex_id]]) {
    float2 p = float2((id << 1) & 2, id & 2);
    return {float4(p * float2(2,-2) + float2(-1,1), 0, 1), p};
}
float luminance(float3 x) { return dot(x, float3(.2126,.7152,.0722)); }
float zValue(texture2d<float> z, float2 uv) { return z.sample(pointSampler, uv).r; }
float3 positionAt(texture2d<float> z, float2 uv, constant Settings &s) {
    return float3((uv.x*2-1)*s.aspect*s.tanHalfFov, (1-uv.y*2)*s.tanHalfFov, 1) * zValue(z,uv);
}
float2 project(float3 p, constant Settings &s) {
    return float2(.5 + p.x/(2*p.z*s.aspect*s.tanHalfFov), .5 - p.y/(2*p.z*s.tanHalfFov));
}
float3 surfaceNormal(texture2d<float> z, float2 uv, float3 p, constant Settings &s) {
    float2 dx = float2(s.effectTexel.x,0), dy = float2(0,s.effectTexel.y);
    float3 left = positionAt(z,uv-dx,s), right = positionAt(z,uv+dx,s);
    float3 top = positionAt(z,uv-dy,s), bottom = positionAt(z,uv+dy,s);
    float3 x = abs(right.z-p.z) < abs(left.z-p.z) ? right-p : p-left;
    float3 y = abs(bottom.z-p.z) < abs(top.z-p.z) ? bottom-p : p-top;
    float3 n = normalize(cross(x,y) + float3(0,0,1e-8));
    return dot(n,-p) < 0 ? -n : n;
}
float sceneMask(float2 uv) {
    float topLeft = (1-smoothstep(.26,.33,uv.x)) * (1-smoothstep(.18,.24,uv.y));
    float topRight = smoothstep(.85,.90,uv.x) * (1-smoothstep(.36,.44,uv.y));
    float bottom = smoothstep(.90,.98,uv.y);
    return 1-max(max(topLeft,topRight),bottom);
}

// View-space depth at effect resolution. Every later tap reads this small
// texture instead of the emulator's full-size depth buffer.
fragment float4 linearDepth(VertexOut in [[stage_in]], depth2d<float> depth [[texture(0)]]) {
    return float4(1.0 / max(1.0 - depth.sample(pointSampler, in.uv), 0.00002));
}

// Box-filtered half-effect-resolution scene color: the source for bloom,
// bounce light, and rough reflections.
fragment float4 sceneDown(VertexOut in [[stage_in]], texture2d<float> color [[texture(0)]], constant Settings &s [[buffer(0)]]) {
    int n = int(s.taps);
    float3 c = 0;
    for (int y=0; y<n; ++y)
        for (int x=0; x<n; ++x)
            c += color.sample(linearSampler, in.uv + (float2(x,y)-float(n-1)*.5)*s.sourceTexel).rgb;
    return float4(max(c/float(n*n),0.0),1);
}

fragment float4 surfaceEffects(VertexOut in [[stage_in]],
    texture2d<float> color [[texture(0)]], texture2d<float> z [[texture(1)]],
    texture2d<float> scene [[texture(2)]], constant Settings &s [[buffer(0)]]) {
    float2 uv = in.uv;
    float centerZ = zValue(z,uv);
    if (centerZ > skyZ || centerZ < 1.001) return float4(0,0,0,1);
    float mask = sceneMask(uv);
    if (mask < .002 && s.debugView < .5) return float4(0,0,0,1);
    float3 base = max(color.sample(linearSampler,uv).rgb,0.0);
    float3 p = float3((uv.x*2-1)*s.aspect*s.tanHalfFov, (1-uv.y*2)*s.tanHalfFov, 1) * centerZ;
    float3 n = surfaceNormal(z,uv,p,s);
    // Interleaved-gradient noise gives every pixel its own sample offset.
    // The resolve pass removes the resulting grain.
    float jitter = fract(52.9829189*fract(dot(floor(in.position.xy),float2(.06711056,.00583715))));

    float ao = 0;
    float3 bounce = 0;
    if (s.occlusion > .005) {
        float radius = p.z * .10;
        float rotation = jitter*6.2831853;
        for (uint i=0; i<8; ++i) {
            float angle = float(i)*2.39996323 + rotation;
            float scale = sqrt((float(i)+.5)/8.0);
            float2 delta = float2(cos(angle)/s.aspect,sin(angle)) * (.032*scale);
            float2 sampleUV = clamp(uv+delta,0.001,.999);
            float3 v = positionAt(z,sampleUV,s)-p;
            float distance = length(v);
            float weight = max(dot(n,v/max(distance,.0001))-.075,0.0) * (1-smoothstep(radius*.25,radius,distance));
            ao += weight;
            if (weight > .001) bounce += scene.sample(linearSampler,sampleUV).rgb*weight;
        }
        bounce *= .0525;
    }
    ao = clamp(1-ao*(s.occlusion*.375),.58,1.0);

    float3 view = normalize(p);
    float3 ray = reflect(view,n);
    float warm = smoothstep(.02,.14,base.r-base.b) * (1-smoothstep(base.r*1.03,base.r*1.40,base.g));
    // Water needs real blue content, or green grass reads as water.
    float water = smoothstep(.025,.12,base.b-base.r) * smoothstep(base.g*.5,base.g*.75,base.b);
    float grass = smoothstep(.03,.12,base.g-max(base.r,base.b)) * .30;
    float neutral = (1-smoothstep(.06,.16,max(max(base.r,base.g),base.b)-min(min(base.r,base.g),base.b))) * .18;
    float material = clamp(max(max(warm,water*.8),grass)+neutral,0.0,1.0);
    float fresnel = .20 + .80*pow(1-clamp(dot(n,-view),0.0,1.0),4.0);
    float reflectivity = s.reflections * material * fresnel;
    float3 reflection = 0;
    float confidence = 0;
    // A ray that points back at the camera can only hit the unseen side of
    // visible geometry, so those reflections fade out and skip the march.
    float towardScene = smoothstep(-.6,-.3,ray.z);
    if (reflectivity*towardScene > .015) {
        float previousDistance = p.z*.018;
        float3 origin = p+n*(p.z*.0015);
        float previousDelta = -1;
        for (uint step=0; step<28; ++step) {
            float distance = p.z*.025*exp2((float(step)+jitter)*.2486);
            float3 sampleP = origin + ray*distance;
            if (sampleP.z < .1) break;
            float2 sampleUV = project(sampleP,s);
            if (any(sampleUV < .002) || any(sampleUV > .998)) break;
            float surfaceZ = zValue(z,sampleUV);
            float delta = sampleP.z-surfaceZ;
            // Accept only a front-to-back crossing, so rays already behind a
            // surface cannot leak reflections through foreground geometry.
            if (delta > 0 && previousDelta <= 0) {
                float lo=previousDistance, hi=distance, hitDelta=delta;
                for (uint refine=0; refine<6; ++refine) {
                    float mid=(lo+hi)*.5;
                    float3 testP=origin+ray*mid;
                    float testDelta=testP.z-zValue(z,project(testP,s));
                    if (testDelta > 0) { hi=mid; hitDelta=testDelta; } else lo=mid;
                }
                // A true hit refines to a small depth gap. A large gap means the
                // ray crossed a silhouette edge, so that reflection fades out.
                float thickness = max(.12,surfaceZ*.026);
                float valid = 1-smoothstep(thickness*.5,thickness*2.0,hitDelta);
                if (valid > .01) {
                    float hitDistance=(lo+hi)*.5;
                    float2 hitUV=project(origin+ray*hitDistance,s);
                    float edge = smoothstep(0.0,.09,min(min(hitUV.x,hitUV.y),min(1-hitUV.x,1-hitUV.y)));
                    float separation=smoothstep(.006,.035,length(hitUV-uv));
                    confidence=valid*edge*separation*towardScene*sceneMask(hitUV)*(1-smoothstep(p.z*2,p.z*4,distance));
                    // Glossy cone: rough materials and long rays read the
                    // prefiltered scene; water keeps the full-resolution image.
                    float roughness = mix(.11,.02,smoothstep(.15,.70,water));
                    float spread = clamp(roughness*hitDistance/max(p.z,.1),.0007,.0060);
                    float rough = smoothstep(.0012,.0032,spread);
                    float2 blur = float2(spread*.55,spread);
                    if (rough < .99) {
                        float2 b = min(blur,s.texel*2.5);
                        reflection = (color.sample(linearSampler,hitUV+b).rgb+color.sample(linearSampler,hitUV-b).rgb+
                            color.sample(linearSampler,hitUV+float2(b.x,-b.y)).rgb+color.sample(linearSampler,hitUV+float2(-b.x,b.y)).rgb)*.25;
                    }
                    if (rough > .01) {
                        float3 soft = (scene.sample(linearSampler,hitUV+blur).rgb+scene.sample(linearSampler,hitUV-blur).rgb+
                            scene.sample(linearSampler,hitUV+float2(blur.x,-blur.y)).rgb+scene.sample(linearSampler,hitUV+float2(-blur.x,blur.y)).rgb)*.25;
                        reflection = mix(reflection,soft,rough);
                    }
                    break;
                }
            }
            previousDelta=delta;
            previousDistance=distance;
        }
    }

    // Screen-space contact shadows toward the sun: march the depth buffer
    // along the sun direction and darken pixels whose light path is blocked
    // by nearby geometry.
    float sunShadow = 1;
    if (s.shadows > .005) {
        float3 sun = float3(s.sunX,s.sunY,s.sunZ);
        float facing = smoothstep(.02,.22,dot(n,sun));
        if (facing > .01) {
            float range = p.z*.40;
            float blocked = 0;
            float3 start = p + n*(p.z*.003);
            for (uint i=0; i<10; ++i) {
                // Quadratic spacing keeps most taps near the contact point.
                float f = (float(i)+jitter)/10.0;
                float t = range*f*f + p.z*.006;
                float3 q = start + sun*t;
                if (q.z < .1) break;
                float2 sampleUV = project(q,s);
                if (any(sampleUV < .002) || any(sampleUV > .998)) break;
                float blockerZ = zValue(z,sampleUV);
                float gap = q.z-blockerZ;
                // Only nearby, plausibly thin blockers count. A huge gap means
                // the ray passed far behind something that already casts a
                // real shadow in the game.
                if (gap > 0 && gap < max(.12,blockerZ*.05)+t*.12) {
                    blocked = max(blocked, 1-t/range);
                    if (blocked > .97) break;
                }
            }
            sunShadow = 1 - min(blocked,1.0)*facing*min(s.shadows*.55,.90);
        }
    }

    // Reflections that darken the surface read as hard-edged shadow blotches
    // on bright sand. Keep the full highlight shimmer but soften darkening.
    float darkening = smoothstep(0.0,.25,luminance(base)-luminance(reflection));
    float strength = min(reflectivity*confidence,.68) * (1 - .70*darkening);
    float3 addition=(max(reflection,0.0)-base)*strength+bounce;
    if (s.debugView > 3.5) return float4(float3(sunShadow),1);
    if (s.debugView > 2.5) return float4(float3(confidence*material),1);
    if (s.debugView > 1.5) return float4(n*.5+.5,1);
    if (s.debugView > .5) return float4(float3(log2(p.z)/12),1);
    return float4(addition*mask,mix(1.0,max(ao*sunShadow,.30),mask));
}

// Depth-aware 5-tap filter. Removes the per-pixel sampling grain without
// blurring effects across silhouettes.
fragment float4 resolveEffects(VertexOut in [[stage_in]], texture2d<float> effects [[texture(0)]],
    texture2d<float> z [[texture(1)]], constant Settings &s [[buffer(0)]]) {
    float4 center = effects.sample(linearSampler,in.uv);
    if (s.debugView > .5) return center;
    float centerZ = zValue(z,in.uv);
    float4 sum = center*2;
    float total = 2;
    for (uint i=0; i<4; ++i) {
        float2 uv = in.uv + float2(i & 1 ? 1.5 : -1.5, i & 2 ? 1.5 : -1.5)*s.effectTexel;
        float w = 1-smoothstep(.01,.04,abs(zValue(z,uv)-centerZ)/centerZ);
        sum += effects.sample(linearSampler,uv)*w;
        total += w;
    }
    return sum/total;
}

fragment float4 bloomExtract(VertexOut in [[stage_in]], texture2d<float> scene [[texture(0)]], constant Settings &s [[buffer(0)]]) {
    // Luminance-weighted taps stop single bright pixels from flickering.
    float3 c = 0;
    float total = 0;
    for (uint i=0; i<4; ++i) {
        float3 tap = scene.sample(linearSampler,in.uv+float2(i & 1 ? 1 : -1, i & 2 ? 1 : -1)*s.sourceTexel).rgb;
        float w = 1/(1+luminance(tap));
        c += tap*w;
        total += w;
    }
    c /= total;
    float brightness=max(c.r,max(c.g,c.b));
    float knee=clamp(brightness-.35,0.0,.40);
    float contribution=max(brightness-.55,knee*knee/.80)/max(brightness,.0001);
    return float4(c*contribution*sceneMask(in.uv),1);
}
fragment float4 bloomDown(VertexOut in [[stage_in]], texture2d<float> source [[texture(0)]], constant Settings &s [[buffer(0)]]) {
    float2 t=s.sourceTexel;
    float3 c=source.sample(linearSampler,in.uv).rgb*.5;
    c+=(source.sample(linearSampler,in.uv+t).rgb+source.sample(linearSampler,in.uv-t).rgb+
        source.sample(linearSampler,in.uv+float2(t.x,-t.y)).rgb+source.sample(linearSampler,in.uv+float2(-t.x,t.y)).rgb)*.125;
    return float4(c,1);
}
// Alpha-blended onto the next larger level, so the wide levels accumulate
// into the narrow one without an extra target.
fragment float4 bloomUp(VertexOut in [[stage_in]], texture2d<float> source [[texture(0)]], constant Settings &s [[buffer(0)]]) {
    float2 t=s.sourceTexel*.75;
    float3 c=(source.sample(linearSampler,in.uv+t).rgb+source.sample(linearSampler,in.uv-t).rgb+
        source.sample(linearSampler,in.uv+float2(t.x,-t.y)).rgb+source.sample(linearSampler,in.uv+float2(-t.x,t.y)).rgb)*.25;
    return float4(c,s.blend);
}

// Reads the drawable through framebuffer fetch, so the frame is never copied.
fragment float4 composite(VertexOut in [[stage_in]], float4 destination [[color(0)]],
    texture2d<float> effects [[texture(0)]], texture2d<float> bloomTex [[texture(1)]], constant Settings &s [[buffer(0)]]) {
    float mask=sceneMask(in.uv);
    if (mask < .002 && s.debugView < .5) return destination;
    float3 raw=destination.rgb;
    float4 fx=s.effectsValid > .5 ? effects.sample(linearSampler,in.uv) : float4(0,0,0,1);
    if (s.debugView > .5) return float4(fx.rgb,1);
    float3 c=max(raw*fx.a+fx.rgb,0.0);
    if (s.bloom > 0) c += bloomTex.sample(linearSampler,in.uv).rgb*s.bloom;
    c *= s.exposure;
    float l=luminance(c);
    c=mix(float3(l),c,s.saturation);
    c=(c-.18)*s.contrast+.18;
    float shadow=1-smoothstep(.015,.18,l);
    float highlight=smoothstep(.28,.95,l);
    c += float3(-.0015,.001,.0035)*shadow+float3(.008,.002,-.003)*highlight;
    // Preserve the emulator's HDR range. Its output already has a tone curve.
    return float4(mix(raw,max(c,0.0),mask),destination.a);
}

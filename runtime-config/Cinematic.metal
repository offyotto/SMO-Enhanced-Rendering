#include <metal_stdlib>
using namespace metal;

struct VertexOut { float4 position [[position]]; float2 uv; };
struct Settings {
    float2 texel;
    float2 depthTexel;
    float aspect;
    float tanHalfFov;
    float reflections;
    float occlusion;
    float bloom;
    float exposure;
    float saturation;
    float contrast;
    float debugView;
    float depthAvailable;
    float2 direction;
    float shadows;
    float sunAzimuth;
    float sunElevation;
};
constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);
constexpr sampler depthSampler(coord::normalized, address::clamp_to_edge, filter::nearest);

vertex VertexOut fullScreen(uint id [[vertex_id]]) {
    float2 p = float2((id << 1) & 2, id & 2);
    return {float4(p * float2(2,-2) + float2(-1,1), 0, 1), p};
}
float luminance(float3 x) { return dot(x, float3(.2126,.7152,.0722)); }
float zValue(depth2d<float> depth, float2 uv) {
    return 1.0 / max(1.0 - depth.sample(depthSampler, uv), 0.00002);
}
float3 positionAt(depth2d<float> depth, float2 uv, constant Settings &s) {
    return float3((uv.x*2-1)*s.aspect*s.tanHalfFov, (1-uv.y*2)*s.tanHalfFov, 1) * zValue(depth,uv);
}
float2 project(float3 p, constant Settings &s) {
    return float2(.5 + p.x/(2*p.z*s.aspect*s.tanHalfFov), .5 - p.y/(2*p.z*s.tanHalfFov));
}
float3 surfaceNormal(depth2d<float> depth, float2 uv, float3 p, constant Settings &s) {
    float2 dx = float2(s.depthTexel.x * 2,0), dy = float2(0,s.depthTexel.y * 2);
    float3 left = positionAt(depth,uv-dx,s), right = positionAt(depth,uv+dx,s);
    float3 top = positionAt(depth,uv-dy,s), bottom = positionAt(depth,uv+dy,s);
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

fragment float4 surfaceEffects(VertexOut in [[stage_in]],
    texture2d<float> color [[texture(0)]], depth2d<float> depth [[texture(1)]],
    constant Settings &s [[buffer(0)]]) {
    float2 uv = in.uv;
    float rawDepth = depth.sample(depthSampler,uv);
    if (s.depthAvailable < .5 || rawDepth > .99998 || rawDepth < .001) return float4(0,0,0,1);
    float3 base = max(color.sample(linearSampler,uv).rgb,0.0);
    float3 p = positionAt(depth,uv,s);
    float3 n = surfaceNormal(depth,uv,p,s);
    // Interleaved-gradient noise: de-bands the reflection march and the
    // sun-shadow march by giving every pixel its own step offset.
    float jitter = fract(52.9829189*fract(dot(floor(in.position.xy),float2(.06711056,.00583715))));
    float ao = 0;
    float3 bounce = 0;
    float radius = p.z * .10;
    for (uint i=0; i<12; ++i) {
        float angle = float(i)*2.39996323;
        float scale = sqrt((float(i)+.5)/12.0);
        float2 delta = float2(cos(angle)/s.aspect,sin(angle)) * (.032*scale);
        float2 sampleUV = clamp(uv+delta,0.001,.999);
        float3 q = positionAt(depth,sampleUV,s);
        float3 v = q-p;
        float distance = length(v);
        float weight = max(dot(n,v/max(distance,.0001))-.075,0.0) * (1-smoothstep(radius*.25,radius,distance));
        ao += weight;
        bounce += max(color.sample(linearSampler,sampleUV).rgb,0.0)*weight;
    }
    ao = clamp(1-ao*(s.occlusion/4.0),.58,1.0);
    bounce *= .035;

    float3 view = normalize(p);
    float3 ray = reflect(view,n);
    float warm = smoothstep(.02,.14,base.r-base.b) * (1-smoothstep(base.r*1.03,base.r*1.40,base.g));
    // Water needs real blue content. The previous max(b,g) test classified
    // green grass as full-strength water and painted it with cliff reflections.
    float water = smoothstep(.025,.12,base.b-base.r) * smoothstep(base.g*.5,base.g*.75,base.b);
    float grass = smoothstep(.03,.12,base.g-max(base.r,base.b)) * .30;
    float neutral = (1-smoothstep(.06,.16,max(max(base.r,base.g),base.b)-min(min(base.r,base.g),base.b))) * .18;
    float material = clamp(max(max(warm,water*.8),grass)+neutral,0.0,1.0);
    float fresnel = .20 + .80*pow(1-clamp(dot(n,-view),0.0,1.0),4.0);
    float reflectivity = s.reflections * material * fresnel;
    float3 reflection = 0;
    float confidence = 0;
    if (reflectivity > .015) {
        float previousDistance = p.z*.018;
        float3 origin = p+n*(p.z*.0015);
        float previousDelta = -1;
        for (uint step=0; step<48; ++step) {
            float distance = p.z*.025*exp2((float(step)+jitter)*.145);
            float3 sampleP = origin + ray*distance;
            if (sampleP.z < .1) break;
            float2 sampleUV = project(sampleP,s);
            if (any(sampleUV < .002) || any(sampleUV > .998)) break;
            float surfaceZ = zValue(depth,sampleUV);
            float delta = sampleP.z-surfaceZ;
            float thickness = max(.12,surfaceZ*.026);
            // Accept only a front-to-back crossing. The previous window test
            // also accepted rays that were already behind a surface, which
            // leaked reflections through foreground geometry.
            if (delta > 0 && previousDelta <= 0) {
                float lo=previousDistance, hi=distance, hitDelta=delta;
                for (uint refine=0; refine<5; ++refine) {
                    float mid=(lo+hi)*.5;
                    float3 testP=origin+ray*mid;
                    float testDelta=testP.z-zValue(depth,project(testP,s));
                    if (testDelta > 0) { hi=mid; hitDelta=testDelta; } else lo=mid;
                }
                // A true hit refines to a small depth gap. A large gap means the
                // ray crossed a silhouette edge, so let that reflection fade out.
                float valid = 1-smoothstep(thickness*.5,thickness*2.0,hitDelta);
                if (valid > .01) {
                    float2 hitUV=project(origin+ray*((lo+hi)*.5),s);
                    float edge = smoothstep(0.0,.09,min(min(hitUV.x,hitUV.y),min(1-hitUV.x,1-hitUV.y)));
                    float separation=smoothstep(.006,.035,length(hitUV-uv));
                    confidence=valid*edge*separation*sceneMask(hitUV)*(1-smoothstep(p.z*2,p.z*4,distance));
                    // Cone-traced glossy blur: rougher materials and longer ray
                    // hits sample a wider, vertically stretched neighborhood so
                    // sand and grass get soft sheen while water stays sharp.
                    float roughness = mix(.11,.02,smoothstep(.15,.70,water));
                    float spread = clamp(roughness*(lo+hi)*.5/max(p.z,.1),.0007,.0060);
                    float2 blur = float2(spread*.55,spread);
                    float3 sum = 0;
                    sum += color.sample(linearSampler,hitUV+float2( blur.x, blur.y)).rgb;
                    sum += color.sample(linearSampler,hitUV+float2(-blur.x, blur.y)).rgb;
                    sum += color.sample(linearSampler,hitUV+float2( blur.x,-blur.y)).rgb;
                    sum += color.sample(linearSampler,hitUV+float2(-blur.x,-blur.y)).rgb;
                    sum += color.sample(linearSampler,hitUV+float2(0, 1.8*blur.y)).rgb;
                    sum += color.sample(linearSampler,hitUV+float2(0,-1.8*blur.y)).rgb;
                    sum += color.sample(linearSampler,hitUV+float2( 1.8*blur.x,0)).rgb;
                    sum += color.sample(linearSampler,hitUV+float2(-1.8*blur.x,0)).rgb;
                    reflection = sum*.125;
                    break;
                }
            }
            previousDelta=delta;
            previousDistance=distance;
        }
    }
    // Screen-space contact shadows toward the sun: march the depth buffer
    // along the sun direction and darken pixels whose light path is blocked
    // by nearby geometry. Complements the game's own shadow maps with the
    // small-scale grounding that ray-traced shadows provide.
    float sunShadow = 1;
    if (s.shadows > .005) {
        float az = s.sunAzimuth*.0174532925, el = s.sunElevation*.0174532925;
        float3 sun = float3(cos(el)*sin(az), sin(el), -cos(el)*cos(az));
        float facing = smoothstep(.02,.22,dot(n,sun));
        if (facing > .01) {
            float range = p.z*.40;
            float blocked = 0;
            float3 start = p + n*(p.z*.003);
            for (uint i=0; i<16; ++i) {
                float t = range*(float(i)+jitter)/16.0 + p.z*.006;
                float3 q = start + sun*t;
                if (q.z < .1) break;
                float2 sampleUV = project(q,s);
                if (any(sampleUV < .002) || any(sampleUV > .998)) break;
                float blockerZ = zValue(depth,sampleUV);
                float gap = q.z-blockerZ;
                // Only nearby, plausibly thin blockers count; a huge gap means
                // the ray passed far behind something that already casts a
                // real shadow in the game.
                if (gap > 0 && gap < max(.12,blockerZ*.05)+t*.12)
                    blocked = max(blocked, 1-t/range);
            }
            sunShadow = 1 - min(blocked,1.0)*facing*min(s.shadows*.55,.90);
        }
    }
    float mask=sceneMask(uv);
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

fragment float4 bloomExtract(VertexOut in [[stage_in]], texture2d<float> color [[texture(0)]], constant Settings &s [[buffer(0)]]) {
    float2 t=s.texel*2;
    float3 c=(color.sample(linearSampler,in.uv+t).rgb+color.sample(linearSampler,in.uv-t).rgb+
        color.sample(linearSampler,in.uv+float2(t.x,-t.y)).rgb+color.sample(linearSampler,in.uv+float2(-t.x,t.y)).rgb)*.25;
    c=max(c,0.0);
    float brightness=max(c.r,max(c.g,c.b));
    float knee=clamp(brightness-.35,0.0,.40);
    float contribution=max(brightness-.55,knee*knee/.80)/max(brightness,.0001);
    return float4(c*contribution*sceneMask(in.uv),1);
}
fragment float4 bloomBlur(VertexOut in [[stage_in]], texture2d<float> color [[texture(0)]], constant Settings &s [[buffer(0)]]) {
    float3 c=color.sample(linearSampler,in.uv).rgb*.19648255;
    c+=(color.sample(linearSampler,in.uv+s.direction*1.4117647).rgb+color.sample(linearSampler,in.uv-s.direction*1.4117647).rgb)*.29690696;
    c+=(color.sample(linearSampler,in.uv+s.direction*3.2941176).rgb+color.sample(linearSampler,in.uv-s.direction*3.2941176).rgb)*.09447040;
    c+=(color.sample(linearSampler,in.uv+s.direction*5.1764706).rgb+color.sample(linearSampler,in.uv-s.direction*5.1764706).rgb)*.01038136;
    return float4(c,1);
}
fragment float4 composite(VertexOut in [[stage_in]], texture2d<float> original [[texture(0)]],
    texture2d<float> effects [[texture(1)]], texture2d<float> bloomTex [[texture(2)]], constant Settings &s [[buffer(0)]]) {
    float3 raw=original.sample(linearSampler,in.uv).rgb;
    float4 fx=effects.sample(linearSampler,in.uv);
    if (s.debugView > .5) return float4(fx.rgb,1);
    float3 c=max(raw*fx.a+fx.rgb,0.0);
    c += bloomTex.sample(linearSampler,in.uv).rgb*s.bloom;
    float mask=sceneMask(in.uv);
    c *= s.exposure;
    float l=luminance(c);
    c=mix(float3(l),c,s.saturation);
    c=(c-.18)*s.contrast+.18;
    float shadow=1-smoothstep(.015,.18,l);
    float highlight=smoothstep(.28,.95,l);
    c += float3(-.0015,.001,.0035)*shadow+float3(.008,.002,-.003)*highlight;
    // Preserve the emulator's HDR range. Its output already has a tone curve.
    return float4(mix(raw,max(c,0.0),mask),1);
}

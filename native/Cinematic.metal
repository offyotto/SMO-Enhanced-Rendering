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
};
struct GeometryInfo {
    float3 normal;
    float continuity;
    float curvature;
};
struct MaterialInfo {
    float weight;
    float roughness;
};

constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);
constexpr sampler depthSampler(coord::normalized, address::clamp_to_edge, filter::nearest);

vertex VertexOut fullScreen(uint id [[vertex_id]]) {
    float2 p=float2((id<<1)&2,id&2);
    return {float4(p*float2(2,-2)+float2(-1,1),0,1),p};
}

float luminance(float3 x) { return dot(x,float3(.2126,.7152,.0722)); }
float zValue(depth2d<float> depth,float2 uv) {
    return 1.0/max(1.0-depth.sample(depthSampler,uv),.00002);
}
float3 positionAt(depth2d<float> depth,float2 uv,constant Settings &s) {
    return float3((uv.x*2-1)*s.aspect*s.tanHalfFov,(1-uv.y*2)*s.tanHalfFov,1)*zValue(depth,uv);
}
float2 project(float3 p,constant Settings &s) {
    return float2(.5+p.x/(2*p.z*s.aspect*s.tanHalfFov),.5-p.y/(2*p.z*s.tanHalfFov));
}
bool validUV(float2 uv) {
    return all(uv>float2(.002))&&all(uv<float2(.998));
}
float farDepthTrust(float rawDepth) {
    // zValue() is proportional to 1/(1-depth). Near the far end of a
    // perspective depth buffer tiny depth changes therefore become enormous
    // pseudo-view-space distances. Do not let those unstable values create
    // kilometre-long SSR rays, huge AO radii, or giant hit-thickness windows.
    // The fade starts late enough that nearby/mid-range floors and water retain
    // their full reflection energy.
    return 1-smoothstep(.9960,.99935,rawDepth);
}
GeometryInfo geometryAt(depth2d<float> depth,float2 uv,float3 p,constant Settings &s) {
    float2 dx=float2(s.depthTexel.x*2,0),dy=float2(0,s.depthTexel.y*2);
    float3 left=positionAt(depth,clamp(uv-dx,.001,.999),s);
    float3 right=positionAt(depth,clamp(uv+dx,.001,.999),s);
    float3 top=positionAt(depth,clamp(uv-dy,.001,.999),s);
    float3 bottom=positionAt(depth,clamp(uv+dy,.001,.999),s);

    float3 x=abs(right.z-p.z)<abs(left.z-p.z)?right-p:p-left;
    float3 y=abs(bottom.z-p.z)<abs(top.z-p.z)?bottom-p:p-top;
    float3 n=normalize(cross(x,y)+float3(0,0,1e-8));
    if (dot(n,-p)<0) n=-n;

    float scale=max(p.z,.1);
    float nearestX=min(abs(right.z-p.z),abs(left.z-p.z))/scale;
    float nearestY=min(abs(bottom.z-p.z),abs(top.z-p.z))/scale;
    float secondX=abs(left.z+right.z-2*p.z)/scale;
    float secondY=abs(top.z+bottom.z-2*p.z)/scale;
    float continuity=1-smoothstep(.035,.14,max(nearestX,nearestY));
    float curvature=smoothstep(.003,.035,max(secondX,secondY));
    continuity*=1-.72*curvature;

    GeometryInfo info;
    info.normal=n;
    info.continuity=clamp(continuity,0.0,1.0);
    info.curvature=clamp(curvature,0.0,1.0);
    return info;
}
MaterialInfo classifyMaterial(float3 base,GeometryInfo geometry) {
    float maxC=max(max(base.r,base.g),base.b);
    float minC=min(min(base.r,base.g),base.b);
    float chroma=maxC-minC;

    float warm=smoothstep(.02,.14,base.r-base.b)*(1-smoothstep(base.r*1.03,base.r*1.40,base.g));
    float water=smoothstep(.025,.12,base.b-base.r)*smoothstep(base.g*.52,base.g*.78,base.b)*.82;
    float grass=smoothstep(.03,.12,base.g-max(base.r,base.b))*.30;
    float neutral=(1-smoothstep(.06,.16,chroma))*.18;

    float weight=clamp(max(max(warm,water),grass)+neutral,0.0,1.0);
    float sum=warm+water+grass+neutral+.0001;
    float roughness=(warm*.46+water*.08+grass*.78+neutral*.58)/sum;
    roughness=clamp(roughness+geometry.curvature*.18+(1-geometry.continuity)*.10,.06,.88);

    MaterialInfo info;
    info.weight=weight;
    info.roughness=roughness;
    return info;
}
float sceneMask(float2 uv) {
    float topLeft=(1-smoothstep(.26,.33,uv.x))*(1-smoothstep(.18,.24,uv.y));
    float topRight=smoothstep(.85,.90,uv.x)*(1-smoothstep(.36,.44,uv.y));
    float bottom=smoothstep(.90,.98,uv.y);
    return 1-max(max(topLeft,topRight),bottom);
}
float depthSimilarity(depth2d<float> depth,float2 uv,float centerZ,float tolerance) {
    float raw=depth.sample(depthSampler,clamp(uv,.001,.999));
    if (raw>.99998||raw<.001) return 0;
    float sampleZ=1.0/max(1.0-raw,.00002);
    return 1-smoothstep(tolerance,tolerance*5,abs(sampleZ-centerZ));
}
float3 filteredReflection(texture2d<float> color,depth2d<float> depth,
                          float2 hitUV,float hitZ,float roughness,float travelRatio,
                          constant Settings &s) {
    float radius=mix(1.0,7.0,roughness*roughness)*(1+min(travelRatio,3.0)*.18);
    float2 ox=float2(s.texel.x*radius,0),oy=float2(0,s.texel.y*radius);
    float tolerance=max(.08,min(hitZ,512.0)*(.012+.010*roughness));

    float3 sum=max(color.sample(linearSampler,hitUV).rgb,0.0)*.36;
    float weight=.36;
    float2 tap=clamp(hitUV+ox,.001,.999);
    float w=.16*depthSimilarity(depth,tap,hitZ,tolerance);
    sum+=max(color.sample(linearSampler,tap).rgb,0.0)*w; weight+=w;
    tap=clamp(hitUV-ox,.001,.999);
    w=.16*depthSimilarity(depth,tap,hitZ,tolerance);
    sum+=max(color.sample(linearSampler,tap).rgb,0.0)*w; weight+=w;
    tap=clamp(hitUV+oy,.001,.999);
    w=.16*depthSimilarity(depth,tap,hitZ,tolerance);
    sum+=max(color.sample(linearSampler,tap).rgb,0.0)*w; weight+=w;
    tap=clamp(hitUV-oy,.001,.999);
    w=.16*depthSimilarity(depth,tap,hitZ,tolerance);
    sum+=max(color.sample(linearSampler,tap).rgb,0.0)*w; weight+=w;
    return sum/max(weight,.0001);
}
float3 fireflyClamp(float3 reflection,float3 base) {
    reflection=max(reflection,0.0);
    float reflectedLuma=luminance(reflection);
    float limit=max(6.0,luminance(base)*12.0+2.0);
    return reflection*min(1.0,limit/max(reflectedLuma,.0001));
}
float sourceDistanceConfidence(GeometryInfo hitGeometry,float travelRatio,float pixelTravel,float roughness) {
    float stable=smoothstep(.10,.70,hitGeometry.continuity);
    float planar=1-smoothstep(.28,.88,hitGeometry.curvature);
    float sourceGeometry=mix(.42,1.0,stable*mix(.68,1.0,planar));
    float longRay=smoothstep(.55,1.75,travelRatio);
    float maxPixels=mix(720.0,360.0,roughness);
    float screenTrust=1-smoothstep(maxPixels*.72,maxPixels,pixelTravel);
    return mix(1.0,sourceGeometry,longRay)*screenTrust;
}

fragment float4 surfaceEffects(VertexOut in [[stage_in]],
    texture2d<float> color [[texture(0)]],depth2d<float> depth [[texture(1)]],
    constant Settings &s [[buffer(0)]]) {
    float2 uv=in.uv;
    float rawDepth=depth.sample(depthSampler,uv);
    if (s.depthAvailable<.5||rawDepth>.99998||rawDepth<.001) return float4(0,0,0,1);

    float3 base=max(color.sample(linearSampler,uv).rgb,0.0);
    float3 p=positionAt(depth,uv,s);
    GeometryInfo geometry=geometryAt(depth,uv,p,s);
    float3 n=geometry.normal;
    float farTrust=farDepthTrust(rawDepth);
    float traceZ=min(p.z,512.0);
    float jitter=fract(52.9829189*fract(dot(floor(in.position.xy),float2(.06711056,.00583715))));

    float ao=0;
    float3 bounce=0;
    float radius=traceZ*.10;
    for (uint i=0;i<12;++i) {
        float angle=float(i)*2.39996323;
        float scale=sqrt((float(i)+.5)/12.0);
        float2 delta=float2(cos(angle)/s.aspect,sin(angle))*(.032*scale);
        float2 sampleUV=clamp(uv+delta,.001,.999);
        float3 q=positionAt(depth,sampleUV,s);
        float3 v=q-p;
        float distance=length(v);
        float weight=max(dot(n,v/max(distance,.0001))-.075,0.0)*(1-smoothstep(radius*.25,radius,distance));
        ao+=weight;
        bounce+=max(color.sample(linearSampler,sampleUV).rgb,0.0)*weight;
    }
    ao=clamp(1-ao*(s.occlusion/4.0),.58,1.0);
    ao=mix(1.0,ao,farTrust);
    bounce*=.035*farTrust;

    MaterialInfo materialInfo=classifyMaterial(base,geometry);
    float3 view=normalize(p);
    float3 ray=reflect(view,n);
    float fresnel=.20+.80*pow(1-clamp(dot(n,-view),0.0,1.0),4.0);
    float roughEnergy=mix(1.0,.72,materialInfo.roughness);
    float reflectivity=s.reflections*materialInfo.weight*fresnel*roughEnergy*farTrust;

    float3 reflection=0;
    float confidence=0;
    if (reflectivity>.010&&geometry.continuity>.04) {
        float3 origin=p+n*(traceZ*(.0012+.0010*(1-geometry.continuity)));
        float maxDistance=traceZ*mix(4.1,2.55,materialInfo.roughness);
        float previousDistance=max(traceZ*.008,.015);
        float3 previousP=origin+ray*previousDistance;
        float2 previousUV=project(previousP,s);
        float previousDelta=-1;
        if (previousP.z>.1&&validUV(previousUV)) previousDelta=previousP.z-zValue(depth,previousUV);

        for (uint step=0;step<48;++step) {
            float scheduled=traceZ*.020*exp2((float(step)+jitter)*.145);
            float distance=max(scheduled,previousDistance+max(traceZ*.001,.01));
            if (distance>maxDistance) distance=maxDistance;

            float3 sampleP=origin+ray*distance;
            if (sampleP.z<.1) break;
            float2 sampleUV=project(sampleP,s);
            if (!validUV(sampleUV)) break;

            float targetPixels=mix(1.30,2.65,materialInfo.roughness);
            float pixelAdvance=length((sampleUV-previousUV)/max(s.depthTexel,float2(1e-6)));
            if (pixelAdvance<targetPixels&&distance<maxDistance) {
                float scale=min(targetPixels/max(pixelAdvance,.05),3.5);
                distance=min(maxDistance,previousDistance+(distance-previousDistance)*scale);
                sampleP=origin+ray*distance;
                if (sampleP.z<.1) break;
                sampleUV=project(sampleP,s);
                if (!validUV(sampleUV)) break;
            }

            float surfaceZ=zValue(depth,sampleUV);
            float delta=sampleP.z-surfaceZ;
            if (delta>0&&previousDelta<=0) {
                float lo=previousDistance,hi=distance;
                for (uint refine=0;refine<5;++refine) {
                    float mid=(lo+hi)*.5;
                    float3 testP=origin+ray*mid;
                    float2 testUV=project(testP,s);
                    if (!validUV(testUV)) { hi=mid; continue; }
                    float testDelta=testP.z-zValue(depth,testUV);
                    if (testDelta>0) hi=mid; else lo=mid;
                }

                float hitDistance=(lo+hi)*.5;
                float3 hitRayP=origin+ray*hitDistance;
                float2 hitUV=project(hitRayP,s);
                if (validUV(hitUV)) {
                    float hitRaw=depth.sample(depthSampler,hitUV);
                    if (hitRaw>.001&&hitRaw<.99998) {
                        float3 hitP=positionAt(depth,hitUV,s);
                        float hitZ=hitP.z;
                        float hitGap=max(hitRayP.z-hitZ,0.0);
                        float hitMetricZ=min(hitZ,512.0);
                        float hitThickness=max(.08,hitMetricZ*(.012+.010*materialInfo.roughness));
                        float valid=1-smoothstep(hitThickness*.35,hitThickness*1.8,hitGap);

                        GeometryInfo hitGeometry=geometryAt(depth,hitUV,hitP,s);
                        float frontFace=smoothstep(.02,.28,dot(hitGeometry.normal,-ray));
                        float edge=smoothstep(0.0,.085,min(min(hitUV.x,hitUV.y),min(1-hitUV.x,1-hitUV.y)));
                        float pixelSeparation=length((hitUV-uv)/max(s.depthTexel,float2(1e-6)));
                        float separation=smoothstep(2.5,11.0,pixelSeparation);
                        float travelRatio=hitDistance/max(traceZ,.1);
                        float distanceFade=1-smoothstep(.78,.99,hitDistance/max(maxDistance,.001));
                        float sourceTrust=sourceDistanceConfidence(hitGeometry,travelRatio,pixelSeparation,materialInfo.roughness);
                        float hitFarTrust=farDepthTrust(hitRaw);
                        float candidate=valid*frontFace*edge*separation*sceneMask(hitUV)*distanceFade*geometry.continuity*hitGeometry.continuity*sourceTrust*hitFarTrust;

                        if (candidate>.020) {
                            confidence=candidate;
                            reflection=filteredReflection(color,depth,hitUV,hitZ,materialInfo.roughness,travelRatio,s);
                            reflection=fireflyClamp(reflection,base);
                            break;
                        }
                    }
                }
            }

            previousDelta=delta;
            previousDistance=distance;
            previousUV=sampleUV;
            if (distance>=maxDistance) break;
        }
    }

    float mask=sceneMask(uv);
    float darkening=smoothstep(0.0,.25,luminance(base)-luminance(reflection));
    float strength=min(reflectivity*confidence,.72)*(1-.72*darkening);
    float3 addition=(reflection-base)*strength+bounce;

    if (s.debugView>2.5) return float4(float3(confidence*materialInfo.weight*farTrust),1);
    if (s.debugView>1.5) return float4(n*.5+.5,1);
    if (s.debugView>.5) return float4(float3(log2(p.z)/12),1);
    return float4(addition*mask,mix(1.0,ao,mask));
}

fragment float4 bloomExtract(VertexOut in [[stage_in]],texture2d<float> color [[texture(0)]],constant Settings &s [[buffer(0)]]) {
    float2 t=s.texel*2;
    float3 c=(color.sample(linearSampler,in.uv+t).rgb+color.sample(linearSampler,in.uv-t).rgb+
        color.sample(linearSampler,in.uv+float2(t.x,-t.y)).rgb+color.sample(linearSampler,in.uv+float2(-t.x,t.y)).rgb)*.25;
    c=max(c,0.0);
    float brightness=luminance(c);
    float threshold=.82;
    float knee=.28;
    float soft=clamp((brightness-threshold+knee)/(2*knee),0.0,1.0);
    soft=soft*soft*knee;
    float contribution=(max(brightness-threshold,0.0)+soft)/max(brightness,.0001);
    contribution=min(contribution,.58);
    return float4(c*contribution*sceneMask(in.uv),1);
}
fragment float4 bloomBlur(VertexOut in [[stage_in]],texture2d<float> color [[texture(0)]],constant Settings &s [[buffer(0)]]) {
    float3 c=color.sample(linearSampler,in.uv).rgb*.19648255;
    c+=(color.sample(linearSampler,in.uv+s.direction*1.4117647).rgb+color.sample(linearSampler,in.uv-s.direction*1.4117647).rgb)*.29690696;
    c+=(color.sample(linearSampler,in.uv+s.direction*3.2941176).rgb+color.sample(linearSampler,in.uv-s.direction*3.2941176).rgb)*.09447040;
    c+=(color.sample(linearSampler,in.uv+s.direction*5.1764706).rgb+color.sample(linearSampler,in.uv-s.direction*5.1764706).rgb)*.01038136;
    return float4(c,1);
}
fragment float4 composite(VertexOut in [[stage_in]],texture2d<float> original [[texture(0)]],
    texture2d<float> effects [[texture(1)]],texture2d<float> bloomTex [[texture(2)]],constant Settings &s [[buffer(0)]]) {
    float3 raw=max(original.sample(linearSampler,in.uv).rgb,0.0);
    float4 fx=effects.sample(linearSampler,in.uv);
    if (s.debugView>.5) return float4(fx.rgb,1);

    float3 c=max(raw*fx.a+fx.rgb,0.0);
    float rawL=luminance(raw);
    float highlightProtect=1-smoothstep(.65,1.55,rawL);
    // Once the game's own HDR output is already bright, adding our blurred
    // bloom back on top only destroys waterfall/cloud detail. Fade it fully
    // rather than retaining the old 25% floor.
    float bloomProtect=1-smoothstep(.80,1.55,rawL);
    c+=bloomTex.sample(linearSampler,in.uv).rgb*s.bloom*bloomProtect;

    float mask=sceneMask(in.uv);
    float exposure=mix(1.0,s.exposure,highlightProtect);
    c*=exposure;
    float l=luminance(c);
    float saturation=mix(1.0,s.saturation,highlightProtect);
    float contrast=mix(1.0,s.contrast,highlightProtect);
    c=mix(float3(l),c,saturation);
    c=(c-.18)*contrast+.18;
    float shadow=1-smoothstep(.015,.18,l);
    float highlight=smoothstep(.28,.95,l)*highlightProtect;
    c+=float3(-.0015,.001,.0035)*shadow+float3(.008,.002,-.003)*highlight;
    return float4(mix(raw,max(c,0.0),mask),1);
}
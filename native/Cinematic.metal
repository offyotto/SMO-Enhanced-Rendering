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
    float debugMip;
    float hierarchyLevels;
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
struct SurfaceOutput {
    float4 effects [[color(0)]];     // Bounce RGB, AO alpha.
    float4 reflection [[color(1)]];  // Radiance times weight RGB, composite weight alpha.
};

// Capture evidence: Depth32Float clears to 1. Larger values are farther away.
// R32Float minima retain depth precision near 1 and need half the bandwidth
// of min/max pairs. A ray behind a minimum must descend, never skip that cell.
// These kernels use distinct mip views. Each encoder ends before the next mip.
kernel void depthHierarchyCopy(depth2d<float,access::read> depth [[texture(0)]],
    texture2d<float,access::write> target [[texture(1)]], uint2 tid [[thread_position_in_grid]]) {
    if (any(tid>=uint2(target.get_width(),target.get_height()))) return;
    float d=depth.read(tid);
    target.write(float4(isfinite(d)&&d>.001&&d<.99998 ? d : 1.0),tid);
}
kernel void depthHierarchyReduce(texture2d<float,access::read> source [[texture(0)]],
    texture2d<float,access::write> target [[texture(1)]], uint2 tid [[thread_position_in_grid]]) {
    uint2 dstSize=uint2(target.get_width(),target.get_height());
    if (any(tid>=dstSize)) return;
    uint2 srcSize=uint2(source.get_width(),source.get_height());
    // Native odd mip sizes round down. Cover every overlapping source texel
    // in normalized coordinates, including the last row and column (up to 3x3).
    uint2 first=tid*srcSize/dstSize;
    uint2 end=((tid+1)*srcSize+dstSize-1)/dstSize;
    float closest=1.0;
    for (uint y=first.y;y<end.y;++y)
        for (uint x=first.x;x<end.x;++x)
            closest=min(closest,source.read(uint2(x,y)).r);
    target.write(float4(closest),tid);
}

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
float hdrSurfaceTrust(float3 base) {
    // The source color is an RGBA16F post-render image, so effects such as
    // waterfalls, mist and additive particles can contain very large HDR values
    // while the depth buffer still belongs to opaque geometry behind them.
    // Depth-driven AO/SSGI/SSR must not treat those bright overlays as opaque
    // surfaces. Normal scene colors are untouched; only the HDR tail fades out.
    return 1-smoothstep(.72,1.45,luminance(base));
}
float3 clampIndirectSample(float3 c) {
    // SSGI/bounce used to accumulate raw HDR source values without a limiter.
    // A waterfall texel at several times display white could therefore inject
    // enormous energy even with bloom disabled. Preserve useful indirect color
    // while bounding emissive/transparent outliers.
    c=max(c,0.0);
    float l=luminance(c);
    return c*min(1.0,2.25/max(l,.0001));
}

enum TraceReason : uint {
    TraceMiss=0, TraceHit=1, TraceRejected=2, TraceEdge=3, TraceDepth=4,
    TraceMaterial=5, TraceBudget=6, TraceGeometry=7, TraceHDR=8, TraceUnavailable=9
};
struct ReflectionResult {
    float3 radiance;
    float confidence;
    uint lookups;
    uint fineLookups;
    uint candidates;
    uint reason;
};
struct ProjectedRay {
    float3 start;
    float3 delta;
    float end;
    float epsilon;
    uint endReason;
};
float3 positionWithZ(float2 uv,float z,constant Settings &s) {
    return float3((uv.x*2-1)*s.aspect*s.tanHalfFov,(1-uv.y*2)*s.tanHalfFov,1)*z;
}
bool makeProjectedRay(float3 origin,float3 direction,float firstDistance,float maxDistance,
                      constant Settings &s,thread ProjectedRay &r) {
    r.endReason=TraceMiss;
    float endDistance=maxDistance;
    // The reconstruction model has near Z=1. Clip before projection, including
    // rays toward the camera. Neither a negative Z direction nor a grazing
    // normal alone excludes a receiver.
    if (direction.z<0) {
        float nearDistance=(origin.z-1.0001)/(-direction.z);
        if (nearDistance<endDistance) { endDistance=nearDistance; r.endReason=TraceDepth; }
    }
    if (endDistance<=firstDistance) return false;
    float3 a=origin+direction*firstDistance,b=origin+direction*endDistance;
    r.start=float3(project(a,s),1-1/a.z);
    r.delta=float3(project(b,s),1-1/b.z)-r.start;
    if (!all(isfinite(r.start))||!all(isfinite(r.delta))) return false;
    if (!validUV(r.start.xy)) { r.endReason=TraceEdge; return false; }
    r.end=1;
    for (uint axis=0;axis<2;++axis) {
        float d=r.delta[axis];
        if (abs(d)<1e-10) continue;
        float boundary=d>0 ? .998 : .002;
        float exit=(boundary-r.start[axis])/d;
        if (exit<r.end) { r.end=max(exit,0.0); r.endReason=TraceEdge; }
    }
    float pixelLength=max(abs(r.delta.x)/s.depthTexel.x,abs(r.delta.y)/s.depthTexel.y);
    // Move 0.005 native depth pixels across cell boundaries. A separate
    // lower bound guarantees progress for a ray with almost no screen motion.
    r.epsilon=max(1e-7,.005/max(pixelLength,1.0));
    return r.end>0;
}
float cellExit(ProjectedRay r,uint2 cell,uint2 size,float t) {
    float exit=r.end;
    for (uint axis=0;axis<2;++axis) {
        float d=r.delta[axis];
        if (abs(d)<1e-10) continue;
        float boundary=float(cell[axis]+uint(d>0))/float(size[axis]);
        exit=min(exit,(boundary-r.start[axis])/d);
    }
    return max(exit,t);
}

// Geometry and material acquisition stay outside this function. True G-buffer
// inputs and stochastic directions can replace their estimates later. The
// result contains current-frame radiance only. It has no temporal history.
ReflectionResult traceReflection(texture2d<float> color,depth2d<float> depth,
    texture2d<float,access::read> hierarchy,float2 uv,float3 p,GeometryInfo geometry,
    MaterialInfo material,float3 base,float reflectivity,constant Settings &s) {
    ReflectionResult result={float3(0),0,0,0,0,TraceMiss};
    float raw=1-1/p.z;
    if (farDepthTrust(raw)<.01) { result.reason=TraceDepth; return result; }
    if (hdrSurfaceTrust(base)<.01) { result.reason=TraceHDR; return result; }
    if (reflectivity<=.010) { result.reason=TraceMaterial; return result; }
    if (geometry.continuity<=.04) { result.reason=TraceGeometry; return result; }

    float traceZ=min(p.z,512.0);
    float3 direction=reflect(normalize(p),geometry.normal);
    float3 origin=p+geometry.normal*(traceZ*(.0012+.0010*(1-geometry.continuity)));
    float maxDistance=traceZ*mix(4.1,2.55,material.roughness);
    ProjectedRay ray;
    if (!makeProjectedRay(origin,direction,max(traceZ*.008,.015),maxDistance,s,ray)) {
        result.reason=ray.endReason==TraceEdge ? TraceEdge : TraceDepth;
        return result;
    }

    int maxMip=max(0,int(s.hierarchyLevels)-1);
    int mip=min(maxMip,material.roughness<.25 ? 0 : (material.roughness<.6 ? 1 : 2));
    uint budget=uint(mix(64.0,40.0,material.roughness));
    uint candidateBudget=material.roughness<.25 ? 6 : (material.roughness<.6 ? 4 : 2);
    float t=0;
    for (uint step=0;step<budget && t<ray.end;++step) {
        uint2 size=uint2(hierarchy.get_width(uint(mip)),hierarchy.get_height(uint(mip)));
        float2 cellUV=ray.start.xy+ray.delta.xy*min(t+ray.epsilon,ray.end);
        uint2 cell=min(uint2(clamp(cellUV,0.0,1.0)*float2(size)),size-1);
        float exit=cellExit(ray,cell,size,t);
        float closest=hierarchy.read(cell,uint(mip)).r;
        result.lookups++;
        if (mip==0) result.fineLookups++;
        float entryDepth=ray.start.z+ray.delta.z*t;
        float exitDepth=ray.start.z+ray.delta.z*exit;

        // Both signs of depth slope use the complete cell interval. Only a
        // ray wholly in front of the closest surface can skip this region.
        if (closest>=1 || max(entryDepth,exitDepth)<closest) {
            t=exit+ray.epsilon;
            mip=min(mip+1,maxMip);
            continue;
        }
        if (mip>0) {
            if (entryDepth<closest && ray.delta.z>1e-10)
                t=max(t,min(exit,(closest-ray.start.z)/ray.delta.z));
            mip--;
            continue;
        }

        // Mip 0 is a constant nearest-sampled depth plane within this texel.
        // Solve its intersection analytically in projected depth. Linear
        // view-space Z or binary samples across texel boundaries are incorrect.
        float leafEnd=max(t,exit-ray.epsilon*.5);
        float hitT=t;
        if (abs(ray.delta.z)>1e-10) {
            float planeT=(closest-ray.start.z)/ray.delta.z;
            hitT=ray.delta.z>0 ? max(t,planeT) : min(leafEnd,planeT);
        }
        float hitZ=1/max(1-closest,.00002);
        float thickness=max(.08,min(hitZ,512.0)*(.012+.010*material.roughness));
        float hitRaw=ray.start.z+ray.delta.z*hitT;
        float hitRayZ=1/max(1-hitRaw,.00002);
        float gap=hitRayZ-hitZ;
        if (hitT>=t && hitT<=leafEnd && gap>=-hitZ*1e-5 && gap<thickness*1.8) {
            result.candidates++;
            float2 hitUV=ray.start.xy+ray.delta.xy*hitT;
            float3 hitP=positionWithZ(hitUV,hitZ,s);
            float3 hitRayP=positionWithZ(hitUV,hitRayZ,s);
            float hitDistance=max(0.0,dot(hitRayP-origin,direction));
            float hitFarTrust=farDepthTrust(closest);
            float3 hitColor=max(color.sample(linearSampler,hitUV).rgb,0.0);
            float hitHdrTrust=hdrSurfaceTrust(hitColor);
            float pixelSeparation=length((hitUV-uv)/s.depthTexel);
            float separation=smoothstep(2.5,11.0,pixelSeparation);
            float edge=smoothstep(0.0,.085,min(min(hitUV.x,hitUV.y),min(1-hitUV.x,1-hitUV.y)));
            float travelRatio=hitDistance/max(traceZ,.1);
            float distanceFade=1-smoothstep(.78,.99,hitDistance/maxDistance);
            float valid=1-smoothstep(thickness*.35,thickness*1.8,max(gap,0.0));
            // Keep source reach constant in screen space when guest depth
            // changes from 1600 to 3200 pixels. This is not a distance cut.
            float referencePixels=length((hitUV-uv)*float2(1600,1600/s.aspect));
            GeometryInfo hitGeometry=geometryAt(depth,hitUV,hitP,s);
            float frontFace=smoothstep(.02,.28,dot(hitGeometry.normal,-direction));
            float sourceTrust=sourceDistanceConfidence(hitGeometry,travelRatio,referencePixels,material.roughness);
            float candidate=valid*frontFace*edge*separation*sceneMask(hitUV)*distanceFade*
                geometry.continuity*hitGeometry.continuity*sourceTrust*hitFarTrust*hitHdrTrust;
            if (candidate>.020) {
                result.confidence=candidate;
                result.radiance=fireflyClamp(filteredReflection(color,depth,hitUV,hitZ,material.roughness,travelRatio,s),base);
                result.reason=TraceHit;
                return result;
            }
            result.reason=hitFarTrust<.01 ? TraceDepth : (hitHdrTrust<.01 ? TraceHDR : TraceRejected);
            if (result.candidates>=candidateBudget) return result;
        }
        // A leaf that lies entirely behind its finite thickness is not a hit.
        // Move past this leaf before ascent, including after source rejection.
        t=exit+ray.epsilon;
        mip=min(1,maxMip);
    }
    if (result.reason==TraceMiss) result.reason=t>=ray.end ? ray.endReason : TraceBudget;
    return result;
}
float3 traceReasonColor(uint reason) {
    switch (reason) {
        case TraceHit: return float3(0,1,0);
        case TraceRejected: return float3(1,.08,.03);
        case TraceEdge: return float3(0,.5,1);
        case TraceDepth: return float3(.65,.1,1);
        case TraceMaterial: return float3(1,.7,0);
        case TraceBudget: return float3(1,0,.6);
        case TraceGeometry: return float3(.45,.2,.05);
        case TraceHDR: return float3(0,1,1);
        case TraceUnavailable: return float3(.5);
        default: return float3(0);
    }
}

fragment SurfaceOutput surfaceEffects(VertexOut in [[stage_in]],
    texture2d<float> color [[texture(0)]],depth2d<float> depth [[texture(1)]],
    texture2d<float,access::read> hierarchy [[texture(2)]],constant Settings &s [[buffer(0)]]) {
    float2 uv=in.uv;
    SurfaceOutput output={float4(0,0,0,1),float4(0)};
    uint debug=uint(s.debugView+.5);
    // Check freshness before any depth or hierarchy access.
    if (s.depthAvailable<.5 || s.hierarchyLevels<1) {
        if (debug==8) output.effects=float4(traceReasonColor(TraceUnavailable),1);
        return output;
    }
    float rawDepth=depth.sample(depthSampler,uv);
    if (debug==4) { output.effects=float4(float3(rawDepth),1); return output; }
    if (debug==5) {
        uint mip=min(uint(s.debugMip),uint(s.hierarchyLevels)-1);
        uint2 size=uint2(hierarchy.get_width(mip),hierarchy.get_height(mip));
        float d=hierarchy.read(min(uint2(uv*float2(size)),size-1),mip).r;
        output.effects=float4(float3(d>=1 ? 1 : log2(1/max(1-d,.00002))/12),1);
        return output;
    }
    if (!isfinite(rawDepth)||rawDepth>.99998||rawDepth<.001) {
        if (debug==8) output.effects=float4(traceReasonColor(TraceDepth),1);
        return output;
    }
    float3 base=max(color.sample(linearSampler,uv).rgb,0.0);
    float3 p=positionWithZ(uv,1/max(1-rawDepth,.00002),s);
    GeometryInfo geometry=geometryAt(depth,uv,p,s);
    float3 n=geometry.normal;
    float farTrust=farDepthTrust(rawDepth);
    float hdrTrust=hdrSurfaceTrust(base);
    float surfaceTrust=farTrust*hdrTrust;
    float traceZ=min(p.z,512.0);

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
        bounce+=clampIndirectSample(color.sample(linearSampler,sampleUV).rgb)*weight;
    }
    ao=clamp(1-ao*(s.occlusion/4.0),.58,1.0);
    ao=mix(1.0,ao,surfaceTrust);
    // Bounce is part of the depth-driven indirect-light pass. It now follows
    // the occlusion control, respects HDR/transparent receiver rejection, and
    // cannot remain secretly enabled when occlusion is set to zero.
    bounce*=.035*min(s.occlusion,1.0)*surfaceTrust*geometry.continuity;

    MaterialInfo materialInfo=classifyMaterial(base,geometry);
    float3 view=normalize(p);
    float fresnel=.20+.80*pow(1-clamp(dot(n,-view),0.0,1.0),4.0);
    float roughEnergy=mix(1.0,.72,materialInfo.roughness);
    float reflectivity=s.reflections*materialInfo.weight*fresnel*roughEnergy*surfaceTrust;

    ReflectionResult hit=traceReflection(color,depth,hierarchy,uv,p,geometry,materialInfo,base,reflectivity,s);
    float mask=sceneMask(uv);
    float darkening=smoothstep(0.0,.25,luminance(base)-luminance(hit.radiance));
    float strength=min(reflectivity*hit.confidence,.72)*(1-.72*darkening);
    output.effects=float4(bounce*mask,mix(1.0,ao,mask));
    // Premultiply before the full-resolution pass filters this texture.
    // Separate RGB/alpha interpolation would attenuate hit boundaries twice.
    output.reflection=float4(hit.radiance*strength*mask,strength*mask);

    if (debug==1) output.effects=float4(float3(log2(p.z)/12),1);
    if (debug==2) output.effects=float4(n*.5+.5,1);
    if (debug==3) output.effects=float4(float3(hit.confidence),1);
    // Fixed scales permit comparisons across roughness and scenes. Red is all
    // hierarchy reads / 64, green is mip-0 reads / 64, blue is candidates / 8.
    if (debug==6) output.effects=float4(float3(hit.lookups/64.0,hit.fineLookups/64.0,hit.candidates/8.0),1);
    if (debug==7) output.effects=float4(hit.radiance,1);
    if (debug==8) output.effects=float4(traceReasonColor(hit.reason),1);
    if (debug==9) output.effects=float4(materialInfo.weight,materialInfo.roughness,surfaceTrust,1);
    return output;
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
    texture2d<float> effects [[texture(1)]],texture2d<float> bloomTex [[texture(2)]],
    texture2d<float> reflectionTex [[texture(3)]],constant Settings &s [[buffer(0)]]) {
    float3 raw=max(original.sample(linearSampler,in.uv).rgb,0.0);
    float4 fx=effects.sample(linearSampler,in.uv);
    if (s.debugView>.5) return float4(fx.rgb,1);

    float4 reflection=reflectionTex.sample(linearSampler,in.uv);
    float3 c=max(raw*fx.a+fx.rgb+reflection.rgb-raw*reflection.a,0.0);
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

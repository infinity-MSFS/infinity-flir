// flir-scene — thermal scene model (SM 6.0, compiled with DXC in the bridge).
// Produces per-pixel in-band scene radiance (R16_FLOAT) the sensor model
// develops. Mirrors docs/thermal-model.md step for step. Parameter layout
// matches flir_scene::SceneConstants and material::MaterialTableGpu.

static const uint MATERIAL_COUNT = 9u;
// class indices (match material.rs)
static const uint M_WATER=0,M_VEG=1,M_SOIL=2,M_ASPHALT=3,M_CONCRETE=4,M_URBAN=5,M_SNOW=6,M_METAL=7,M_EMISSIVE=8;

cbuffer SceneCB : register(b0)
{
    float4x4 gInvViewProj;
    float3 gCamWorld;    float _p0;
    float3 gSunDir;      float _p1;

    float gAmbientC;     float gLapse;        float gAircraftAltM; float gGroundAltM;
    float gDiurnal;      float gIsDay;        float gExtinction;   float gWetness;
    float gWaterLoad;    float gSkyDepr;      float gCloudBaseM;    uint  gTier;

    uint2 gSize;         uint  gBand;         uint  gDepthFlags;
    float gNearM;        float gFarM;         float2 _p2;
};

// Material thermal table (one float4-packed array per field; CPU pads each
// entry to its own slot — here declared as float[] for clarity, the bridge
// uploads them 16-byte aligned).
cbuffer MaterialCB : register(b1)
{
    float4 gEmissivity[MATERIAL_COUNT];   // .x used
    float4 gBaseOffset[MATERIAL_COUNT];
    float4 gSunLoading[MATERIAL_COUNT];
    float4 gInertia[MATERIAL_COUNT];
    float4 gWetting[MATERIAL_COUNT];
    float4 gFresnel[MATERIAL_COUNT];
};

Texture2D<float>  gDepth    : register(t0);
Texture2D<float4> gGbufferA : register(t1); // albedo.rgb, a = flags/ao (layout per docs/gbuffer-layout.md)
Texture2D<float4> gGbufferB : register(t2); // normal.xyz (world or view), roughness/metal in some builds
Texture2D<float4> gColour   : register(t3); // HDR lit colour (R11G11B10) for sunlit test / colour-only tier
Texture2D<float4> gCloud    : register(t4); // redirected cloud-merge: premultiplied rgb, a = opacity
Texture1D<float>  gRadiance : register(t5); // band-radiance LUT, normalised, 256 entries over [-80,150]C

RWTexture2D<float> gOut     : register(u0);

static const float T_MIN_C = -80.0, T_MAX_C = 150.0;

float planck(float tC)               // normalised band radiance for temp (C)
{
    float f = saturate((tC - T_MIN_C) / (T_MAX_C - T_MIN_C)) * 255.0;
    return gRadiance.Load(int2((int)(f + 0.5), 0));
}

float airTempAt(float altM)          // lapse from aircraft ambient
{
    return gAmbientC + (gAircraftAltM - altM) * gLapse;
}

// Reconstruct world position from depth + inverse view-projection.
bool worldFromDepth(int2 p, float depth, out float3 world)
{
    world = 0;
    bool reversed = (gDepthFlags & 1u) != 0u;
    // Sky / far plane: reversed-Z stores 0 at infinity; forward stores 1.
    float farVal = reversed ? 0.0 : 1.0;
    if (abs(depth - farVal) < 1e-7) return false;      // sky
    float2 uv = (float2(p) + 0.5) / float2(gSize);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 clip = float4(ndc, depth, 1.0);
    float4 w = mul(gInvViewProj, clip);
    if (abs(w.w) < 1e-9) return false;
    world = w.xyz / w.w;
    return true;
}

float3 normalFromDepth(int2 p)       // depth-derivative normal fallback
{
    float dc = gDepth.Load(int3(p, 0));
    float dx = gDepth.Load(int3(clamp(p + int2(1,0), int2(0,0), int2(gSize)-1), 0)) - dc;
    float dy = gDepth.Load(int3(clamp(p + int2(0,1), int2(0,0), int2(gSize)-1), 0)) - dc;
    // Cross of screen-space tangents, approximate; good enough for N.S terms.
    float3 n = normalize(float3(-dx, -dy, 1e-3));
    return n;
}

// Soft material weights. Full tier reads the G-buffer; depth-only tier guesses
// from lit colour + geometry.
void classify(int2 p, float3 world, float3 normal, out float w[MATERIAL_COUNT])
{
    [unroll] for (uint i = 0; i < MATERIAL_COUNT; ++i) w[i] = 0;

    if (gTier == 1u)  // full: G-buffer
    {
        float4 a = gGbufferA.Load(int3(p, 0));
        float4 b = gGbufferB.Load(int3(p, 0));
        float3 albedo = a.rgb;
        float luma = dot(albedo, float3(0.299, 0.587, 0.114));
        float rough = b.a;                 // placeholder channel per docs
        float metal = a.a;
        float emissive = max(0, luma - 1.0); // HDR emissive hint

        // Flatness → horizontal surfaces (water/roads) vs. vertical (buildings).
        float up = saturate(normal.y);

        // Blue-dominant + flat + smooth → water.
        w[M_WATER]   = saturate((albedo.b - max(albedo.r, albedo.g)) * 3.0) * up * (1.0 - rough);
        // Green-dominant → vegetation.
        w[M_VEG]     = saturate((albedo.g - max(albedo.r, albedo.b)) * 3.0);
        // Dark gray flat → asphalt; light gray → concrete.
        float gray   = 1.0 - saturate(abs(albedo.r-albedo.g)*4 + abs(albedo.g-albedo.b)*4);
        w[M_ASPHALT] = gray * saturate(0.35 - luma) * up;
        w[M_CONCRETE]= gray * saturate(luma - 0.35) * up;
        // Vertical gray → urban/building.
        w[M_URBAN]   = gray * (1.0 - up);
        // Very bright → snow/ice.
        w[M_SNOW]    = saturate((luma - 0.8) * 5.0);
        // Metalness channel.
        w[M_METAL]   = saturate(metal);
        // Brown-ish low-sat → bare soil.
        w[M_SOIL]    = saturate(0.4 - abs(albedo.r - albedo.g*1.2)) * saturate(albedo.r - albedo.b);
        // Emissive hot.
        w[M_EMISSIVE]= saturate(emissive);
    }
    else  // depth-only: coarse guess from lit colour + flatness
    {
        float3 c = gColour.Load(int3(p, 0)).rgb;
        float luma = dot(c, float3(0.299, 0.587, 0.114));
        float up = saturate(normal.y);
        w[M_WATER] = saturate((c.b - max(c.r, c.g)) * 2.0) * up;
        w[M_VEG]   = saturate((c.g - max(c.r, c.b)) * 2.0);
        w[M_CONCRETE] = (1.0 - w[M_WATER] - w[M_VEG]) * up;
        w[M_URBAN] = (1.0 - up) * 0.5;
        w[M_SOIL]  = 0.1;
    }

    // Normalise (soft weights sum to 1); default to concrete if empty.
    float sum = 0; [unroll] for (uint i=0;i<MATERIAL_COUNT;++i) sum += w[i];
    if (sum < 1e-4) { w[M_CONCRETE] = 1.0; sum = 1.0; }
    [unroll] for (uint i=0;i<MATERIAL_COUNT;++i) w[i] /= sum;
}

// Blend a per-class scalar by the weight vector.
float blend(float w[MATERIAL_COUNT], float4 tbl[MATERIAL_COUNT])
{
    float v = 0; [unroll] for (uint i=0;i<MATERIAL_COUNT;++i) v += w[i] * tbl[i].x; return v;
}

// Effective clear-sky brightness temperature by elevation (very cold at
// zenith, approaching air temp at the horizon).
float skyTempC(float3 dir)           // dir: from surface toward sky (world ENU)
{
    float elev = saturate(dir.y);    // 0 horizon, 1 zenith
    return gAmbientC - gSkyDepr * elev;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gSize.x || tid.y >= gSize.y) return;
    int2 p = int2(tid.xy);

    // --- Colour-only tier: weighted-RGB heat proxy, cooling (blue) sky. ---
    if (gTier == 3u)
    {
        float3 c = gColour.Load(int3(p, 0)).rgb;
        float heat = dot(c, float3(0.5, 0.35, 0.15));   // warm-weighted
        // Treat strongly blue pixels as cool sky, not warm.
        float sky = saturate((c.b - max(c.r, c.g)) * 3.0);
        heat = lerp(heat, heat * 0.2, sky);
        gOut[p] = planck(gAmbientC + (heat - 0.3) * 40.0);
        return;
    }

    float depth = gDepth.Load(int3(p, 0));
    float3 world;
    bool isSurface = worldFromDepth(p, depth, world);

    // Camera-to-pixel direction and slant range.
    float3 toCam = gCamWorld - world;
    float range = isSurface ? length(toCam) : gFarM;
    float3 viewDir = isSurface ? normalize(toCam) : float3(0,0,1);

    float Lobserved;

    if (!isSurface)
    {
        // --- Sky ---: look direction is away from camera; approximate with up.
        // Reconstruct a sky ray from NDC for elevation.
        float2 uv = (float2(p) + 0.5) / float2(gSize);
        float2 ndc = float2(uv.x*2-1, 1-uv.y*2);
        float4 farp = mul(gInvViewProj, float4(ndc, 0.5, 1.0));
        float3 rayDir = normalize(farp.xyz / farp.w - gCamWorld);
        float Lsky = planck(skyTempC(rayDir));
        Lobserved = Lsky;
    }
    else
    {
        float3 normal = (gTier == 1u) ? normalize(gGbufferB.Load(int3(p,0)).xyz * 2.0 - 1.0) : normalFromDepth(p);
        float w[MATERIAL_COUNT];
        classify(p, world, normal, w);

        float emis   = blend(w, gEmissivity);
        float baseC  = blend(w, gBaseOffset);
        float sunC   = blend(w, gSunLoading);
        float inertia= blend(w, gInertia);
        float wetting= blend(w, gWetting);
        float fres   = blend(w, gFresnel);

        float altM = world.y;            // ENU up as altitude proxy (sim world Y)
        float Tair = airTempAt(altM);

        // Sunlit test: N.S, optionally confirmed by lit colour brightness.
        float NdotS = saturate(dot(normal, gSunDir));
        float sunlit = NdotS * gIsDay;
        // Diurnal: sun loading when day; retained heat (inertia) when night.
        float dayWarm = sunC * max(0, gDiurnal) * sunlit;
        float nightWarm = inertia * 2.0 * (1.0 - gIsDay);  // retained relative warmth
        float Tsurf = Tair + baseC + dayWarm + nightWarm;

        // Wetness drags toward air temp and water emissivity.
        float wet = saturate(gWetness * wetting + gWetness*0.3);
        Tsurf = lerp(Tsurf, Tair, wet);
        emis  = lerp(emis, 0.98, wet);

        // Emissive/hot sources already carry a big base offset; clamp sane.
        Tsurf = clamp(Tsurf, T_MIN_C, T_MAX_C);

        // Emission + reflection. Fresnel grazing boost for water/metal.
        float grazing = pow(1.0 - saturate(dot(normal, viewDir)), 4.0);
        float reflW = saturate(fres * (0.1 + 0.9 * grazing));
        float3 skyRefl = reflect(-viewDir, normal);
        float Lsky = planck(skyTempC(skyRefl));
        float Lsurface = emis * planck(Tsurf) + (1.0 - emis) * Lsky * reflW + (1.0 - emis) * planck(Tair) * (1.0 - reflW);

        Lobserved = Lsurface;
    }

    // --- Clouds (full/depth tiers) ---
    if (gTier == 1u || gTier == 2u)
    {
        float4 cloud = gCloud.Load(int3(p, 0));
        float opacity = saturate(cloud.a);
        if (opacity > 0.001)
        {
            float Tcloud = airTempAt(gCloudBaseM);
            float Lcloud = planck(Tcloud);
            Lobserved = opacity * Lcloud + (1.0 - opacity) * Lobserved;
            range = min(range, gCloudBaseM);  // atmosphere only to the cloud
        }
    }

    // --- Atmosphere: L = tau*L + (1-tau)*B(Tpath) ---
    float tau = exp(-gExtinction * range);
    float Tpath = airTempAt((gAircraftAltM + max(world.y, gGroundAltM)) * 0.5);
    Lobserved = tau * Lobserved + (1.0 - tau) * planck(Tpath);

    gOut[p] = Lobserved;
}

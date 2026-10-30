// flir-sensor — real-time sensor model (SM 6.0, compiled with DXC in the
// bridge). Mirrors the CPU reference in cpu.rs stage-for-stage. AGC limits and
// the plateau CDF are computed from the PREVIOUS frame's histogram on the CPU
// side (or a prepass) and handed in via the constant buffer, so this pass does
// no cross-pixel reduction and never stalls.
//
// Input  t0 : scene band-radiance, R16_FLOAT (one channel), SRV.
// Input  t1 : plateau CDF, 256-entry R16_FLOAT 1-D texture (from prev frame).
// Output u0 : RGBA8 display image, UAV.

cbuffer SensorCB : register(b0)
{
    uint2 gSize;        // width, height
    float gAgcLo;       // smoothed percentile low
    float gAgcHi;       // smoothed percentile high

    float gPlateau;     // 0..1 blend of plateau-EQ over linear stretch
    float gDetailGain;  // >=0, local-contrast boost
    float gGamma;
    float gNoiseSigma;

    uint  gFrame;       // frame counter for temporal noise
    uint  gPolarity;    // 0 white-hot, 1 black-hot
    uint  gPalette;     // 0 green, 1 gray, 2 ironbow
    uint  gFlags;       // bit0 detail, bit1 noise, bit2 column-fpn, bit3 agc
};

Texture2D<float>   gScene   : register(t0);
Texture1D<float>   gCdf     : register(t1);
RWTexture2D<float4> gOut     : register(u0);

static const uint F_DETAIL = 1u;
static const uint F_NOISE  = 2u;
static const uint F_FPN    = 4u;
static const uint F_AGC    = 8u;

float hash(uint x)
{
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return (float)x / 4294967295.0;
}

float3 palette(uint p, float v)
{
    v = saturate(v);
    if (p == 1u) return float3(v, v, v);                        // gray
    if (p == 2u)                                                // ironbow
    {
        const float stopsP[6] = { 0.00, 0.20, 0.45, 0.70, 0.90, 1.00 };
        const float3 stopsC[6] = {
            float3(0,0,0), float3(0.13,0,0.45), float3(0.70,0.05,0.55),
            float3(0.98,0.45,0.05), float3(1.0,0.85,0.20), float3(1.0,1.0,0.95) };
        [unroll] for (int i = 0; i < 5; ++i)
        {
            if (v <= stopsP[i+1])
            {
                float t = saturate((v - stopsP[i]) / (stopsP[i+1] - stopsP[i]));
                return lerp(stopsC[i], stopsC[i+1], t);
            }
        }
        return stopsC[5];
    }
    return float3(v*v*0.45, v, v*v*0.35);                       // phosphor green
}

// 3x3 box mean as a cheap base estimate. The CPU reference uses a wider blur;
// the bridge runs a separable blur prepass into t-slot and samples it here when
// a larger radius is wanted. This inline version keeps the single-pass path
// correct for small radii.
float baseEstimate(int2 p)
{
    float acc = 0.0;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
    [unroll] for (int dx = -1; dx <= 1; ++dx)
    {
        int2 q = clamp(p + int2(dx, dy), int2(0,0), int2(gSize) - 1);
        acc += gScene.Load(int3(q, 0));
    }
    return acc / 9.0;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gSize.x || tid.y >= gSize.y) return;
    int2 p = int2(tid.xy);

    float scene = gScene.Load(int3(p, 0));

    // 1. AGC linear stretch.
    float span = max(gAgcHi - gAgcLo, 1e-4);
    float v = saturate((scene - gAgcLo) / span);

    // plateau-EQ blend from the handed-in CDF.
    if ((gFlags & F_AGC) && gPlateau > 0.0)
    {
        float eq = gCdf.Load(int2((int)(saturate(v) * 255.0 + 0.5), 0));
        v = lerp(v, eq, gPlateau);
    }

    // 2. detail (DDE).
    if (gFlags & F_DETAIL)
    {
        float b = baseEstimate(p);
        float detail = v - b;
        v = saturate(b + detail * (1.0 + gDetailGain));
    }

    // 3. gamma.
    if (abs(gGamma - 1.0) > 1e-3) v = pow(saturate(v), max(gGamma, 1e-3));

    // 4. column FPN + temporal noise + vignette.
    if (gFlags & F_FPN)
        v = saturate(v + (hash((uint)p.x ^ 0x9e37u) - 0.5) * 0.02);
    if (gFlags & F_NOISE)
        v = saturate(v + (hash(((uint)(p.y * gSize.x + p.x)) ^ (gFrame * 2654435761u)) - 0.5) * 2.0 * gNoiseSigma);

    float2 c = float2(gSize) * 0.5;
    float2 d = (float2(p) - c);
    float r2 = dot(d, d) / dot(c, c);
    v *= 1.0 - 0.18 * r2;

    // 5. polarity + palette.
    v = saturate(v);
    if (gPolarity == 1u) v = 1.0 - v;
    gOut[p] = float4(palette(gPalette, v), 1.0);
}

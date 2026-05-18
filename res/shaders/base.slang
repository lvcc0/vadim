struct VSInput
{
    float2 inPos;
    float3 inColor;
};

struct VSOutput
{
    float4 pos : SV_Position;
    float3 color;
};

struct UniformBuffer
{
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};
ConstantBuffer<UniformBuffer> ubo;

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;

    output.pos = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPos, 0.0, 1.0))));
    output.color = input.inColor;
    
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput inVert) : SV_Target
{
    return float4(inVert.color, 1.0);
}


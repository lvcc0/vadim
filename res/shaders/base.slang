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

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;

    output.pos = float4(input.inPos, 0.0, 1.0);
    output.color = input.inColor;
    
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput inVert) : SV_Target
{
    return float4(inVert.color, 1.0);
}


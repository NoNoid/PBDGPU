kernel void postSolveUpdate(
global float3* predPos,
global float3* posCorr)
{
    size_t i = get_global_id(0);

    predPos[i] += posCorr[i];
}
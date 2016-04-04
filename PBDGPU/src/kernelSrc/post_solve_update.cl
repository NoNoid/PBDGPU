kernel void postSolveUpdate(
global float3* predPos,
global float3* posCorr,
global int* numConstraints)
{
    size_t i = get_global_id(0);

    float overRelaxationParameter = 2.f;

    predPos[i] +=  (overRelaxationParameter/(float)numConstraints[i]  ) * /**/ posCorr[i];
}
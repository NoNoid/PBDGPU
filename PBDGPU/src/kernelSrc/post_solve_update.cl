kernel void postSolveUpdate(
global float3* predPos,
global float3* posCorr,
global int* numConstraints)
{
    size_t i = get_global_id(0);

    float overRelaxationParameter = 1.f;

    int n = numConstraints[i];

    if(n > 0)
    {
        float3 tmp = (overRelaxationParameter/(float)n  ) *  posCorr[i];

        //printf("%d +=  (%f/%d  ) * %2.2v3hlf = %2.2v3hlf\n",i,overRelaxationParameter,numConstraints[i],posCorr[i],tmp);

        predPos[i] += tmp;
    }
}
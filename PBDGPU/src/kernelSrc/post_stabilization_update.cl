kernel void postStabilizationUpdate(
global float3* predPos,
global pbd_particle* p,
global float3* posCorr,
global int* numConstraints)
{
    size_t i = get_global_id(0);

    float overRelaxationParameter = 1.f;

    int n = numConstraints[i];

    if(n > 0)
    {
        float3 tmp = (overRelaxationParameter/(float)n  ) *  posCorr[i];

#ifdef PBDGPU_DEBUG_PRINT
        printf("i = %3d | %8.4v3hlf |  %3d | = %8.4v3hlf\n",i,posCorr[i],numConstraints[i],tmp);
#endif

        predPos[i] += tmp;
        p[i].x += tmp;
    }
}
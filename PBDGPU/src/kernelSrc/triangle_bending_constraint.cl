kernel void triangleBendingConstraint(
global pbd_particle* p,
global float3* pred_pos,
global float3* posCorr,
global int* numConstraints,
global pbd_triangleBendingConstraintData* data,
constant pbd_simulationParameters *params)

{
    size_t i = get_global_id(0);

    pbd_triangleBendingConstraintData cData = data[i];

    float3 b0 = pred_pos[cData.index_b0];
    float3 b1 = pred_pos[cData.index_b1];;
    float3 v  = pred_pos[cData.index_v];

    float3 c = 0.33333333333f * (b0 + b1 +v);
    float h = length(v-c);
    float C_triangle = h -(cData.curvature+cData.restLength);

    if(h > 1e-5f)
    {
        float k_prime = 1.f - pow(1.f - cData.k, 1.f / params-> numIterations);

        float invMass_b0 = p[cData.index_b0].invmass;
        float invMass_b1 = p[cData.index_b1].invmass;
        float invMass_v = p[cData.index_v].invmass;

        // generalized inverse mass of the triangle
        float W = invMass_b0 + invMass_b1 + 2 * invMass_v;



        const float3 constantTerm = (v-c)*(1.f - (cData.curvature + cData.restLength / h));

        float3 dp_b0 = ((2*invMass_b0) / W ) * k_prime * constantTerm;
        float3 dp_b1 = ((2*invMass_b1) / W ) * k_prime * constantTerm;
        float3 dp_v = ((4*invMass_v ) / W ) * k_prime * constantTerm;

#ifdef PBDGPU_DEBUG_PRINT_KERNEL
    printf(\
"i = %d\nh = %f\nk_prime = %f\nW = %f\nconstantTerm = %8.4v3hlf\n %3d | dp_b0 = %8.4v3hlf\n %3d | dp_b1 = %8.4v3hlf\n %3d | dp_v = %8.4v3hlf\n\n\n",\
i,h,k_prime,W,constantTerm,cData.index_b0,dp_b0,cData.index_b1,dp_b1,cData.index_v,dp_v);
#endif

        float3_atomic_add(posCorr+cData.index_b0, dp_b0);
        float3_atomic_add(posCorr+cData.index_b1, dp_b1);
        float3_atomic_add(posCorr+cData.index_v, dp_v);

        atomic_inc(numConstraints+cData.index_b0);
        atomic_inc(numConstraints+cData.index_b1);
        atomic_inc(numConstraints+cData.index_v);
    }
}
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

    float 3 c = 0.33333333333f * (b0 + b1 +v);

    C_triangle = length(v-c)-(cData.curvature+cData.restLength);

    if(C_triangle < 0.0f)
    {
        float k_prime = 1.f - pow(1.f - cData.stiffness, 1.f / params-> numIterations);

        float invMass_b0 = p[cData.index_b0].invmass;
        float invMass_b1 = p[cData.index_b1].invmass;
        float invMass_v = p[cData.index_v].invmass;

        // generalized inverse mass of the triangle
        float W = invMass_b0 + invMass_b1 + 2 * invMass_v;

        const float3 constantTerm = (v-c)*(1.f - (cData.curvature + cData.restLength / length(v-c)));

        float3 dp_b0 = ((2*invMass_b0) / W ) * k_prime * constantTerm;
        float3 dp_b1 = ((2*invMass_b1) / W ) * k_prime * constantTerm;
        float3 dp_v = ((4*invMass_v ) / W ) * k_prime * constantTerm;

        float3_atomic_add(posCorr+cData.index_b0, dp_b0);
        float3_atomic_add(posCorr+cData.index_b1, dp_b1);
        float3_atomic_add(posCorr+cData.index_v, dp_v);

        atomic_inc(numConstraints+cData.index_b0);
        atomic_inc(numConstraints+cData.index_b1);
        atomic_inc(numConstraints+cData.index_v);
    }
}
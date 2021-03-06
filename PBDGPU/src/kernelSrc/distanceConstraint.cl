kernel void distanceConstraint(
global pbd_particle* p,
global float3* pred_pos,
global pbd_distanceConstraintData* data,
constant pbd_simulationParameters *params)

{
    size_t i = get_global_id(0);

    int pi = 0;

    pbd_distanceConstraintData d = data[i];

    float3 p1 = pred_pos[d.index0];
    float3 p2 = pred_pos[d.index1];

    float w1 = p[d.index0].invmass;
    float w2 = p[d.index1].invmass;

    float3 difference = p1 - p2;
    float distance = length(difference);
    float3 derivate = difference/distance;

    float w = w1 + w2;
    float s_pre  = (distance - d.d)/w;
    float s = w < 1e-10f ? 0.f : s_pre;

    float k_2 = pow(1.f-(1.f-0.5f),1.f/params->numIterations);
    float3 dp1 = - (k_2) * s * w1 * derivate;
    float3 dp2 = (k_2) * s * w2 * derivate;


    float3_atomic_add(pred_pos+d.index0,dp1);
    float3_atomic_add(pred_pos+d.index1,dp2);
}

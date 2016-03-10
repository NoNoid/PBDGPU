kernel void distanceConstraint(
global pbd_particle* p,
global float3* pred_pos,
global pbd_distanceConstraintData* data)

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
    float s = (distance - d.d)/(w1 + w2);

    float3 dp1 = - s * w1 * derivate;
    float3 dp2 = s * w2 * derivate;

    pred_pos[d.index0] = dp1;
    pred_pos[d.index1] = dp2;
}

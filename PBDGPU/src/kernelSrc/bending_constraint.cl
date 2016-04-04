kernel void bendingConstraint(
global pbd_particle* p,
global float3* pred_pos,
global float3* posCorr,
global pbd_bendingConstraintData* data,
constant pbd_simulationParameters *params)

{
    size_t i = get_global_id(0);

    pbd_bendingConstraintData cData = data[i];

    float3 p1 = pred_pos[cData.index1];
    float3 p2 = pred_pos[cData.index2];
    float3 p3 = pred_pos[cData.index3];
    float3 p4 = pred_pos[cData.index4];

    float3 n1 = cross(p2-p1,p3-p1)/length(cross(p2-p1,p3-p1));
    float3 n2 = cross(p2-p1,p4-p1)/length(cross(p2-p1,p4-p1));

    float d = dot(n1,n2);

    float3 q3 = (cross(p2,n2)+cross(n1,p2)*d)/length(cross(p2,p3));
    float3 q4 = (cross(p2,n1)+cross(n2,p2)*d)/length(cross(p2,p4));
    float3 q2 = ((cross(p3,n2)+cross(n1,p3)*d)/length(cross(p2,p3))) - ((cross(p4,n1)+cross(n2,p4)*d)/length(cross(p2,p4)));
    float3 q1 = q2-q3-q4;

    float w1 = p[cData.index1].invmass;
    float w2 = p[cData.index2].invmass;
    float w3 = p[cData.index3].invmass;
    float w4 = p[cData.index4].invmass;

    float tmp = w1*length(q1)*length(q1)+w2*length(q2)*length(q2)+w3*length(q3)*length(q3)+w4*length(q4)*length(q4);

    float3 dp1 = ((-w1*sqrt(1-d*d)*(acos(d)-cData.phi))/(tmp))*q1;
    float3 dp2 = ((-w2*sqrt(1-d*d)*(acos(d)-cData.phi))/(tmp))*q2;
    float3 dp3 = ((-w3*sqrt(1-d*d)*(acos(d)-cData.phi))/(tmp))*q3;
    float3 dp4 = ((-w4*sqrt(1-d*d)*(acos(d)-cData.phi))/(tmp))*q4;

    float k_2 = pow(1.f-(1.f-cData.k),1.f/params->numIterations);

    pred_pos[cData.index1] = k_2 * dp1;
    pred_pos[cData.index2] = k_2 * dp2;
    pred_pos[cData.index3] = k_2 * dp3;
    pred_pos[cData.index4] = k_2 * dp4;
}
kernel void bendingConstraint(
global pbd_particle* p,
global float3* pred_pos,
global float3* posCorr,
global int* numConstraints,
global pbd_bendingConstraintData* data,
constant pbd_simulationParameters *params)

{
    size_t i = get_global_id(0);

    pbd_bendingConstraintData cData = data[i];

    float3 p1 = pred_pos[cData.index1];
    float3 p2 = pred_pos[cData.index2] - p1;
    float3 p3 = pred_pos[cData.index3] - p1;
    float3 p4 = pred_pos[cData.index4] - p1;

    float3 n1 = cross(p2,p3)/length(cross(p2,p3));
    float3 n2 = cross(p2,p4)/length(cross(p2,p4));

    float d = dot(n1,n2);

    float3 q3 = (cross(p2,n2)+cross(n1,p2)*d)/length(cross(p2,p3));

/*
printf("\n\
n1 = %v3hlf\n\
n2 = %v3hlf\n\
cross(p2,n2) = %v3hlf\n\
cross(p2,n1) = %v3hlf\n\
(cross(p2,n2)+cross(p2,n1)*d) = %v3hlf\n\
",n1\
,n2\
,cross(p2,n2)\
,cross(p2,n1)\
,(cross(p2,n2)+cross(p2,n1)*d)\
);
*/

    float3 q4 = (cross(p2,n1)+cross(n2,p2)*d)/length(cross(p2,p4));
    float3 q2 = ((cross(p3,n2)+cross(n1,p3)*d)/length(cross(p2,p3))) - ((cross(p4,n1)+cross(n2,p4)*d)/length(cross(p2,p4)));
    float3 q1 = q2-q3-q4;

    float w1 = p[cData.index1].invmass;
    float w2 = p[cData.index2].invmass;
    float w3 = p[cData.index3].invmass;
    float w4 = p[cData.index4].invmass;

    float tmp = w1*length(q1)*length(q1)+w2*length(q2)*length(q2)+w3*length(q3)*length(q3)+w4*length(q4)*length(q4);

    if(tmp > 1e-5f)
    {

        float k_2 = pow(1.f-(1.f-cData.k),1.f/params->numIterations);

        float3 dp1 = k_2 *((-w1*sqrt(1-d*d)*(acos(d)-cData.phi))/(tmp))*q1;
        float3 dp2 = k_2 *((-w2*sqrt(1-d*d)*(acos(d)-cData.phi))/(tmp))*q2;
        float3 dp3 = k_2 *((-w3*sqrt(1-d*d)*(acos(d)-cData.phi))/(tmp))*q3;
        float3 dp4 = k_2 *((-w4*sqrt(1-d*d)*(acos(d)-cData.phi))/(tmp))*q4;


/*        printf("i = %d\n\
 p1 = %2.2v3hlf\n p2 = %2.2v3hlf\n p3 = %2.2v3hlf\n p4 = %2.2v3hlf\n\
n1 = %2.2v3hlf\nn2 = %2.2v3hlf\nd = %f\n\
 q1 = %v3hlf\n q2 = %v3hlf\n q3 = %v3hlf\n q4 = %v3hlf\n\
w1 = %f\nw2 = %f\nw3 = %f\nw4 = %f\n\
 tmp = %f\n k_2 = %f\n\
dp1 = %2.2v3hlf\ndp2 = %2.2v3hlf\ndp3 = %2.2v3hlf\ndp4 = %2.2v3hlf\n\n"\
,i,p1,p2,p3,p4,n1,n2,d,q1,q2,q3,q4,w1,w2,w3,w4,tmp,k_2,dp1,dp2,dp3,dp4);
/**/

        float3_atomic_add(posCorr+cData.index1, dp1);
        float3_atomic_add(posCorr+cData.index2, dp2);
        float3_atomic_add(posCorr+cData.index3, dp3);
        float3_atomic_add(posCorr+cData.index4, dp4);

        atomic_inc(numConstraints+cData.index1);
        atomic_inc(numConstraints+cData.index2);
        atomic_inc(numConstraints+cData.index3);
        atomic_inc(numConstraints+cData.index4);

    }
}
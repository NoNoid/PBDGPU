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

    float3 p0 = pred_pos[cData.index1];
    float3 p1 = pred_pos[cData.index2] - p0;
    float3 p2 = pred_pos[cData.index3] - p0;
    float3 p3 = pred_pos[cData.index4] - p0;

    float invMass0 = p[cData.index1].invmass;
    float invMass1 = p[cData.index2].invmass;
    float invMass2 = p[cData.index3].invmass;
    float invMass3 = p[cData.index4].invmass;

	if (!(invMass0 == 0.0f && invMass1 == 0.0f))
	{

        float3 e = p3-p2;
        float  elen = length(e);
        if (!(elen < 1e-6f))
        {

            float invElen = 1.0f / elen;

            float3 n1 = normalize(cross(p2-p0,p3-p0)); n1 *= n1;
            float3 n2 = normalize(cross(p3 - p1,p2 - p1)); n2 *= n2;

            float3 d0 = elen*n1;
            float3 d1 = elen*n2;
            float3 d2 = dot(p0-p3,e) * invElen * n1 + dot(p1-p3,e) * invElen * n2;
            float3 d3 = dot(p2-p0,e) * invElen * n1 + dot(p2-p1,e) * invElen * n2;

            normalize(n1);
            normalize(n2);

            float dotprod = dot(n1,n2);

            if (dotprod < -1.0f) dotprod = -1.0f;
            if (dotprod >  1.0f) dotprod =  1.0f;
            float phi = acos(dotprod);

            // float phi = (-0.6981317f * dotprod * dotprod - 0.8726646f) * dotprod + 1.570796f;	// fast approximation

            float d0len = length(d0);
            float d1len = length(d1);
            float d2len = length(d2);
            float d3len = length(d3);

            float lambda =
                invMass0 * d0len * d0len +
                invMass1 * d1len * d1len +
                invMass2 * d2len * d2len +
                invMass3 * d3len * d3len;

            if (!(lambda == 0.0f))
            {

                // stability
                // 1.5 is the largest magic number I found to be stable in all cases :-)
                //if (stiffness > 0.5f && fabs(phi - b.restAngle) > 1.5f)
                //	stiffness = 0.5f;

                // maybe use k_prime
                lambda = (phi - cData.phi) / lambda * pow(1.f - (1.f - cData.k), 1.f / params->numIterations);

                if (dot(cross(n1,n2),e) > 0.0f)
                    lambda = -lambda;

                float3 dp0 = - invMass0 * lambda * d0;
                float3 dp1 = - invMass1 * lambda * d1;
                float3 dp2 = - invMass2 * lambda * d2;
                float3 dp3 = - invMass3 * lambda * d3;

                /*    printf(\
                "i = %d\ndp0 = %2.2v3hlf\ndp1 = %2.2v3hlf\ndp2 = %2.2v3hlf\ndp3 = %2.2v3hlf\n\n\n",\
                i,dp0,dp1,dp2,dp3);
                /**/

                float3_atomic_add(posCorr+cData.index1, dp0);
                float3_atomic_add(posCorr+cData.index2, dp1);
                float3_atomic_add(posCorr+cData.index3, dp2);
                float3_atomic_add(posCorr+cData.index4, dp3);

                atomic_inc(numConstraints+cData.index1);
                atomic_inc(numConstraints+cData.index2);
                atomic_inc(numConstraints+cData.index3);
                atomic_inc(numConstraints+cData.index4);
            }
        }
    }
}
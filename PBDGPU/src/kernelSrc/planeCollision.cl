kernel void planeCollision(
global pbd_particle *p,
global float3 *pred_x,
global float3 *posCorr,
global float4 *planes,
private uint numPlanes,
constant pbd_simulationParameters *params)
{
    size_t i = get_global_id(0);

    //if(i==0) printf("planeKernelHello");

    float3 currentPos = pred_x[i];

	for (int planeIndex = 0; planeIndex < numPlanes; ++planeIndex)
	{
		// layout = n.x n.y n.z d | n = plane normal
		float4 plane = planes[0];
		float3 n = (float3)(plane.x, plane.y, plane.z);
		float d = plane.w;

		float C = dot(n, currentPos) - d;

		if (C < 0)
		{
			float w = p[i].invmass;
			float s = C / w;
			//float3 dX = - (C/dot(n,n)*n);
			float3 dX = -s*w*n;

			float k_2 = pow(1.f - (1.f - 0.9f), 1.f / params->numIterations);
			posCorr[i] += k_2 * dX;
		}
	}
}
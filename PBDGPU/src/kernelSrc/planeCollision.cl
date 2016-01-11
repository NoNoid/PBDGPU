kernel void planeCollision(
global pbd_particle *p,
global float3 *pred_x,
global float4 *planes,
private uint numPlanes)
{
    size_t i = get_global_id(0);

    //if(i==0) printf("planeKernelHello");

    float3 currentPos = pred_x[i];

    // layout = n.x n.y n.z d | n = plane normal
    float4 plane = planes[0];
    float3 n = (float3)(plane.x,plane.y,plane.z);
    float d = plane.w;

    float C = dot(n,currentPos)-d;

    if(C < 0)
    {
        float w = p[i].invmass;
        float s = C/(w*dot(n,n));
        float3 dX = - (C/dot(n,n)*n);

        pred_x[i] = currentPos + dX;
    }

}
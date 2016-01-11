kernel void planeCollision(
global pbd_particle *p,
global float3 *pred_x,
global float4 *planes,
private uint numPlanes)
{
    size_t i = get_global_id(0);

    if(i==0) printf("planeKernelHello");
}
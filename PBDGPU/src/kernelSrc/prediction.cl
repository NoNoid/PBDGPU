kernel void prediction(
    global pbd_particle *p,
    global float3 *fext,
    global float3 *pred_x,
    global float *m,
    global float *scaled_m,
    private float3 g,
    private float dt)
{
    int i = get_global_id(0);

    p[i].v      = p[i].v + (fext[i] + g) * p[i].invmass * dt;
    pred_x[i]   = p[i].x + (p[i].v * dt);
    scaled_m[i] = m[i] * exp(-1.0f/*-k*/ * pred_x[i].z);
}

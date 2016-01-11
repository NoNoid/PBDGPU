kernel void prediction(
    global pbd_particle *p,
    global float3 *fext,
    global float3 *pred_x,
    global float *m,
    global float *scaled_m,
    private float3 g,
    private float dt)
{
    size_t i = get_global_id(0);

    pbd_particle particle = p[i];
    float3 externalForce = fext[i];    
  
    float3 vel = particle.v + ((externalForce + g) * particle.invmass * dt);
    p[i].v = vel;
    pred_x[i] = particle.x + (vel * dt);

    float mass = m[i];
    float height = pred_x[i].z;

    //if(i==0) printf("test");
    scaled_m[i] = mass * exp(-1.0f/*-k*/ * height);
}

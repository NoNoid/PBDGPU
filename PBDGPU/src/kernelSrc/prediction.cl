kernel void prediction(
    global pbd_particle *p,
    global float3 *fext,
    global float3 *pred_x,
    global float *m,
    global float *scaled_m,
    constant pbd_simulationParameters *params)
{
    size_t i = get_global_id(0);

    pbd_particle particle = p[i];
    float3 externalForce = fext[i];    
  
    float3 vel = particle.v + ((externalForce + params->gravity) * particle.invmass * params->timeStep);
    float3 predictedDelta = (.95f * vel * params->timeStep);

#ifdef PBDGPU_DEBUG_PRINT
    printf("i = %3d | pD = %8.4v3hlf\n",i,predictedDelta);
#endif

    pred_x[i] = particle.x + predictedDelta;

    float mass = m[i];
    float height = pred_x[i].z;

    scaled_m[i] = mass * exp(-1.0f/*-k*/ * height);
}

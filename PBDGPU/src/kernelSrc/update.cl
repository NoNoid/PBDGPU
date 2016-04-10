kernel void update(
        global pbd_particle *p,
        global float3 *pred_x,
        constant pbd_simulationParameters *params)
{
    size_t i = get_global_id(0);

    float3 position = p[i].x;
    float3 predictedPosition = pred_x[i];
    float3 finalDelta = predictedPosition-position;

#ifdef PBDGPU_DEBUG_PRINT
    printf("i = %3d | fD = %8.4v3hlf\n",i,finalDelta);
#endif

    float3 updatedVelocity = (1.0f/params->timeStep)  * finalDelta;

    p[i].v = updatedVelocity;
    p[i].x = predictedPosition;
}
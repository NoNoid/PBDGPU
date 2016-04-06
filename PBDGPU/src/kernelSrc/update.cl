kernel void update(
        global pbd_particle *p,
        global float3 *pred_x,
        constant pbd_simulationParameters *params)
{
    size_t i = get_global_id(0);

    float3 position = p[i].x;
    float3 predictedPosition = pred_x[i];
    float3 finalDelta = predictedPosition-position;

    //printf("finalDelta = %2.2v3hlf\n",finalDelta);

    float3 updatedVelocity = (1.0f/params->timeStep)  * finalDelta;

    //printf("updatedVelocity = %2.2v3hlf\n",updatedVelocity);

    p[i].v = updatedVelocity;
    p[i].x = predictedPosition;
}
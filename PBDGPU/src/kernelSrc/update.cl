kernel void update(
        global pbd_particle *p,
        global float3 *pred_x,
        constant pbd_simulationParameters *params)
{
    size_t i = get_global_id(0);

    float3 position = p[i].x;
    float3 predictedPosition = pred_x[i];
    float3 updatedVelocity = 1/params->timeStep * (predictedPosition-position);

    p[i].v = updatedVelocity;
    p[i].x = predictedPosition;
}
kernel void update(
        global pbd_particle *p,
        global float3 *pred_x,
        global int numConstraints,
        constant pbd_simulationParameters *params)
{
    size_t i = get_global_id(0);

    float3 position = p[i].x;
    float3 predictedPosition = pred_x[i];
    float3 finalDelta = predictedPosition-position;

    finalDelta *= 1/float(numConstraints[i]);

    //printf("finalDelta = %2.2v3hlf\n",finalDelta);

    float3 updatedVelocity = (1/params->timeStep) * .95f * finalDelta;

    //printf("updatedVelocity = %2.2v3hlf\n",updatedVelocity);

    p[i].v = updatedVelocity;
    p[i].x = predictedPosition;
}
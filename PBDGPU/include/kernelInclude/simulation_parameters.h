//
// Created by tobi on 11.03.16.
//

#ifndef PBDGPU_SIMULATION_PARAMETERS_H
#define PBDGPU_SIMULATION_PARAMETERS_H

#ifndef __OPENCL_C_VERSION__
#include <clew.h>
#endif

typedef struct pbd_simulationParameters
{
    cl_float timeStep;
    cl_uint numIterations;
    cl_float3 gravity;

} pbd_simulationParameters;

#endif //PBDGPU_SIMULATION_PARAMETERS_H

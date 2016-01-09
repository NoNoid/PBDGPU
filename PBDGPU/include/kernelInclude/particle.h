#ifndef PBD_PARTICLE
#define PBD_PARTICLE

#ifndef __OPENCL_C_VERSION__
#include <clew.h>
#endif

typedef struct pbd_particle
{
    cl_float3 x;
    cl_float3 v;
    cl_float invmass;
    cl_int phase;
} pbd_particle;

#endif

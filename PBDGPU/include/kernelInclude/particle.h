#ifndef OCL_PARTICLE
#define OCL_PARTICLE

#include <clew.h>

struct particle
{
    cl_float4 x;
    cl_float4 v;
    cl_float invmass;
    cl_int phase;
};

#endif

//
// Created by tobi on 13.03.16.
//

#ifndef PBDGPU_BENDING_CONSTRAINT_DATA_H
#define PBDGPU_BENDING_CONSTRAINT_DATA_H

#ifndef __OPENCL_C_VERSION__
#include <clew.h>
#endif

typedef struct pbd_bendingConstraintData
{
    cl_uint index1;
    cl_uint index2;
    cl_uint index3;
    cl_uint index4;
    cl_float phi;
    cl_float k;
} pbd_bendingConstraintData;

#endif //PBDGPU_BENDING_CONSTRAINT_DATA_H

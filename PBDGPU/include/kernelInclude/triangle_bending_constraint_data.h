//
// Created by tobi on 07.04.16.
//

#ifndef PBDGPU_TRIANGLE_BENDING_CONSTRAINT_DATA_H
#define PBDGPU_TRIANGLE_BENDING_CONSTRAINT_DATA_H

#ifndef __OPENCL_C_VERSION__
#include <clew.h>
#endif

typedef struct pbd_trianglebBendingConstraintData
{
    cl_uint index_b0;
    cl_uint index_b1;
    cl_uint index_v;
    cl_float restLength;
    cl_float k;
    cl_float curvature;
} pbd_trianglebBendingConstraintData;

#endif //PBDGPU_TRIANGLE_BENDING_CONSTRAINT_DATA_H

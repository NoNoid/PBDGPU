//
// Created by tobi on 10.03.16.
//

#ifndef PBDGPU_DISTANCECONSTRAINTDATA_H
#define PBDGPU_DISTANCECONSTRAINTDATA_H

#ifndef __OPENCL_C_VERSION__
#include <clew.h>
#endif

typedef struct pbd_distanceConstraintData
{
    cl_int index0;
    cl_int index1;
    cl_float d;
} pbd_distanceConstraintData;

#endif //PBDGPU_DISTANCECONSTRAINTDATA_H

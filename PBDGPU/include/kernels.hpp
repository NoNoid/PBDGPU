#ifndef KERNELS_HPP_
#define KERNELS_HPP_

#include <clew.h>

namespace pbdgpu {

    cl_kernel buildPredictionKernel(cl_context context, cl_device_id device);

    cl_kernel buildUpdateKernel(cl_context context, cl_device_id device);

    cl_kernel buildPlaneCollisionKernel(cl_context context, cl_device_id device);

    cl_kernel buildDistanceConstraintKernel(cl_context context, cl_device_id device);

    cl_kernel buildBendingConstraintKernel(cl_context context, cl_device_id device);
}

#endif

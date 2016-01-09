#ifndef KERNELS_HPP_
#define KERNELS_HPP_

#include <clew.h>

namespace pbdgpu {

    cl_kernel buildPredictionKernel(cl_context context, cl_device_id device);

}

#endif

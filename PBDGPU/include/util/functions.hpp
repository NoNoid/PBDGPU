#ifndef _UTILITY_FUNCTIONS_HPP_
#define _UTILITY_FUNCTIONS_HPP_

#include <clew.h>

namespace pbdgpu
{
    cl_context_properties * getOGLInteropInfo(cl_device_id &out_device);
}
#endif

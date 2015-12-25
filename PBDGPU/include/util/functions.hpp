#ifndef _UTILITY_FUNCTIONS_HPP_
#define _UTILITY_FUNCTIONS_HPP_

#include <clew.h>

/** @file
 * Contains utility functions
*/

/**
 * @namespace pbdgpu
 */
namespace pbdgpu
{

    /** @fn cl_context_properties * getOGLInteropInfo(cl_device_id &out_device)
     * A function to search for the OpenCL Device which is associated with the current OpenGL context.
     * Can Only be called in thread which has a current OpenGL context.
     * @param out_device returns the device id of the OpenCL device, currently associated with OpenGL context.
     * @return OpenCl context properties which are needed to correctly initialse the a GL-CL-Interop context.
     */
    cl_context_properties * getOGLInteropInfo(cl_device_id &out_device);
}
#endif

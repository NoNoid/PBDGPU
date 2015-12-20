#include <stdio.h>
#include <util/functions.hpp>

#ifdef _WIN32
#   include <windows.h>
#elif __linux__
#include <GL/glx.h>
#elif __APPLE__

#endif

cl_context_properties *pbdgpu::getOGLInteropInfo(cl_device_id &out_device)
{
	cl_device_id currentOGLDevice;
	cl_platform_id arbitraryplatform;
	clGetPlatformIDs(1, &arbitraryplatform, NULL);

	if (!clGetGLContextInfoKHR)
	{
        clGetGLContextInfoKHR = reinterpret_cast<clGetGLContextInfoKHR_fn>(clGetExtensionFunctionAddressForPlatform(arbitraryplatform, "clGetGLContextInfoKHR"));
		if (!clGetGLContextInfoKHR)
		{
			printf("pbdgpu::getCurrentOGLDevice(): Failed to query proc address for clGetGLContextInfoKHR");
		}
	}

	cl_context_properties properties[] = {
#ifdef _WIN32
        CL_GL_CONTEXT_KHR, reinterpret_cast<cl_context_properties>(wglGetCurrentContext()),
        CL_WGL_HDC_KHR, reinterpret_cast<cl_context_properties>(wglGetCurrentDC()),
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(arbitraryplatform),
#elif __linux__
        CL_GL_CONTEXT_KHR, reinterpret_cast<cl_context_properties>(glXGetCurrentContext()),
        CL_GLX_DISPLAY_KHR, reinterpret_cast<cl_context_properties>(glXGetCurrentDisplay()),
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(arbitraryplatform),
#elif __APPLE__
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
        reinterpret_cast<cl_context_properties>(CGLGetShareGroup(CGLGetCurrentContext())),
#endif
	0
	};

	cl_int err = clGetGLContextInfoKHR(properties, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, sizeof(cl_device_id), &currentOGLDevice, NULL);

    uint length = (sizeof(properties)/sizeof(*properties));
    cl_context_properties *out_properties = new cl_context_properties[length];
    for(uint i = 0; i < length; ++i)
    {
        out_properties[i] = properties[i];
    }

    out_device =  currentOGLDevice;

    return out_properties;
}

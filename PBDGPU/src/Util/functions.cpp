#include <stdio.h>
#include <util\functions.hpp>

#ifdef _WIN32
#   include <windows.h>
#elif __linux__

#elif __APPLE__

#endif

cl_device_id pbdgpu::getCurrentOGLDevice()
{
	cl_device_id currentOGLDevice;
	cl_platform_id arbitraryplatform;
	clGetPlatformIDs(1, &arbitraryplatform, NULL);

	if (!clGetGLContextInfoKHR)
	{
		clGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn)
			clGetExtensionFunctionAddressForPlatform(arbitraryplatform, "clGetGLContextInfoKHR");
		if (!clGetGLContextInfoKHR)
		{
			printf("pbdgpu::getCurrentOGLDevice(): Failed to query proc address for clGetGLContextInfoKHR");
		}
	}

	cl_context_properties properties[] = {
#ifdef _WIN32
		CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
		CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)arbitraryplatform,
#elif __linux__
	cl_context_properties linuxproperties[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
		CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)arbitraryplatform,
#elif __APPLE__
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
		(cl_context_properties)CGLGetShareGroup(CGLGetCurrentContext()),
#endif
	0
	};

	cl_int err = clGetGLContextInfoKHR(properties, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, sizeof(cl_device_id), &currentOGLDevice, NULL);

	return currentOGLDevice;
}
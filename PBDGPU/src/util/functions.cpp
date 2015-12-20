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
	cl_uint num_platforms;

	clGetPlatformIDs(0, nullptr, &num_platforms);

	cl_platform_id *platforms = new cl_platform_id[num_platforms];

	clGetPlatformIDs(num_platforms, platforms, nullptr);

	for (unsigned int i = 0; i < num_platforms; ++i)
	{
		// get the function pointer for 'clGetGLContextInfoKHR' from the current platform
		clGetGLContextInfoKHR = reinterpret_cast<clGetGLContextInfoKHR_fn>(clGetExtensionFunctionAddressForPlatform(platforms[i], "clGetGLContextInfoKHR"));
		if (!clGetGLContextInfoKHR)
		{
			continue;
		}

		// set up the required properties for 'clGetGLContextInfoKHR' and creation of a shared GL CL Context
		cl_context_properties properties[] = {
#ifdef _WIN32
			CL_GL_CONTEXT_KHR, reinterpret_cast<cl_context_properties>(wglGetCurrentContext()),
			CL_WGL_HDC_KHR, reinterpret_cast<cl_context_properties>(wglGetCurrentDC()),
			CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[i]),
#elif __linux__
			CL_GL_CONTEXT_KHR, reinterpret_cast<cl_context_properties>(glXGetCurrentContext()),
			CL_GLX_DISPLAY_KHR, reinterpret_cast<cl_context_properties>(glXGetCurrentDisplay()),
			CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[i]),
#elif __APPLE__
			CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
			reinterpret_cast<cl_context_properties>(CGLGetShareGroup(CGLGetCurrentContext())),
#endif
			0
		};

		size_t param_value_size_ret;
		clGetGLContextInfoKHR(properties, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, sizeof(cl_device_id), &currentOGLDevice, &param_value_size_ret);

		// This means for the current platform exists no which is currently associated with a OpenGL Context
		// -> Continue with next platform
		if (param_value_size_ret == 0) continue;

		// A suitable device was found
		out_device = currentOGLDevice;

		// Write found Context properties to output
		unsigned int length = (sizeof(properties) / sizeof(*properties));
		cl_context_properties *out_properties = new cl_context_properties[length];
		for (unsigned int i = 0; i < length; ++i)
		{
			out_properties[i] = properties[i];
		}
		return out_properties;
	}

	printf("Could not find a OpenCL platform with active OpenGL interop device");
	
	return nullptr;
}

#include <stdio.h>
#include <fstream>
#include <vector>
#include <util/functions.hpp>

#include <GL/glew.h>

#ifdef _WIN32
#   include <windows.h>
#elif __linux__
#include <GL/glx.h>
#elif __APPLE__

#endif

using std::vector;

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

string pbdgpu::readFile(const string filename)
{
    std::ifstream ifs(filename);
    string content( (std::istreambuf_iterator<char>(ifs) ),
                    (std::istreambuf_iterator<char>()    ) );
    ifs.close();
    return content;
}

unsigned int pbdgpu::createShader(const std::string filename, const unsigned int shaderType)
{
    unsigned int shaderID = glCreateShader(shaderType);
    string shaderSource = readFile(filename);
    const char* shaderSourcePtr = shaderSource.c_str();
    const int sourceSize = shaderSource.size();

    glShaderSource(shaderID, 1, &shaderSourcePtr, &sourceSize);
    glCompileShader(shaderID);

    GLint compilationSuccess = GL_FALSE;
    glGetShaderiv(shaderID, GL_COMPILE_STATUS, &compilationSuccess);
    if(compilationSuccess == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &maxLength);

        vector<char> errorLog(maxLength);
        glGetShaderInfoLog(shaderID, maxLength, &maxLength, &errorLog[0]);

        printf("%s",&errorLog[0]);

        glDeleteShader(shaderID);
        return 0;
    }
    return shaderID;
}

unsigned int pbdgpu::createProgram(const unsigned int vertexShader, const unsigned int hullShader, const unsigned int domainShader, const unsigned int fragmentShader)
{
    unsigned int programID = glCreateProgram();

    if(glIsShader(vertexShader))
    {
      glAttachShader(programID,vertexShader);
    }else{
        printf("createProgram: vertex shader invalid but program needs vertex shader. Linking aborted\n");
        return 0;
    }

    if(glIsShader(hullShader))
    {
        glAttachShader(programID,hullShader);
    }

    if(glIsShader(domainShader))
    {
        glAttachShader(programID,domainShader);
    }

    if(glIsShader(fragmentShader))
    {
        glAttachShader(programID,fragmentShader);
    }else{
        printf("createProgram: fragment shader invalid but program needs fragment shader. Linking aborted\n");
        return 0;
    }

    glLinkProgram(programID);

    GLint linkingSuccess = GL_FALSE;
    glGetProgramiv(programID, GL_LINK_STATUS, &linkingSuccess);
    if(linkingSuccess == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &maxLength);

        vector<char> errorLog(maxLength);
        glGetProgramInfoLog(programID, maxLength, &maxLength, &errorLog[0]);

        printf("%s",&errorLog[0]);

        glDeleteProgram(programID);

        return 0;
    }

    glDetachShader(programID,vertexShader);
    glDetachShader(programID,hullShader);
    glDetachShader(programID,domainShader);
    glDetachShader(programID,fragmentShader);

    return programID;
}

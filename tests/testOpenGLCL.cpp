#include <memory>
#include <iostream>


#include <GL/glew.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <clew.h>
#include <util\functions.hpp>

int main(int argc, char **argv)
{	

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(320, 320);
	glutCreateWindow("Nothing to see here");

	glewInit();

	bool clpresent = 0 == clewInit();
	if (!clpresent) {
		throw std::runtime_error("OpenCL library not found");
	}

	printf("OpenGL Version:\t\t%s\nOpenGL Vendor:\t\t%s\nOpenGL Renderer:\t%s\nGLSL Version:\t\t%s\n",
		glGetString(GL_VERSION),
		glGetString(GL_VENDOR),
		glGetString(GL_RENDERER),
		glGetString(GL_SHADING_LANGUAGE_VERSION)
		);

	cl_int err;	

	cl_uint numPlaforms = 0;

	if (clGetPlatformIDs(0, NULL, &numPlaforms) != CL_SUCCESS)
	{
		printf("cannot find OpenCL Platforms");
		exit(1);
	}

	printf("\nOpenCL\n");
	printf("Number of Platforms: %d\n", numPlaforms);

	cl_platform_id* platforms = new cl_platform_id[numPlaforms];

	clGetPlatformIDs(numPlaforms, platforms, NULL);

	char buffer[10240];
	for (cl_uint i = 0; i < numPlaforms; ++i)
	{
		printf("\n\nPlatform: %d\n", i);
		err = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_NAME,
			10240,
			buffer,
			NULL);
		printf("Platformnname:\t\t%s\n", buffer);

		err = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_VENDOR,
			10240,
			buffer,
			NULL);
		printf("Platformvendor:\t\t%s\n", buffer);

		err = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_VERSION,
			10240,
			buffer,
			NULL);
		printf("Platformversion:\t%s\n", buffer);

		err = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_PROFILE,
			10240,
			buffer,
			NULL);
		printf("Platformprofile:\t%s\n", buffer);

		cl_uint numDevices;

		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);

		printf("\n\t%d Device(s) detected.\n", numDevices);

		cl_device_id* devices = new cl_device_id[numDevices];

		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numDevices, devices, &numDevices);

		for (cl_uint i = 0; i < numDevices; ++i)
		{
			printf("\n\n\tDevice: %d\n", i);
			err = clGetDeviceInfo(
				devices[i],
				CL_DEVICE_NAME,
				10240,
				buffer,
				NULL);
			printf("\tDevice Name:\t%s\n", buffer);

			err = clGetDeviceInfo(
				devices[i],
				CL_DEVICE_VENDOR,
				10240,
				buffer,
				NULL);
			printf("\tDevice Vendor:\t%s\n", buffer);

			err = clGetDeviceInfo(
				devices[i],
				CL_DEVICE_VERSION,
				10240,
				buffer,
				NULL);
			printf("\tDevice Version:\t%s\n", buffer);

			err = clGetDeviceInfo(
				devices[i],
				CL_DRIVER_VERSION,
				10240,
				buffer,
				NULL);
			printf("\tDriver Version:\t%s\n", buffer);
		}
	}

	cl_device_id currentOGLDevice = pbdgpu::getCurrentOGLDevice();

	cl_context GLCLContext = clCreateContext(NULL, 1, &currentOGLDevice, NULL, NULL, &err);

	const size_t testsize = 1024;

	int* testdata = new int[testsize];

	for (int i = 0; i < testsize; ++i)
	{
		testdata[i] = i*2;
	}

	GLuint id = 12488;
	glGenBuffers(1, &id);
	glBindBuffer(GL_ARRAY_BUFFER, id);
	glBufferData(GL_ARRAY_BUFFER, testsize*sizeof(int), testdata, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glFinish();

	cl_mem sharedMem = clCreateFromGLBuffer(
		GLCLContext,
		CL_MEM_READ_WRITE,
		id,
		&err);

	cl_command_queue queue = clCreateCommandQueue(GLCLContext, currentOGLDevice, 0, &err);

	const char* kernelSrc = "\n		__kernel void rev(__global int *data)\n	{\n		int gid = get_global_id(0);\n		int getGid = (get_global_size(0)-1-gid);\n		data[gid] = data[getGid];\n	}";

	cl_program program = clCreateProgramWithSource(GLCLContext, 1, &kernelSrc, NULL, &err);

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	//size_t len;
	//char * BLog;
	//clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
	//BLog = new char[len];
	//clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, len, BLog, NULL);
	//printf("%s\n", BLog);

	cl_kernel kernel = clCreateKernel(program, "rev", &err);

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &sharedMem);

	err = clEnqueueAcquireGLObjects(queue, 1, &sharedMem, 0, NULL, NULL);

	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &testsize, NULL, 0, NULL, NULL);

	clFinish(queue);

	err = clEnqueueReleaseGLObjects(queue, 1, &sharedMem, 0, NULL, NULL);

	glBindBuffer(GL_ARRAY_BUFFER, id);

	int* data = (int*)glMapBufferRange(GL_ARRAY_BUFFER, 0, testsize*sizeof(int), GL_MAP_READ_BIT);

	printf("\n");
	for (int i = 0; i < 10; ++i)
	{
		printf("%d ", data[i]);
	}
	printf("... ");
	for (int i = testsize-10; i < testsize; ++i)
	{
		printf("%d ", data[i]);
	}
	printf("\n");
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
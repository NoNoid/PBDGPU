#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <time.h>
#include <limits>

#include "../include/sort/cli.h"
#include "../include/sort/bitonicSortKernels.h"
#include "../include/sort/stl.h"


#include <GL/glew.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <clew.h>
#include <util/functions.hpp>
#include <util/gl_buffer_allocator.hpp>


#define CL_ERRORS 1
#define NORMALS
//#define DATA_TYPE int 
//#define DATA_SIZE 1024
#define WORK_GROUP_SIZE 64 // logical errors occur after work group size > 128

#ifndef _WIN32
#ifndef __APPLE__
#define TIME 1
#define BENCHSIZE 16
#endif
#endif

#include <vector>


int main(int argc, char **argv)
{

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(320, 320);
	glutCreateWindow("Nothing to see here");

	glewInit();

	GLenum gl_err = GL_NO_ERROR;

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



	std::vector<cl_int> errors;
	std::vector<float> verticies;

	int vert_count = 149;
	for (int i = 0; i < vert_count; ++i)
	{
		//minimalist randomNumber-Generator
		float randomValue;
		randomValue = fmod((i * 10007.0F) / 7.0f, 100.0F);
		verticies.push_back(randomValue);
		//printf("i=%d: \t%f \n", i, randomValue);
	}
	
	//  --------------------------
	//
	// pad our verticies with FLT_MAX's
	//
	//  --------------------------
	unsigned int n = verticies.size() - 1;
	unsigned int p2 = 1;

	size_t original_vertex_size = verticies.size();
	do ++p2; while ((n >>= 0x1) != 0);
	size_t padded_size = 0x1 << p2;

	unsigned int padd = 0;

	// it just needs to be larger really
	while (verticies.size() < padded_size)
	{
		verticies.push_back(FLT_MAX);
		++padd;
	}

	//  --------------------------
	//
	// OpenCL stuff
	//
	//  --------------------------
	cl_int clStatus;

	CLI *cli_bsort = (CLI*)malloc(sizeof(CLI));
	cliInitialize(cli_bsort, errors);
	cliBuild(
		cli_bsort,
		bitonic_STL_sort_source,
		"_kbitonic_stl_sort",
		errors);

	// Basic initialization and declaration...
	// Execute the OpenCL kernel on the list
	// Each work item shall compare two elements.
	size_t global_size = padded_size / 2;
	// This is the size of the work group.
	size_t local_size = WORK_GROUP_SIZE;
	// Calculate the Number of work groups.
	size_t num_of_work_groups = global_size / local_size;

	size_t bufferSize = verticies.size()*sizeof(verticies[0]);
	//Create memory buffers on the device for each vector
	cl_mem pInputBuffer_clmem = clCreateBuffer(
		cli_bsort->context,
		CL_MEM_READ_WRITE,
		bufferSize,
		NULL,
		&clStatus);
	errors.push_back(clStatus);
	// create kernel
	//PrintCLIStatus(errors);
	clStatus = clEnqueueWriteBuffer(cli_bsort->cmdQueue, pInputBuffer_clmem, CL_TRUE, 0, bufferSize, &verticies[0], NULL, NULL, NULL);
	errors.push_back(clStatus);

	clStatus = clSetKernelArg(
		cli_bsort->kernel,
		0,
		sizeof(cl_mem),
		(void *)&pInputBuffer_clmem);
	errors.push_back(clStatus);

	unsigned int stage, passOfStage, numStages, temp;
	stage = passOfStage = numStages = 0;

	for (temp = padded_size; temp > 1; temp >>= 1)
		++numStages;

	global_size = padded_size >> 1;
	local_size = WORK_GROUP_SIZE;

	for (stage = 0; stage < numStages; ++stage)
	{
		// stage of the algorithm
		clStatus = clSetKernelArg(
			cli_bsort->kernel,
			1,
			sizeof(int),
			(void *)&stage);
		errors.push_back(clStatus);
		// Every stage has stage + 1 passes
		for (passOfStage = 0; passOfStage < stage + 1; ++passOfStage)
		{
			// pass of the current stage
			//printf("Pass no: %d\n", passOfStage);
			clStatus = clSetKernelArg(
				cli_bsort->kernel,
				2,
				sizeof(int),
				(void *)&passOfStage);
			
			errors.push_back(clStatus);

			//
			// Enqueue a kernel run call.
			// Each thread writes a sorted pair.
			// So, the number of threads (global) should be half the 
			// length of the input buffer.
			//
			clStatus = clEnqueueNDRangeKernel(
				cli_bsort->cmdQueue,
				cli_bsort->kernel,
				1,
				NULL,
				&global_size,
				&local_size,
				0,
				NULL,
				NULL);
			errors.push_back(clStatus);
			
			clFinish(cli_bsort->cmdQueue);
		} //end of for passStage = 0:stage-1
	} //end of for stage = 0:numStage-1

	float *mapped_input_buffer =
		(float *)clEnqueueMapBuffer(
		cli_bsort->cmdQueue,
		pInputBuffer_clmem,
		true,
		CL_MAP_READ,
		0,
		sizeof(float) * padded_size,
		0,
		NULL,
		NULL,
		&clStatus);

	errors.push_back(clStatus);


	//  --------------------------
	//
	// Done
	//
	//  --------------------------

	//PrintCLIStatus(errors);
	std::vector<float> output;

	//Check for sortedness
	bool everythingIsSorted = true;
	float prev, val;
	for (int i = 0; i < vert_count; i++)
	{
		val = mapped_input_buffer[i];
		//printf("i=%d: \t%f \n", i, val);
		if(i > 0 && prev > val)
		{
			everythingIsSorted = false;
			break;
		}
		prev = val;
	}
	if (everythingIsSorted)
		printf("everything is sorted");
	else
		printf("everything is NOT sorted");

	// cleanup...
	int a;
	a = 0;
	return 0;
}

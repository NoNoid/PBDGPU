#include <stdio.h>
#include <cstring>
#include <random>

#include <GL/glew.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <clew.h>
#include <util/functions.hpp>
#include <util/gl_buffer_allocator.hpp>
#include <util/cl_buffer_allocator.hpp>

template <typename T>
bool testBuffer(pbdgpu::GPUMemAllocator* buffer, size_t numElems, void* data)
{
    buffer->write(numElems,data);

    T* mappedPtr =  reinterpret_cast<T*>(buffer->map());
    T* castedPtr = reinterpret_cast<T*>(data);

    bool result = true;
    for(size_t i = 0; i < numElems; ++i)
    {
        result &= (mappedPtr[i] == castedPtr[i]);
    }

    buffer->unmap();

    return result;
}

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
        printf("OpenCL library not found");
        return -1;
    }

    const size_t numELems = 1000;
    const size_t sizeOfElement = sizeof(int);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(-100, 100);

    int* data = new int[numELems];

    for (size_t i=0; i<numELems; ++i)
    {
        data[i] = dist(mt);
    }

    bool testSuccessful = true;
    pbdgpu::GPUMemAllocator *buffer =  new pbdgpu::GLBufferAllocator(sizeOfElement);

    testSuccessful = testBuffer<int>(buffer,numELems,data);
    delete buffer;

    if(!testSuccessful)
    {
        return -1;
    }

    cl_device_id currentOGLDevice;
    cl_context_properties* properties = pbdgpu::getOGLInteropInfo(currentOGLDevice);
	if (!properties)
	{
		// if properties if null abort test because no CLGL interop device could be found
		return -1;
	}
    cl_context GLCLContext = clCreateContext(properties, 1, &currentOGLDevice, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(GLCLContext, currentOGLDevice, 0, nullptr);

    buffer = new pbdgpu::CLBufferAllocator(sizeOfElement,GLCLContext,queue);

    testSuccessful = testBuffer<int>(buffer,numELems,data);
    delete buffer;

    if(!testSuccessful)
    {
        return -1;
    }

    clReleaseCommandQueue(queue);
    clReleaseContext(GLCLContext);
    clReleaseDevice(currentOGLDevice);
}

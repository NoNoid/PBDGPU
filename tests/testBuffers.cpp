#include <stdio.h>
#include <cstring>
#include <random>
#include <vector>

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

using std::vector;

template <typename T>
bool testBuffer(pbdgpu::GPUMemAllocator* buffer, size_t numElems, vector<T> data)
{
    buffer->write(numElems,&data[0]);

    T* mappedPtr =  reinterpret_cast<T*>(buffer->map());

    bool result = true;
    for(size_t i = 0; i < numElems; ++i)
    {
        result &= (mappedPtr[i] == data[i]);
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

    vector<int> data(numELems);

    for (size_t i=0; i<numELems; ++i)
    {
        data[i] = dist(mt);
    }

    bool testSuccessful = true;
    pbdgpu::GLBufferAllocator *buffer1 =  new pbdgpu::GLBufferAllocator(sizeOfElement,numELems);

    testSuccessful = testBuffer<int>(buffer1,numELems,data);

    buffer1->free();

    if(glIsBuffer(buffer1->getBufferID()))
    {
        return -1;
    }

    delete buffer1;

    if(!testSuccessful)
    {
        return -1;
    }

    cl_device_id currentOGLDevice;
    vector<cl_context_properties> properties = pbdgpu::getOGLInteropInfo(currentOGLDevice);
    if (properties.empty())
	{
		// if properties if null abort test because no CLGL interop device could be found
		return -1;
	}
    cl_context GLCLContext = clCreateContext(&properties[0], 1, &currentOGLDevice, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(GLCLContext, currentOGLDevice, 0, nullptr);

    pbdgpu::CLBufferAllocator* buffer2 = new pbdgpu::CLBufferAllocator(GLCLContext,queue,sizeOfElement,numELems);

    testSuccessful = testBuffer<int>(buffer2,numELems,data);

    buffer2->free();

    if(clReleaseMemObject(buffer2->getCLMem()) != CL_INVALID_MEM_OBJECT)
    {
        return -1;
    }

    delete buffer2;

    if(!testSuccessful)
    {
        return -1;
    }

    clReleaseCommandQueue(queue);
    clReleaseContext(GLCLContext);
    clReleaseDevice(currentOGLDevice);
}

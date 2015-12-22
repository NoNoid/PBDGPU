#include <util/cl_buffer_allocator.hpp>

namespace pbdgpu
{

    CLBufferAllocator::~CLBufferAllocator()
    {
        clReleaseMemObject(buffer);
    }

    void CLBufferAllocator::write(size_t numElems, const void *data)
    {
        if(length != numElems)
            setLength(numElems);

        clEnqueueWriteBuffer(
            commandQueue,
            buffer,
            true,
            0,
            length*sizeOfElement,
            data,
            0,
            nullptr,
            nullptr);
    }

    void *CLBufferAllocator::map()
    {
        mappedPtr =  clEnqueueMapBuffer(
            commandQueue,
            buffer,
            true,
            CL_MAP_WRITE,
            0,
            sizeOfElement*length,
            0,
            nullptr,
            nullptr,
            nullptr);
        return mappedPtr;
    }

    void CLBufferAllocator::unmap()
    {
        clEnqueueUnmapMemObject(
           commandQueue,
           buffer,
           mappedPtr,
           0,
           nullptr,
           nullptr);
        mappedPtr = nullptr;
    }

    void CLBufferAllocator::allocate(const size_t length)
    {
        buffer = clCreateBuffer(context, memFlags, sizeOfElement*length, nullptr, nullptr);
    }

}

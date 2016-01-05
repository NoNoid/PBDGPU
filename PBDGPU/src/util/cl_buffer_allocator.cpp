#include <util/cl_buffer_allocator.hpp>

namespace pbdgpu
{

    CLBufferAllocator::~CLBufferAllocator()
    {
        free();
    }

    void CLBufferAllocator::write(size_t numElems, const void *data)
    {
        if(size != numElems)
            return;

        clEnqueueWriteBuffer(
            commandQueue,
            buffer,
            true,
            0,
            size*sizeOfElement,
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
            sizeOfElement*size,
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

    void CLBufferAllocator::allocate(const size_t sizeOfElement, const size_t length)
    {
        GPUMemAllocator::allocate(sizeOfElement,length);
        buffer = clCreateBuffer(context, memFlags, getSizeinBytes(), nullptr, nullptr);
    }

    void CLBufferAllocator::free()
    {
        GPUMemAllocator::free();
        if(buffer)
        {
            clReleaseMemObject(buffer);
            buffer = nullptr;
        }
    }

}



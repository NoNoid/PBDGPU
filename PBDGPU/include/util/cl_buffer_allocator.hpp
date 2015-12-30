#ifndef CL_BUFFER_ALLOCATOR
#define CL_BUFFER_ALLOCATOR

#include <util/gpu_mem_allocator.hpp>
#include <clew.h>

namespace pbdgpu {

/**
 * @brief A GPU memory allocator class, which uses OpenCL buffers to allocate memory.
 */
class CLBufferAllocator : public GPUMemAllocator
{
public:
    /** @fn CLBufferAllocator(const size_t sizeOfElement,cl_context context,cl_command_queue commandQueue)
     * @brief Constructor of CLBufferAllocator
     * @param sizeOfElement The size in bytes of one element of the buffer.
     * @param context An OpenCL context.
     * @param commandQueue An OpenCL command queue.
     */
    CLBufferAllocator(const size_t sizeOfElement,cl_context context,cl_command_queue commandQueue) :
        GPUMemAllocator(sizeOfElement),
        context(context),
        commandQueue(commandQueue),
        memFlags(CL_MEM_READ_WRITE)

    {}
    virtual ~CLBufferAllocator();

    // GPUMemAllocator interface

    virtual void write(size_t numElems, const void *data) override;

    virtual void *map() override;

    virtual void unmap() override;

    virtual void free() override;

protected:

    virtual void allocate(const size_t length) override;

private:
    cl_mem buffer = nullptr;
    cl_context context;
    cl_command_queue commandQueue;
    cl_mem_flags memFlags;
    void* mappedPtr;
};

}

#endif

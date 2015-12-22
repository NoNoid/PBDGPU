#ifndef CL_BUFFER_ALLOCATOR
#define CL_BUFFER_ALLOCATOR

#include <util/gpu_mem_allocator.hpp>
#include <clew.h>

class CLBufferAllocator : public pbdgpu::GPUMemAllocator
{

    CLBufferAllocator(const size_t sizeOfElement,cl_context context,cl_command_queue commandQueue) :
        GPUMemAllocator(sizeOfElement),
        context(context),
        commandQueue(commandQueue),
        memFlags(CL_MEM_READ_WRITE)

    {}
    virtual ~CLBufferAllocator() {}

    // GPUMemAllocator interface
public:
    virtual void write(size_t numElems, const void *data) override;
    virtual void *map() override;
    virtual void unmap() override;

    cl_event write(size_t numElems, const void *data, cl_uint numEventsInWaitList, const cl_event *eventWaitList);
protected:
    virtual void allocate(const size_t length) override;

private:
    cl_mem buffer;
    cl_context context;
    cl_command_queue commandQueue;
    cl_mem_flags memFlags;
    void* mappedPtr;
};

#endif

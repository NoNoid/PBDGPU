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
     * @param size Number of Elements in the buffer.
     */
    CLBufferAllocator(cl_context context = nullptr, cl_command_queue commandQueue = nullptr, const size_t sizeOfElement = 0, const size_t size = 0) :
        context(context),
        commandQueue(commandQueue),
        memFlags(CL_MEM_READ_WRITE)
    {allocate(sizeOfElement,size);}

    virtual ~CLBufferAllocator();

    void setOpenCLContext(cl_context context)
    {
        this->context = context;
    }


    void setOpenCLCommandQueue(cl_command_queue queue)
    {
        this->commandQueue = queue;
    }

    /**
     * @fn const cl_mem &getCLMem()
     * @return Const Reference to the underlying cl_mem.
     */
    inline const cl_mem &getCLMem() override {return buffer;}

    // GPUMemAllocator interface

    virtual void write(size_t numElems, const void *data) override;

    virtual void *map() override;

    virtual void unmap() override;

    virtual void free() override;

    virtual void allocate(const size_t sizeOfElement, const size_t size) override;


    virtual void acquireForCL(cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);


    virtual void releaseFromCL(cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);

// Do not allow copies
    CLBufferAllocator (const CLBufferAllocator &obj) = delete;
    CLBufferAllocator & operator= (const CLBufferAllocator &obj) = delete;

private:
    cl_mem buffer = nullptr;
    cl_context context;
    cl_command_queue commandQueue;
    cl_mem_flags memFlags;
    void* mappedPtr;
};

}

#endif

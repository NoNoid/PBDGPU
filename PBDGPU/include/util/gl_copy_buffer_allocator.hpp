#ifndef GL_COPY_BUFFER_ALLOCATOR_H_
#define GL_COPY_BUFFER_ALLOCATOR_H_

#include "gl_buffer_allocator.hpp"

namespace pbdgpu {

    class GLCopyBufferAllocator : public GLBufferAllocator {
    public:
        GLCopyBufferAllocator(const size_t sizeOfElement, const size_t size, const cl_context context = nullptr, const cl_command_queue queue = nullptr)
        : GLBufferAllocator()
        {
            memFlags = CL_MEM_READ_WRITE;
            bufferID = 0;
            bufferTarget = GL_ARRAY_BUFFER;
            bufferTargetBinding = GL_ARRAY_BUFFER_BINDING;
            bufferUsage = GL_DYNAMIC_DRAW;
            clSharingMem = nullptr;
            initCLSharing(context,queue);
            allocate(sizeOfElement,size);
        }

        virtual ~GLCopyBufferAllocator();

        virtual void initCLSharing(cl_context context, cl_command_queue queue) override;

        virtual void acquireForCL(cl_uint num_events_in_wait_list,
                                  const cl_event *event_wait_list,
                                  cl_event *event) override;

        virtual void releaseFromCL(cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) override;

        virtual void write(size_t numElems, const void *data) override;

        virtual void free() override;

        virtual void allocate(const size_t sizeOfElement, const size_t length) override;

        GLCopyBufferAllocator (const GLCopyBufferAllocator &obj) = delete;
        GLCopyBufferAllocator & operator= (const GLCopyBufferAllocator &obj) = delete;

    protected:
        cl_mem_flags memFlags;
    };

}

#endif
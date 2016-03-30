#include <util/gl_copy_buffer_allocator.hpp>
#include <cassert>

namespace pbdgpu {
    GLCopyBufferAllocator::~GLCopyBufferAllocator()
    {

    }

    void GLCopyBufferAllocator::initCLSharing(cl_context context, cl_command_queue queue)
    {
        this->sharingContext = context;
        this->queue = queue;
        //clSharingMem = clCreateBuffer(context, memFlags, getSizeinBytes(), nullptr, nullptr);
    }

    void GLCopyBufferAllocator::releaseFromCL(cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                                              cl_event *event) {

        cl_int error;

        void* ptr = clEnqueueMapBuffer (queue,
                                         clSharingMem,
                                         CL_TRUE,
                                         CL_MAP_READ,
                                         0,
                                         getSizeinBytes(),
                                         num_events_in_wait_list,
                                         event_wait_list,
                                         nullptr,
                                         &error);

        assert(error==0 && "Error in GLCopyBufferAllocator::releaseFromCL while mapping");

        GLBufferAllocator::write(getSize(),ptr);

        error = clEnqueueUnmapMemObject(queue,
                                 clSharingMem,
                                 ptr,
                                 0,
                                 nullptr,
                                 event);

        assert(error==0 && "Error in GLCopyBufferAllocator::releaseFromCL while unmapping");
    }

    void GLCopyBufferAllocator::acquireForCL(cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                                             cl_event *event)
    {

    }

    void GLCopyBufferAllocator::free() {
        GLBufferAllocator::free();
    }

    void GLCopyBufferAllocator::allocate(const size_t sizeOfElement, const size_t length) {
        GLBufferAllocator::allocate(sizeOfElement,length);
        cl_int error = 0;
        clSharingMem = clCreateBuffer(sharingContext,
                                      memFlags,
                                      getSizeinBytes(),
                                      nullptr,
                                      &error);

        assert(error == 0 && "Error while creating buffer.");

    }

    void GLCopyBufferAllocator::write(size_t numElems, const void *data) {

        GLBufferAllocator::write(numElems, data);

        clEnqueueWriteBuffer(
                queue,
                clSharingMem,
                true,
                0,
                size*sizeOfElement,
                data,
                0,
                nullptr,
                nullptr);
    }
}
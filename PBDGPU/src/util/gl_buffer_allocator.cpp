#include <util/gl_buffer_allocator.hpp>
#include <cassert>

namespace pbdgpu
{
	void GLBufferAllocator::initCLSharing(cl_context context,cl_command_queue comqueue)
	{
		sharingContext = context;
		queue = comqueue;
		clSharingMem = clCreateFromGLBuffer(sharingContext, CL_MEM_READ_WRITE, bufferID, nullptr);
	}

	void* GLBufferAllocator::map()
	{
		void* data;
		GLint previousBuffer;
		glGetIntegerv(bufferTargetBinding, &previousBuffer);
		glBindBuffer(bufferTarget, bufferID);
		data = glMapBuffer(bufferTarget, GL_READ_WRITE);
		glBindBuffer(bufferTarget, previousBuffer);
		return data;
	}

	void GLBufferAllocator::unmap()
	{
		
		GLint previousBuffer;
		glGetIntegerv(bufferTargetBinding, &previousBuffer);
		glBindBuffer(bufferTarget, bufferID);
		glUnmapBuffer(bufferTarget);
		glBindBuffer(bufferTarget, previousBuffer);
	}

    void GLBufferAllocator::allocate(const size_t sizeOfElement, const size_t length)
	{
        GPUMemAllocator::allocate(sizeOfElement,length);

        assert(getSizeinBytes() > 0 && "Invalid BufferSize");

		if (!glIsBuffer(bufferID))
		{
			glGenBuffers(1, &bufferID);
		}

		GLint previousBuffer;
		glGetIntegerv(bufferTargetBinding, &previousBuffer);
		glBindBuffer(bufferTarget, bufferID);
        glBufferData(bufferTarget, getSizeinBytes(), nullptr, bufferUsage);
		glBindBuffer(bufferTarget, previousBuffer);
	}

    GLBufferAllocator::~GLBufferAllocator()
    {        
        free();
    }

    void GLBufferAllocator::write(size_t numElems, const void *data)
	{
		if (numElems > size)
		{
            return;
		}
		GLint previousBuffer;
		glGetIntegerv(bufferTargetBinding, &previousBuffer);
		glBindBuffer(bufferTarget, bufferID);
        glBufferSubData(bufferTarget, 0, getSizeinBytes(), data);
		glBindBuffer(bufferTarget, previousBuffer);
    }


    void GLBufferAllocator::free()
    {
        GPUMemAllocator::free();
        if(clSharingMem)
        {
            clReleaseMemObject(clSharingMem);
            clSharingMem = nullptr;
        }
        glDeleteBuffers(1,&bufferID);
    }

	void GLBufferAllocator::acquireForCL(cl_uint num_events_in_wait_list,
										 const cl_event *event_wait_list,
										 cl_event *event) {
		clEnqueueAcquireGLObjects(queue,1,&clSharingMem,num_events_in_wait_list, event_wait_list, event);
	}

	void GLBufferAllocator::releaseFromCL(cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
										  cl_event *event) {
		clEnqueueReleaseGLObjects(queue,1,&clSharingMem,num_events_in_wait_list,event_wait_list,event);
	}
}

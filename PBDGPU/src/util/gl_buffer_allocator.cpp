#include <util/gl_buffer_allocator.hpp>

namespace pbdgpu
{
	void GLBufferAllocator::initCLSharing(const cl_context &context)
	{
		clSharingMem = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, bufferID, nullptr);
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

        if(getSizeinBytes() <= 0) return;

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
}

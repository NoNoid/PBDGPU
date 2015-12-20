#ifndef _GL_BUFFER_ALLOCATOR_
#define _GL_BUFFER_ALLOCATOR_

#include <util/gpu_mem_allocator.hpp>
#include <GL/glew.h>
#include <clew.h>

namespace pbdgpu
{
	class GLBufferAllocator : public GPUMemAllocator
	{
	public:
		GLBufferAllocator(const size_t sizeOfElement) : 
			GPUMemAllocator(sizeOfElement),
			bufferID(0),
			bufferTarget(GL_ARRAY_BUFFER),
			bufferTargetBinding(GL_ARRAY_BUFFER_BINDING),
			bufferUsage(GL_DYNAMIC_DRAW)
		{}
        virtual ~GLBufferAllocator();

		inline GLuint getBufferID() { return bufferID; }

		virtual void write(size_t numElems, const void *data) override;
		virtual void* map() override;
		virtual void unmap() override;

		void initCLSharing(const cl_context &context);
		inline const cl_mem &getCLMem() { return clSharingMem; }

	protected:

		void allocate(const size_t length) override;

	private:
		GLuint bufferID;
		GLenum bufferTarget;
		GLenum bufferTargetBinding;
		GLenum bufferUsage;

		cl_mem clSharingMem;
	};
}

#endif

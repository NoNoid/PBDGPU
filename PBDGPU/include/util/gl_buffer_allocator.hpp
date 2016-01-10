#ifndef _GL_BUFFER_ALLOCATOR_
#define _GL_BUFFER_ALLOCATOR_

#include <util/gpu_mem_allocator.hpp>
#include <GL/glew.h>
#include <clew.h>

namespace pbdgpu
{
    /**
     * @brief A GPU memory allocator class, which uses OpenGL buffers to allocate memory.
     */
	class GLBufferAllocator : public GPUMemAllocator
	{
	public:
        /** @fn GLBufferAllocator(const size_t sizeOfElement)
         * @brief A Constructor for GLBufferAllocator.
         * @param sizeOfElement Size of one Element in bytes.
         * @param size Number of Elements in the buffer.
         */
        GLBufferAllocator(const size_t sizeOfElement = 0, const size_t size = 0) :
			bufferID(0),
			bufferTarget(GL_ARRAY_BUFFER),
			bufferTargetBinding(GL_ARRAY_BUFFER_BINDING),
            bufferUsage(GL_DYNAMIC_DRAW),
            clSharingMem(nullptr)
        {allocate(sizeOfElement,size);}

        virtual ~GLBufferAllocator();

        /** @fn inline GLuint getBufferID()
         * @brief Get the buffer ID.
         * @return ID the of the buffer.
         */
		inline GLuint getBufferID() { return bufferID; }

        /** @fn void initCLSharing
         * @brief Inits sharing with OpenCL for the buffer.
         * @param context An OpenCL buffer with CLGL-interop activated.
         */
        void initCLSharing(const cl_context &context);

        /** @fn inline const cl_mem &getCLMem()
         * @brief Get the OpenCL memory object for the buffer.
         * @return OpenCL memory object.
         */
        inline const cl_mem &getCLMem() override { return clSharingMem; }

        // GPUMemAllocator interface

		virtual void write(size_t numElems, const void *data) override;

		virtual void* map() override;

		virtual void unmap() override;

        virtual void free() override;

        void allocate(const size_t sizeOfElement, const size_t size) override;

// Do not allow copies
		GLBufferAllocator (const GLBufferAllocator &obj) = delete;
		GLBufferAllocator & operator= (const GLBufferAllocator &obj) = delete;

	private:
		GLuint bufferID;
		GLenum bufferTarget;
		GLenum bufferTargetBinding;
		GLenum bufferUsage;

        cl_mem clSharingMem = nullptr;

    };
}

#endif

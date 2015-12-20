#ifndef _GPU_MEM_ALLOCATOR_
#define  _GPU_MEM_ALLOCATOR_

#include <cstddef>

namespace pbdgpu
{
	class GPUMemAllocator
	{
	public:
		GPUMemAllocator() : sizeOfElement(0),length(0) {}
		GPUMemAllocator(const size_t sizeOfElement) : sizeOfElement(sizeOfElement), length(0){}

        virtual ~GPUMemAllocator() {}

		inline void setLength(const size_t newLength) { length = newLength; allocate(length); }
		inline size_t getLength() const { return length; }
		inline size_t getSizeOfElement() const { return sizeOfElement; }

		virtual void write(size_t numElems, const void *data) = 0;

		virtual void *map() = 0;
		virtual void unmap() = 0;

	protected:
		const size_t sizeOfElement;
		size_t length;

        virtual void allocate(const size_t length) = 0;
    };

}

#endif

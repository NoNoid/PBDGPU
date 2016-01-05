#include <util/gpu_mem_allocator.hpp>

void pbdgpu::GPUMemAllocator::allocate(const size_t sizeOfElement, const size_t size)
{
    this->sizeOfElement = sizeOfElement;
    this->size = size;
}

void pbdgpu::GPUMemAllocator::free()
{
    sizeOfElement = 0;
    size = 0;
}

size_t pbdgpu::GPUMemAllocator::getSizeinBytes()
{
    return sizeOfElement*size;
}

//
// Created by tobi on 08.03.16.
//

#include <constraint.hpp>

shared_ptr<pbdgpu::GPUMemAllocator> pbdgpu::getBufferChecked(unordered_map<string, shared_ptr<pbdgpu::GPUMemAllocator> > BufferMap, string bufferName)
{
    auto result = BufferMap.find(bufferName);
    assert(result != BufferMap.end() && "Buffername not found");
    return result->second;
}

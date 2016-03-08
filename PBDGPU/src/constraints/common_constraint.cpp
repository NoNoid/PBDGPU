//
// Created by tobi on 07.03.16.
//

#include <constraints/common_constraint.hpp>
//#include <util/gpu_mem_allocator.hpp>

void pbdgpu::CommonConstraint::getSharedBuffers(
        unordered_map<string, shared_ptr<pbdgpu::GPUMemAllocator> > sharedBuffers)
{
    particleBuffer = getBufferChecked(sharedBuffers,PARTICLE_BUFFER_NAME);
}

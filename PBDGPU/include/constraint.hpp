//
// Created by tobi on 05.03.16.
//

#ifndef PBDGPU_CONSTRAINT_HPP
#define PBDGPU_CONSTRAINT_HPP

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

using std::string;
using std::vector;
using std::shared_ptr;
using std::unordered_map;

#include <util/gpu_mem_allocator.hpp>
#include <cassert>

namespace pbdgpu
{
    shared_ptr<GPUMemAllocator> getBufferChecked(unordered_map<string, shared_ptr<GPUMemAllocator> > BufferMap, string bufferName);

    class Constraint
    {
    public:
        virtual void acquire() = 0;
        virtual void release() = 0;
        virtual void getSharedBuffers(unordered_map<string, shared_ptr<GPUMemAllocator> > sharedBuffers) = 0;
        virtual void update() = 0;
        virtual bool needsAcquisition() = 0;
        virtual bool needsStabilization() = 0;

        virtual void initKernel(const cl_context context, const cl_device_id device, const cl_command_queue queue) = 0;
    };

}

#endif //PBDGPU_CONSTRAINT_HPP

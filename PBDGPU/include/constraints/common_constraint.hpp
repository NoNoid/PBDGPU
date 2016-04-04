//
// Created by tobi on 07.03.16.
//

#ifndef PBDGPU_COMMON_CONSTRAINT_HPP
#define PBDGPU_COMMON_CONSTRAINT_HPP

#include <constraint.hpp>
#include <clew.h>

namespace pbdgpu
{
    class CommonConstraint : public Constraint
    {

    public:
        virtual void getSharedBuffers(unordered_map<string, shared_ptr<GPUMemAllocator> > sharedBuffers) override;

    protected:

        shared_ptr<GPUMemAllocator> particleBuffer;
        shared_ptr<GPUMemAllocator> simParamBuffer;
        shared_ptr<GPUMemAllocator> predictedPositionBuffer;
        shared_ptr<GPUMemAllocator> positionCorrectionsBuffer;
    };
}

#endif //PBDGPU_COMMON_CONSTRAINT_HPP

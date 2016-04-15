//
// Created by tobi on 10.03.16.
//

#ifndef PBDGPU_DISTANCE_CONSTRAINT_HPP
#define PBDGPU_DISTANCE_CONSTRAINT_HPP

#include <constraints/common_constraint.hpp>
#include <kernelInclude/distanceConstraintData.h>

namespace pbdgpu
{

    class DistanceConstraint : public CommonConstraint
    {

    public:

        DistanceConstraint(shared_ptr<GPUMemAllocator>distanceConstraintData) : dataBuffer(distanceConstraintData)
        {
            assert(dataBuffer->getSizeOfElement() == sizeof(pbd_distanceConstraintData));
            needsStabilizationFlag = true;
        }

        virtual void acquire() override {}

        virtual void release() override {}

        virtual void update() override;

        virtual bool needsAcquisition() override;

        virtual void initKernel(const cl_context context, const cl_device_id device,
                                const cl_command_queue queue) override;

    protected:

        cl_kernel kernel;
        cl_command_queue queue;
        shared_ptr<GPUMemAllocator> dataBuffer;
    };
}

#endif //PBDGPU_DISTANCE_CONSTRAINT_HPP

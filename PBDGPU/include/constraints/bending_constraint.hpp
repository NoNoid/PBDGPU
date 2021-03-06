//
// Created by tobi on 14.03.16.
//

#ifndef PBDGPU_BENDING_CONSTRAINT_HPP
#define PBDGPU_BENDING_CONSTRAINT_HPP

#include "common_constraint.hpp"

namespace pbdgpu
{
    class BendingConstraint : public CommonConstraint
    {

    public:
        virtual void initKernel(const cl_context context, const cl_device_id device, const cl_command_queue queue);


    protected:
        cl_command_queue queue;
        cl_kernel kernel;

        shared_ptr<GPUMemAllocator> dataBuffer;
    public:
        virtual void acquire();

        virtual void release();

        virtual bool needsAcquisition();

        virtual void update();
    };

}

#endif //PBDGPU_BENDING_CONSTRAINT_HPP

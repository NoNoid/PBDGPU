//
// Created by tobi on 07.04.16.
//

#ifndef PBDGPU_TRIANGLE_BENDING_CONSTRAINT_HPP
#define PBDGPU_TRIANGLE_BENDING_CONSTRAINT_HPP

#include <kernelInclude/triangle_bending_constraint_data.h>
#include "common_constraint.hpp"

namespace pbdgpu
{
    class TriangleBendingConstraint : public CommonConstraint
    {

    public:
        TriangleBendingConstraint(shared_ptr<GPUMemAllocator> data) : dataBuffer(data)
        {
            assert(data->getSizeOfElement() == sizeof(pbd_triangleBendingConstraintData));
        }

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

#endif //PBDGPU_TRIANGLE_BENDING_CONSTRAINT_HPP

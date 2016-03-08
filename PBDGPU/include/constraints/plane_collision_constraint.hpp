//
// Created by tobi on 05.03.16.
//

#ifndef PBDGPU_PLANE_COLLISION_CONSTRAINT_HPP
#define PBDGPU_PLANE_COLLISION_CONSTRAINT_HPP

#include <constraints/common_constraint.hpp>

namespace pbdgpu
{
    class PlaneCollisionConstraint : public CommonConstraint
    {

    protected:
        cl_kernel kernel = nullptr;
        cl_command_queue queue = nullptr;

        shared_ptr<GPUMemAllocator> predictedPositionBuffer;
        shared_ptr<GPUMemAllocator> planeBuffer;

    public:

        virtual void getSharedBuffers(unordered_map<string, shared_ptr<GPUMemAllocator> > sharedBuffers) override;

        PlaneCollisionConstraint(
                shared_ptr<GPUMemAllocator> planeBuffer);

/*        PlaneCollisionConstraint(
                shared_ptr<GPUMemAllocator> predictedPositionBuffer,
                shared_ptr<GPUMemAllocator> planeBuffer,
                const cl_context context,
                const cl_device_id device,
                const cl_command_queue queue);*/

        virtual void initKernel(
                const cl_context context,
                const cl_device_id device,
                const cl_command_queue queue);

        virtual bool needsAcquisition();

        virtual void update();


        virtual void acquire();

        virtual void release();
    };
}

#endif //PBDGPU_PLANE_COLLISION_CONSTRAINT_HPP

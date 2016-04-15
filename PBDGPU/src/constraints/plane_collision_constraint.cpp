//
// Created by tobi on 05.03.16.
//

#include <constraints/plane_collision_constraint.hpp>
#include <kernels.hpp>
#include <cassert>

void pbdgpu::PlaneCollisionConstraint::initKernel(
        const cl_context context,
        const cl_device_id device,
        const cl_command_queue queue)
{
    kernel = pbdgpu::buildPlaneCollisionKernel(context,device);

    this->queue = queue;

    assert(particleBuffer && "particleBuffer is null");

    int cl_err;
    cl_err = clSetKernelArg(kernel,0,sizeof(cl_mem),&particleBuffer->getCLMem());
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"cl error %d\n",cl_err);
        assert(cl_err == CL_SUCCESS && "Error while setting kernel arguments" );
    }

    cl_err = clSetKernelArg(kernel,1,sizeof(cl_mem),&predictedPositionBuffer->getCLMem());
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"cl error %d\n",cl_err);
        assert(cl_err == CL_SUCCESS && "Error while setting kernel arguments" );
    }

    assert(positionCorrectionsBuffer && "data buffer is null");
    assert(clSetKernelArg(kernel,2,sizeof(cl_mem),&positionCorrectionsBuffer->getCLMem()) == CL_SUCCESS);

    assert(numConstraintsBuffer && "num constraints buffer is null");
    assert(clSetKernelArg(kernel,3,sizeof(cl_mem),&numConstraintsBuffer->getCLMem()) == CL_SUCCESS);

    cl_err = clSetKernelArg(kernel,4,sizeof(cl_mem),&planeBuffer->getCLMem());
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"cl error %d\n",cl_err);
        assert(cl_err == CL_SUCCESS && "Error while setting kernel arguments" );
    }

    size_t numPlanes = planeBuffer->getSize();
    cl_err = clSetKernelArg(kernel, 5, sizeof(cl_uint), &numPlanes);
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"cl error %d\n",cl_err);
        assert(cl_err == CL_SUCCESS && "Error while setting kernel arguments" );
    }

    assert(simParamBuffer && "data buffer is null");
    assert(clSetKernelArg(kernel,6,sizeof(cl_mem),&simParamBuffer->getCLMem()) == CL_SUCCESS);

}

void pbdgpu::PlaneCollisionConstraint::update()
{
    int cl_err;

    assert(particleBuffer && "particleBuffer is null");
    assert(queue && "queue is null");
    assert(kernel && "kernel is null");

    const size_t numParticles = particleBuffer->getSize();
    cl_err = clEnqueueNDRangeKernel(
            queue,
            kernel,
            1, nullptr, &numParticles, nullptr,
            0, nullptr, nullptr);
    if(cl_err != CL_SUCCESS)
    {
        printf("Error on Plane Collision Kernel Execution:%d \n",cl_err);
    }
}

bool pbdgpu::PlaneCollisionConstraint::needsAcquisition()
{
    return false;
}

pbdgpu::PlaneCollisionConstraint::PlaneCollisionConstraint(shared_ptr<GPUMemAllocator> planeBuffer)
{
    this->planeBuffer = planeBuffer;
    this->needsStabilizationFlag = true;
}

/*pbdgpu::PlaneCollisionConstraint::PlaneCollisionConstraint(shared_ptr<GPUMemAllocator> predictedPositionBuffer,
                                                           shared_ptr<GPUMemAllocator> planeBuffer,
                                                           const cl_context context,
                                                           const cl_device_id device,
                                                           const cl_command_queue queue)
        : PlaneCollisionConstraint(predictedPositionBuffer,planeBuffer)
{
    initKernel(context,device,queue);
}*/

void pbdgpu::PlaneCollisionConstraint::release() {

}

void pbdgpu::PlaneCollisionConstraint::acquire() {

}

//
// Created by tobi on 14.03.16.
//

#include <constraints/bending_constraint.hpp>
#include <kernels.hpp>

bool pbdgpu::BendingConstraint::needsAcquisition() {
    return false;
}

void pbdgpu::BendingConstraint::release() {

}

void pbdgpu::BendingConstraint::acquire() {

}

void pbdgpu::BendingConstraint::update() {

    assert(kernel && "kernel is null");
    assert(dataBuffer && "dataBuffer is null");

    const size_t numBendingConstraints = dataBuffer->getSize();
    cl_int cl_err = clEnqueueNDRangeKernel(
            queue,
            kernel,
            1, nullptr, &numBendingConstraints, nullptr,
            0, nullptr, nullptr);

    assert(cl_err == CL_SUCCESS);
}

void pbdgpu::BendingConstraint::initKernel(const cl_context context, const cl_device_id device,
                                   const cl_command_queue queue) {
    this->queue =  queue;

    kernel = pbdgpu::buildBendingConstraintKernel(context,device);

    assert(particleBuffer && "particleBuffer is null");
    assert(clSetKernelArg(kernel,0,sizeof(cl_mem),&particleBuffer->getCLMem()) == CL_SUCCESS);

    assert(predictedPositionBuffer && "predicted positions buffer is null");
    assert(clSetKernelArg(kernel,1,sizeof(cl_mem),&predictedPositionBuffer->getCLMem()) == CL_SUCCESS);

    assert(dataBuffer && "data buffer is null");
    assert(clSetKernelArg(kernel,2,sizeof(cl_mem),&dataBuffer->getCLMem()) == CL_SUCCESS);

    assert(simParamBuffer && "simParam buffer is null");
    assert(clSetKernelArg(kernel,3,sizeof(cl_mem),&simParamBuffer->getCLMem()) == CL_SUCCESS);

}
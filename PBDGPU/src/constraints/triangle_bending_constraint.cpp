//
// Created by tobi on 07.04.16.
//

#include <constraints/triangle_bending_constraint.hpp>
#include <kernels.hpp>


void pbdgpu::TriangleBendingConstraint::initKernel(const cl_context context, const cl_device_id device,
                                                   const cl_command_queue queue) {
    this->queue =  queue;

    kernel = pbdgpu::buildTriangleBendingKernel(context,device);

    assert(particleBuffer && "particleBuffer is null");
    assert(clSetKernelArg(kernel,0,sizeof(cl_mem),&particleBuffer->getCLMem()) == CL_SUCCESS);

    assert(predictedPositionBuffer && "predicted positions buffer is null");
    assert(clSetKernelArg(kernel,1,sizeof(cl_mem),&predictedPositionBuffer->getCLMem()) == CL_SUCCESS);

    assert(positionCorrectionsBuffer && "position corrections buffer is null");
    assert(clSetKernelArg(kernel,2,sizeof(cl_mem),&positionCorrectionsBuffer->getCLMem()) == CL_SUCCESS);

    assert(numConstraintsBuffer && "num constraints buffer is null");
    assert(clSetKernelArg(kernel,3,sizeof(cl_mem),&numConstraintsBuffer->getCLMem()) == CL_SUCCESS);

    assert(dataBuffer && "data buffer is null");
    assert(clSetKernelArg(kernel,4,sizeof(cl_mem),&dataBuffer->getCLMem()) == CL_SUCCESS);

    assert(simParamBuffer && "simParam buffer is null");
    assert(clSetKernelArg(kernel,5,sizeof(cl_mem),&simParamBuffer->getCLMem()) == CL_SUCCESS);
}

void pbdgpu::TriangleBendingConstraint::acquire() {

}

void pbdgpu::TriangleBendingConstraint::release() {

}

bool pbdgpu::TriangleBendingConstraint::needsAcquisition() {
    return false;
}

void pbdgpu::TriangleBendingConstraint::update() {
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


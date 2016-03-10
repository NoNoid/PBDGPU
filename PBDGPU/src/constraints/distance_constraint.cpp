//
// Created by tobi on 10.03.16.
//

#include <constraints/distance_constraint.hpp>
#include <kernels.hpp>

void pbdgpu::DistanceConstraint::update() {

    assert(kernel && "kernel is null");
    assert(dataBuffer && "dataBuffer is null");

    const size_t numDistanceConstraints = dataBuffer->getSize();
    cl_int cl_err = clEnqueueNDRangeKernel(
            queue,
            kernel,
            1, nullptr, &numDistanceConstraints, nullptr,
            0, nullptr, nullptr);

    assert(cl_err == CL_SUCCESS);
}

bool pbdgpu::DistanceConstraint::needsAcquisition() {
    return false;
}

void pbdgpu::DistanceConstraint::getSharedBuffers(unordered_map<string, shared_ptr<GPUMemAllocator> > sharedBuffers) {
    CommonConstraint::getSharedBuffers(sharedBuffers);
    predictedPositionBuffer = getBufferChecked(sharedBuffers,PREDICTED_POSITIONS_BUFFER_NAME);
}

void pbdgpu::DistanceConstraint::initKernel(const cl_context context, const cl_device_id device,
                                    const cl_command_queue queue) {

    this->queue = queue;

    kernel = buildDistanceConstraintKernel(context,device);

    assert(particleBuffer && "particleBuffer is null");
    assert(clSetKernelArg(kernel,0,sizeof(cl_mem),&particleBuffer->getCLMem()) == CL_SUCCESS);

    assert(predictedPositionBuffer && "predicted positions buffer is null");
    assert(clSetKernelArg(kernel,1,sizeof(cl_mem),&predictedPositionBuffer->getCLMem()) == CL_SUCCESS);

    assert(dataBuffer && "data buffer is null");
    assert(clSetKernelArg(kernel,2,sizeof(cl_mem),&dataBuffer->getCLMem()) == CL_SUCCESS);

}
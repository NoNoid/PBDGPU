//
// Created by tobi on 05.03.16.
//

#include <simulation_data.hpp>
#include <kernels.hpp>
#include <cassert>
#include <util/cl_buffer_allocator.hpp>
#include <kernelInclude/simulation_parameters.h>

void pbdgpu::SimulationData::addConstraint(shared_ptr<Constraint> Constraint)
{
    Constraint->getSharedBuffers(sharedBuffers);
    Constraints.push_back(Constraint);

    if(Constraint->needsAcquisition())
    {
        ConstraintsNeedingAcquisition.push_back(Constraint);
    }

    Constraint->initKernel(context,device,kernel_queue);
}

void pbdgpu::SimulationData::update()
{
    acquireResources();

    int cl_err = clEnqueueNDRangeKernel(
            kernel_queue,
            predictionKernel,
            1, nullptr, &numParticles, nullptr,
            0, nullptr, nullptr);
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"Error on Prediction Kernel Execution:%d \n",cl_err);
    }

    for(int i = 0; i < numIterations; ++i) {
        projectConstraints();
#ifdef PBDGPU_DEBUG_PRINT
        printf("\n######  sub_it = %d ######\n",i);
#endif
    }

    cl_err = clEnqueueNDRangeKernel(
            kernel_queue,
            updateKernel,
            1, nullptr, &numParticles, nullptr,
            0, nullptr, nullptr);
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"Error on Update Kernel Execution:%d \n",cl_err);
    }

    releaseResources();

#ifdef PBDGPU_DEBUG_PRINT
    static int iter = 0;
    printf("\n--- iter = %d --------------------------------------------------------\n",iter++);
#endif
}

void pbdgpu::SimulationData::nullBuffers() const
{
    cl_int cl_err;

    cl_float3 nullVector = (cl_float3){0.f,0.f,0.f};
    auto tmpBuffer = getBufferChecked(sharedBuffers,POSITION_CORRECTIONS_BUFFER_NAME);
    cl_err =  clEnqueueFillBuffer(this->kernel_queue, tmpBuffer->getCLMem(),
                                  &nullVector, sizeof(cl_float3), 0, numParticles*sizeof(cl_float3),
                                  0, nullptr, nullptr);
    assert(cl_err == 0 && "Error while nulling position corrections buffer");

    cl_int null = 0;
    tmpBuffer = getBufferChecked(sharedBuffers,pbdgpu::NUM_CONSTRAINTS_BUFFER_NAME);
    cl_err =  clEnqueueFillBuffer(this->kernel_queue, tmpBuffer->getCLMem(),
                                  &null, sizeof(cl_int), 0, numParticles*sizeof(cl_int),
                                  0, nullptr, nullptr);
    assert(cl_err == 0 && "Error while nulling position corrections buffer");
}

void pbdgpu::SimulationData::postSolveUpdate() const {
    cl_int cl_err;
    cl_err = clEnqueueNDRangeKernel(
            kernel_queue,
            postSolveUpdateKernel,
            1, nullptr, &numParticles, nullptr,
            0, nullptr, nullptr);
    assert(cl_err == CL_SUCCESS && "Error on execution of postSolveUpdateKernel");
}

void pbdgpu::SimulationData::projectConstraints() const {

    for(shared_ptr<pbdgpu::Constraint> Constraint : this->Constraints)
    {
        nullBuffers();
        Constraint->update();
        postSolveUpdate();

#ifdef PBDGPU_DEBUG_PRINT
        printf("..........................................................\n");
#endif

    }

}

void pbdgpu::SimulationData::releaseResources() const {
    for(shared_ptr<pbdgpu::Constraint> Constraint : this->ConstraintsNeedingAcquisition)
    {
        Constraint->release();
    }
}

void pbdgpu::SimulationData::acquireResources() const {
    for(shared_ptr<pbdgpu::Constraint> Constraint : this->ConstraintsNeedingAcquisition)
    {
        Constraint->acquire();
    }
}

void pbdgpu::SimulationData::addSharedBuffer(shared_ptr<GPUMemAllocator> buffer, string key)
{
    std::pair<string,shared_ptr<GPUMemAllocator> > input(key,buffer);
    sharedBuffers.insert(input);
}

void pbdgpu::SimulationData::initStandardKernels()
{
    initPredictionKernel();
    initUpdateKernel();
    initPostSolveUpdateKernel();
}

void pbdgpu::SimulationData::initUpdateKernel() {
    updateKernel = pbdgpu::buildUpdateKernel(this->context, this->device);

    cl_int cl_err;

    auto buffer = pbdgpu::getBufferChecked(this->sharedBuffers, pbdgpu::PARTICLE_BUFFER_NAME);
    cl_err = clSetKernelArg(this->updateKernel, 0, sizeof(cl_mem), &buffer->getCLMem());
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"cl error %d\n",cl_err);
        assert(cl_err == CL_SUCCESS && "Error while setting kernel arguments" );
    }

    buffer = pbdgpu::getBufferChecked(this->sharedBuffers, pbdgpu::PREDICTED_POSITIONS_BUFFER_NAME);
    cl_err = clSetKernelArg(this->updateKernel, 1, sizeof(cl_mem), &buffer->getCLMem());
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"cl error %d\n",cl_err);
        assert(cl_err == CL_SUCCESS && "Error while setting kernel arguments" );
    }

    buffer = pbdgpu::getBufferChecked(this->sharedBuffers, pbdgpu::SIMULATION_PARAMETERS);
    cl_err = clSetKernelArg(this->updateKernel, 2, sizeof(cl_mem), &buffer->getCLMem());
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"cl error %d\n",cl_err);
        assert(cl_err == CL_SUCCESS && "Error while setting kernel arguments" );
    }
}

void pbdgpu::SimulationData::initPredictionKernel() {
    predictionKernel = pbdgpu::buildPredictionKernel(this->context, this->device);
    cl_int cl_err;

    auto buffer = pbdgpu::getBufferChecked(this->sharedBuffers, pbdgpu::PARTICLE_BUFFER_NAME);

    cl_err = clSetKernelArg(this->predictionKernel, 0, sizeof(cl_mem), &buffer->getCLMem());
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"cl error %d\n",cl_err);
        assert(cl_err == CL_SUCCESS && "Error while setting kernel arguments" );
    }

    buffer = pbdgpu::getBufferChecked(this->sharedBuffers, pbdgpu::EXTERNAL_FORCES_BUFFER_NAME);
    cl_err = clSetKernelArg(this->predictionKernel, 1, sizeof(cl_mem), &buffer->getCLMem());
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"cl error %d\n",cl_err);
        assert(cl_err == CL_SUCCESS && "Error while setting kernel arguments" );
    }

    buffer = pbdgpu::getBufferChecked(this->sharedBuffers, pbdgpu::PREDICTED_POSITIONS_BUFFER_NAME);
    cl_err = clSetKernelArg(this->predictionKernel, 2, sizeof(cl_mem), &buffer->getCLMem());
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"cl error %d\n",cl_err);
        assert(cl_err == CL_SUCCESS && "Error while setting kernel arguments" );
    }

    buffer = pbdgpu::getBufferChecked(this->sharedBuffers, pbdgpu::MASSES_BUFFER_NAME);
    cl_err = clSetKernelArg(this->predictionKernel, 3, sizeof(cl_mem), &buffer->getCLMem());
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"cl error %d\n",cl_err);
        assert(cl_err == CL_SUCCESS && "Error while setting kernel arguments" );
    }

    buffer = pbdgpu::getBufferChecked(this->sharedBuffers, pbdgpu::SCALED_MASSES_BUFFER_NAME);
    cl_err = clSetKernelArg(this->predictionKernel, 4, sizeof(cl_mem), &buffer->getCLMem());
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"cl error %d\n",cl_err);
        assert(cl_err == CL_SUCCESS && "Error while setting kernel arguments" );
    }

    buffer = pbdgpu::getBufferChecked(this->sharedBuffers, pbdgpu::SIMULATION_PARAMETERS);
    cl_err = clSetKernelArg(this->predictionKernel, 5, sizeof(cl_mem), &buffer->getCLMem());
    if(cl_err != CL_SUCCESS)
    {
        fprintf(stderr,"cl error %d\n",cl_err);
        assert(cl_err == CL_SUCCESS && "Error while setting kernel arguments" );
    }

}

void pbdgpu::SimulationData::initSimParamMemory()
{
    simParamBuffer = std::make_shared<CLBufferAllocator>(context,kernel_queue,sizeof(pbd_simulationParameters),1,CL_MEM_READ_WRITE);

    pbd_simulationParameters d;
    d.gravity = gravityVector;
    d.numIterations = numIterations;
    d.timeStep = timeStep;

    simParamBuffer->write(1,&d);

    sharedBuffers.insert(std::pair<string,shared_ptr<GPUMemAllocator> >(SIMULATION_PARAMETERS,simParamBuffer));
}

pbdgpu::SimulationData::~SimulationData()
{
    simParamBuffer->free();
}

void pbdgpu::SimulationData::initPostSolveUpdateKernel() {
    postSolveUpdateKernel = pbdgpu::buildPostSolveUpdateKernel(context,device);

    cl_int cl_err;

    auto tmpBuffer = getBufferChecked(sharedBuffers,pbdgpu::PREDICTED_POSITIONS_BUFFER_NAME);
    cl_err = clSetKernelArg(postSolveUpdateKernel,0,sizeof(cl_mem),&tmpBuffer->getCLMem());
    assert(cl_err == CL_SUCCESS);

    tmpBuffer = getBufferChecked(sharedBuffers,pbdgpu::POSITION_CORRECTIONS_BUFFER_NAME);
    cl_err = clSetKernelArg(postSolveUpdateKernel,1,sizeof(cl_mem),&tmpBuffer->getCLMem());
    assert(cl_err == CL_SUCCESS);

    tmpBuffer = getBufferChecked(sharedBuffers,pbdgpu::NUM_CONSTRAINTS_BUFFER_NAME);
    cl_err = clSetKernelArg(postSolveUpdateKernel,2,sizeof(cl_mem),&tmpBuffer->getCLMem());
    assert(cl_err == CL_SUCCESS);
}



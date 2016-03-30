//
// Created by tobi on 05.03.16.
//

#ifndef PBDGPU_SIMULATION_DATA_HPP
#define PBDGPU_SIMULATION_DATA_HPP

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

using std::string;
using std::vector;
using std::shared_ptr;
using std::unordered_map;

#include <util/gpu_mem_allocator.hpp>
#include <constraint.hpp>
#include <util/cl_buffer_allocator.hpp>

namespace pbdgpu
{

    class SimulationData
    {
    public:
        SimulationData(
                size_t numParticles,
                cl_context context,
                cl_device_id device,
                cl_command_queue kernel_queue,
                cl_uint solverIterations
        ) : numParticles(numParticles),
            context(context),
            device(device),
            kernel_queue(kernel_queue),
            timeStep(1.f/30.f),
            numIterations(solverIterations)
        {
            gravityVector.x = 0.f;
            gravityVector.y = -9.81f;
            gravityVector.z = 0.f;
            gravityVector.w = 0.f;

            initSimParamMemory();
        };

        ~SimulationData();

        template<typename _Tp, typename... _Args>
        void buildConstraint(_Args&&... __args)
        {
            addConstraint(std::make_shared<_Tp>(std::forward<_Args>(__args)...));
        };

        void addSharedBuffer(shared_ptr<GPUMemAllocator> buffer, string key);

        void acquireResources() const;
        void releaseResources() const;
        void projectConstraints() const;

        void initStandardKernels();

        void update();

    protected:
        size_t numParticles = 0;
        cl_float3 gravityVector;
        cl_float timeStep;
        cl_uint numIterations;

        unordered_map<string,shared_ptr<GPUMemAllocator> > sharedBuffers;
        vector<shared_ptr<Constraint> > Constraints;
        vector<shared_ptr<Constraint> > ConstraintsNeedingAcquisition;
        shared_ptr<CLBufferAllocator> simParamBuffer;

        cl_context context;
        cl_device_id device;
        cl_command_queue kernel_queue;

        cl_kernel updateKernel;
        cl_kernel predictionKernel;

        void addConstraint(shared_ptr<Constraint> Constraint);
        void initPredictionKernel();
        void initUpdateKernel();

        void initSimParamMemory();


    };
}

#endif //PBDGPU_SIMULATION_DATA_HPP

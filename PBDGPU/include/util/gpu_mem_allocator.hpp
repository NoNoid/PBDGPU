#ifndef _GPU_MEM_ALLOCATOR_
#define  _GPU_MEM_ALLOCATOR_

#include <cstddef>
#include <clew.h>
#include <string>

using std::string;

namespace pbdgpu
{
    const static string PARTICLE_BUFFER_NAME = "particles";
    const static string EXTERNAL_FORCES_BUFFER_NAME = "externalForces";
    const static string PREDICTED_POSITIONS_BUFFER_NAME = "predictedPositions";
    const static string MASSES_BUFFER_NAME = "masses";
    const static string SCALED_MASSES_BUFFER_NAME = "scaledMasses";
    const static string SIMULATION_PARAMETERS = "simulationParameters";

    /**
     * @brief An abstract class to define an interface for GPU memory Allocators.
     */
	class GPUMemAllocator
	{
	public:   
        GPUMemAllocator() : sizeOfElement(0),size(0) {}

        virtual ~GPUMemAllocator() {}

        /** @fn inline size_t getSize() const
         * @brief Get number of elements in the buffer.
         * @return The number of ements in the buffer.
         */
        inline size_t getSize() const { return size; }

        /** @fn inline size_t getSizeOfElement() const
         * @brief Gets the size in bytes of one element of the buffer.
         * @return The size on bytes of one elment.
         */
		inline size_t getSizeOfElement() const { return sizeOfElement; }

        /** @fn virtual void write(size_t numElems, const void *data) override
         * @brief writes data to the buffer.
         * @param numElems Number of elements to write.
         * @param data pointer to the data to write to the buffer. Has to have size = numElems*this->sizeOfElements in bytes.
         */
        virtual void write(size_t numElems, const void *data) = 0;

        /** @fn virtual void *map() override
         * @brief Maps the buffer data to the host memory.
         * @return A pointer to the mapped data. Has size = numElems*this->sizeOfElements in bytes.
         */
        virtual void *map() = 0;

        /** @fn virtual void unmap() override
         * @brief Unmaps the buffers data from the most memory.
         */
        virtual void unmap() = 0;

        /**
         * @brief Allocates memory for the buffer.
         * @param size number of Elemnts the new buffer should have.
         * @param sizeOfElement Size in bytes of a single element.
         */
        virtual void allocate(const size_t sizeOfElement, const size_t size);

        /**
         * @fn virtual void free()
         * Deallocates the buffer.
         */
        virtual void free();

        /**
         * @fn size_t getSizeinBytes()
         * @return Size of the buffer in bytes.
         */
        size_t getSizeinBytes();

        virtual const cl_mem &getCLMem() = 0;

        virtual void acquireForCL(cl_uint num_events_in_wait_list,
                                  const cl_event *event_wait_list,
                                  cl_event *event) = 0;

        virtual void releaseFromCL(cl_uint num_events_in_wait_list,
                                   const cl_event *event_wait_list,
                                   cl_event *event) = 0;

	protected:
        /**
         * @brief sizeOfElement
         * The size in bytes of one element.
         */
        size_t sizeOfElement;
        /**
         * @brief size
         * Number of elements in buffer.
         */
        size_t size;

    };

}

#endif

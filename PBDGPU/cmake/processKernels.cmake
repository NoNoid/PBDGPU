set(AUTO_GENERATED_WARNING
"########################################################################
This file is AUTOGENERATED, changes you make will not persist.
Do not check into Souce Control.

Edit ../cmake/kernels.cpp.in instead.
To add Kernel Sources edit ../cmake/processKernels.cmake
########################################################################"
)

# read kernel source files and headers
file(READ ${DIR}/src/kernelSrc/prediction.cl PREDICTION_KERNEL)
file(READ ${DIR}/include/kernelInclude/particle.h PARTICLE_HEADER)
file(READ ${DIR}/src/kernelSrc/update.cl UPDATE_KERNEL)
file(READ ${DIR}/src/kernelSrc/planeCollision.cl PLANE_COLL_KERNEL)
file(READ ${DIR}/src/kernelSrc/float_atomic_add.cl ATOMIC_FLOAT_FUNCTIONS)
file(READ ${DIR}/include/kernelInclude/distanceConstraintData.h DISTANCE_CONSTRAINT_DATA_HEADER)
file(READ ${DIR}/src/kernelSrc/distanceConstraint.cl DISTANCE_CONSTRAINT_KERNEL_SOURCE)
file(READ ${DIR}/include/kernelInclude/simulation_parameters.h SIMULATION_PARAMETERS_HEADER)
file(READ ${DIR}/include/kernelInclude/bending_constraint_data.h BENDING_CONSTRAINT_DATA_HEADER)
file(READ ${DIR}/src/kernelSrc/bending_constraint.cl BENDING_CONSTRAINT_KERNEL_SOURCE)

configure_file(${DIR}/cmake/kernels.cpp.in ${DIR}/src/kernels.cpp @ONLY)
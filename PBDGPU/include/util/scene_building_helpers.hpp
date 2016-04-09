//
// Created by tobi on 14.03.16.
//

#ifndef PBDGPU_SCENE_BUILDING_HELPERS_HPP
#define PBDGPU_SCENE_BUILDING_HELPERS_HPP

#include <vector>

#include <glm/vec3.hpp>

#include <kernelInclude/particle.h>
#include <kernelInclude/distanceConstraintData.h>
#include <kernelInclude/bending_constraint_data.h>
#include <kernelInclude/triangle_bending_constraint_data.h>


using glm::vec3;
using std::vector;

namespace pbdgpu
{
    vec3 bilinearInterp(const vec3 &q11, const vec3 &q21, const vec3 &q12, const vec3 &q22, const float x, const float y);

    void buildClothSheet(vector<pbd_particle> &out_particles, const vec3 &p1, const vec3 &p2, const bool suspended,
                             const vec3 &dp, const unsigned int vn, const unsigned int hn, const float mass,
                             const int phase, vector<pbd_distanceConstraintData> &out_distConData,
                             vector<pbd_bendingConstraintData> &out_bendConData, const float bendingStiffness,
                             vector<pbd_triangleBendingConstraintData> &out_triagBendConData);

	void deriveStandardBuffers(
			const vector<pbd_particle> &particles,
			vector<cl_float3> &predPosData,
			vector<cl_float> &masses,
			vector<cl_float> &scaledMasses,
			vector<cl_float3> &extForces,
			vector<cl_float3> &positionCorrections,
			vector<cl_int> &numConstraints);

}

#endif //PBDGPU_SCENE_BUILDING_HELPERS_HPP

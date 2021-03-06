//
// Created by tobi on 14.03.16.
//

#include <util/scene_building_helpers.hpp>
#include <cmath>
#include <stdio.h>
#include <glm/detail/func_geometric.hpp>

vec3 pbdgpu::bilinearInterp(const vec3 &q11, const vec3 &q21, const vec3 &q12, const vec3 &q22, const float x, const float y)
{
    const float y2 = 1.0f;
    const float y1 = 0.f;
    const float x1 = 0.f;
    const float x2 = 1.0f;

    vec3 r1 = ((x2-x)/(x2-x1))*q11+((x-x1)/(x2-x1))*q21;
    vec3 r2 = ((x2-x)/(x2-x1))*q12+((x-x1)/(x2-x1))*q22;
    vec3 res = ((y2-y)/(y2-y1))*r1+((y-y1)/(y2-y1))*r2;

    return res;
}

void pbdgpu::buildClothSheet(vector<pbd_particle> &out_particles, vector<pbd_distanceConstraintData> &out_distConData,
                             const vec3 &p1, const vec3 &p2, const vec3 &dp, const unsigned int hn,
                             const unsigned int vn, const float invmass, const int phase, const bool suspended) {
    const vec3 p3 = p1 + dp;
    const vec3 p4 = p2 + dp;
    const unsigned int numParticles = hn*vn;
    //printf("numParticles = %d\n",numParticles);

    out_particles.resize(numParticles);
    out_distConData.reserve((hn-1)*vn+(vn-1)*hn);


    for(int i = 0; i < numParticles; ++i) {
        const float x = float(i % hn)/float(hn-1);
        const float y = floor(float(i/vn)+0.1f)/float(vn-1);

        //printf("(%f,%f)",x,y);

        vec3 p = bilinearInterp(p1,p2,p3,p4,x,y);

        //printf("(%f,%f,%f)\n",p.x,p.y,p.z);

        out_particles[i].x.x = p.x;
        out_particles[i].x.y = p.y;
        out_particles[i].x.z = p.z;
        out_particles[i].phase = phase;
        out_particles[i].invmass = suspended && i < hn ? 0.f : invmass;


        if((i-1) >= 0 && i % hn != 0 && !(suspended && i < hn))
        {
            pbd_distanceConstraintData data;

            vec3 other_p;

            other_p.x = out_particles[i - 1].x.x;
            other_p.y = out_particles[i - 1].x.y;
            other_p.z = out_particles[i - 1].x.z;

            data.index0 = i-1;
            data.index1 = i;
            data.d = glm::distance(p, other_p);

            //printf("(%d\t,%d\t,%f)\n",data.index0,data.index1,data.d);

            out_distConData.push_back(data);
        }

        if(i-int(hn) >= 0)
        {
            pbd_distanceConstraintData data;

            vec3 other_p;

            other_p.x = out_particles[i - hn].x.x;
            other_p.y = out_particles[i - hn].x.y;
            other_p.z = out_particles[i - hn].x.z;

            data.index0 = i-hn;
            data.index1 = i;
            data.d = glm::distance(p, other_p);

            //printf("  (%d\t,%d\t,%f)\n",data.index0,data.index1,data.d);

            out_distConData.push_back(data);
        }
    }
}

void ::pbdgpu::deriveStandardBuffers(const vector<pbd_particle> &particles, vector<cl_float3> &predPosData,
                                     vector<cl_float> &masses, vector<cl_float> &scaledMasses,
                                     vector<cl_float3> &extForces) {
    const size_t numParticles = particles.size();

    predPosData.resize(numParticles);
    masses.resize(numParticles);
    scaledMasses.resize(numParticles);
    extForces.resize(numParticles);

    for(int i = 0; i  < numParticles; ++i) {
        masses[i] = particles[i].invmass < 1e-8 ? CL_FLT_MAX : 1.f/particles[i].invmass;
    }

}

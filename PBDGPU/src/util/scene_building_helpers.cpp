//
// Created by tobi on 14.03.16.
//

#include <util/scene_building_helpers.hpp>
#include <cmath>
#include <stdio.h>
#include <glm/detail/func_geometric.hpp>

vec3 getVec3(const pbd_particle &particles) {
    vec3 other_p;

    other_p.x = particles.x.x;
    other_p.y = particles.x.y;
    other_p.z = particles.x.z;
    return other_p;
}

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

void pbdgpu::buildBox(vector<pbd_particle> &out_particles, const vec3 &leftBottomFront,
                      const vec3 &rightTopBack, const int numWidthPoints, const int numHeightPoints,
                      const int numDepthPoints, const bool filled) {

    const int numFrontParticles = numWidthPoints*numHeightPoints;
    const vec3& p1 = leftBottomFront;
    const vec3& p7 = rightTopBack;
    const vec3 p2 = vec3(p7.x,p1.y,p1.z);
    const vec3 p6 = glm::vec3(p7.x,p7.y,p1.z);
    const vec3 p5 = glm::vec3(p1.x,p7.y,p1.z);
    const vec3 center = p1 + .5f * (p7-p1);
    const vec3 dir = glm::normalize(p7-p6);
    const float l = glm::distance(p7,p6);

    vector<vec3> frontPoints;
    vector<vec3> middlePoints;
    vector<vec3> tmp;

    for(int i = 0; i < numFrontParticles; ++i) {
        const float x = float(i % numWidthPoints) / float(numWidthPoints - 1);
        const float y = floor(float(i / numHeightPoints) + 0.1f) / float(numHeightPoints - 1);

        vec3 p = bilinearInterp(p1, p2, p5, p6, x, y);

        frontPoints.push_back(p);
    }

    if(!filled) {
        for (int i = 0; i < numFrontParticles; ++i) {
            if (!filled &&
                (i / numWidthPoints == 0 || i / numWidthPoints == numHeightPoints - 1 || i % numWidthPoints == 0 ||
                 i % numWidthPoints == numWidthPoints - 1)) {
                const float x = float(i % numWidthPoints) / float(numWidthPoints - 1);
                const float y = floor(float(i / numHeightPoints) + 0.1f) / float(numHeightPoints - 1);

                vec3 p = bilinearInterp(p1, p2, p5, p6, x, y);

                middlePoints.push_back(p);
            }
        }
    }

    tmp = frontPoints;

    for(int i = 0; i < numDepthPoints; ++i) {
        const float z = (float(i)/float(numDepthPoints))*l;
        const vec3 offset = z * dir;

        if(filled) {
            for (int j = 0; j < frontPoints.size(); ++j) {
                tmp.push_back(frontPoints[j] + offset);
            }
        }else{
            for (int j = 0; j < middlePoints.size(); ++j) {
                tmp.push_back(middlePoints[j] + offset);
            }
        }
    }

    const vec3 offset = l * dir;
    for (int j = 0; j < frontPoints.size(); ++j) {
            tmp.push_back(frontPoints[j] + offset);
    }



    out_particles.resize(tmp.size());
    for(int i = 0; i < tmp.size(); ++i)
    {
        const vec3 &p = tmp[i];
        out_particles[i].x.x = p.x;
        out_particles[i].x.y = p.y;
        out_particles[i].x.z = p.z;
        out_particles[i].phase = 1;
        out_particles[i].invmass = 1.0f;
    }
}

void pbdgpu::buildClothSheet(vector<pbd_particle> &out_particles, const vec3 &p1, const vec3 &p2, const bool suspended,
                             const vec3 &dp, const unsigned int vn, const unsigned int hn, const float mass,
                             const int phase, vector<pbd_distanceConstraintData> &out_distConData,
                             vector<pbd_bendingConstraintData> &out_bendConData, const float bendingStiffness,
                             vector<pbd_triangleBendingConstraintData> &out_triagBendConData) {
    const vec3 p3 = p1 + dp;
    const vec3 p4 = p2 + dp;
    const unsigned int numParticles = hn*vn;
    const float invmass = 1.f/mass;
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

        // Distance Constraints
        if((i-1) >= 0 && i % hn != 0 && !(suspended && i < hn))
        {
            pbd_distanceConstraintData data;

            vec3 other_p = getVec3(out_particles[i-1]);

            data.index0 = i-1;
            data.index1 = i;
            data.d = glm::distance(p, other_p);

            //printf("(%d\t,%d\t,%f)\n",data.index0,data.index1,data.d);

            out_distConData.push_back(data);
        }

        if(i-int(hn) >= 0)
        {
            pbd_distanceConstraintData data;

            vec3 other_p = getVec3(out_particles[i-hn]);

            data.index0 = i-hn;
            data.index1 = i;
            data.d = glm::distance(p, other_p);

            //printf("  (%d\t,%d\t,%f)\n",data.index0,data.index1,data.d);

            out_distConData.push_back(data);
        }

        // Dihedral Bending Constraint
        const float phi = 180.f* 0.0174532925f; // in radians

        // Center
        if(int(i-hn-1) >= 0 && i - 1 >= 0 && int(i-hn) >= 0 && i % hn != 0)
        {
            pbd_bendingConstraintData data;
            data.index1 = i;
            data.index2 = i-hn-1;
            data.index3 = i-1;
            data.index4 = i-hn;
            data.k = bendingStiffness;
            data.phi = phi;

            out_bendConData.push_back(data);
        }/**/


        //Down
        if(i-int(hn)-1 >= 0 && i - int(hn) >= 0 && i-2*int(hn)-1 >= 0 && i % hn != 0)
        {
            pbd_bendingConstraintData data;
            data.index1 = i-hn-1;
            data.index2 = i-hn;
            data.index3 = i-2*hn-1;
            data.index4 = i;
            data.k = bendingStiffness;
            data.phi = phi;

            out_bendConData.push_back(data);
        }/**/

        // Right
        if(i-1 >= 0 && i - int(hn) -1 >= 0 && i-int(hn)-2 >= 0 && i % hn != 0 && (i-1) % hn != 0) {
            pbd_bendingConstraintData data;
            data.index1 = i - 1;
            data.index2 = i - hn - 1;
            data.index3 = i - hn - 2;
            data.index4 = i;
            data.k = bendingStiffness;
            data.phi = phi;

            out_bendConData.push_back(data);
        }/**/

        // Triangle Bending Constraint
        const float stiffness = 0.1f;
        const float curvature = 0.f;
        const float restLength = 0.f;
        // Down
        if(i-2*int(hn) >= 0)
        {
            pbd_triangleBendingConstraintData data;
            data.index_b0 = i;
            data.index_v = i-hn;
            data.index_b1 = i-2*hn;
            data.k = stiffness;
            data.curvature = curvature;
            data.restLength = restLength;

            out_triagBendConData.push_back(data);
        }/**/

        /*/Down Right
        if(i-2*int(hn)+2 >= 0 && i % (hn-1) != 0 && i % (hn-2) != 0)
        {
            pbd_triangleBendingConstraintData data;
            data.index_b0 = i;
            data.index_v = i-hn+1;
            data.index_b1 = i-2*hn+2;
            data.k = stiffness;
            data.curvature = curvature;
            data.restLength = restLength;

            out_triagBendConData.push_back(data);
        }/**/

        /*/ Down Left
        if(i-2*int(hn)-2 >= 0 && i % (hn+1) != 0 && i % (hn+2) != 0)
        {
            pbd_triangleBendingConstraintData data;
            data.index_b0 = i;
            data.index_v = i-hn-1;
            data.index_b1 = i-2*hn-2;
            data.k = stiffness;
            data.curvature = curvature;
            data.restLength = restLength;

            out_triagBendConData.push_back(data);
        }/**/

        }
}



void pbdgpu::deriveStandardBuffers(const vector<pbd_particle> &particles, vector<cl_float3> &predPosData,
                                     vector<cl_float> &masses, vector<cl_float> &scaledMasses,
                                     vector<cl_float3> &extForces, vector<cl_float3> &positionCorrections,vector<cl_int> &numConstraints) {
    const size_t numParticles = particles.size();

    predPosData.resize(numParticles);
    masses.resize(numParticles);
    scaledMasses.resize(numParticles);
    extForces.resize(numParticles);
    positionCorrections.resize(numParticles);
    numConstraints.resize(numParticles);

    for(int i = 0; i  < numParticles; ++i) {
        masses[i] = particles[i].invmass < 1e-8 ? CL_FLT_MAX : 1.f/particles[i].invmass;

        positionCorrections[i].x = 0.f;
        positionCorrections[i].y = 0.f;
        positionCorrections[i].z = 0.f;
        numConstraints[i] = 0;
    }

}

#include <stdio.h>
#include <chrono>
#include <memory>

#include <GL/glew.h>

#ifdef __APPLE__
#include <GLUT/freeglut.h>
#else
#include <GL/freeglut.h>
#endif

#define GLM_FORCE_RADIANS
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <clew.h>
#include <util/functions.hpp>
#include <util/cl_buffer_allocator.hpp>

#include <kernelInclude/particle.h>
#include <kernels.hpp>

#include <constraints/plane_collision_constraint.hpp>

#include <simulation_data.hpp>
#include <kernelInclude/distanceConstraintData.h>
#include <constraints/distance_constraint.hpp>

using glm::vec3;
using glm::mat4;

static struct simData
{
    const size_t particles_size = 200;
    std::shared_ptr<pbdgpu::GLBufferAllocator> particles;
    std::shared_ptr<pbdgpu::CLBufferAllocator> externalForces;
    std::shared_ptr<pbdgpu::CLBufferAllocator> predictedPositions;
    std::shared_ptr<pbdgpu::CLBufferAllocator> masses;
    std::shared_ptr<pbdgpu::CLBufferAllocator> scaledMasses;
    std::shared_ptr<pbdgpu::CLBufferAllocator> planes;
    std::shared_ptr<pbdgpu::CLBufferAllocator> distanceData;

    std::shared_ptr<pbdgpu::SimulationData> SData;

    cl_uint numPlanes = 1;

} simData;

static struct simulationParameters
{
    std::chrono::time_point<std::chrono::high_resolution_clock> lastTime;
}simParams;

static struct renderData
{
    unsigned int particlesVAO = 0;
    unsigned int cameraBuffer = 0;
    unsigned int planeBuffer = 0;
    unsigned int planeVAO = 0;
} renderData;

static struct gpuprograms
{
    GLuint particleVertexShader = 0;
    GLuint particleFragmentShader = 0;
    GLuint planeVertexShader = 0;
    GLuint planeFragmentShader = 0;
    GLuint particleShaderProgram = 0;
    GLuint planeShaderProgram = 0;
    cl_kernel predictionKernel = nullptr;
    cl_kernel updateKernel = nullptr;
    cl_kernel planeCollKernel = nullptr;
    GLint particleShaderProgramVertexAttribLocation;
    GLint planeShaderProgramVertexAttribLocation;
} progs;

static struct oclvars
{
    cl_device_id currentOGLDevice;
    vector<cl_context_properties> properties;
    cl_context GLCLContext;
    cl_command_queue queue;
} oclvars;

static struct mouse
{
    int lx;
    int ly;
} mouse;

static struct camera
{
    vec3 pos = vec3(0.0f,40.0f,-100.0f);
    vec3 cpos = vec3(0.0f,0.0f,0.0f);
    vec3 forward = glm::normalize(cpos-pos);
    vec3 up = vec3(0.0f,1.0f,0.0f);
    float fovy = 45.0f;
    float aspectRatio = 0.0f;
    float zNear = 0.1f;
    float zFar = 1000.0f;
    float speed = 0.51f;

    void move(vec3 delta)
    {
        pos += delta;
        cpos = pos + forward;
    }

    vec3 getRight() const
    {
        return glm::cross(forward,up);
    }
} cam;

static struct window
{
    unsigned int displayMode =  GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA;
    int x = 100;
    int y = 100;
    int width = 640;
    int height = 480;
    int cx = width/2;
    int cy = height/2;
    const char* title = "SimplePBD";
} win;

void draw();

void reshape(const int newWidth, const int newHeight) {

    win.width = newWidth;
    win.height = newHeight;

    if(win.height == 0)
        win.height = 1;

    cam.aspectRatio = 1.0f * newWidth / newHeight;

    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();
    glViewport(0,0, newWidth, newHeight);

    mat4 projMat = glm::perspective(cam.fovy,cam.aspectRatio,cam.zNear,cam.zFar);

    glLoadMatrixf(glm::value_ptr(projMat));

    glBindBuffer(GL_UNIFORM_BUFFER, renderData.cameraBuffer);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(float) * 16, glm::value_ptr(projMat));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glMatrixMode(GL_MODELVIEW);
}

void keyboard(const unsigned char key, const int x, const int y) {

    if(key == 27)
        glutLeaveMainLoop();

    if(key == 'w')
    {
        vec3 d = cam.speed * cam.forward;
        cam.move(d);
    }

    if(key == 's')
    {
        vec3 d = -cam.speed * cam.forward;
        cam.move(d);
    }

    if(key == 'd')
    {
        vec3 d = cam.speed * cam.getRight();
        cam.move(d);
    }

    if(key == 'a')
    {
        vec3 d = -cam.speed * cam.getRight();
        cam.move(d);
    }
}

void passiveMotion(int x, int y)
{
    mouse.lx = x;
    mouse.ly = y;
}

void motion(int x, int y)
{
    if(y-mouse.ly!=0 || x-mouse.lx!=0)
    {
        cam.forward = glm::normalize(glm::rotate(cam.forward,glm::radians(cam.speed),vec3(mouse.ly-y,x-mouse.lx,0)));
        cam.move(vec3());
    }

    passiveMotion(x,y);
}

void display(void) {

    auto now = std::chrono::high_resolution_clock::now();
    long elapsed_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now - simParams.lastTime).count();
    simParams.lastTime = now;
    float elapsed_seconds = float(elapsed_microseconds) / 1000000.0f;
    //printf("Elapsed time: %f \n",elapsed_seconds);

    // Compute

    simData.particles->acquireForCL(0,nullptr, nullptr);

    for (unsigned int i = 0; i < 1; ++i)
    {
        simData.SData->update();
    }

    simData.particles->releaseFromCL(0, nullptr, nullptr);

    draw();

}

void draw() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    mat4 viewMat = glm::lookAt(cam.pos, cam.cpos, cam.up);

    glLoadMatrixf(glm::value_ptr(viewMat));

    glBindBuffer(GL_UNIFORM_BUFFER, renderData.cameraBuffer);
    glBufferSubData(GL_UNIFORM_BUFFER, sizeof(float) * 16, sizeof(float) * 16, glm::value_ptr(viewMat));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    // draw groundplane
    glUseProgram(progs.planeShaderProgram);
    glBindVertexArray(renderData.planeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // draw particles
    glUseProgram(progs.particleShaderProgram);
    glBindVertexArray(renderData.particlesVAO);
    glDrawArrays(GL_POINTS, 0, simData.particles_size);

    //glBindBuffer(GL_ARRAY_BUFFER,0);
    glUseProgram(0);

    glutSwapBuffers();
}

void atClose()
{
    glDeleteProgram(progs.particleShaderProgram);

    glDeleteBuffers(1,&renderData.cameraBuffer);

    simData.particles->free();
    simData.externalForces->free();
    simData.predictedPositions->free();
    simData.masses->free();
    simData.scaledMasses->free();
    simData.planes->free();
    simData.distanceData->free();

    simData.SData.reset();
}

int main(int argc, char *argv[])
{
    setbuf(stdout, nullptr);

    // init OpenGL
    glutInit(&argc, argv);
    glutInitContextVersion(3,3);
    glutInitContextProfile(GLUT_CORE_PROFILE);
    glutInitDisplayMode(win.displayMode);
    glutInitWindowPosition(win.x,win.y);
    glutInitWindowSize(win.width, win.height);
    glutCreateWindow(win.title);

    std::printf("%s\n%s\n",
                glGetString(GL_RENDERER),
                glGetString(GL_VERSION)
    );

    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutPassiveMotionFunc(passiveMotion);
    glutCloseFunc(atClose);

    glewExperimental = GL_TRUE;
    glewInit();

    glEnable(GL_POINT_SMOOTH);
    glPointSize(5.0f);

    progs.particleVertexShader = pbdgpu::createShader(pbdgpu::readFile("shaders/particle.vert"), GL_VERTEX_SHADER);
    progs.particleFragmentShader = pbdgpu::createShader(pbdgpu::readFile("shaders/particle.frag"), GL_FRAGMENT_SHADER);
    progs.planeVertexShader = pbdgpu::createShader(pbdgpu::readFile("shaders/plane.vert"),GL_VERTEX_SHADER);
    progs.planeFragmentShader = pbdgpu::createShader(pbdgpu::readFile("shaders/plane.frag"),GL_FRAGMENT_SHADER);

    progs.particleShaderProgram = pbdgpu::createProgram(progs.particleVertexShader, 0, 0, progs.particleFragmentShader);
    progs.particleShaderProgramVertexAttribLocation = glGetAttribLocation(progs.particleShaderProgram,"position");

    progs.planeShaderProgram = pbdgpu::createProgram(progs.planeVertexShader, 0, 0, progs.planeFragmentShader);
    progs.planeShaderProgramVertexAttribLocation = glGetAttribLocation(progs.planeShaderProgram,"position");

    glDeleteShader(progs.particleVertexShader);
    glDeleteShader(progs.particleFragmentShader);
    glDeleteShader(progs.planeVertexShader);
    glDeleteShader(progs.planeFragmentShader);

    // init OpenCL

    bool clpresent = 0 == clewInit();
    if (!clpresent) {
        printf("OpenCL library not found");
        return -1;
    }

    int cl_err;

    bool useSharing = false;

    oclvars.properties = pbdgpu::getOGLInteropInfo(oclvars.currentOGLDevice);
    if (oclvars.properties.empty())
    {
        // if properties if null abort test because no CLGL interop device could be found
        oclvars.properties.push_back(0);

        cl_uint num_platforms;

        clGetPlatformIDs(0, nullptr, &num_platforms);

        vector<cl_platform_id> platforms(num_platforms);

        clGetPlatformIDs(num_platforms, &platforms[0], nullptr);

        cl_int error = 0;
        for (unsigned int i = 0; i < num_platforms; ++i)
        {
            error = clGetDeviceIDs(platforms[i],CL_DEVICE_TYPE_GPU,1,&oclvars.currentOGLDevice,nullptr);
            assert(error == 0 && "Error while getting Devices");
        }

        useSharing = false;
    }else{
        useSharing = true;
    }

    oclvars.GLCLContext = clCreateContext(&oclvars.properties[0], 1, &oclvars.currentOGLDevice, nullptr, nullptr, nullptr);
    oclvars.queue = clCreateCommandQueue(oclvars.GLCLContext, oclvars.currentOGLDevice, 0, &cl_err);
    assert(cl_err == 0 && "Error while creating commandqueue");

    // init buffers

    // init particle buffer
    vector<pbd_particle> pos(simData.particles_size);
    for(size_t i = 0; i < simData.particles_size; ++i)
    {
        pos[i].x.x = float(i)-100.0f;
        pos[i].x.y = 10.0f;
        pos[i].x.z = 0.0f;
        pos[i].v.x = 0.0f;
        pos[i].v.y = 0.0f;
        pos[i].v.z = 0.0f;
        pos[i].invmass = 1.0f;
        pos[i].phase = 1;
    }

    simData.particles = pbdgpu::createSharedBuffer(useSharing,sizeof(pbd_particle), simData.particles_size,oclvars.GLCLContext,oclvars.queue);
    simData.particles->write(simData.particles_size, &pos[0]);

    glGenVertexArrays(1,&renderData.particlesVAO);
    glBindBuffer(GL_ARRAY_BUFFER, simData.particles->getBufferID());
    glBindVertexArray(renderData.particlesVAO);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(pbd_particle),NULL);
    glEnableVertexAttribArray(0);

    // init external forces buffer
    vector<cl_float3> extForces(simData.particles_size);
    for(size_t i = 0; i < simData.particles_size; ++i)
    {
        extForces[i].x = 0.0f;
        extForces[i].y = 0.0f;
        extForces[i].z = 0.0f;
    }

    simData.externalForces = std::make_shared<pbdgpu::CLBufferAllocator>(oclvars.GLCLContext,oclvars.queue,sizeof(cl_float3), simData.particles_size);
    simData.externalForces->write(simData.particles_size, &extForces[0]);

    // init predicted positions buffer
    vector<cl_float3> predPos(simData.particles_size);
    for(size_t i = 0; i < simData.particles_size; ++i)
    {
        predPos[i].x = 0.0f;
        predPos[i].y = 0.0f;
        predPos[i].z = 0.0f;
    }
    simData.predictedPositions = std::make_shared<pbdgpu::CLBufferAllocator>(oclvars.GLCLContext,oclvars.queue,sizeof(cl_float3), simData.particles_size);
    simData.predictedPositions->write(simData.particles_size, &predPos[0]);

    // init masses buffer
    vector<cl_float> masses(simData.particles_size);
    for(size_t i = 0; i < simData.particles_size; ++i)
    {
        masses[i] = 1.0f;
    }
    simData.masses = std::make_shared<pbdgpu::CLBufferAllocator>(oclvars.GLCLContext,oclvars.queue,sizeof(cl_float), simData.particles_size);
    simData.masses->write(simData.particles_size, &masses[0]);

    // init scaled masses buffer
    vector<cl_float> scaledMasses(simData.particles_size);
    for(size_t i = 0; i < simData.particles_size; ++i)
    {
        scaledMasses[i] = 1.0f;
    }

    simData.scaledMasses = std::make_shared<pbdgpu::CLBufferAllocator>(oclvars.GLCLContext, oclvars.queue, sizeof(cl_float), simData.particles_size);
    simData.scaledMasses->write(simData.particles_size, &scaledMasses[0]);

    // init plane buffer
    // layout = n.x n.y n.z d | n = plane normal
    cl_float4 plane;
    plane.x = 0.0f;
    plane.y = 1.0f;
    plane.z = 0.0f;
    plane.w = 0.0f;

    simData.planes = std::make_shared<pbdgpu::CLBufferAllocator>(oclvars.GLCLContext, oclvars.queue, sizeof(cl_float4),simData.numPlanes);
    simData.planes->write(simData.numPlanes,&plane);

    // init additional buffers
    glGenBuffers(1, &renderData.cameraBuffer);
    glBindBuffer(GL_UNIFORM_BUFFER, renderData.cameraBuffer);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 32, NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    GLuint uniformBlockIndex = glGetUniformBlockIndex(progs.particleShaderProgram,"globalMatrices");
    if(uniformBlockIndex == GL_INVALID_INDEX)
    {
        fprintf(stderr, "Failed to get Uniform Block Index: %s\n", "globalMatrices");
        exit(1);
    }
    glUniformBlockBinding(progs.particleShaderProgram, uniformBlockIndex, 0);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, renderData.cameraBuffer, 0, sizeof(float) * 32);

    vector<float> planeVerts = {
            -100.0f,0.0f,100.0f,
            -100.0f,0.0f,-100.0f,
            100.0f,0.0f,100.0f,
            -100.0f,0.0f,-100.0f,
            100.0f,0.0f,-100.0f,
            100.0f,0.0f,100.0f};

    glGenBuffers(1, &renderData.planeBuffer);
    glBindBuffer(GL_UNIFORM_BUFFER, renderData.planeBuffer);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 18, &planeVerts[0], GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glGenVertexArrays(1,&renderData.planeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, renderData.planeBuffer);
    glBindVertexArray(renderData.planeVAO);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,0,NULL);
    glEnableVertexAttribArray(1);

    // init distance constraint data

    vector<pbd_distanceConstraintData> distanceConstraintData;

    for(int i = 0; i < 100; i=i+2){
        pbd_distanceConstraintData data;
        data.d = 0.5f;
        data.index0 = i;
        data.index1 = i+1;
        distanceConstraintData.push_back(data);
    }

    simData.distanceData = std::make_shared<pbdgpu::CLBufferAllocator>(oclvars.GLCLContext, oclvars.queue, sizeof(pbd_distanceConstraintData),distanceConstraintData.size());
    simData.distanceData->write(distanceConstraintData.size(),&distanceConstraintData[0]);


    // init constraints

    simData.SData = std::make_shared<pbdgpu::SimulationData>(simData.particles_size,oclvars.GLCLContext,oclvars.currentOGLDevice,oclvars.queue,1);

    simData.SData->addSharedBuffer(simData.particles,pbdgpu::PARTICLE_BUFFER_NAME);
    simData.SData->addSharedBuffer(simData.predictedPositions,pbdgpu::PREDICTED_POSITIONS_BUFFER_NAME);
    simData.SData->addSharedBuffer(simData.externalForces,pbdgpu::EXTERNAL_FORCES_BUFFER_NAME);
    simData.SData->addSharedBuffer(simData.masses,pbdgpu::MASSES_BUFFER_NAME);
    simData.SData->addSharedBuffer(simData.scaledMasses,pbdgpu::SCALED_MASSES_BUFFER_NAME);
    simData.SData->initStandardKernels();

    simData.SData->buildConstraint<pbdgpu::PlaneCollisionConstraint>(simData.planes);
    //simData.SData->buildConstraint<pbdgpu::DistanceConstraint>(simData.distanceData);

    // start app

    glutMainLoop();
}

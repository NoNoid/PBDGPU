#include <stdio.h>

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
#include <util/gl_buffer_allocator.hpp>
#include <util/cl_buffer_allocator.hpp>

#include <kernelInclude/particle.h>
#include <kernels.hpp>

using glm::vec3;
using glm::mat4;

static struct simData
{
    simData()
    {
        g.x = 0.0f;
        g.y = -0.01f;
        g.z = 0.0f;
    }

    const size_t particles_size = 200;
    pbdgpu::GLBufferAllocator particles;
    pbdgpu::CLBufferAllocator externalForces;
    pbdgpu::CLBufferAllocator predictedPositions;
    pbdgpu::CLBufferAllocator masses;
    pbdgpu::CLBufferAllocator scaledMasses;

    cl_float3 g;
    cl_float dt = 0.03333333333f;
} simData;

static struct renderData
{
    unsigned int particlesVAO = 0;
    unsigned int cameraBuffer = 0;
} renderData;

static struct gpuprograms
{
    GLuint vertexShader = 0;
    GLuint fragmentShader = 0;
    GLuint particleShaderProgram = 0;
    cl_kernel predictionKernel = nullptr;
    cl_kernel updateKernel;
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
    unsigned int displayMode = GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA;
    int x = 100;
    int y = 100;
    int width = 640;
    int height = 480;
    int cx = width/2;
    int cy = height/2;
    const char* title = "SimplePBD";
} win;

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

    // Compute
    cl_event acquireEvent = nullptr;
    simData.particles.acquireForCL(0,nullptr,&acquireEvent);

    cl_event predKernelEvent = nullptr;
    int cl_err = clEnqueueNDRangeKernel(oclvars.queue, progs.predictionKernel, 1, nullptr, &simData.particles_size, nullptr, 1,&acquireEvent , &predKernelEvent);
    if(cl_err != CL_SUCCESS)
    {
        printf("Error on Prediction Kernel Execution:%d",cl_err);
    }

    cl_event updateKernelEvent = nullptr;
    cl_err = clEnqueueNDRangeKernel(oclvars.queue, progs.updateKernel, 1, nullptr, &simData.particles_size, nullptr, 1,&predKernelEvent , &updateKernelEvent);
    if(cl_err != CL_SUCCESS)
    {
        printf("Error on Update Kernel Execution:%d",cl_err);
    }

    simData.particles.releaseFromCL(1,&updateKernelEvent, nullptr);

    // Draw
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    mat4 viewMat = glm::lookAt(cam.pos,cam.cpos,cam.up);

    glLoadMatrixf(glm::value_ptr(viewMat));

    glBindBuffer(GL_UNIFORM_BUFFER, renderData.cameraBuffer);
    glBufferSubData(GL_UNIFORM_BUFFER, sizeof(float) * 16, sizeof(float) * 16, glm::value_ptr(viewMat));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    // draw groundplane
    glColor3f(0.5f,0.5f,0.5f);
    glBegin(GL_QUADS);
        glVertex3f(-100.0f,0.0f,100.0f);
        glVertex3f(100.0f,0.0f,100.0f);
        glVertex3f(100.0f,0.0f,-100.0f);
        glVertex3f(-100.0f,0.0f,-100.0f);
    glEnd();
    glColor3f(1.0f,1.0f,1.0f);

    // draw particles
    glUseProgram(progs.particleShaderProgram);
    //glBindBuffer(GL_ARRAY_BUFFER,buffers.positions.getBufferID());

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

    simData.particles.free();
    simData.externalForces.free();
    simData.predictedPositions.free();
    simData.masses.free();
    simData.scaledMasses.free();
}

int main(int argc, char *argv[])
{
    setbuf(stdout, nullptr);

    // init OpenGL
    glutInit(&argc, argv);
    glutInitDisplayMode(win.displayMode);
    glutInitWindowPosition(win.x,win.y);
    glutInitWindowSize(win.width, win.height);
    glutCreateWindow(win.title);

    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutPassiveMotionFunc(passiveMotion);
    glutCloseFunc(atClose);

    glewInit();

    glEnable(GL_POINT_SMOOTH);
    glPointSize(5.0f);

    progs.vertexShader = pbdgpu::createShader(pbdgpu::readFile("shaders/particle.vert"),GL_VERTEX_SHADER);
    progs.fragmentShader = pbdgpu::createShader(pbdgpu::readFile("shaders/particle.frag"),GL_FRAGMENT_SHADER);
    progs.particleShaderProgram = pbdgpu::createProgram(progs.vertexShader,0,0,progs.fragmentShader);

    glDeleteShader(progs.vertexShader);
    glDeleteShader(progs.fragmentShader);

    // init OpenCL

    bool clpresent = 0 == clewInit();
    if (!clpresent) {
        printf("OpenCL library not found");
        return -1;
    }

    oclvars.properties = pbdgpu::getOGLInteropInfo(oclvars.currentOGLDevice);
    if (oclvars.properties.empty())
    {
        // if properties if null abort test because no CLGL interop device could be found
        return -1;
    }
    oclvars.GLCLContext = clCreateContext(&oclvars.properties[0], 1, &oclvars.currentOGLDevice, nullptr, nullptr, nullptr);
    oclvars.queue = clCreateCommandQueue(oclvars.GLCLContext, oclvars.currentOGLDevice, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, nullptr);
    progs.predictionKernel = pbdgpu::buildPredictionKernel(oclvars.GLCLContext,oclvars.currentOGLDevice);
    clSetKernelArg(progs.predictionKernel, 5, sizeof(cl_float3), &simData.g);
    clSetKernelArg(progs.predictionKernel, 6, sizeof(cl_float), &simData.dt);

    progs.updateKernel = pbdgpu::buildUpdateKernel(oclvars.GLCLContext,oclvars.currentOGLDevice);
    clSetKernelArg(progs.updateKernel,2,sizeof(cl_float),&simData.dt);

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

    simData.particles.allocate(sizeof(pbd_particle), simData.particles_size);
    simData.particles.write(simData.particles_size, &pos[0]);
    simData.particles.initCLSharing(oclvars.GLCLContext,oclvars.queue);
    clSetKernelArg(progs.predictionKernel, 0, sizeof(cl_mem), &simData.particles.getCLMem());
    clSetKernelArg(progs.updateKernel,0,sizeof(cl_mem), &simData.particles.getCLMem());

    glGenVertexArrays(1,&renderData.particlesVAO);
    glBindBuffer(GL_ARRAY_BUFFER, simData.particles.getBufferID());
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

    simData.externalForces.setOpenCLContext(oclvars.GLCLContext);
    simData.externalForces.setOpenCLCommandQueue(oclvars.queue);
    simData.externalForces.allocate(sizeof(cl_float3), simData.particles_size);
    simData.externalForces.write(simData.particles_size, &extForces[0]);
    clSetKernelArg(progs.predictionKernel, 1, sizeof(cl_mem), &simData.externalForces.getCLMem());

    // init predicted positions buffer
    vector<cl_float3> predPos(simData.particles_size);
    for(size_t i = 0; i < simData.particles_size; ++i)
    {
        predPos[i].x = 0.0f;
        predPos[i].y = 0.0f;
        predPos[i].z = 0.0f;
    }
    simData.predictedPositions.setOpenCLContext(oclvars.GLCLContext);
    simData.predictedPositions.setOpenCLCommandQueue(oclvars.queue);
    simData.predictedPositions.allocate(sizeof(cl_float3), simData.particles_size);
    simData.predictedPositions.write(simData.particles_size, &predPos[0]);
    clSetKernelArg(progs.predictionKernel, 2, sizeof(cl_mem), &simData.predictedPositions.getCLMem());
    clSetKernelArg(progs.updateKernel,1,sizeof(cl_mem),&simData.predictedPositions.getCLMem());

    // init masses buffer
    vector<cl_float> masses(simData.particles_size);
    for(size_t i = 0; i < simData.particles_size; ++i)
    {
        masses[i] = 1.0f;
    }
    simData.masses.setOpenCLContext(oclvars.GLCLContext);
    simData.masses.setOpenCLCommandQueue(oclvars.queue);
    simData.masses.allocate(sizeof(cl_float), simData.particles_size);
    simData.masses.write(simData.particles_size, &masses[0]);
    clSetKernelArg(progs.predictionKernel, 3, sizeof(cl_mem), &simData.masses.getCLMem());

    // init scaled masses buffer
    vector<cl_float> scaledMasses(simData.particles_size);
    for(size_t i = 0; i < simData.particles_size; ++i)
    {
        scaledMasses[i] = 1.0f;
    }
    simData.scaledMasses.setOpenCLContext(oclvars.GLCLContext);
    simData.scaledMasses.setOpenCLCommandQueue(oclvars.queue);
    simData.scaledMasses.allocate(sizeof(cl_float), simData.particles_size);
    simData.scaledMasses.write(simData.particles_size, &scaledMasses[0]);
    clSetKernelArg(progs.predictionKernel, 4, sizeof(cl_mem), &simData.scaledMasses.getCLMem());

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

    // start app

    glutMainLoop();
}

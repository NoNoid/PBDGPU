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

using glm::vec3;
using glm::mat4;

static struct buffers
{
    unsigned int particlesVAO = 0;
    const size_t particles_size = 200;
    pbdgpu::GLBufferAllocator particles = pbdgpu::GLBufferAllocator();

    unsigned int cameraBuffer = 0;
} buffers;

static struct shaders
{
    GLuint vertexShader = 0;
    GLuint fragmentShader = 0;
    GLuint particleShaderProgram = 0;
} shaders;

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

    glBindBuffer(GL_UNIFORM_BUFFER, buffers.cameraBuffer);
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

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    mat4 viewMat = glm::lookAt(cam.pos,cam.cpos,cam.up);

    glLoadMatrixf(glm::value_ptr(viewMat));

    glBindBuffer(GL_UNIFORM_BUFFER, buffers.cameraBuffer);
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
    glUseProgram(shaders.particleShaderProgram);
    //glBindBuffer(GL_ARRAY_BUFFER,buffers.positions.getBufferID());

    glBindVertexArray(buffers.particlesVAO);
    glDrawArrays(GL_POINTS,0,buffers.particles_size);

    //glBindBuffer(GL_ARRAY_BUFFER,0);
    glUseProgram(0);

    glutSwapBuffers();
}

void atClose()
{
    glDeleteProgram(shaders.particleShaderProgram);

    glDeleteBuffers(1,&buffers.cameraBuffer);

    buffers.particles.free();
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

    shaders.vertexShader = pbdgpu::createShader("shaders/particle.vert",GL_VERTEX_SHADER);
    shaders.fragmentShader = pbdgpu::createShader("shaders/particle.frag",GL_FRAGMENT_SHADER);
    shaders.particleShaderProgram = pbdgpu::createProgram(shaders.vertexShader,0,0,shaders.fragmentShader);

    glDeleteShader(shaders.vertexShader);
    glDeleteShader(shaders.fragmentShader);

    // init OpenCL

    bool clpresent = 0 == clewInit();
    if (!clpresent) {
        printf("OpenCL library not found");
        return -1;
    }

    // init buffers

    particle* pos = new particle[buffers.particles_size];

    for( int i = 0; i < buffers.particles_size; ++i)
    {
        pos[i].x.x = float(i)-100.0f;
        pos[i].x.y = 10.0f;
        pos[i].x.z = 0.0f;
        pos[i].x.w = 1.0f;
    }

    buffers.particles.allocate(sizeof(particle),buffers.particles_size);
    buffers.particles.write(buffers.particles_size,pos);

    glGenVertexArrays(1,&buffers.particlesVAO);
    glBindBuffer(GL_ARRAY_BUFFER,buffers.particles.getBufferID());
    glBindVertexArray(buffers.particlesVAO);
    glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,sizeof(particle),NULL);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &buffers.cameraBuffer);
    glBindBuffer(GL_UNIFORM_BUFFER, buffers.cameraBuffer);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 32, NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    GLuint uniformBlockIndex = glGetUniformBlockIndex(shaders.particleShaderProgram,"globalMatrices");
    if(uniformBlockIndex == GL_INVALID_INDEX)
    {
        fprintf(stderr, "Failed to get Uniform Block Index: %s\n", "globalMatrices");
        exit(1);
    }
    glUniformBlockBinding(shaders.particleShaderProgram, uniformBlockIndex, 0);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, buffers.cameraBuffer, 0, sizeof(float)*32);

    // start app

    glutMainLoop();
}

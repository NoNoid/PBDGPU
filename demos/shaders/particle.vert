#version 330

layout(location = 0) in vec4 position;

layout(std140) uniform globalMatrices
{
        mat4 Projection;
        mat4 View;
};

void main()
{
    gl_Position = Projection * View * position;
    gl_PointSize = 20.0f;
}

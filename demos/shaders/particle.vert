#version 330

layout(location = 0) in vec3 position;

layout(std140) uniform globalMatrices
{
        mat4 Projection;
        mat4 View;
};

void main()
{
    gl_Position = Projection * View * vec4(position,1.0f);
}

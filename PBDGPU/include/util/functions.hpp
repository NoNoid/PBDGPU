#ifndef _UTILITY_FUNCTIONS_HPP_
#define _UTILITY_FUNCTIONS_HPP_

#include <string>
#include <vector>

#include <clew.h>

using std::string;
using std::vector;

/** @file
 * Contains utility functions
*/

/**
 * @namespace pbdgpu
 */
namespace pbdgpu
{

    /** @fn vector<cl_context_properties> getOGLInteropInfo(cl_device_id &out_device);
     * A function to search for the OpenCL Device which is associated with the current OpenGL context.
     * Can Only be called in thread which has a current OpenGL context.
     * @param out_device returns the device id of the OpenCL device, currently associated with OpenGL context.
     * @return OpenCl context properties which are needed to correctly initialse the a GL-CL-Interop context.
     */
    vector<cl_context_properties> getOGLInteropInfo(cl_device_id &out_device);

    /** @fn string readFile(const string filename);
     * Open the specified file and reads its contents.
     * @param filename The filename can be relative or absolut.
     * @return contents of the file
     */
    string readFile(const string filename);

    /**
     * @fn unsigned int createShader(const string filename, const unsigned int shaderType)
     * @param filename The filename of the Shader can be realtive or absolute.
     * @param shaderType Type of the Shader. Allowed values are GL_COMPUTE_SHADER, GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, or GL_FRAGMENT_SHADER.
     * @return id of the shader or 0 if creation failed.
     */
    unsigned int createShader(const string shaderSource, const unsigned int shaderType);

    /**
     * @fn unsigned int createProgram(
            const unsigned int vertexShader,
            const unsigned int hullShader,
            const unsigned int domainShader,
            const unsigned int fragmentShader)
     * @param vertexShader Id of a shader of type GL_VERTEX_SHADER. Must be valid.
     * @param hullShader Id of a shader of type GL_TESS_CONTROL_SHADER. Will be ignored if invalid.
     * @param domainShader Id of a shader of type GL_TESS_EVALUATION_SHADER. Will be ignored if invalid.
     * @param fragmentShader Id of Shader of type GL_FRAGMENT_SHADER. Must be valid.
     * @return id of the program or 0 if creation failed.
     */
    unsigned int createProgram(
            const unsigned int vertexShader,
            const unsigned int hullShader,
            const unsigned int domainShader,
            const unsigned int fragmentShader);

    cl_kernel createKernel(string kernelSource,string buildOptions, string kernelName, const cl_context context, const cl_device_id device);
}
#endif

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

inline void float_atomic_add(global float* address, float value)
{
    float old = value;

    while ((old = atomic_xchg(address, atomic_xchg(address, 0.0f)+old))!=0.0f);

}

inline void float3_atomic_add(global float3* address, float3 value)
{
    global float* singleAddress = address;
    float_atomic_add(singleAddress,value.x);
    float_atomic_add(singleAddress+1,value.y);
    float_atomic_add(singleAddress+2,value.z);
}
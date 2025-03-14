#pragma once

#include "vec3.h"

class ray
{
public:
    __device__ ray() {}
    __device__ ray(const vec3& a, const vec3& b, float time = 0)
        : A(a), B(b), tm(time) {}
    __device__ vec3 origin() const { return A; }
    __device__ vec3 direction() const { return B; }
    __device__ float time() const { return tm; }
    __device__ vec3 point_at_parameter(float t) const { return A + t * B; }

    vec3 A;
    vec3 B;
    float tm;
};
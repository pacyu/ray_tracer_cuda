#pragma once

#include "ray.h"
#include <corecrt_math_defines.h>

__device__ vec3 random_in_unit_disk(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

__device__ float random_float(float min, float max, curandState* local_rand_state) {
    return min + (max - min) * curand_uniform(local_rand_state);
}

class camera {
public:
    __device__ camera(
        vec3 lookfrom,
        vec3 lookat,
        vec3 vup,
        float vfov,
        float aspect_ratio,
        float aperture,
        float focus_dist,
        float _time0 = 0,
        float _time1 = 0
    ) {
        float theta = vfov * ((float)M_PI) / 180.0f;
        float viewport_height = tan(theta / 2.0f);
        float viewport_width = aspect_ratio * viewport_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;
        lens_radius = aperture / 2.0f;
        time0 = _time0;
        time1 = _time1;
    }

    __device__ ray get_ray(float s, float t, curandState* local_rand_state) {
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(
            origin + offset,
            lower_left_corner + s * horizontal + t * vertical - origin - offset,
            random_float(time0, time1, local_rand_state));
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
    float time0, time1;
};
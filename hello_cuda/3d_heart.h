#pragma once
#include "hittable.h"


class heart : public hittable {
public:
    __device__ heart() {}
    __device__ heart(float x0, float x1, float y0, float y1, float z0, float z1, vec3 cen, material* m)
        : x0(x0), x1(x1), y0(y0), y1(y1), z0(z0), z1(z1), center(cen), mat_ptr(m)
    {}
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

    float x0, x1, y0, y1, z0, z1;
    vec3 center;
    material* mat_ptr;
};

__device__ bool heart::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
    float t = 1.0;
    if (t < tmin || t > tmax) return false;
    float x = (r.origin().x() + t * r.direction().x()) - center.x();
    float y = (r.origin().y() + t * r.direction().y()) - center.y();
    float z = (r.origin().z() + t * r.direction().z()) - center.z();
    if (x0 > x || x > x1 || y0 > y || y > y1 || z0 > z || z > z1)
        return false;
    rec.t = t;
    rec.p = r.point_at_parameter(t);
    rec.normal = unit_vector(rec.p - center);
    return true;
}
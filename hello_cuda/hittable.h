#pragma once

#include "ray.h"
#include "aabb.h"

class material;

struct hit_record
{
  float t;
  float u;
  float v;
  vec3 p;
  vec3 normal;
  material* mat_ptr;
  bool front_face;

  __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
    front_face = dot(r.direction(), outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
  }
};

class hittable {
public:
  __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
  __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const = 0;
};
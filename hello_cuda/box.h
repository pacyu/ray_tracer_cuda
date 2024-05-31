#pragma once

#include "arect.h"
#include "hittable_list.h"
#include "aabb.h"

class box : public hittable {
public:
  __device__ box() : box_min(), box_max(), sides() {}
  __device__ box(const vec3& p0, const vec3& p1, material* ptr);

  __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

  __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const {
    output_box = aabb(box_min, box_max);
    return true;
  }

public:
  vec3 box_min;
  vec3 box_max;
  hittable_list* sides;
};

__device__ box::box(const vec3& p0, const vec3& p1, material* ptr)
  : box_min(p0), box_max(p1) {
  hittable *plane[6];
  plane[0] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr);
  plane[1] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr);

  plane[2] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr);
  plane[3] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr);

  plane[4] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr);
  plane[5] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr);
  sides = new hittable_list(plane, 6);
}

__device__ bool box::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
  return sides->hit(r, t_min, t_max, rec);
}
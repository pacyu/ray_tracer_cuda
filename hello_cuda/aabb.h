#pragma once

#include "ray.h"
#include <crt/math_functions.h>

class aabb {
public:
  __device__ aabb() {}
  __device__ aabb(const vec3& a, const vec3& b) { minimum = a; maximum = b; }

  __device__ vec3 min() const { return minimum; }
  __device__ vec3 max() const { return maximum; }

  __device__ bool hit(const ray& r, float t_min, float t_max) const {
    for (int a = 0; a < 3; a++) {
      auto t0 = fmin((minimum[a] - r.origin()[a]) / r.direction()[a],
        (maximum[a] - r.origin()[a]) / r.direction()[a]);
      auto t1 = fmax((minimum[a] - r.origin()[a]) / r.direction()[a],
        (maximum[a] - r.origin()[a]) / r.direction()[a]);
      t_min = fmax(t0, t_min);
      t_max = fmin(t1, t_max);
      if (t_max <= t_min)
        return false;
    }
    return true;
  }

  vec3 minimum;
  vec3 maximum;
};

__device__ aabb surrounding_box(aabb box0, aabb box1) {
  return aabb(
    vec3(
      min(box0.minimum.x(), box1.minimum.x()),
      min(box0.minimum.y(), box1.minimum.y()),
      min(box0.minimum.z(), box1.minimum.z())),
    vec3(
      max(box0.maximum.x(), box1.maximum.x()),
      max(box0.maximum.y(), box1.maximum.y()),
      max(box0.maximum.z(), box1.maximum.z())));
}
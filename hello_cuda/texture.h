#pragma once
#include "vec3.h"

class texture_ {
public:
  __device__ texture_() {}
  __device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class solid_color : public texture_ {
public:
  __device__ solid_color() {}
  __device__ solid_color(vec3 c) : color_value(c) {}

  __device__ solid_color(float red, float green, float blue)
    : solid_color(vec3(red, green, blue)) {}

  __device__ virtual vec3 value(float u, float v, const vec3& p) const {
    return color_value;
  }

private:
  vec3 color_value;
};
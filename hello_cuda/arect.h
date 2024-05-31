#pragma once

#include "hittable.h"

class xy_rect : public hittable {
public:
	__device__ xy_rect() {}
	__device__ xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material* m)
		: x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mat_ptr(m)
	{}
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
	__device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const {
		output_box = aabb(vec3(x0, y0, k - 0.0001), vec3(x1, y1, k + 0.0001));
		return true;
	}
	float x0, x1, y0, y1, k;
	material* mat_ptr;
};

__device__ bool xy_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
	float t = (k - r.origin().z()) / r.direction().z();
	if (t < t_min || t > t_max)
		return false;
	float xt = r.origin().x() + t * r.direction().x();
	float yt = r.origin().y() + t * r.direction().y();
	if (xt < x0 || xt > x1 || yt < y0 || yt > y1)
		return false;
	rec.u = (xt - x0) / (x1 - x0);
	rec.v = (yt - y0) / (y1 - y0);
	rec.t = t;
	vec3 outward_normal = vec3(0, 0, 1);
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mat_ptr;
	rec.p = r.point_at_parameter(t);
	return true;
}

class xz_rect : public hittable {
public:
	__device__ xz_rect() {}
	__device__ xz_rect(float _x0, float _x1, float _z0, float _z1, float _k, material* m)
		: x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mat_ptr(m)
	{}
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
	__device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const {
		output_box = aabb(vec3(x0, k - 0.0001, z0), vec3(x1, k + 0.0001, z1));
		return true;
	}
	float x0, x1, z0, z1, k;
	material* mat_ptr;
};

__device__ bool xz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
	float t = (k - r.origin().y()) / r.direction().y();
	if (t < t_min || t > t_max)
		return false;
	float xt = r.origin().x() + t * r.direction().x();
	float zt = r.origin().z() + t * r.direction().z();
	if (xt < x0 || xt > x1 || zt < z0 || zt > z1)
		return false;
	rec.u = (xt - x0) / (x1 - x0);
	rec.v = (zt - z0) / (z1 - z0);
	rec.t = t;
	vec3 outward_normal = vec3(0, 1, 0);
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mat_ptr;
	rec.p = r.point_at_parameter(t);
	return true;
}

class yz_rect : public hittable {
public:
	__device__ yz_rect() {}
	__device__ yz_rect(float _y0, float _y1, float _z0, float _z1, float _k, material* m)
		: y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mat_ptr(m)
	{}
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
	__device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const {
		output_box = aabb(vec3(y0, k - 0.0001, z0), vec3(y1, k + 0.0001, z1));
		return true;
	}
	float y0, y1, z0, z1, k;
	material* mat_ptr;
};

__device__ bool yz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
	float t = (k - r.origin().x()) / r.direction().x();
	if (t < t_min || t > t_max)
		return false;
	float yt = r.origin().y() + t * r.direction().y();
	float zt = r.origin().z() + t * r.direction().z();
	if (yt < y0 || yt > y1 || zt < z0 || zt > z1)
		return false;
	rec.u = (yt - y0) / (y1 - y0);
	rec.v = (zt - z0) / (z1 - z0);
	rec.t = t;
	vec3 outward_normal = vec3(1, 0, 0);
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mat_ptr;
	rec.p = r.point_at_parameter(t);
	return true;
}
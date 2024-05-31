#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include "sphere.h"
#include "box.h"
#include "3d_heart.h"
#include "camera.h"
#include "hittable_list.h"
#include "material.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
      << file << ":" << line << " '" << func << "' \n";
    cudaDeviceReset();
    exit(99);
  }
}

__global__ void rand_init(curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(1984, 0, 0, rand_state);
  }
}

__device__ inline float clamp(float x, float min, float max)
{
  if (x < min) return min;
  else if (x > max) return max;
  return x;
}

__device__ vec3 color(const ray& r, hittable** world, curandState* local_rand_state)
{
  ray cur_ray = r;
  vec3 emitted(0, 0, 0);
  vec3 cur_attenuation(1, 1, 1);
  for (int i = 0; i < 50; i++) {
    hit_record rec;
    if (!(*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) return vec3(0.0, 0.0, 0.0);
    
    ray scattered;
    vec3 attenuation;
    float pdf;
    emitted += cur_attenuation * rec.mat_ptr->scattering_pdf(cur_ray, rec, scattered)
      * rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

    if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, pdf, local_rand_state)) {
      cur_attenuation *= attenuation;
      cur_ray = scattered;
    }
    else {
      return emitted / pdf;
    }

    //vec3 unit_direction = unit_vector(cur_ray.direction());
    //float t = 0.5f * (unit_direction.y() + 1.0f);
    //vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    //return cur_attenuation * c;
  }
  return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j * max_x + i;
  //Each thread gets same seed, a different sequence number, no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, const int max_x, const int max_y,
  int ns, camera** cam, hittable** world, curandState* rand_state)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j * max_x + i;
  curandState local_rand_state = rand_state[pixel_index];
  vec3 col(0, 0, 0);
  for (int s = 0; s < ns; s++) {
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x - 1);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y - 1);
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    col += color(r, world, &local_rand_state);
  }
  rand_state[pixel_index] = local_rand_state;
  col /= float(ns);
  col[0] = clamp(sqrt(col[0]), 0, 0.999);
  col[1] = clamp(sqrt(col[1]), 0, 0.999);
  col[2] = clamp(sqrt(col[2]), 0, 0.999);
  fb[pixel_index] = col;
}

__global__ void create_world(
  hittable** d_list,
  hittable** d_world,
  camera** d_camera,
  int nx, int ny,
  vec3 lookfrom,
  vec3 lookat,
  curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;
    int i = 0;
    d_list[i++] = new yz_rect(0, 555, 0, 555, 555, new lambertian(vec3(.12, .45, .15)));
    d_list[i++] = new yz_rect(0, 555, 0, 555, 0, new lambertian(vec3(.65, .05, .05)));
    d_list[i++] = new xz_rect(113, 443, 127, 432, 554, new diffuse_light(vec3(4, 4, 4)));
    d_list[i++] = new xz_rect(0, 555, 0, 555, 555, new lambertian(vec3(.73, .73, .73)));
    d_list[i++] = new xz_rect(0, 555, 0, 555, 0, new lambertian(vec3(.73, .73, .73)));
    d_list[i++] = new xy_rect(0, 555, 0, 555, 555, new lambertian(vec3(.73, .73, .73)));
    d_list[i++] = new sphere(vec3(271.4, 271.4, 0), 50, new lambertian(vec3(0.8, 0.5, 0.1)));
    //d_list[i++] = new sphere(vec3(90, 66, 21), 25, new lambertian(vec3(0.1, 0.5, 0.8)));

    *rand_state = local_rand_state;
    *d_world = new hittable_list(d_list, i);

    float dist_to_focus = (lookfrom - lookat).length();
    *d_camera = new camera(
      lookfrom,
      lookat,
      vec3(0, 1, 0),
      40,
      float(nx) / float(ny),
      0.1f,
      dist_to_focus);
  }
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera) {
  int i = 0;
  delete((yz_rect*)d_list[i])->mat_ptr;
  delete d_list[i++];
  delete((yz_rect*)d_list[i])->mat_ptr;
  delete d_list[i++];
  
  delete((xz_rect*)d_list[i])->mat_ptr;
  delete d_list[i++];
  delete((xz_rect*)d_list[i])->mat_ptr;
  delete d_list[i++];
  delete((xz_rect*)d_list[i])->mat_ptr;
  delete d_list[i++];

  delete((xy_rect*)d_list[i])->mat_ptr;
  delete d_list[i++];
  
  delete((sphere*)d_list[i])->mat_ptr;
  delete d_list[i++];
  //delete((sphere*)d_list[i])->mat_ptr;
  //delete d_list[i++];

  delete* d_world;
  delete* d_camera;
}

int main()
{
  int nx = 800,
    ny = 800,
    ns = 100,
    tx = 8,
    ty = 8;

  int pixels = nx * ny;
  size_t fb_size = pixels * sizeof(vec3);

  vec3* fb;
  checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

  curandState* d_rand_state;
  checkCudaErrors(cudaMalloc((void**)&d_rand_state, pixels * sizeof(curandState)));
  curandState* d_rand_state2;
  checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));
  // we need that 2nd random state to be initialized for the world creation
  rand_init<<<1, 1>>>(d_rand_state2);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // make our world of hittables & the camera
  hittable** d_list;
  int num_hittables = 7;
  checkCudaErrors(cudaMalloc((void**)&d_list, num_hittables * sizeof(hittable*)));
  hittable** d_world;
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
  camera** d_camera;
  checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
  create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny,
    vec3(278, 278, -800), vec3(278, 278, 0), d_rand_state2);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  clock_t start, stop;
  start = clock();

  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);
  render_init <<<blocks, threads>>> (nx, ny, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  render <<<blocks, threads>>> (fb, nx, ny, ns,
    d_camera,
    d_world,
    d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  FILE* fp = fopen("output.ppm", "wb");
  fprintf(fp, "P6 %d %d 255 ", nx, ny);
  for (int j = ny - 1; j >= 0; j--)
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j * nx + i;
      unsigned char r = int(256 * fb[pixel_index].r());
      unsigned char g = int(256 * fb[pixel_index].g());
      unsigned char b = int(256 * fb[pixel_index].b());
      fprintf(fp, "%c%c%c", r, g, b);
    }
  fclose(fp);

  checkCudaErrors(cudaDeviceSynchronize());
  free_world <<<1, 1>>> (d_list, d_world, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(d_rand_state2));
  checkCudaErrors(cudaFree(fb));
  return 0;
}
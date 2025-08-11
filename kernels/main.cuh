#pragma once
#include <cuda_runtime.h>
#include <cuda.h>


#include <vector>
#define MI(r, c, width) ((r) * (width) + (c))
#define P_SIZE (9+9+3)*4
#define CUDA_ALIGNMENT 16
#define CHK(code) \
do { \
    if ((code) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s %s %i\n", \
                        cudaGetErrorString((code)), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)
// This is the public interface of our cuda function, called directly in main.cpp
void wrap_test_vectorAdd();
float* wrap_sweeping_plane_device(double* h_ref_cam, double* h_cam, uint8_t * luma,
                                const int width, const int height, const int n_planes, const int n_cam, int window);
// parameters array 
//extern __constant__ __device__ double cam_param_array[P_SIZE];
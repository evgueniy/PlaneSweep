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
template<class T>
float* wrap_sweeping_plane_device(T * h_ref_cam, T * h_cam, uint8_t * luma,
                                const int width, const int height, const int n_planes, const int n_cam, int window, float& cuda_ms_time);
// The tile size needs to accommodate the block size plus the window halo
// TILE_SIZE = BLOCK_SIZE + WINDOW_SIZE - 1. For a 5x5 window, this is 16 + 5 - 1 = 20.
#define TILE_SIZE 20
// parameters array 
//extern __constant__ __device__ double cam_param_array[P_SIZE];
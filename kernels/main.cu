#include "main.cuh"

#include <cstdio>
//__constant__ int    offset;
__constant__ __device__ double cam_param_array[P_SIZE];
__constant__ __device__ float cam_param_array_f[P_SIZE];
// Those functions are an example on how to call cuda functions from the main.cpp
__global__ void dev_test_vecAdd(int* A, int* B, int* C, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	C[i] = A[i] + B[i];
}


__global__ void sweeping_plane_device_opti(const uint8_t* __restrict__ luma, float* __restrict__ cost_cube, int height, int width, int n_cam,
    int zplanes, int window) {

    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x >= width || idx_y >= height) return;
    const int idx = idx_y * width + idx_x;
    const float ZNear = 0.3f;
    const float ZFar = 1.1f;

    for (int cam = 1; cam < n_cam; cam++) {
        for (int zi = 0; zi < zplanes; zi++) {
            // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
            double z = ZNear * ZFar / (ZNear + (((double)zi / (double)zplanes) * (ZFar - ZNear)));
            // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
            double* ref_K_inv = cam_param_array;
            double X_ref = fma((fma(ref_K_inv[0], (double)idx_x, fma(ref_K_inv[1], (double)idx_y, ref_K_inv[2]))) , z, 0.0);
            double Y_ref = fma((fma(ref_K_inv[3], (double)idx_x, fma(ref_K_inv[4], (double)idx_y, ref_K_inv[5]))) , z, 0.0);
            double Z_ref = fma((fma(ref_K_inv[6], (double)idx_x, fma(ref_K_inv[7], (double)idx_y, ref_K_inv[8]))) , z, 0.0);
            // 3D in ref camera coordinates to 3D world
            double* ref_R_inv = ref_K_inv + 9;
            double* ref_t_inv = ref_R_inv + 9;
            double X = fma(ref_R_inv[0], X_ref, fma(ref_R_inv[1], Y_ref, fma(ref_R_inv[2], Z_ref, -ref_t_inv[0])));
            double Y = fma(ref_R_inv[3], X_ref, fma(ref_R_inv[4], Y_ref, fma(ref_R_inv[5], Z_ref, -ref_t_inv[1])));
            double Z = fma(ref_R_inv[6], X_ref, fma(ref_R_inv[7], Y_ref, fma(ref_R_inv[8], Z_ref, -ref_t_inv[2])));
            // cam data pointer
            double* cam_K = (ref_t_inv + 3) + 21 * (cam - 1);
            double* cam_R = cam_K + 9;
            double* cam_t = cam_R + 9;
            // 3D world to projected camera 3D coordinates
            double X_proj = fma(cam_R[0], X, fma(cam_R[1], Y, fma(cam_R[2], Z, -cam_t[0])));
            double Y_proj = fma(cam_R[3], X, fma(cam_R[4], Y, fma(cam_R[5], Z, -cam_t[1])));
            double Z_proj = fma(cam_R[6], X, fma(cam_R[7], Y, fma(cam_R[8], Z, -cam_t[2])));
            // Projected camera 3D coordinates to projected camera 2D coordinates
            // inverse of Z_proj to use multiplications of fma
            const double inv_Z_proj = 1.0 / Z_proj;
            double x_proj = fma(cam_K[0], X_proj * inv_Z_proj, fma(cam_K[1], Y_proj * inv_Z_proj, cam_K[2]));
            double y_proj = fma(cam_K[3], X_proj * inv_Z_proj, fma(cam_K[4], Y_proj * inv_Z_proj, cam_K[5]));
            //double z_proj = Z_proj;

            x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
            y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);
            // (ii) calculate cost against reference
                    // Calculating cost in a window
            float cost = 0.0f;
            float cc = 0.0f;
            for (int k = -window / 2; k <= window / 2; k++)
            {
                for (int l = -window / 2; l <= window / 2; l++)
                {
                    if (idx_x + l < 0 || idx_x + l >= width || idx_y + k < 0 || idx_y + k >= height)  continue;

                    int proj_x_win = (int)x_proj + l;
                    int proj_y_win = (int)y_proj + k;

                    if (proj_x_win < 0 || proj_x_win >= width || proj_y_win < 0 || proj_y_win >= height) continue;
                    // Y
                    int offset = cam * width * height;
                    cost += fabsf(luma[(idx_y + k) * width + (idx_x + l)] - luma[(proj_y_win)*width + (proj_x_win)+offset]);
                    cc += 1.0f;
                }
            }
            cost /= cc;
            //  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
            // only the minimum cost for all the cameras is stored
            cost_cube[idx + (width * height * zi)] = fminf(cost_cube[idx + (width * height * zi)], cost);
        }
    }
}


__global__ void sweeping_plane_device_opti_float(const uint8_t* __restrict__ luma, float* __restrict__ cost_cube, int height, int width, int n_cam,
    int zplanes, int window) {

    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x >= width || idx_y >= height) return;
    const int idx = idx_y * width + idx_x;
    const float ZNear = 0.3f;
    const float ZFar = 1.1f;

    for (int cam = 1; cam < n_cam; cam++) {
        for (int zi = 0; zi < zplanes; zi++) {
            // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
            float z = ZNear * ZFar / (ZNear + (((float)zi / (float)zplanes) * (ZFar - ZNear)));

            // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
            float* ref_K_inv = cam_param_array_f;
            float X_ref = fmaf((fmaf(ref_K_inv[0], (float)idx_x, fmaf(ref_K_inv[1], (float)idx_y, ref_K_inv[2]))), z, 0.0f);
            float Y_ref = fmaf((fmaf(ref_K_inv[3], (float)idx_x, fmaf(ref_K_inv[4], (float)idx_y, ref_K_inv[5]))), z, 0.0f);
            float Z_ref = fmaf((fmaf(ref_K_inv[6], (float)idx_x, fmaf(ref_K_inv[7], (float)idx_y, ref_K_inv[8]))), z, 0.0f);
            // 3D in ref camera coordinates to 3D world
            float* ref_R_inv = ref_K_inv + 9;
            float* ref_t_inv = ref_R_inv + 9;
            float X = fmaf(ref_R_inv[0], X_ref, fmaf(ref_R_inv[1], Y_ref, fmaf(ref_R_inv[2], Z_ref, -ref_t_inv[0])));
            float Y = fmaf(ref_R_inv[3], X_ref, fmaf(ref_R_inv[4], Y_ref, fmaf(ref_R_inv[5], Z_ref, -ref_t_inv[1])));
            float Z = fmaf(ref_R_inv[6], X_ref, fmaf(ref_R_inv[7], Y_ref, fmaf(ref_R_inv[8], Z_ref, -ref_t_inv[2])));
            // cam data pointer
            float* cam_K = (ref_t_inv + 3) + 21 * (cam - 1);
            float* cam_R = cam_K + 9;
            float* cam_t = cam_R + 9;
            // 3D world to projected camera 3D coordinates
            float X_proj = fmaf(cam_R[0], X, fmaf(cam_R[1], Y, fmaf(cam_R[2], Z, -cam_t[0])));
            float Y_proj = fmaf(cam_R[3], X, fmaf(cam_R[4], Y, fmaf(cam_R[5], Z, -cam_t[1])));
            float Z_proj = fmaf(cam_R[6], X, fmaf(cam_R[7], Y, fmaf(cam_R[8], Z, -cam_t[2])));
            // Projected camera 3D coordinates to projected camera 2D coordinates
            // inverse of Z_proj to use multiplications of fma
            const float inv_Z_proj = 1.0 / Z_proj;
            float x_proj = fmaf(cam_K[0], X_proj * inv_Z_proj, fmaf(cam_K[1], Y_proj * inv_Z_proj, cam_K[2]));
            float y_proj = fmaf(cam_K[3], X_proj * inv_Z_proj, fmaf(cam_K[4], Y_proj * inv_Z_proj, cam_K[5]));
            //double z_proj = Z_proj;

            x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
            y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);
            // (ii) calculate cost against reference
                    // Calculating cost in a window
            float cost = 0.0f;
            float cc = 0.0f;
            for (int k = -window / 2; k <= window / 2; k++)
            {
                for (int l = -window / 2; l <= window / 2; l++)
                {
                    if (idx_x + l < 0 || idx_x + l >= width || idx_y + k < 0 || idx_y + k >= height)  continue;

                    int proj_x_win = (int)x_proj + l;
                    int proj_y_win = (int)y_proj + k;

                    if (proj_x_win < 0 || proj_x_win >= width || proj_y_win < 0 || proj_y_win >= height) continue;
                    // Y
                    int offset = cam * width * height;
                    cost += fabsf(luma[(idx_y + k) * width + (idx_x + l)] - luma[(proj_y_win)*width + (proj_x_win)+offset]);
                    cc += 1.0f;
                }
            }
            cost /= cc;
            //  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
            // only the minimum cost for all the cameras is stored
            cost_cube[idx + (width * height * zi)] = fminf(cost_cube[idx + (width * height * zi)], cost);
        }
    }
}
__global__ void sweeping_plane_device_old(const uint8_t* __restrict__ luma, float* __restrict__ cost_cube, int height, int width, int n_cam,
    int zplanes, int window) {
    
    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x >= width || idx_y >= height) return;
    const int idx = idx_y * width + idx_x;
    //if (idx == 0) printf("Running standard version\n");
    const float ZNear = 0.3f;
    const float ZFar = 1.1f;

    for (int cam = 1; cam < n_cam; cam++) {
        for (int zi = 0; zi < zplanes; zi++) {
            // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
            double z = ZNear * ZFar / (ZNear + (((double)zi / (double)zplanes) * (ZFar - ZNear)));
            // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
            double* ref_K_inv = cam_param_array;
            double X_ref = (ref_K_inv[0] * idx_x + ref_K_inv[1] * idx_y + ref_K_inv[2]) * z;
            double Y_ref = (ref_K_inv[3] * idx_x + ref_K_inv[4] * idx_y + ref_K_inv[5]) * z;
            double Z_ref = (ref_K_inv[6] * idx_x + ref_K_inv[7] * idx_y + ref_K_inv[8]) * z;
            // 3D in ref camera coordinates to 3D world
            double* ref_R_inv = ref_K_inv + 9;
            double* ref_t_inv = ref_R_inv + 9;
            double X =  ref_R_inv[0] * X_ref +  ref_R_inv[1] * Y_ref +  ref_R_inv[2] * Z_ref - ref_t_inv[0];
            double Y =  ref_R_inv[3] * X_ref +  ref_R_inv[4] * Y_ref +  ref_R_inv[5] * Z_ref - ref_t_inv[1];
            double Z =  ref_R_inv[6] * X_ref +  ref_R_inv[7] * Y_ref +  ref_R_inv[8] * Z_ref - ref_t_inv[2];
            // cam data pointer
            double* cam_K = (ref_t_inv + 3) + 21 * (cam - 1);
            double* cam_R = cam_K + 9;
            double* cam_t = cam_R + 9;
            // 3D world to projected camera 3D coordinates
            double X_proj = cam_R[0] * X + cam_R[1] * Y + cam_R[2] * Z - cam_t[0];
            double Y_proj = cam_R[3] * X + cam_R[4] * Y + cam_R[5] * Z - cam_t[1];
            double Z_proj = cam_R[6] * X + cam_R[7] * Y + cam_R[8] * Z - cam_t[2];
            // Projected camera 3D coordinates to projected camera 2D coordinates
            double x_proj = (cam_K[0] * X_proj / Z_proj + cam_K[1] * Y_proj / Z_proj + cam_K[2]);
            double y_proj = (cam_K[3] * X_proj / Z_proj + cam_K[4] * Y_proj / Z_proj + cam_K[5]);
            //double z_proj = Z_proj;

            x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
            y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);
            // (ii) calculate cost against reference
                    // Calculating cost in a window
            float cost = 0.0f;
            float cc = 0.0f;
            for (int k = -window / 2; k <= window / 2; k++)
            {
                for (int l = -window / 2; l <= window / 2; l++)
                {
                    if (idx_x + l < 0 || idx_x + l >= width || idx_y + k < 0 || idx_y + k >= height)  continue;

                    if (idx_x + l < 0 || idx_x + l >= width)
                        continue;
                    if (idx_y + k < 0 || idx_y + k >= height)
                        continue;
                    if (x_proj + l < 0 || x_proj + l >= width)
                        continue;
                    if (y_proj + k < 0 || y_proj + k >= height)
                        continue;
                    // Y
                    int offset = cam * width * height;
                    cost += fabsf(luma[(idx_y + k) * width + (idx_x + l)] - luma[((int)y_proj + k) * width + ((int)x_proj + l) + offset]);
                    cc += 1.0f;
                }
            }
            cost /= cc;
            //  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
            // only the minimum cost for all the cameras is stored
            cost_cube[idx + (width * height * zi)] = fminf(cost_cube[idx + (width * height * zi)], cost);
        }
    }
}

__global__ void sweeping_plane_device_standard(const uint8_t* __restrict__ luma, float* __restrict__ cost_cube, int height, int width, int n_cam,
    int zplanes, int window) {

    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x >= width || idx_y >= height) return;
    const int idx = idx_y * width + idx_x;
    //if (idx == 0) printf("Running standard version\n");
    const float ZNear = 0.3f;
    const float ZFar = 1.1f;

    for (int cam = 1; cam < n_cam; cam++) {
        for (int zi = 0; zi < zplanes; zi++) {
            // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
            double z = ZNear * ZFar / (ZNear + (((double)zi / (double)zplanes) * (ZFar - ZNear)));
            // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
            double* ref_K_inv = cam_param_array;
            double X_ref = (ref_K_inv[0] * idx_x + ref_K_inv[1] * idx_y + ref_K_inv[2]) * z;
            double Y_ref = (ref_K_inv[3] * idx_x + ref_K_inv[4] * idx_y + ref_K_inv[5]) * z;
            double Z_ref = (ref_K_inv[6] * idx_x + ref_K_inv[7] * idx_y + ref_K_inv[8]) * z;
            // 3D in ref camera coordinates to 3D world
            double* ref_R_inv = ref_K_inv + 9;
            double* ref_t_inv = ref_R_inv + 9;
            double X = ref_R_inv[0] * X_ref + ref_R_inv[1] * Y_ref + ref_R_inv[2] * Z_ref - ref_t_inv[0];
            double Y = ref_R_inv[3] * X_ref + ref_R_inv[4] * Y_ref + ref_R_inv[5] * Z_ref - ref_t_inv[1];
            double Z = ref_R_inv[6] * X_ref + ref_R_inv[7] * Y_ref + ref_R_inv[8] * Z_ref - ref_t_inv[2];
            // cam data pointer
            double* cam_K = (ref_t_inv + 3) + 21 * (cam - 1);
            double* cam_R = cam_K + 9;
            double* cam_t = cam_R + 9;
            // 3D world to projected camera 3D coordinates
            double X_proj = cam_R[0] * X + cam_R[1] * Y + cam_R[2] * Z - cam_t[0];
            double Y_proj = cam_R[3] * X + cam_R[4] * Y + cam_R[5] * Z - cam_t[1];
            double Z_proj = cam_R[6] * X + cam_R[7] * Y + cam_R[8] * Z - cam_t[2];
            // Projected camera 3D coordinates to projected camera 2D coordinates
            double x_proj = (cam_K[0] * X_proj / Z_proj + cam_K[1] * Y_proj / Z_proj + cam_K[2]);
            double y_proj = (cam_K[3] * X_proj / Z_proj + cam_K[4] * Y_proj / Z_proj + cam_K[5]);
            //double z_proj = Z_proj;

            x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
            y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);
            // (ii) calculate cost against reference
                    // Calculating cost in a window
            float cost = 0.0f;
            float cc = 0.0f;
            for (int k = -window / 2; k <= window / 2; k++)
            {
                for (int l = -window / 2; l <= window / 2; l++)
                {
                    if (idx_x + l < 0 || idx_x + l >= width || idx_y + k < 0 || idx_y + k >= height)  continue;

                    int proj_x_win = (int)x_proj + l;
                    int proj_y_win = (int)y_proj + k;

                    if (proj_x_win < 0 || proj_x_win >= width || proj_y_win < 0 || proj_y_win >= height) continue;
                    // Y
                    int offset = cam * width * height;
                    cost += fabsf(luma[(idx_y + k) * width + (idx_x + l)] - luma[(proj_y_win)*width + (proj_x_win)+offset]);
                    cc += 1.0f;
                }
            }
            cost /= cc;
            //  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
            // only the minimum cost for all the cameras is stored
            cost_cube[idx + (width * height * zi)] = fminf(cost_cube[idx + (width * height * zi)], cost);
        }
    }
}

__global__ void sweeping_plane_device_old_float(const uint8_t* luma, float* __restrict__ cost_cube, int height, int width, int n_cam,
    int zplanes, int window) {
    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x >= width || idx_y >= height) return;
    const int idx = idx_y * width + idx_x;
    const float ZNear = 0.3f;
    const float ZFar = 1.1f;
    for (int cam = 1; cam < n_cam; cam++) {
        for (int zi = 0; zi < zplanes; zi++) {
            // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
            float z = ZNear * ZFar / (ZNear + (((float)zi / (float)zplanes) * (ZFar - ZNear)));
            // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
            float* ref_K_inv = cam_param_array_f;
            float X_ref = (ref_K_inv[0] * idx_x + ref_K_inv[1] * idx_y + ref_K_inv[2]) * z;
            float Y_ref = (ref_K_inv[3] * idx_x + ref_K_inv[4] * idx_y + ref_K_inv[5]) * z;
            float Z_ref = (ref_K_inv[6] * idx_x + ref_K_inv[7] * idx_y + ref_K_inv[8]) * z;
            // 3D in ref camera coordinates to 3D world
            float* ref_R_inv = ref_K_inv + 9;
            float* ref_t_inv = ref_R_inv + 9;
            float X = ref_R_inv[0] * X_ref + ref_R_inv[1] * Y_ref + ref_R_inv[2] * Z_ref - ref_t_inv[0];
            float Y = ref_R_inv[3] * X_ref + ref_R_inv[4] * Y_ref + ref_R_inv[5] * Z_ref - ref_t_inv[1];
            float Z = ref_R_inv[6] * X_ref + ref_R_inv[7] * Y_ref + ref_R_inv[8] * Z_ref - ref_t_inv[2];
            // cam data pointer
            float* cam_K = (ref_t_inv + 3) + 21 * (cam - 1);
            float* cam_R = cam_K + 9;
            float* cam_t = cam_R + 9;
            // 3D world to projected camera 3D coordinates
            float X_proj = cam_R[0] * X + cam_R[1] * Y + cam_R[2] * Z - cam_t[0];
            float Y_proj = cam_R[3] * X + cam_R[4] * Y + cam_R[5] * Z - cam_t[1];
            float Z_proj = cam_R[6] * X + cam_R[7] * Y + cam_R[8] * Z - cam_t[2];
            // Projected camera 3D coordinates to projected camera 2D coordinates
            float x_proj = (cam_K[0] * X_proj / Z_proj + cam_K[1] * Y_proj / Z_proj + cam_K[2]);
            float y_proj = (cam_K[3] * X_proj / Z_proj + cam_K[4] * Y_proj / Z_proj + cam_K[5]);
            float z_proj = Z_proj;

            x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
            y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);
            // (ii) calculate cost against reference
                    // Calculating cost in a window
            float cost = 0.0f;
            float cc = 0.0f;
            for (int k = -window / 2; k <= window / 2; k++)
            {
                for (int l = -window / 2; l <= window / 2; l++)
                {
                    if (idx_x + l < 0 || idx_x + l >= width || idx_y + k < 0 || idx_y + k >= height)  continue;

                    int proj_x_win = (int)x_proj + l;
                    int proj_y_win = (int)y_proj + k;

                    if (proj_x_win < 0 || proj_x_win >= width || proj_y_win < 0 || proj_y_win >= height) continue;
                    // Y
                    int offset = cam * width * height;
                    cost += fabsf(luma[(idx_y + k) * width + (idx_x + l)] - luma[(proj_y_win)*width + (proj_x_win)+offset]);
                    cc += 1.0f;
                }
            }
            cost /= cc;
            //  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
            // only the minimum cost for all the cameras is stored
            cost_cube[idx + (width * height * zi)] = fminf(cost_cube[idx + (width * height * zi)], cost);
        }
    }
}

__global__ void sweeping_plane_device_float(const uint8_t* luma, float* __restrict__ cost_cube, int height, int width, int n_cam,
    int zplanes, int window) {
    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x >= width || idx_y >= height) return;
    const int idx = idx_y * width + idx_x;
    const float ZNear = 0.3f;
    const float ZFar = 1.1f;
    for (int cam = 1; cam < n_cam; cam++) {
        for (int zi = 0; zi < zplanes; zi++) {
            // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
            float z = ZNear * ZFar / (ZNear + (((float)zi / (float)zplanes) * (ZFar - ZNear)));
            // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
            float* ref_K_inv = cam_param_array_f;
            float X_ref = (ref_K_inv[0] * idx_x + ref_K_inv[1] * idx_y + ref_K_inv[2]) * z;
            float Y_ref = (ref_K_inv[3] * idx_x + ref_K_inv[4] * idx_y + ref_K_inv[5]) * z;
            float Z_ref = (ref_K_inv[6] * idx_x + ref_K_inv[7] * idx_y + ref_K_inv[8]) * z;
            // 3D in ref camera coordinates to 3D world
            float* ref_R_inv = ref_K_inv + 9;
            float* ref_t_inv = ref_R_inv + 9;
            float X = ref_R_inv[0] * X_ref + ref_R_inv[1] * Y_ref + ref_R_inv[2] * Z_ref - ref_t_inv[0];
            float Y = ref_R_inv[3] * X_ref + ref_R_inv[4] * Y_ref + ref_R_inv[5] * Z_ref - ref_t_inv[1];
            float Z = ref_R_inv[6] * X_ref + ref_R_inv[7] * Y_ref + ref_R_inv[8] * Z_ref - ref_t_inv[2];
            // cam data pointer
            float* cam_K = (ref_t_inv + 3) + 21 * (cam - 1);
            float* cam_R = cam_K + 9;
            float* cam_t = cam_R + 9;
            // 3D world to projected camera 3D coordinates
            float X_proj = cam_R[0] * X + cam_R[1] * Y + cam_R[2] * Z - cam_t[0];
            float Y_proj = cam_R[3] * X + cam_R[4] * Y + cam_R[5] * Z - cam_t[1];
            float Z_proj = cam_R[6] * X + cam_R[7] * Y + cam_R[8] * Z - cam_t[2];
            // Projected camera 3D coordinates to projected camera 2D coordinates
            float x_proj = (cam_K[0] * X_proj / Z_proj + cam_K[1] * Y_proj / Z_proj + cam_K[2]);
            float y_proj = (cam_K[3] * X_proj / Z_proj + cam_K[4] * Y_proj / Z_proj + cam_K[5]);
            float z_proj = Z_proj;

            x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
            y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);
            // (ii) calculate cost against reference
                    // Calculating cost in a window
            float cost = 0.0f;
            float cc = 0.0f;
            for (int k = -window / 2; k <= window / 2; k++)
            {
                for (int l = -window / 2; l <= window / 2; l++)
                {
                    if (idx_x + l < 0 || idx_x + l >= width || idx_y + k < 0 || idx_y + k >= height)  continue;

                    int proj_x_win = (int)x_proj + l;
                    int proj_y_win = (int)y_proj + k;

                    if (proj_x_win < 0 || proj_x_win >= width || proj_y_win < 0 || proj_y_win >= height) continue;
                    // Y
                    int offset = cam * width * height;
                    cost += fabsf(luma[(idx_y + k) * width + (idx_x + l)] - luma[(proj_y_win)*width + (proj_x_win)+offset]);
                    cc += 1.0f;
                }
            }
            cost /= cc;
            //  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
            // only the minimum cost for all the cameras is stored
            cost_cube[idx + (width * height * zi)] = fminf(cost_cube[idx + (width * height * zi)], cost);
        }
    }
}


__global__ void sweeping_plane_device_shared(const uint8_t* __restrict__ luma, float* __restrict__ cost_cube, int height, int width, int n_cam,
    int zplanes, int window) {

    // Calculate tile dimensions inside the kernel
    const int tile_width = blockDim.x + window - 1;
    const int tile_height = blockDim.y + window - 1;

    extern __shared__ uint8_t ref_cam_tile[];

    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int t_idx_x = threadIdx.x;
    const int t_idx_y = threadIdx.y;

    const int tile_start_x = blockIdx.x * blockDim.x - (window / 2);
    const int tile_start_y = blockIdx.y * blockDim.y - (window / 2);
    // loading ref cam tile into memory
    for (int y = t_idx_y; y < tile_height; y += blockDim.y) {
        for (int x = t_idx_x; x < tile_width; x += blockDim.x) {
            int load_x = tile_start_x + x;
            int load_y = tile_start_y + y;
            if (load_x >= 0 && load_x < width && load_y >= 0 && load_y < height) {
                ref_cam_tile[y * tile_width + x] = luma[load_y * width + load_x];
            }
            else {
                ref_cam_tile[y * tile_width + x] = 0;
            }
        }
    }

    __syncthreads();

    if (idx_x >= width || idx_y >= height) return;

    const int idx = idx_y * width + idx_x;
    //if (idx == 0) printf("Running shared version\n");
    const float ZNear = 0.3f;
    const float ZFar = 1.1f;

    for (int cam = 1; cam < n_cam; cam++) {
        for (int zi = 0; zi < zplanes; zi++) {
            // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
            double z = ZNear * ZFar / (ZNear + (((double)zi / (double)zplanes) * (ZFar - ZNear)));
            //2D ref camera point to 3D in ref camera coordinates (p * K_inv)
            double* ref_K_inv = cam_param_array;
            double X_ref = (fma(ref_K_inv[0], (double)idx_x, fma(ref_K_inv[1], (double)idx_y, ref_K_inv[2]))) * z;
            double Y_ref = (fma(ref_K_inv[3], (double)idx_x, fma(ref_K_inv[4], (double)idx_y, ref_K_inv[5]))) * z;
            double Z_ref = (fma(ref_K_inv[6], (double)idx_x, fma(ref_K_inv[7], (double)idx_y, ref_K_inv[8]))) * z;
            // 3D in ref camera coordinates to 3D world
            double* ref_R_inv = ref_K_inv + 9;
            double* ref_t_inv = ref_R_inv + 9;
            double X = fma(ref_R_inv[0], X_ref, fma(ref_R_inv[1], Y_ref, fma(ref_R_inv[2], Z_ref, -ref_t_inv[0])));
            double Y = fma(ref_R_inv[3], X_ref, fma(ref_R_inv[4], Y_ref, fma(ref_R_inv[5], Z_ref, -ref_t_inv[1])));
            double Z = fma(ref_R_inv[6], X_ref, fma(ref_R_inv[7], Y_ref, fma(ref_R_inv[8], Z_ref, -ref_t_inv[2])));
            // cam data pointer
            double* cam_K = (ref_t_inv + 3) + 21 * (cam - 1);
            double* cam_R = cam_K + 9;
            double* cam_t = cam_R + 9;
            // 3D world to projected camera 3D coordinates
            double X_proj = fma(cam_R[0], X, fma(cam_R[1], Y, fma(cam_R[2], Z, -cam_t[0])));
            double Y_proj = fma(cam_R[3], X, fma(cam_R[4], Y, fma(cam_R[5], Z, -cam_t[1])));
            double Z_proj = fma(cam_R[6], X, fma(cam_R[7], Y, fma(cam_R[8], Z, -cam_t[2])));
            // Projected camera 3D coordinates to projected camera 2D coordinates
            // inverse of Z_proj to use multiplications of fma
            const double inv_Z_proj = 1.0 / Z_proj;
            double x_proj = fma(cam_K[0], X_proj * inv_Z_proj, fma(cam_K[1], Y_proj * inv_Z_proj, cam_K[2]));
            double y_proj = fma(cam_K[3], X_proj * inv_Z_proj, fma(cam_K[4], Y_proj * inv_Z_proj, cam_K[5]));
            //double z_proj = Z_proj;

            x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
            y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);
            // (ii) calculate cost against reference
                    // Calculating cost in a window
            float cost = 0.0f;
            float cc = 0.0f;
            for (int k = -window / 2; k <= window / 2; k++) {
                for (int l = -window / 2; l <= window / 2; l++) {
                    
                    if (idx_x + l < 0 || idx_x + l >= width || idx_y + k < 0 || idx_y + k >= height)  continue;

                    int proj_x_win = (int)x_proj + l;
                    int proj_y_win = (int)y_proj + k;

                    if (proj_x_win < 0 || proj_x_win >= width || proj_y_win < 0 || proj_y_win >= height) continue;

                    uint8_t ref_val = ref_cam_tile[(t_idx_y + k + (window / 2)) * tile_width + (t_idx_x + l + (window / 2))];

                    int offset = cam * width * height;
                    uint8_t cam_val = luma[proj_y_win * width + proj_x_win + offset];

                    cost += fabsf(ref_val - cam_val);
                    cc += 1.0f;
                }
            }
            cost /= cc;
            //  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
            // only the minimum cost for all the cameras is stored
            cost_cube[idx + (width * height * zi)] = fminf(cost_cube[idx + (width * height * zi)], cost);
        }
    }
}

inline unsigned divUp(unsigned x, unsigned y) { return (x + y - 1) / y; }

void wrap_test_vectorAdd() {
	printf("Vector Add:\n");

	int N = 3;
	int a[] = { 1, 2, 3 };
	int b[] = { 1, 2, 3 };
	int c[] = { 0, 0, 0 };

	int* dev_a, * dev_b, * dev_c;

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	cudaMemcpy(dev_a, a, N * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int),
		cudaMemcpyHostToDevice);

	dev_test_vecAdd <<<1, N>>> (dev_a, dev_b, dev_c, N);

	cudaMemcpy(c, dev_c, N * sizeof(int),
		cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	for (int i = 0; i < N; ++i) {
		printf("%i + %i = %i\n", a[i], b[i], c[i]);
	}
}
template<class T>
float* wrap_sweeping_plane_device(T* h_ref_cam, T* h_cams, uint8_t* h_luma, const int width, const int height,
    const int n_planes, const int n_cam, int window, float& cuda_ms_time){
    printf("Sweeping_plane_device standard:\n");
    // cuda timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    size_t n_luma = width * height * n_cam;
    size_t n_cost_cube = width * height * n_planes;

    float* h_cost_cube = new float[n_cost_cube];
    std::fill_n(h_cost_cube, n_cost_cube, 255.f);

    // device buffers
    uint8_t* d_luma = nullptr;
    float* d_cost_cube = nullptr;

    CHK(cudaSetDevice(0));
    // copy camera params to constant memory

    if constexpr (std::is_same_v<T, double>) {
        printf("Double version\n");
        CHK(cudaMemcpyToSymbol(cam_param_array,
            h_ref_cam,
            (P_SIZE / 4) * sizeof(T),
            0,
            cudaMemcpyHostToDevice));
        CHK(cudaMemcpyToSymbol(cam_param_array,
            h_cams,
            (P_SIZE / 4) * 3 * sizeof(T),
            (P_SIZE / 4) * sizeof(T),
            cudaMemcpyHostToDevice));
    }
    else {
        printf("Float version\n");
        CHK(cudaMemcpyToSymbol(cam_param_array_f,
            h_ref_cam,
            (P_SIZE / 4) * sizeof(T),
            0,
            cudaMemcpyHostToDevice));
        CHK(cudaMemcpyToSymbol(cam_param_array_f,
            h_cams,
            (P_SIZE / 4) * 3 * sizeof(T),
            (P_SIZE / 4) * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    // flat allocations
    CHK(cudaMalloc(&d_luma, n_luma * sizeof(uint8_t)));
    CHK(cudaMalloc(&d_cost_cube, n_cost_cube * sizeof(float)));

    // upload luma & initial cost
    CHK(cudaMemcpy(d_luma,
        h_luma,
        n_luma * sizeof(uint8_t),
        cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_cost_cube,
        h_cost_cube,
        n_cost_cube * sizeof(float),
        cudaMemcpyHostToDevice));

    // launch configuration: one thread per (x,y)
    dim3 threads(128,1);
    dim3 blocks(divUp(width, threads.x),
        divUp(height, threads.y));
    int tile_width = blocks.x + window - 1;
    int tile_height = blocks.y + window - 1;
    size_t shared_mem_size = tile_width * tile_height * sizeof(uint8_t);
    // start timer
    cudaEventRecord(start);
    if constexpr (std::is_same_v<T, double>) {
        sweeping_plane_device_opti << <blocks, threads>> > (
            d_luma,
            d_cost_cube,
            height,
            width,
            n_cam,
            n_planes,
            window
            );
    }
    else {
        sweeping_plane_device_opti_float << <blocks, threads >> > (
            d_luma,
            d_cost_cube,
            height,
            width,
            n_cam,
            n_planes,
            window
            );
    }
    // stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);
    cuda_ms_time = milliseconds;
    // destroying the events for timers
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHK(cudaGetLastError());
    // getting the device array copied to host
    CHK(cudaMemcpy(h_cost_cube, d_cost_cube, n_cost_cube *sizeof(float),
                   cudaMemcpyDeviceToHost));

    // cleanup
    cudaFree(d_luma);
    cudaFree(d_cost_cube);
    return h_cost_cube;
}


template float* wrap_sweeping_plane_device<float>(float*, float*, unsigned char*, int, int, int, int, int, float&);
template float* wrap_sweeping_plane_device<double>(double*, double*, unsigned char*, int, int, int, int, int, float&);

__global__ void sweeping_plane_device_3d_opti_float(const uint8_t* __restrict__ luma, float* __restrict__ cost_cube, int height, int width, int n_cam,
    int zplanes, int window) {

    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zi = blockIdx.z;

    if (idx_x >= width || idx_y >= height) return;

    const int idx = idx_y * width + idx_x;
    const float ZNear = 0.3f;
    const float ZFar = 1.1f;

    for (int cam = 1; cam < n_cam; cam++) {
        // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
        float z = ZNear * ZFar / (ZNear + (((float)zi / (float)zplanes) * (ZFar - ZNear)));

        // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
        float* ref_K_inv = cam_param_array_f;
        float X_ref = fmaf((fmaf(ref_K_inv[0], (float)idx_x, fmaf(ref_K_inv[1], (float)idx_y, ref_K_inv[2]))), z, 0.0f);
        float Y_ref = fmaf((fmaf(ref_K_inv[3], (float)idx_x, fmaf(ref_K_inv[4], (float)idx_y, ref_K_inv[5]))), z, 0.0f);
        float Z_ref = fmaf((fmaf(ref_K_inv[6], (float)idx_x, fmaf(ref_K_inv[7], (float)idx_y, ref_K_inv[8]))), z, 0.0f);
        // 3D in ref camera coordinates to 3D world
        float* ref_R_inv = ref_K_inv + 9;
        float* ref_t_inv = ref_R_inv + 9;
        float X = fmaf(ref_R_inv[0], X_ref, fmaf(ref_R_inv[1], Y_ref, fmaf(ref_R_inv[2], Z_ref, -ref_t_inv[0])));
        float Y = fmaf(ref_R_inv[3], X_ref, fmaf(ref_R_inv[4], Y_ref, fmaf(ref_R_inv[5], Z_ref, -ref_t_inv[1])));
        float Z = fmaf(ref_R_inv[6], X_ref, fmaf(ref_R_inv[7], Y_ref, fmaf(ref_R_inv[8], Z_ref, -ref_t_inv[2])));
        // cam data pointer
        float* cam_K = (ref_t_inv + 3) + 21 * (cam - 1);
        float* cam_R = cam_K + 9;
        float* cam_t = cam_R + 9;
        // 3D world to projected camera 3D coordinates
        float X_proj = fmaf(cam_R[0], X, fmaf(cam_R[1], Y, fmaf(cam_R[2], Z, -cam_t[0])));
        float Y_proj = fmaf(cam_R[3], X, fmaf(cam_R[4], Y, fmaf(cam_R[5], Z, -cam_t[1])));
        float Z_proj = fmaf(cam_R[6], X, fmaf(cam_R[7], Y, fmaf(cam_R[8], Z, -cam_t[2])));
        // Projected camera 3D coordinates to projected camera 2D coordinates
        // inverse of Z_proj to use multiplications of fma
        const float inv_Z_proj = 1.0f / Z_proj;
        float x_proj = fmaf(cam_K[0], X_proj * inv_Z_proj, fmaf(cam_K[1], Y_proj * inv_Z_proj, cam_K[2]));
        float y_proj = fmaf(cam_K[3], X_proj * inv_Z_proj, fmaf(cam_K[4], Y_proj * inv_Z_proj, cam_K[5]));

        x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
        y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);

        // (ii) calculate cost against reference
        // Calculating cost in a window
        float cost = 0.0f;
        float cc = 0.0f;
        for (int k = -window / 2; k <= window / 2; k++)
        {
            for (int l = -window / 2; l <= window / 2; l++)
            {
                if (idx_x + l < 0 || idx_x + l >= width || idx_y + k < 0 || idx_y + k >= height)  continue;

                int proj_x_win = (int)x_proj + l;
                int proj_y_win = (int)y_proj + k;

                if (proj_x_win < 0 || proj_x_win >= width || proj_y_win < 0 || proj_y_win >= height) continue;

                // Y
                int offset = cam * width * height;
                cost += fabsf(luma[(idx_y + k) * width + (idx_x + l)] - luma[(proj_y_win)*width + (proj_x_win)+offset]);
                cc += 1.0f;
            }
        }
        cost /= cc;

        //  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
        // only the minimum cost for all the cameras is stored
        cost_cube[idx + (width * height * zi)] = fminf(cost_cube[idx + (width * height * zi)], cost);
    }
}


__global__ void sweeping_plane_device_3d_opti(const uint8_t* __restrict__ luma, float* __restrict__ cost_cube, int height, int width, int n_cam,
    int zplanes, int window) {

    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zi = blockIdx.z;

    if (idx_x >= width || idx_y >= height) return;

    const int idx = idx_y * width + idx_x;
    const double ZNear = 0.3;
    const double ZFar = 1.1;

    for (int cam = 1; cam < n_cam; cam++) {
        // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
        double z = ZNear * ZFar / (ZNear + (((double)zi / (double)zplanes) * (ZFar - ZNear)));
        // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
        double* ref_K_inv = cam_param_array;
        double X_ref = fma((fma(ref_K_inv[0], (double)idx_x, fma(ref_K_inv[1], (double)idx_y, ref_K_inv[2]))), z, 0.0);
        double Y_ref = fma((fma(ref_K_inv[3], (double)idx_x, fma(ref_K_inv[4], (double)idx_y, ref_K_inv[5]))), z, 0.0);
        double Z_ref = fma((fma(ref_K_inv[6], (double)idx_x, fma(ref_K_inv[7], (double)idx_y, ref_K_inv[8]))), z, 0.0);
        // 3D in ref camera coordinates to 3D world
        double* ref_R_inv = ref_K_inv + 9;
        double* ref_t_inv = ref_R_inv + 9;
        double X = fma(ref_R_inv[0], X_ref, fma(ref_R_inv[1], Y_ref, fma(ref_R_inv[2], Z_ref, -ref_t_inv[0])));
        double Y = fma(ref_R_inv[3], X_ref, fma(ref_R_inv[4], Y_ref, fma(ref_R_inv[5], Z_ref, -ref_t_inv[1])));
        double Z = fma(ref_R_inv[6], X_ref, fma(ref_R_inv[7], Y_ref, fma(ref_R_inv[8], Z_ref, -ref_t_inv[2])));
        // cam data pointer
        double* cam_K = (ref_t_inv + 3) + 21 * (cam - 1);
        double* cam_R = cam_K + 9;
        double* cam_t = cam_R + 9;
        // 3D world to projected camera 3D coordinates
        double X_proj = fma(cam_R[0], X, fma(cam_R[1], Y, fma(cam_R[2], Z, -cam_t[0])));
        double Y_proj = fma(cam_R[3], X, fma(cam_R[4], Y, fma(cam_R[5], Z, -cam_t[1])));
        double Z_proj = fma(cam_R[6], X, fma(cam_R[7], Y, fma(cam_R[8], Z, -cam_t[2])));
        // Projected camera 3D coordinates to projected camera 2D coordinates
        // inverse of Z_proj to use multiplications of fma
        const double inv_Z_proj = 1.0 / Z_proj;
        double x_proj = fma(cam_K[0], X_proj * inv_Z_proj, fma(cam_K[1], Y_proj * inv_Z_proj, cam_K[2]));
        double y_proj = fma(cam_K[3], X_proj * inv_Z_proj, fma(cam_K[4], Y_proj * inv_Z_proj, cam_K[5]));

        x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
        y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);

        // (ii) calculate cost against reference
        // Calculating cost in a window
        float cost = 0.0f;
        float cc = 0.0f;
        for (int k = -window / 2; k <= window / 2; k++)
        {
            for (int l = -window / 2; l <= window / 2; l++)
            {
                if (idx_x + l < 0 || idx_x + l >= width || idx_y + k < 0 || idx_y + k >= height)  continue;

                int proj_x_win = (int)x_proj + l;
                int proj_y_win = (int)y_proj + k;

                if (proj_x_win < 0 || proj_x_win >= width || proj_y_win < 0 || proj_y_win >= height) continue;

                // Y
                int offset = cam * width * height;
                cost += fabsf(luma[(idx_y + k) * width + (idx_x + l)] - luma[(proj_y_win)*width + (proj_x_win)+offset]);
                cc += 1.0f;
            }
        }
        cost /= cc;

        //  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
        // only the minimum cost for all the cameras is stored
        cost_cube[idx + (width * height * zi)] = fminf(cost_cube[idx + (width * height * zi)], cost);
    }
}
__global__ void sweeping_plane_device_3d_shared_float(const uint8_t* __restrict__ luma, float* __restrict__ cost_cube, int height, int width, int n_cam,
    int zplanes, int window) {

    // Calculate tile dimensions inside the kernel
    const int tile_width = blockDim.x + window - 1;
    const int tile_height = blockDim.y + window - 1;

    extern __shared__ uint8_t ref_cam_tile[];

    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zi = blockIdx.z;

    const int t_idx_x = threadIdx.x;
    const int t_idx_y = threadIdx.y;

    const int tile_start_x = blockIdx.x * blockDim.x - (window / 2);
    const int tile_start_y = blockIdx.y * blockDim.y - (window / 2);
    // loading ref cam tile into memory
    for (int y = t_idx_y; y < tile_height; y += blockDim.y) {
        for (int x = t_idx_x; x < tile_width; x += blockDim.x) {
            int load_x = tile_start_x + x;
            int load_y = tile_start_y + y;
            if (load_x >= 0 && load_x < width && load_y >= 0 && load_y < height) {
                ref_cam_tile[y * tile_width + x] = luma[load_y * width + load_x];
            }
            else {
                ref_cam_tile[y * tile_width + x] = 0;
            }
        }
    }

    __syncthreads();

    if (idx_x >= width || idx_y >= height) return;

    const int idx = idx_y * width + idx_x;
    const float ZNear = 0.3f;
    const float ZFar = 1.1f;

    for (int cam = 1; cam < n_cam; cam++) {
        // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
        float z = ZNear * ZFar / (ZNear + (((float)zi / (float)zplanes) * (ZFar - ZNear)));

        // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
        float* ref_K_inv = cam_param_array_f;
        float X_ref = fmaf((fmaf(ref_K_inv[0], (float)idx_x, fmaf(ref_K_inv[1], (float)idx_y, ref_K_inv[2]))), z, 0.0f);
        float Y_ref = fmaf((fmaf(ref_K_inv[3], (float)idx_x, fmaf(ref_K_inv[4], (float)idx_y, ref_K_inv[5]))), z, 0.0f);
        float Z_ref = fmaf((fmaf(ref_K_inv[6], (float)idx_x, fmaf(ref_K_inv[7], (float)idx_y, ref_K_inv[8]))), z, 0.0f);
        // 3D in ref camera coordinates to 3D world
        float* ref_R_inv = ref_K_inv + 9;
        float* ref_t_inv = ref_R_inv + 9;
        float X = fmaf(ref_R_inv[0], X_ref, fmaf(ref_R_inv[1], Y_ref, fmaf(ref_R_inv[2], Z_ref, -ref_t_inv[0])));
        float Y = fmaf(ref_R_inv[3], X_ref, fmaf(ref_R_inv[4], Y_ref, fmaf(ref_R_inv[5], Z_ref, -ref_t_inv[1])));
        float Z = fmaf(ref_R_inv[6], X_ref, fmaf(ref_R_inv[7], Y_ref, fmaf(ref_R_inv[8], Z_ref, -ref_t_inv[2])));
        // cam data pointer
        float* cam_K = (ref_t_inv + 3) + 21 * (cam - 1);
        float* cam_R = cam_K + 9;
        float* cam_t = cam_R + 9;
        // 3D world to projected camera 3D coordinates
        float X_proj = fmaf(cam_R[0], X, fmaf(cam_R[1], Y, fmaf(cam_R[2], Z, -cam_t[0])));
        float Y_proj = fmaf(cam_R[3], X, fmaf(cam_R[4], Y, fmaf(cam_R[5], Z, -cam_t[1])));
        float Z_proj = fmaf(cam_R[6], X, fmaf(cam_R[7], Y, fmaf(cam_R[8], Z, -cam_t[2])));
        // Projected camera 3D coordinates to projected camera 2D coordinates
        // inverse of Z_proj to use multiplications of fma
        const float inv_Z_proj = 1.0f / Z_proj;
        float x_proj = fmaf(cam_K[0], X_proj * inv_Z_proj, fmaf(cam_K[1], Y_proj * inv_Z_proj, cam_K[2]));
        float y_proj = fmaf(cam_K[3], X_proj * inv_Z_proj, fmaf(cam_K[4], Y_proj * inv_Z_proj, cam_K[5]));

        x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
        y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);

        // (ii) calculate cost against reference
        // Calculating cost in a window
        float cost = 0.0f;
        float cc = 0.0f;
        for (int k = -window / 2; k <= window / 2; k++)
        {
            for (int l = -window / 2; l <= window / 2; l++)
            {
                if (idx_x + l < 0 || idx_x + l >= width || idx_y + k < 0 || idx_y + k >= height)  continue;

                int proj_x_win = (int)x_proj + l;
                int proj_y_win = (int)y_proj + k;

                if (proj_x_win < 0 || proj_x_win >= width || proj_y_win < 0 || proj_y_win >= height) continue;

                uint8_t ref_val = ref_cam_tile[(t_idx_y + k + (window / 2)) * tile_width + (t_idx_x + l + (window / 2))];

                // Y
                int offset = cam * width * height;
                uint8_t cam_val = luma[proj_y_win * width + proj_x_win + offset];

                cost += fabsf(ref_val - cam_val);
                cc += 1.0f;
            }
        }
        cost /= cc;

        //  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
        // only the minimum cost for all the cameras is stored
        cost_cube[idx + (width * height * zi)] = fminf(cost_cube[idx + (width * height * zi)], cost);
    }
}


__global__ void sweeping_plane_device_3d_shared(const uint8_t* __restrict__ luma, float* __restrict__ cost_cube, int height, int width, int n_cam,
    int zplanes, int window) {

    // Calculate tile dimensions inside the kernel
    const int tile_width = blockDim.x + window - 1;
    const int tile_height = blockDim.y + window - 1;

    extern __shared__ uint8_t ref_cam_tile[];

    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zi = blockIdx.z;

    const int t_idx_x = threadIdx.x;
    const int t_idx_y = threadIdx.y;

    const int tile_start_x = blockIdx.x * blockDim.x - (window / 2);
    const int tile_start_y = blockIdx.y * blockDim.y - (window / 2);
    // loading ref cam tile into memory
    for (int y = t_idx_y; y < tile_height; y += blockDim.y) {
        for (int x = t_idx_x; x < tile_width; x += blockDim.x) {
            int load_x = tile_start_x + x;
            int load_y = tile_start_y + y;
            if (load_x >= 0 && load_x < width && load_y >= 0 && load_y < height) {
                ref_cam_tile[y * tile_width + x] = luma[load_y * width + load_x];
            }
            else {
                ref_cam_tile[y * tile_width + x] = 0;
            }
        }
    }

    __syncthreads();

    if (idx_x >= width || idx_y >= height) return;

    const int idx = idx_y * width + idx_x;
    const double ZNear = 0.3;
    const double ZFar = 1.1;

    for (int cam = 1; cam < n_cam; cam++) {
        // Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
        double z = ZNear * ZFar / (ZNear + (((double)zi / (double)zplanes) * (ZFar - ZNear)));
        // 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
        double* ref_K_inv = cam_param_array;
        double X_ref = fma((fma(ref_K_inv[0], (double)idx_x, fma(ref_K_inv[1], (double)idx_y, ref_K_inv[2]))), z, 0.0);
        double Y_ref = fma((fma(ref_K_inv[3], (double)idx_x, fma(ref_K_inv[4], (double)idx_y, ref_K_inv[5]))), z, 0.0);
        double Z_ref = fma((fma(ref_K_inv[6], (double)idx_x, fma(ref_K_inv[7], (double)idx_y, ref_K_inv[8]))), z, 0.0);
        // 3D in ref camera coordinates to 3D world
        double* ref_R_inv = ref_K_inv + 9;
        double* ref_t_inv = ref_R_inv + 9;
        double X = fma(ref_R_inv[0], X_ref, fma(ref_R_inv[1], Y_ref, fma(ref_R_inv[2], Z_ref, -ref_t_inv[0])));
        double Y = fma(ref_R_inv[3], X_ref, fma(ref_R_inv[4], Y_ref, fma(ref_R_inv[5], Z_ref, -ref_t_inv[1])));
        double Z = fma(ref_R_inv[6], X_ref, fma(ref_R_inv[7], Y_ref, fma(ref_R_inv[8], Z_ref, -ref_t_inv[2])));
        // cam data pointer
        double* cam_K = (ref_t_inv + 3) + 21 * (cam - 1);
        double* cam_R = cam_K + 9;
        double* cam_t = cam_R + 9;
        // 3D world to projected camera 3D coordinates
        double X_proj = fma(cam_R[0], X, fma(cam_R[1], Y, fma(cam_R[2], Z, -cam_t[0])));
        double Y_proj = fma(cam_R[3], X, fma(cam_R[4], Y, fma(cam_R[5], Z, -cam_t[1])));
        double Z_proj = fma(cam_R[6], X, fma(cam_R[7], Y, fma(cam_R[8], Z, -cam_t[2])));
        // Projected camera 3D coordinates to projected camera 2D coordinates
        // inverse of Z_proj to use multiplications of fma
        const double inv_Z_proj = 1.0 / Z_proj;
        double x_proj = fma(cam_K[0], X_proj * inv_Z_proj, fma(cam_K[1], Y_proj * inv_Z_proj, cam_K[2]));
        double y_proj = fma(cam_K[3], X_proj * inv_Z_proj, fma(cam_K[4], Y_proj * inv_Z_proj, cam_K[5]));

        x_proj = x_proj < 0 || x_proj >= width ? 0 : roundf(x_proj);
        y_proj = y_proj < 0 || y_proj >= height ? 0 : roundf(y_proj);

        // (ii) calculate cost against reference
        // Calculating cost in a window
        float cost = 0.0f;
        float cc = 0.0f;
        for (int k = -window / 2; k <= window / 2; k++)
        {
            for (int l = -window / 2; l <= window / 2; l++)
            {
                if (idx_x + l < 0 || idx_x + l >= width || idx_y + k < 0 || idx_y + k >= height)  continue;

                int proj_x_win = (int)x_proj + l;
                int proj_y_win = (int)y_proj + k;

                if (proj_x_win < 0 || proj_x_win >= width || proj_y_win < 0 || proj_y_win >= height) continue;

                uint8_t ref_val = ref_cam_tile[(t_idx_y + k + (window / 2)) * tile_width + (t_idx_x + l + (window / 2))];

                // Y
                int offset = cam * width * height;
                uint8_t cam_val = luma[proj_y_win * width + proj_x_win + offset];

                cost += fabsf(ref_val - cam_val);
                cc += 1.0f;
            }
        }
        cost /= cc;

        //  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
        // only the minimum cost for all the cameras is stored
        cost_cube[idx + (width * height * zi)] = fminf(cost_cube[idx + (width * height * zi)], cost);
    }
}


template<class T>
float* wrap_sweeping_plane_device_3d(T* h_ref_cam, T* h_cams, uint8_t* h_luma, const int width, const int height,
    const int n_planes, const int n_cam, int window, float& cuda_ms_time) {
    printf("Sweeping_plane_device 3D:\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t n_luma = width * height * n_cam;
    size_t n_cost_cube = width * height * n_planes;

    float* h_cost_cube = new float[n_cost_cube];
    std::fill_n(h_cost_cube, n_cost_cube, 255.f);

    uint8_t* d_luma = nullptr;
    float* d_cost_cube = nullptr;

    CHK(cudaSetDevice(0));

    if constexpr (std::is_same_v<T, double>) {
        printf("Double version\n");
        CHK(cudaMemcpyToSymbol(cam_param_array,
            h_ref_cam,
            (P_SIZE / 4) * sizeof(T), 
            0,
            cudaMemcpyHostToDevice));
        CHK(cudaMemcpyToSymbol(cam_param_array, 
            h_cams, (P_SIZE / 4) * 3 * sizeof(T),
            (P_SIZE / 4) * sizeof(T), 
            cudaMemcpyHostToDevice));
    }
    else {
        printf("Float version\n");
        CHK(cudaMemcpyToSymbol(cam_param_array_f, 
            h_ref_cam, (P_SIZE / 4) * sizeof(T), 
            0,
            cudaMemcpyHostToDevice));
        CHK(cudaMemcpyToSymbol(cam_param_array_f, 
            h_cams, 
            (P_SIZE / 4) * 3 * sizeof(T), (P_SIZE / 4) * sizeof(T), 
            cudaMemcpyHostToDevice));
    }

    CHK(cudaMalloc(&d_luma, n_luma * sizeof(uint8_t)));
    CHK(cudaMalloc(&d_cost_cube, n_cost_cube * sizeof(float)));

    CHK(cudaMemcpy(d_luma, h_luma, n_luma * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_cost_cube, h_cost_cube, n_cost_cube * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threads(16, 16, 1);
    dim3 blocks(divUp(width, threads.x), divUp(height, threads.y), n_planes);
    int tile_width = threads.x + window - 1;
    int tile_height = threads.y + window - 1;
    size_t shared_mem_size = tile_width * tile_height * sizeof(uint8_t);

    cudaEventRecord(start);
    if constexpr (std::is_same_v<T, double>) {
        sweeping_plane_device_3d_shared << <blocks, threads, shared_mem_size >> > (
            d_luma,
            d_cost_cube,
            height,
            width,
            n_cam,
            n_planes,
            window
            );
    }
    else {
        sweeping_plane_device_3d_shared_float << <blocks, threads, shared_mem_size >> > (
            d_luma,
            d_cost_cube,
            height,
            width,
            n_cam,
            n_planes,
            window
            );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);
    cuda_ms_time = milliseconds;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHK(cudaGetLastError());
    CHK(cudaMemcpy(h_cost_cube, d_cost_cube, n_cost_cube * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_luma);
    cudaFree(d_cost_cube);
    return h_cost_cube;
}
template float* wrap_sweeping_plane_device_3d<float>(float*, float*, unsigned char*, int, int, int, int, int, float&);
template float* wrap_sweeping_plane_device_3d<double>(double*, double*, unsigned char*, int, int, int, int, int, float&);
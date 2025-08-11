#include "main.cuh"

#include <cstdio>
//__constant__ int    offset;
__constant__ __device__ double cam_param_array[P_SIZE];

// Those functions are an example on how to call cuda functions from the main.cpp
__global__ void dev_test_vecAdd(int* A, int* B, int* C, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	C[i] = A[i] + B[i];
}

//__global__ void sweeping_plane_device(const uint8_t* __restrict__ luma, float* __restrict__ cost_cube, int height, int width,int n_cam,
//	int zplanes, int window, unsigned pitch_luma, unsigned pitch_cost) {
//	const unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
//	const unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
//	if (idx_x >= (unsigned)width || idx_y >= (unsigned)height) return;
//	unsigned idx_luma = idx_y * pitch_luma + idx_x * n_cam;
//	for (int cam = 1; cam < n_cam; cam++) {
//		for (int zi = 0; zi < zplanes; zi++) {
//						
//		}
//	}
//}
__device__ void debugTrap() { asm("brkpt;"); }
__global__ void sweeping_plane_device_standard(const uint8_t* luma, float* __restrict__ cost_cube, int height, int width, int n_cam,
    int zplanes, int window) {
    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    /*if (idx_y != 0 || idx_x != 0) return;
    else printf("Hello from kernel at block(%d,%d) thread(%d,%d)\n",
        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);*/
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
            double z_proj = Z_proj;

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
                    //if (zi == 0 && idx_x == 0 && idx_y == 0 && cam == 1) {
                    //    //printf("Index test: %d\n", (idx_y + k) * width);
                    //    //printf("Simple test: %d\n", luma[(idx_y + k) * width]);
                    //    printf("Cuda Ref value at coordinates Y: %d - X: %d = %d\n", idx_y + k, idx_x + l, luma[(idx_y + k) * width  +  (idx_x + l)]);
                    //    printf("Cuda Cam value at coordinates Y: %d - X: %d = %d\n", (int)y_proj + k, (int)x_proj + l, luma[((int)y_proj + k) * width  + ((int)x_proj + l) + offset]);
                    //}
                    cost += fabsf(luma[(idx_y + k) * width + (idx_x + l)] - luma[((int)y_proj + k) * width +  ((int)x_proj + l) + offset]);
                    cc += 1.0f;
                }
            }
            /*if (zi == 0 && idx == 0) {
                printf("cost: %f\n", cost);
                printf("cc: %f\n", cc);
                printf("cost/cc: %f\n", cost/cc);
            }*/
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

float* wrap_sweeping_plane_device(double* h_ref_cam, double* h_cams, uint8_t* h_luma, const int width, const int height,
    const int n_planes, const int n_cam, int window){
    printf("Sweeping_plane_device standard:\n");

    size_t n_luma = width * height * n_cam;
    size_t n_cost_cube = width * height * n_planes;

    float* h_cost_cube = new float[n_cost_cube];
    std::fill_n(h_cost_cube, n_cost_cube, 255.f);

    // device buffers
    uint8_t* d_luma = nullptr;
    float* d_cost_cube = nullptr;

    CHK(cudaSetDevice(0));
    // copy camera params to constant memory
    CHK(cudaMemcpyToSymbol(cam_param_array,
        h_ref_cam,
        (P_SIZE / 4) * sizeof(double),
        0,
        cudaMemcpyHostToDevice));
    CHK(cudaMemcpyToSymbol(cam_param_array,
        h_cams,
        (P_SIZE / 4) * 3 * sizeof(double),
        (P_SIZE / 4) * sizeof(double),
        cudaMemcpyHostToDevice));

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
    dim3 threads(16, 16);
    dim3 blocks(divUp(width, threads.x),
        divUp(height, threads.y));

    sweeping_plane_device_standard << <blocks, threads >> > (
        d_luma,
        d_cost_cube,
        height,
        width,
        n_cam,
        n_planes,
        window
        );
    CHK(cudaGetLastError());
    // getting the device array copied to host
    CHK(cudaMemcpy(h_cost_cube, d_cost_cube, n_cost_cube *sizeof(float),
                   cudaMemcpyDeviceToHost));

    // cleanup
    cudaFree(d_luma);
    cudaFree(d_cost_cube);
    return h_cost_cube;
}

//void wrap_sweeping_plane_device(double* h_ref_cam, double* h_cams, uint8_t* h_luma, 
//								const int width, const int height, const int n_planes, const int n_cam, int window = 3) {
//    printf("Sweeping_plane_device:\n");
//
//    uint8_t* d_luma = nullptr;
//    float* d_cost_cube = nullptr;
//    float* h_cost_cube = new float[height * width * n_planes];
//    std::fill_n(h_cost_cube, height * width * n_planes, 255.0f);
//
//    CHK(cudaSetDevice(0));
//    CHK(cudaMemcpyToSymbol(cam_param_array,
//        h_ref_cam,
//        (P_SIZE / 4) * sizeof(double),
//        0,
//        cudaMemcpyHostToDevice));
//    CHK(cudaMemcpyToSymbol(cam_param_array,
//        h_cams,
//        (P_SIZE / 4) * 3 * sizeof(double),
//        (P_SIZE / 4) * sizeof(double),
//        cudaMemcpyHostToDevice));
//
//    size_t pitch_luma_bytes, pitch_cost_bytes;
//    CHK(cudaMallocPitch((void**)&d_luma,
//        &pitch_luma_bytes,
//        width * n_cam * sizeof(uint8_t),
//        height));
//    CHK(cudaMallocPitch((void**)&d_cost_cube,
//        &pitch_cost_bytes,
//        width * n_planes * sizeof(float),
//        height));
//
//    unsigned d_pitch_luma = pitch_luma_bytes / sizeof(uint8_t);
//    unsigned d_pitch_cost = pitch_cost_bytes / sizeof(float);
//
//   
//    CHK(cudaMemcpy2D(d_luma,
//        pitch_luma_bytes,
//        h_luma,
//        width * n_cam * sizeof(uint8_t),
//        width * n_cam * sizeof(uint8_t),
//        height,
//        cudaMemcpyHostToDevice));
//
//    CHK(cudaMemcpy2D(d_cost_cube,
//        pitch_cost_bytes,
//        h_cost_cube,
//        width * n_planes * sizeof(float),
//        width * n_planes * sizeof(float),
//        height,
//        cudaMemcpyHostToDevice));
//
//    dim3 threads(256, 1);
//    dim3 blocks(divUp(width, threads.x),
//        divUp(height, threads.y));
//    sweeping_plane_device << <blocks, threads >> > (d_luma,
//        d_cost_cube,
//        height,
//        width,
//        n_cam,
//        n_planes,
//        window,
//        d_pitch_luma,
//        d_pitch_cost);
//
//    delete[] h_cost_cube;
//}




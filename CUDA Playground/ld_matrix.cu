#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_fp8.h>
#include <vector>
#include <ctime>
#include <random>

template<typename T, int M, int N>
struct ptx_fragment {
    T data[M * N / 32]; 
    uint32_t raw[M * N / 32];
    static constexpr int total_elements = M * N;
    static constexpr int total_bytes = M * N * sizeof(T);
    static constexpr int bytes_per_thread = total_bytes / 32;
    static constexpr int elements_per_thread = total_elements / 32;
};

/*
Uses a wrap (32 threads) to load 1x 16x16 8-bit matrix. 
@result - destination
@shared_addr - source
*/
__device__ __forceinline__ void load_matrix_ptx_16x16_x1_b8(uint32_t* result, __nv_fp8_e4m3* shared_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m16n16.trans.x1.shared.b8 {%0,%1}, [%2];"
        : "=r"(result[0]), "=r"(result[1])
        : "l"(shared_addr)
        );
}

__global__ void kernel_load_1x_16x16_8bit(int M, int N, __nv_fp8_e4m3* data) {
    ptx_fragment<__nv_fp8x4_e4m3, 16, 16> frag;

    // Need 16x16 = 256 fp8 elements = 64 __nv_fp8x4_e4m3 elements
    __shared__ __nv_fp8_e4m3 shared_data[256];


    // Load enough data (assuming you have it)

    for (int i = 0; i < 256; i++) {
        shared_data[i] = data[i];
    }
    if (threadIdx.x == 0) {
        printf("First few shared values as hex: 0x%02x 0x%02x 0x%02x 0x%02x\n",
            *(uint8_t*)&shared_data[0], *(uint8_t*)&shared_data[1],
            *(uint8_t*)&shared_data[2], *(uint8_t*)&shared_data[3]);
    }
    uint32_t result[2];
    load_matrix_ptx_16x16_x1_b8(result, shared_data);
    __syncthreads();
    // Print raw hex values first to see what you're getting
    if (threadIdx.x == 0) {
        printf("Raw result[0] = 0x%08x, result[1] = 0x%08x\n", result[0], result[1]);
    }

    uint8_t val0 = (result[0] >> 0) & 0xFF;
    uint8_t val1 = (result[0] >> 8) & 0xFF;
    uint8_t val2 = (result[0] >> 16) & 0xFF;
    uint8_t val3 = (result[0] >> 24) & 0xFF;

    if (threadIdx.x == 0) {
        printf("val0=%d, val1=%d, val2=%d, val3=%d\n", val0, val1, val2, val3);
    }
}

/*
Uses a wrap (32 threads) to load 2x 16x16 8-bit matrix.
@result - destination
@shared_addr - source
*/
__device__ __forceinline__ void load_matrix_ptx_16x16_x2_b8(uint32_t* result, __nv_fp8_e4m3* shared_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m16n16.trans.x2.shared.b8 {%0,%1,%2,%3}, [%4];"
        : "=r"(result[0]), "=r"(result[1]), "=r"(result[2]), "=r"(result[3])
        : "l"(shared_addr)
        );
}


__global__ void kernel_load_2x_16x16_8bit(int M, int N, __nv_fp8_e4m3* data) {
    ptx_fragment<__nv_fp8x4_e4m3, 16, 16> frag;

    // Use 2 vectors of size 256, or 1 vector of 512
    __shared__ __nv_fp8_e4m3 shared_A[256];
    __shared__ __nv_fp8_e4m3 shared_B[256];



    // Load data concurently 

    for (int i = threadIdx.x; i < 256; i+=8) {
        shared_A[i] = data[i];
        shared_B[i] = data[i + 256];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("First few shared values as hex: 0x%02x 0x%02x 0x%02x 0x%02x\n",
            *(uint8_t*)&shared_A[0], *(uint8_t*)&shared_A[1],
            *(uint8_t*)&shared_A[2], *(uint8_t*)&shared_A[3]);
    }
    uint32_t result[4];
    load_matrix_ptx_16x16_x2_b8(result, shared_A);
    __syncthreads();
    // Print raw hex values first to see what you're getting
    if (threadIdx.x == 0) {
        printf("Raw result[0] = 0x%08x, result[1] = 0x%08x, result[1] = 0x%08x, result[1] = 0x%08x\n",
            result[0], result[1], result[2], result[3]);
    }

    uint8_t val0 = (result[0] >> 0) & 0xFF;
    uint8_t val1 = (result[0] >> 8) & 0xFF;
    uint8_t val2 = (result[0] >> 16) & 0xFF;
    uint8_t val3 = (result[0] >> 24) & 0xFF;

    if (threadIdx.x == 0) {
        printf("val0=%d, val1=%d, val2=%d, val3=%d\n", val0, val1, val2, val3);
    }
}


int main(){
    std::mt19937            rng{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<__nv_fp8_e4m3> mat;
    for (int i = 0; i < 16 * 16 * 2; i+=1) {
        mat.push_back(__nv_fp8_e4m3(1.0f));
    }
    __nv_fp8_e4m3* d_data;
    
    cudaMalloc(&d_data, 16 * 16 * 2);
    cudaMemcpy(d_data, mat.data(), 16 * 16 * 2, cudaMemcpyHostToDevice);

    //kernel_load_1x_16x16_8bit << <1, 32 >> > (16, 16, d_data);
    kernel_load_2x_16x16_8bit << <1, 32 >> > (16, 16, d_data);

    cudaFree(d_data);
    
}
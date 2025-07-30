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

__device__ __forceinline__ uint32_t load_matrix_ptx_8x8(__nv_fp8x4_e4m3* shared_addr) {
    uint32_t result;
    uint32_t source = *reinterpret_cast<uint32_t*>(&shared_addr);
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x1.shared.b32 {%0}, [%1];"
        : "=r"(result)           // output: 32-bit register
        : "r"(shared_addr)       // input: 32-bit address register  
        );
    return result;
}

__global__ void kernel_launch(int M, int N, __nv_fp8x4_e4m3* data) {
    ptx_fragment<__nv_fp8x4_e4m3, 16, 16> frag;

    // Need 16x16 = 256 fp8 elements = 64 __nv_fp8x4_e4m3 elements
    __shared__ __nv_fp8x4_e4m3 shared_data[16];

    // Load enough data (assuming you have it)
    for (int i = 0; i < 64 && i < M * N / 4; i++) {
        shared_data[i] = data[i];
    }


    uint32_t res = load_matrix_ptx_8x8(&shared_data[0]);

    }



int main(){
    std::mt19937            rng{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<__nv_fp8x4_e4m3> mat;
    for (int i = 0; i < 16 * 16; i+=4) {
        float4 f4 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        mat.push_back(__nv_fp8x4_e4m3(f4));
    }
    __nv_fp8x4_e4m3* d_data;
    
    cudaMalloc(&d_data, 16 * 16);
    cudaMemcpy(d_data, mat.data(), 16 * 16, cudaMemcpyHostToDevice);

    kernel_launch << <1, 1 >> > (16, 16, d_data);

    cudaFree(d_data);
    
}
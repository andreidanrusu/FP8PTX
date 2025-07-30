
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include <ctime>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <mma.h>
#define TILE_SIZE 16
using namespace nvcuda::wmma;

#define TILE_X 16
#define TILE_Y 16
__device__ void naiveMatmul(int N, int K,int M, float* mat1, float* mat2, float* res) {
    
}

__device__ void wmma_gemm_16x16(__half* a, __half* b, float* c) {
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    load_matrix_sync(a_frag, a, 16);  // stride = 16
    load_matrix_sync(b_frag, b, 16);
    fill_fragment(c_frag, 0.0f);      // initialize accumulator

    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(c, c_frag, 16, mem_row_major);

}

__device__ void matmul(int N, int K, int M, float* mat1, float* mat2, float* res) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    //Thread row within a block
    int y = threadIdx.y;
    //Thread col within a block
    int x = threadIdx.x;
    
    //Thread row within all blocks
    int row = blockIdx.y * TILE_SIZE + y;

    //Thread col within all blocks
    int col = blockIdx.x * TILE_SIZE + x;

    int tile_count = (K + TILE_SIZE - 1) / TILE_SIZE;

    float sum = 0.0f;

    for (int t = 0; t < tile_count; t++) {
        int aRow = row;
        int aCol = t * TILE_SIZE + x;
        tileA[y][x] = (aRow < N && aCol < K) ? mat1[aRow * K + aCol] : 0;
        
        int bRow = t * TILE_SIZE + y;
        int bCol = col;
        tileB[y][x] = (bRow < K && bCol < M) ? mat2[bCol * K + bRow] : 0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[y][k] * tileB[k][x];
        }

        __syncthreads();
        
    }
    if (row < N && col < M) {
        res[row * M + col] = sum;
        printf("[%d,%d] - %f\n",y, x, sum);
    }

}

__global__ void printTID()
{
    int i = threadIdx.x;
    printf("Thread id: %d\n", i);
}

__global__ void mul1DVectors(float* vec1, float* vec2, float* res, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;   
    if (i < N) {
        res[i] = vec1[i] * vec2[i];
        printf("%f * %f = %f from thread %d\n", vec1[i], vec2[i], res[i], i);
    }
}

__global__ void matadd(float *mat1, float *mat2, float* res, int N, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < M) {
        int idx = i * M + j;
        res[idx] = mat1 [idx]+ mat2[idx];
        printf("%f + %f = %f from thread %d\n", mat1[idx], mat2[idx], res[idx], idx);
    }
}

__global__ void fusedKernel(float* mat1, float* mat2, float* res, int N, int K, int M) {
    matmul(N, K, M, mat1, mat2, res);
}

void printVector(float* vec, int size) {
    printf("[%f", vec[0]);
    if (size > 1) {
        for (int i = 1; i < size; i++) {
            printf(",%f", vec[i]);
        }
    }
    printf("]\n");
}

void multiplyVectors(int N) {
    float* h_v1 = new float[N];
    float* h_v2 = new float[N];
    float* h_res = new float[N];
    float* d_v1, * d_v2, * d_res;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    std::srand(static_cast<unsigned>(std::time(0)));
    for (int i = 0; i < N; ++i) {
        h_v1[i] = static_cast<float>(rand()) / RAND_MAX;
        h_v2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    cudaMalloc(&d_v1, N * sizeof(float));
    cudaMalloc(&d_v2, N * sizeof(float));
    cudaMalloc(&d_res, N * sizeof(float));

    cudaStream_t async_cpy_stream;
    cudaStreamCreate(&async_cpy_stream);

    cudaMemcpyAsync(d_v1, h_v1, N * sizeof(float), cudaMemcpyHostToDevice, async_cpy_stream);
    cudaMemcpyAsync(d_v2, h_v2, N * sizeof(float), cudaMemcpyHostToDevice, async_cpy_stream);

    cudaStreamSynchronize(async_cpy_stream);

    mul1DVectors << <blocks, threads >> > (d_v1, d_v2, d_res, N);

    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_res);
}

void addMatrices(int N, int M) {
    float* h_m1 = new float[N * M];
    float* h_m2 = new float[N * M];
     
    float* d_m1, * d_m2, * d_res;

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);


    std::srand(static_cast<unsigned>(std::time(0)));
    for (int i = 0; i < N * M; ++i) {
        h_m1[i] = static_cast<float>(rand()) / RAND_MAX;
        h_m2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    cudaMalloc(&d_m1, N * M * sizeof(float));
    cudaMalloc(&d_m2, N * M * sizeof(float));
    cudaMalloc(&d_res, N * M * sizeof(float));

    cudaStream_t async_cpy_stream;
    cudaStreamCreate(&async_cpy_stream);

    cudaMemcpyAsync(d_m1, h_m1, N * M * sizeof(float), cudaMemcpyHostToDevice, async_cpy_stream);
    cudaMemcpyAsync(d_m2, h_m2, N * M * sizeof(float), cudaMemcpyHostToDevice, async_cpy_stream);

    cudaStreamSynchronize(async_cpy_stream);

    matadd << <numBlocks, threadsPerBlock >> > (d_m1, d_m2, d_res, N, M);

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_res);
}

void multiplyRandMatrices(int N, int K, int M) {
    float* h_m1 = new float[N * K];
    float* h_m2 = new float[K * M];
    float* h_res = new float[N * M];

    float* d_m1, * d_m2, * d_res;

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);


    std::srand(static_cast<unsigned>(std::time(0)));
    for (int i = 0; i < N * K; ++i) {
        h_m1[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < K * M; ++i) {
        h_m2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    cudaMalloc(&d_m1, N * K * sizeof(float));
    cudaMalloc(&d_m2, K * M * sizeof(float));
    cudaMalloc(&d_res, N * M * sizeof(float));

    cudaStream_t async_cpy_stream;
    cudaStreamCreate(&async_cpy_stream);

    cudaMemcpyAsync(d_m1, h_m1, N * K * sizeof(float), cudaMemcpyHostToDevice, async_cpy_stream);
    cudaMemcpyAsync(d_m2, h_m2, K * M * sizeof(float), cudaMemcpyHostToDevice, async_cpy_stream);

    cudaStreamSynchronize(async_cpy_stream);

    fusedKernel << <numBlocks, threadsPerBlock >> > (d_m1, d_m2, d_res, N, K, M);

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_res);
}

void multiplyFloatMatrices(int N, int K, int M) {
    float* h_m1 = new float[N * K];
    float* h_m2 = new float[K * M];
    float* h_res = new float[N * M];

    float* d_m1, * d_m2, * d_res;

    dim3 threadsPerBlock(TILE_X, TILE_Y);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);


    for (int i = 0; i < N * K; ++i) {
        h_m1[i] = i;
    }

    float acc = 0;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            int index = j * K + i;
            printf("Index is: %d with acc: %f\n", index, acc);
            h_m2[index] = acc;
            acc += 1;
        }
    }
     
    printVector(h_m1, N * K);
    printVector(h_m2, K * M);

    cudaMalloc(&d_m1, N * K * sizeof(float));
    cudaMalloc(&d_m2, K * M * sizeof(float));
    cudaMalloc(&d_res, N * M * sizeof(float));

    cudaStream_t async_cpy_stream;
    cudaStreamCreate(&async_cpy_stream);

    cudaMemcpyAsync(d_m1, h_m1, N * K * sizeof(float), cudaMemcpyHostToDevice, async_cpy_stream);
    cudaMemcpyAsync(d_m2, h_m2, K * M * sizeof(float), cudaMemcpyHostToDevice, async_cpy_stream);

    cudaStreamSynchronize(async_cpy_stream);

    fusedKernel << <numBlocks, threadsPerBlock >> > (d_m1, d_m2, d_res, N, K, M);

    cudaMemcpy(h_res, d_res, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    printVector(h_res, N * M);

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_res);
    delete[] h_m1;
    delete[] h_m2;
    delete[] h_res;

}



//int main()
//{
//    //Multiplies 2 vectors of passed size.
//    //multiplyVectors(50);
//
//    // Add two random matrices of size N x M 
//    //addMatrices(5, 5);
//
//    //Multiply two random matrices of sizes N x K and K x M
//    //multiplyRandMatrices(10, 5, 10);
//
//    //Multiplay matrices
//    multiplyFloatMatrices(32, 32, 64);
//
//    // printTID << <1, 256 >> > ();
//}




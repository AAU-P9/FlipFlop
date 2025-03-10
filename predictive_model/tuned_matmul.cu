#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

const int TILE_SIZE = 32;
const int M = 1024, N = 1024, K = 1024;

template <typename T>
__global__ void tuned_matmul(T *a, T *b, T *c, int M, int N, int K) {
    __shared__ __align__(32) T a_tile[TILE_SIZE][TILE_SIZE+1]; // +1 to avoid bank conflicts
    __shared__ __align__(32) T b_tile[TILE_SIZE][TILE_SIZE+1];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    T sum = 0;

    for(int t = 0; t < (K + TILE_SIZE - 1)/TILE_SIZE; ++t) {
        int a_col = t*TILE_SIZE + threadIdx.x;
        int b_row = t*TILE_SIZE + threadIdx.y;
        
        // Load tiles with boundary checks
        if(row < M && a_col < K)
            a_tile[threadIdx.y][threadIdx.x] = a[row*K + a_col];
        
        if(col < N && b_row < K)
            b_tile[threadIdx.y][threadIdx.x] = b[b_row*N + col];
        
        __syncthreads();

        // Process tile with loop unrolling
        #pragma unroll
        for(int k = 0; k < TILE_SIZE; k++) {
            sum += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Store result with alpha=2.0 and beta=0.5
    if(row < M && col < N)
        c[row*N + col] = 2.0f * sum + 0.5f * c[row*N + col];
}

int main() {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, M*K*sizeof(float));
    cudaMalloc(&d_b, K*N*sizeof(float));
    cudaMalloc(&d_c, M*N*sizeof(float));

    // Initialize matrices
    float *h_a = new float[M*K];
    float *h_b = new float[K*N];
    float *h_c = new float[M*N];
    
    // Fill matrices with sample data
    for(int i = 0; i < M*K; i++) h_a[i] = 1.0f;
    for(int i = 0; i < K*N; i++) h_b[i] = 1.0f;
    for(int i = 0; i < M*N; i++) h_c[i] = 0.5f;

    cudaMemcpy(d_a, h_a, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, M*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1)/TILE_SIZE, (M + TILE_SIZE - 1)/TILE_SIZE);

    // Warmup
    tuned_matmul<float><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();

    // Timing
    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int trials = 100;
    cudaEventRecord(start);
    for(int i = 0; i < trials; i++) {
        tuned_matmul<float><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "Average execution time: " << elapsed_time/trials << " ms" << std::endl;

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}

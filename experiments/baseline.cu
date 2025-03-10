__global__ void kernel_func(float *C, float *A, float *B, int N) {
    int row = blockIdx.y * block_size_y + threadIdx.y;
    int col = blockIdx.x * block_size_x + threadIdx.x;
    float sum = 0.0f;
    
    if (row < N && col < N) {
        #if use_shared_mem
            __shared__ float As[32][32];
            __shared__ float Bs[32][32];
            
            for (int tile = 0; tile < N; tile += 32) {
                if (threadIdx.x < 32 && threadIdx.y < 32) {
                    As[threadIdx.y][threadIdx.x] = A[row*N + tile + threadIdx.x];
                    Bs[threadIdx.y][threadIdx.x] = B[(tile + threadIdx.y)*N + col];
                }
                __syncthreads();
                
                #if unroll_factor==1
                    for (int k = 0; k < 32; k++) {
                        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                    }
                #elif unroll_factor==2
                    #pragma unroll 2
                    for (int k = 0; k < 32; k++) {
                        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                    }
                #elif unroll_factor==4
                    #pragma unroll 4
                    for (int k = 0; k < 32; k++) {
                        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                    }
                #elif unroll_factor==8
                    #pragma unroll 8
                    for (int k = 0; k < 32; k++) {
                        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                    }
                #endif
                __syncthreads();
            }
        #else
            #if unroll_factor==1
                for (int k = 0; k < N; k++) {
                    sum += A[row*N + k] * B[k*N + col];
                }
            #elif unroll_factor==2
                #pragma unroll 2
                for (int k = 0; k < N; k++) {
                    sum += A[row*N + k] * B[k*N + col];
                }
            #elif unroll_factor==4
                #pragma unroll 4
                for (int k = 0; k < N; k++) {
                    sum += A[row*N + k] * B[k*N + col];
                }
            #elif unroll_factor==8
                #pragma unroll 8
                for (int k = 0; k < N; k++) {
                    sum += A[row*N + k] * B[k*N + col];
                }
            #endif
        #endif
        
        C[row*N + col] = sum;
    }
}
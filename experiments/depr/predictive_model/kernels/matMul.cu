extern "C" {
    __global__ void matMul(float* A, float* B, float* C, int M, int N, int K) {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if(row < M && col < K) {
            float sum = 0.0f;
            for(int i = 0; i < N; i++) {
                sum += A[row * N + i] * B[i * K + col];
            }
            C[row * K + col] = sum;
        }
    }
    }
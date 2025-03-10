#define grid_size_x 128
#define grid_size_y 128
#define grid_size_z 1
#define block_size_x 32
#define block_size_y 32
#define block_size_z 1
#define tile_size_x 16
#define tile_size_y 4
#define nvml_pwr_limit 175
#define kernel_tuner 1
#line 1
#include <cuda_fp16.h>  // Must include for __half support

// template <typename T>
// __global__ void tuned_matmul(T* a, T* b, T* c, int M, int K, int N) {

//     const T alpha = T(2.0);
//     const T beta  = T(0.5);
//     extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
//     T* a_tile = reinterpret_cast<T*>(shared_mem);
//     T* b_tile = reinterpret_cast<T*>(shared_mem + blockDim.y * blockDim.x * sizeof(T));
    
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
    
//     T sum = 0;
//     for (int t = 0; t < (K + blockDim.x - 1)/blockDim.x; ++t) {
//         // Collaborative loading
//         if (row < M && (t*blockDim.x + threadIdx.x) < K)
//             a_tile[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + t*blockDim.x + threadIdx.x];
//         if ((t*blockDim.y + threadIdx.y) < K && col < N)
//             b_tile[threadIdx.y * blockDim.x + threadIdx.x] = b[(t*blockDim.y + threadIdx.y) * N + col];
        
//         __syncthreads();
        
//         #pragma unroll
//         for (int k = 0; k < blockDim.x; k++) {
//             // Use half-precision intrinsics when T is __half
//             if constexpr (std::is_same_v<T, __half>) {
//                 sum = __hadd(sum, __hmul(a_tile[threadIdx.y * blockDim.x + k], 
//                                        b_tile[k * blockDim.x + threadIdx.x]));
//             } else {
//                 sum += a_tile[threadIdx.y * blockDim.x + k] * 
//                        b_tile[k * blockDim.x + threadIdx.x];
//             }
//         }
//         __syncthreads();
//     }
    
//     if (row < M && col < N) {
//         c[row * N + col] = alpha * sum + beta * c[row * N + col];
//     }
// }



template <typename T>
__global__ void tuned_matmul(T *a, T *b, T *c, int M, int K, int N) {
  const T alpha = T(2.0);
  const T beta  = T(0.5);
  int row = blockIdx.y * tile_size_y + threadIdx.y;
  int col = blockIdx.x * tile_size_x + threadIdx.x;
  if (row < M && col < N) {
    T s = 0;
    for (int k = 0; k < K; k++)
      s += a[row * K + k] * b[k * N + col];
    c[row * N + col] = alpha * s + beta * c[row * N + col];
  }
}
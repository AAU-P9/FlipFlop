// mha.cu

// Device headers only:
#include <cmath>
#include <cuda.h>
#include <cub/block/block_reduce.cuh>

// Host-only headers should be wrapped so they’re not included in device code.
#ifndef __CUDA_ARCH__
  #include <nvml.h>
  #include <cstdio>
  #include <cstdlib>
  #include <chrono>
  #include <iostream>
#endif

#define FINAL_MASK 0xffffffff

__inline__ __device__
float warpReduceSum(float val)
{
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

__inline__ __device__
float blockReduceSum(float val)
{
  static __shared__ float shared[32]; 
  int lane = threadIdx.x & 0x1f; 
  int wid = threadIdx.x >> 5;  
  val = warpReduceSum(val);
  if(lane == 0)
    shared[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
  val = warpReduceSum(val);
  return val;
}

__inline__ __device__
float warpReduceMax(float val)
{
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

__inline__ __device__
float blockReduceMax(float val)
{
  static __shared__ float shared[32]; 
  int lane = threadIdx.x & 0x1f; 
  int wid = threadIdx.x >> 5;
  val = warpReduceMax(val);
  if(lane == 0)
    shared[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
  val = warpReduceMax(val);
  return val;
}

__global__
void mha (
   const float *__restrict__ q, 
   const float *__restrict__ k, 
   const float *__restrict__ v, 
   const int beam_size, 
   const int n_steps, 
   const int qk_col, 
   const int v_col, 
   const int nhead, 
   const float scale,
   const int THRESHOLD,
   float *__restrict__ dst)
{
  // Each block processes one head from one candidate.
  int dim_per_head = qk_col / nhead;
  int candidate_id = blockIdx.x / nhead;
  int head_id = blockIdx.x % nhead;

  typedef cub::BlockReduce<float, 256> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // Shared memory buffer: first part for query vector, then logits.
  extern __shared__ float buffer[];
  float *sq = buffer;
  float *logits = buffer + dim_per_head;

  // Each block loads its portion of the query vector.
  int pos = candidate_id * qk_col + head_id * dim_per_head + threadIdx.x;
  if(threadIdx.x < dim_per_head)
    sq[threadIdx.x] = q[pos];
  __syncthreads();

  // Compute dot product (query dot key) for each time step.
  float summ = 0.f;
  if(threadIdx.x < n_steps)
  {   
    const float *k2 = k + candidate_id * qk_col * n_steps + head_id * dim_per_head + threadIdx.x * qk_col;
    for (int i = 0; i < dim_per_head; i++)
      summ += sq[i] * k2[i];
    summ *= scale;
  }   

  // Softmax reduction using block reductions.
  __shared__ float s_max_val;
  __shared__ float s_sum;
  float local_i = threadIdx.x < n_steps ? summ : -1e-20f;
  float local_o;
  float max_val = BlockReduce(temp_storage).Reduce(local_i, cub::Max());
  if(threadIdx.x == 0)
    s_max_val = max_val;
  __syncthreads();
  local_i -= s_max_val;
  if(local_i < -THRESHOLD) local_i = -THRESHOLD;
  local_o = expf(local_i);
  float val = (threadIdx.x < n_steps) ? local_o : 0.f;
  val = BlockReduce(temp_storage).Sum(val);
  if(threadIdx.x == 0) s_sum = val;
  __syncthreads();
  if(threadIdx.x < n_steps) logits[threadIdx.x] = local_o / s_sum;
  __syncthreads();

  // Compute weighted sum over the value matrix.
  summ = 0.f;
  if(threadIdx.x < dim_per_head)
  {
    int tid = candidate_id * v_col * n_steps + head_id * dim_per_head + threadIdx.x;
    for(int i = 0; i < n_steps; ++i)
      summ += logits[i] * v[tid + i * v_col];
    dst[candidate_id * v_col + head_id * dim_per_head + threadIdx.x] = summ;
  }
}

#ifndef __CUDA_ARCH__  // Host-only code follows.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <iostream>
int main(int argc, char* argv[])
{
  if(argc != 2){
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::atoi(argv[1]);

  const int beamsize = 4;
  const int nhead = 16;
  const int dim_feature = nhead * 256;
  const int n_steps = 9;
  const float scaler = std::sqrt(nhead * 1.f / dim_feature);
  const int qk_col = dim_feature;
  const int v_col = dim_feature;
  const int THRESHOLD = 64;

  const int q_size = beamsize * dim_feature;
  const int q_size_bytes = sizeof(float) * q_size;
  const int k_size = beamsize * dim_feature * n_steps;
  const int k_size_bytes = sizeof(float) * k_size;
  const int v_size = beamsize * dim_feature * n_steps;
  const int v_size_bytes = sizeof(float) * v_size;

  float *dq, *dk, *dv, *dst;
  cudaMalloc((void**)&dq, q_size_bytes);
  cudaMalloc((void**)&dk, k_size_bytes);
  cudaMalloc((void**)&dv, v_size_bytes);
  cudaMalloc((void**)&dst, q_size_bytes);

  float *hq = (float*)malloc(q_size_bytes);
  float *hk = (float*)malloc(k_size_bytes);
  float *hv = (float*)malloc(v_size_bytes);
  float *h_dst = (float*)malloc(q_size_bytes);

  srand(123);
  for(int i = 0; i < q_size; ++i)
    hq[i] = rand() / (float)RAND_MAX;
  for(int i = 0; i < k_size; ++i)
    hk[i] = rand() / (float)RAND_MAX;
  for(int i = 0; i < v_size; ++i)
    hv[i] = rand() / (float)RAND_MAX;

  cudaMemcpy(dq, hq, q_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dk, hk, k_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv, v_size_bytes, cudaMemcpyHostToDevice);

  dim3 grid(nhead * beamsize);
  dim3 block(qk_col / nhead);
  const int shared_size = sizeof(float) * ((qk_col / nhead) + n_steps);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    mha<<<grid, block, shared_size>>>(dq, dk, dv,
      beamsize, n_steps, qk_col, v_col, nhead, scaler, THRESHOLD, dst);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  cudaMemcpy(h_dst, dst, q_size_bytes, cudaMemcpyDeviceToHost);
  cudaFree(dq);
  cudaFree(dk);
  cudaFree(dv);
  cudaFree(dst);

  for (int i = 0; i < beamsize - 1; i++) {
    float sum = 0.f;
    for (int j = 0; j < dim_feature; j++) {
       float d = h_dst[i * dim_feature + j] - h_dst[(i + 1) * dim_feature + j];
       sum += d * d;
    }
    std::printf("Distance between beams %d and %d: %f\n", i, i+1, std::sqrt(sum));
  }
  free(hq);
  free(hk);
  free(hv);
  free(h_dst);
  return 0;
}
#endif

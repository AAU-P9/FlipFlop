// mha_kernel.cu
// This file contains only the device code for the multi–head attention (MHA) kernel.
// It is meant to be compiled by nvcc (via PyCUDA) and does not include any host main().
// Use your Python launcher (with configuration from config_mha.json) to compile and run this kernel.

#include <cuda.h>
#include <cub/block/block_reduce.cuh>
#include <math.h>   // Use math.h (not <cmath>) for device math functions

// Define a constant mask for warp-level primitives.
#define FINAL_MASK 0xffffffff

// Warp-level reduction for sum.
__inline__ __device__
float warpReduceSum(float val) {
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

// Block-level reduction for sum using dynamic shared memory.
__inline__ __device__
float blockReduceSum(float val) {
  extern __shared__ float shared[];
  int lane = threadIdx.x & 0x1f;
  int wid  = threadIdx.x >> 5;
  val = warpReduceSum(val);
  if (lane == 0)
    shared[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
  val = warpReduceSum(val);
  return val;
}

// Warp-level reduction for maximum.
__inline__ __device__
float warpReduceMax(float val) {
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

// Block-level reduction for maximum.
__inline__ __device__
float blockReduceMax(float val) {
  extern __shared__ float shared[];
  int lane = threadIdx.x & 0x1f;
  int wid  = threadIdx.x >> 5;
  val = warpReduceMax(val);
  if (lane == 0)
    shared[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
  val = warpReduceMax(val);
  return val;
}

// -----------------------------
// Multi–Head Attention (MHA) kernel
// -----------------------------
// This kernel computes the softmax–weighted attention output for the query, key, and value matrices.
// It is declared with extern "C" so that its name is not mangled for the host (PyCUDA) to access.
extern "C" __global__
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

  // Use CUB's block reduction.
  typedef cub::BlockReduce<float, 256> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // Use externally allocated shared memory.
  // The first part holds the query vector (sq); the second part holds the logits.
  extern __shared__ float buffer[];
  float *sq = buffer;
  float *logits = buffer + dim_per_head;

  // Load the query vector for this candidate/head.
  int pos = candidate_id * qk_col + head_id * dim_per_head + threadIdx.x;
  if (threadIdx.x < dim_per_head)
    sq[threadIdx.x] = q[pos];
  __syncthreads();

  // For each time step, compute the dot product (query · key) and scale.
  float summ = 0.f;
  if (threadIdx.x < n_steps) {   
    const float *k2 = k + candidate_id * qk_col * n_steps + head_id * dim_per_head + threadIdx.x * qk_col;
    for (int i = 0; i < dim_per_head; i++)
      summ += sq[i] * k2[i];
    summ *= scale;
  }   

  // Compute the softmax normalization using block-level reductions.
  __shared__ float s_max_val;
  __shared__ float s_sum;
  float local_i = (threadIdx.x < n_steps) ? summ : -1e-20f;
  float local_o;
  float max_val = BlockReduce(temp_storage).Reduce(local_i, cub::Max());
  if (threadIdx.x == 0)
    s_max_val = max_val;
  __syncthreads();
  
  local_i -= s_max_val;
  if (local_i < -THRESHOLD) local_i = -THRESHOLD;
  local_o = expf(local_i);
  float val = (threadIdx.x < n_steps) ? local_o : 0.f;
  val = BlockReduce(temp_storage).Sum(val);
  if (threadIdx.x == 0)
    s_sum = val;
  __syncthreads();
  
  if (threadIdx.x < n_steps)
    logits[threadIdx.x] = local_o / s_sum;
  __syncthreads();

  // Compute the weighted sum over the value matrix.
  summ = 0.f;
  if (threadIdx.x < dim_per_head) {
    int tid = candidate_id * v_col * n_steps + head_id * dim_per_head + threadIdx.x;
    for (int i = 0; i < n_steps; ++i)
      summ += logits[i] * v[tid + i * v_col];
    dst[candidate_id * v_col + head_id * dim_per_head + threadIdx.x] = summ;
  }
}

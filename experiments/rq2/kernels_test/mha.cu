// mha_kernel.cu
// This CUDA kernel implements multi-head attention (MHA) using warp- and block-level reductions.
// It is intended to be compiled with nvcc (via PyCUDA/Kernel Tuner).
//
// Input layouts:
//    - q: [beam_size, qk_col] where qk_col = nhead * dim_per_head
//    - k: [beam_size, n_steps, qk_col]
//    - v: [beam_size, n_steps, v_col] (with v_col = qk_col)
//    - dst: [beam_size, v_col]
// Externally allocated shared memory must have size >= (dim_per_head + n_steps) * sizeof(float).
// THRESHOLD is passed as an argument.
// 
// Assumes that n_steps <= blockDim.x (blockDim.x is fixed to 256).
// For larger sequence lengths, further looping is needed.

#include <cuda.h>
#include <cub/block/block_reduce.cuh>
#include <math.h>

// Define constant mask for warp-level primitives.
#define FINAL_MASK 0xffffffff

// Warp-level reduction for sum.
__inline__ __device__
float warpReduceSum(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
}

// Block-level reduction for sum using CUB.
template <int BLOCK_SIZE>
__inline__ __device__
float blockReduceSum(float val, typename cub::BlockReduce<float, BLOCK_SIZE>::TempStorage &temp_storage) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    return BlockReduce(temp_storage).Reduce(val, cub::Sum());
}

// Warp-level reduction for maximum.
__inline__ __device__
float warpReduceMax(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

// Block-level reduction for maximum using CUB.
template <int BLOCK_SIZE>
__inline__ __device__
float blockReduceMax(float val, typename cub::BlockReduce<float, BLOCK_SIZE>::TempStorage &temp_storage) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    return BlockReduce(temp_storage).Reduce(val, cub::Max());
}

// -----------------------------
// Multi–Head Attention (MHA) kernel
// -----------------------------
// Computes the softmax–weighted attention output for query, key, and value matrices.
// Assumes:
//   - q: [beam_size, qk_col] where qk_col = nhead * dim_per_head.
//   - k: [beam_size, n_steps, qk_col].
//   - v: [beam_size, n_steps, v_col] (v_col = qk_col).
//   - dst: [beam_size, v_col].
// Uses externally allocated shared memory:
//    - 'sq': query vector of size dim_per_head
//    - 'logits': softmax logits of size n_steps.
// THRESHOLD is passed as a kernel argument.
extern "C" __global__
void mha(
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
    // Fixed block size of 256 threads is enforced.
    const int BLOCK_SIZE = 256;
    int dim_per_head = qk_col / nhead;

    // Each block is responsible for one candidate and one head.
    int candidate_id = blockIdx.x / nhead;
    int head_id = blockIdx.x % nhead;

    // Create a CUB BlockReduce instance.
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // Externally allocated shared memory.
    // Layout: first dim_per_head floats for 'sq', next n_steps floats for 'logits'.
    extern __shared__ float shared_buffer[];
    float *sq = shared_buffer;
    float *logits = shared_buffer + dim_per_head;

    // Load query vector for this candidate and head.
    int q_index = candidate_id * qk_col + head_id * dim_per_head + threadIdx.x;
    if (threadIdx.x < dim_per_head)
        sq[threadIdx.x] = q[q_index];
    __syncthreads();

    // Compute dot-product for each time step (only threads with index < n_steps participate).
    float dot_val = 0.f;
    if (threadIdx.x < n_steps) {
        const float *k_ptr = k + candidate_id * (qk_col * n_steps)
                             + threadIdx.x * qk_col
                             + head_id * dim_per_head;
        for (int i = 0; i < dim_per_head; i++) {
            dot_val += sq[i] * k_ptr[i];
        }
        dot_val *= scale;
    }

    // Compute maximum value for softmax stabilization.
    __shared__ float s_max_val;
    __shared__ float s_sum;
    float local_val = (threadIdx.x < n_steps) ? dot_val : -1e20f;
    float max_val = blockReduceMax<BLOCK_SIZE>(local_val, temp_storage);
    if (threadIdx.x == 0)
        s_max_val = max_val;
    __syncthreads();

    // Subtract maximum and clip using THRESHOLD.
    local_val -= s_max_val;
    if (local_val < -THRESHOLD)
        local_val = -THRESHOLD;
    float exp_val = expf(local_val);

    // Compute sum of exponentials.
    float sum_exp = blockReduceSum<BLOCK_SIZE>((threadIdx.x < n_steps) ? exp_val : 0.f, temp_storage);
    if (threadIdx.x == 0)
        s_sum = sum_exp;
    __syncthreads();

    // Write softmax probabilities (logits) to shared memory.
    if (threadIdx.x < n_steps)
        logits[threadIdx.x] = exp_val / s_sum;
    __syncthreads();

    // Compute weighted sum over value matrix.
    float weighted_sum = 0.f;
    if (threadIdx.x < dim_per_head) {
        int v_index = candidate_id * (v_col * n_steps) + head_id * dim_per_head + threadIdx.x;
        for (int t = 0; t < n_steps; t++) {
            weighted_sum += logits[t] * v[v_index + t * v_col];
        }
        dst[candidate_id * v_col + head_id * dim_per_head + threadIdx.x] = weighted_sum;
    }
}

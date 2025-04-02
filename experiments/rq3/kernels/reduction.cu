extern "C" {
    __global__ void reduction(float* input, float* output, int size) {
        extern __shared__ float sdata[];
        
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        sdata[tid] = (i < size) ? input[i] : 0;
        __syncthreads();
    
        for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
            if(tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        if(tid == 0) output[blockIdx.x] = sdata[0];
    }
    }
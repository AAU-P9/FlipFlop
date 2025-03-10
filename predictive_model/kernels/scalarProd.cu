extern "C" {
    __global__ void scalarProd(float* a, float* b, float* c, int size) {
        __shared__ float cache[256];
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int cacheIndex = threadIdx.x;
        
        float temp = 0;
        while(tid < size) {
            temp += a[tid] * b[tid];
            tid += blockDim.x * gridDim.x;
        }
        
        cache[cacheIndex] = temp;
        __syncthreads();
        
        for(int i = blockDim.x/2; i > 0; i >>= 1) {
            if(cacheIndex < i) {
                cache[cacheIndex] += cache[cacheIndex + i];
            }
            __syncthreads();
        }
        
        if(cacheIndex == 0) {
            atomicAdd(c, cache[0]);
        }
    }
    }
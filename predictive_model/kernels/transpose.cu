extern "C" {
    __global__ void transpose(float* input, float* output, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if(x < width && y < height) {
            output[x * height + y] = input[y * width + x];
        }
    }
    }
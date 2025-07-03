#define FILTER_RADIUS 3
__constant__ float filter[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

extern "C" {
__global__ void convolution(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x >= FILTER_RADIUS && x < width-FILTER_RADIUS && 
       y >= FILTER_RADIUS && y < height-FILTER_RADIUS) {
        float sum = 0.0f;
        for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
            for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
                sum += input[(y+dy)*width + (x+dx)] * 
                       filter[dy+FILTER_RADIUS][dx+FILTER_RADIUS];
            }
        }
        output[y*width + x] = sum;
    }
}
}
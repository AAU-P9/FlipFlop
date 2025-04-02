extern "C" {
    __global__ void laplace3d(float* in, float* out, int nx, int ny, int nz) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        
        if(x > 0 && x < nx-1 && y > 0 && y < ny-1 && z > 0 && z < nz-1) {
            int idx = z*nx*ny + y*nx + x;
            out[idx] = (in[idx+1] + in[idx-1] +
                        in[idx+nx] + in[idx-nx] +
                        in[idx+nx*ny] + in[idx-nx*ny] -
                        6.0f * in[idx]);
        }
    }
    }
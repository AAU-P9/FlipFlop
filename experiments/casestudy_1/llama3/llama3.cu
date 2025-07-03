/*
 * llama3.cuda is a pure C/CUDA implementation for Llama 3 model.
 */
 #include <stdio.h>
 #include <stdlib.h>
 #include <ctype.h>
 #include <time.h>
 #include <string.h>
 #include <fcntl.h>
 #include <unistd.h>
 #include <sys/mman.h>
 
 #include <cuda_runtime.h>
 #include <cub/cub.cuh>
 #include <cublas_v2.h>
 #include <stdarg.h>


#ifndef BLOCK_RMSNORM_X
#define BLOCK_RMSNORM_X 1024
#endif

#ifndef BLOCK_RMSNORM_Y
#define BLOCK_RMSNORM_Y 1
#endif

#ifndef BLOCK_MATMUL_X
#define BLOCK_MATMUL_X 64
#endif

#ifndef BLOCK_MATMUL_Y
#define BLOCK_MATMUL_Y 1
#endif

#ifndef BLOCK_MHA_X
#define BLOCK_MHA_X 256
#endif

#ifndef BLOCK_MHA_Y
#define BLOCK_MHA_Y 1
#endif


#define CUDA_CHECK(val) { \
    if (val != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", val, cudaGetErrorString(val), __FILE__, __LINE__); \
        fflush(stderr); \
        exit(val); \
    } \
}

FILE* debug_file = NULL;

void debug_init() {
    debug_file = fopen("debug_log.txt", "w");
    if (!debug_file) {
        fprintf(stderr, "Failed to open debug file\n");
        exit(EXIT_FAILURE);
    }
}

#ifdef __CUDA_ARCH__
__device__ void debug_print(const char* format, ...) {
    // Device code cannot perform file I/O; leave empty.
}
#else
__host__ void debug_print(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(debug_file, format, args);
    va_end(args);
    fflush(debug_file);
}
#endif


void dump_tensor(const char* name, float* tensor, int size) {
    float* cpu_data = (float*)malloc(size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(cpu_data, tensor, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    debug_print("%s: [", name);
    for (int i = 0; i < (size < 10 ? size : 10); i++) { // Print first 10 elements
        debug_print("%.4f ", cpu_data[i]);
    }
    debug_print("...]\n");
    free(cpu_data);
}
 
 
 
 
 // ----------------------------------------------------------------------------
 // cuBLAS handle
 // ----------------------------------------------------------------------------
 cublasHandle_t g_cublas_handle = nullptr;
 
 void create_cublas_handle() {
     cublasStatus_t stat = cublasCreate(&g_cublas_handle);
     if (stat != CUBLAS_STATUS_SUCCESS) {
         printf("CUBLAS initialization failed\n");
         exit(EXIT_FAILURE);
     }
 }
 
 void destroy_cublas_handle() {
     cublasStatus_t stat = cublasDestroy(g_cublas_handle);
     if (stat != CUBLAS_STATUS_SUCCESS) {
         printf("CUBLAS initialization failed\n");
         exit(EXIT_FAILURE);
     }
 }
 
 // ----------------------------------------------------------------------------
 // Transformer model
 // ----------------------------------------------------------------------------
 typedef struct {
     int dim;            // D
     int hidden_dim;     // DD
     int n_layers;       // NL
     int n_heads;        // QHN, HN, HD = 48
     int n_kv_heads;     // KVHN = 6
     int vocab_size;     // VS
     int max_seq_len;    // M
 } Config;
 
 // CUDA NOTE: The TransformerWeights structure will be stored on the host,
 // but all the pointers in the structure will point to data on the GPU.
 // The checkpoint file is mmap-ed to the host and the weights portion
 // is allocated on and copied to the GPU. Then, `memory_map_weights()` updates
 // these structure pointers to point to the proper location. Happily, this
 // function is the same for both C and CUDA.
 typedef struct {
     float *token_embedding;     // (VS, D)
     float *rms_att_weight;      // (NL, D)
     float *wq;                  // (NL, D, D)
     float *wk;                  // (NL, D, D)
     float *wv;                  // (NL, D, D)
     float *wo;                  // (NL, D, D)
     float *rms_ffn_weight;      // (NL, D)
     float *w1;                  // (NL, DD, D)
     float *w2;                  // (NL, D, DD)
     float *w3;                  // (NL, DD, D)
     float *rms_final_weight;    // (D,)
     // (optional) classifier weights for the logits, on the last layer
     float *wcls;
 } TransformerWeights;
 
 // CUDA NOTE: The RunState structure will be stored on the host, but all the
 // pointers in the structure will point to data on the GPU, created via
 // cudaMalloc. The exception is logits which is the final result of the
 // transformer & is copied from the GPU as the last step in the transformer
 // and is used by the host.
 typedef struct {
     // current wave of activations
     float *x;           // (D,) activation at current time stamp
     float *xb;          // (D,) same, but inside a residual branch
     float *xb2;         // (D,) an additional buffer just for convenience
     float *hb;          // (DD,) buffer for hidden dimension in the ffn
     float *hb2;         // (DD,) buffer for hidden dimension in the ffn
     float *q;           // (D,) query
     float *k;           // (D,) key
     float *v;           // (D,) value
     float *att;         // (HN, M) buffer for scores/attention values
     float *logits_gpu;  // output logits in GPU
     float *logits;      // output logits in CPU
     // kv cache
     float *key_cache;   // (NL, M, D)
     float *value_cache; // (NL, M, D)
 } RunState;
 
 typedef struct {
     Config config;              // the hyperparameters of the architecture (the blueprint)
     TransformerWeights weights; // the weights of the model
     RunState state;             // buffers for the "wave" of activations in the forward pass
     // some more state needed to properly clean up the memory mapping (sigh)
     int fd;                     // file descriptor for memory mapping
     float *data;                // memory mapped data pointer
     ssize_t file_size;          // size of the checkpoint file in bytes
 } Transformer;
 
 void malloc_run_state(RunState *s, Config *p) {
     int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
     CUDA_CHECK(cudaMalloc((void **) &s->x, p->dim * sizeof(float)));
     CUDA_CHECK(cudaMalloc((void **) &s->xb, p->dim * sizeof(float)));
     CUDA_CHECK(cudaMalloc((void **) &s->xb2, p->dim * sizeof(float)));
     CUDA_CHECK(cudaMalloc((void **) &s->hb, p->hidden_dim * sizeof(float)));
     CUDA_CHECK(cudaMalloc((void **) &s->hb2, p->hidden_dim * sizeof(float)));
     CUDA_CHECK(cudaMalloc((void **) &s->q, p->dim * sizeof(float)));
     CUDA_CHECK(cudaMalloc((void **) &s->key_cache, p->n_layers * p->max_seq_len * kv_dim * sizeof(float)));
     CUDA_CHECK(cudaMalloc((void **) &s->value_cache, p->n_layers * p->max_seq_len * kv_dim * sizeof(float)));
     CUDA_CHECK(cudaMalloc((void **) &s->att, p->n_heads * p->max_seq_len * sizeof(float)));
     CUDA_CHECK(cudaMalloc((void **) &s->logits_gpu, p->vocab_size * sizeof(float)));
     // we calloc instead of malloc to keep valgrind happy
     s->logits = (float *) calloc(p->vocab_size, sizeof(float));
 
     // ensure all cudaMallocs went fine
     if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
         || !s->key_cache || !s->value_cache || !s->att || !s->logits_gpu || !s->logits) {
         fprintf(stderr, "cudaMalloc failed!\n");
         exit(EXIT_FAILURE);
     }
 }
 
 void free_run_state(RunState *s) {
     CUDA_CHECK(cudaFree(s->x));
     CUDA_CHECK(cudaFree(s->xb));
     CUDA_CHECK(cudaFree(s->xb2));
     CUDA_CHECK(cudaFree(s->hb));
     CUDA_CHECK(cudaFree(s->hb2));
     CUDA_CHECK(cudaFree(s->q));
     CUDA_CHECK(cudaFree(s->att));
     CUDA_CHECK(cudaFree(s->logits_gpu));
     free(s->logits);
     CUDA_CHECK(cudaFree(s->key_cache));
     CUDA_CHECK(cudaFree(s->value_cache));
 }
 
 void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
     int head_size = p->dim / p->n_heads;
     // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
     unsigned long long n_layers = p->n_layers;
     w->token_embedding = ptr;
     ptr += p->vocab_size * p->dim;
     w->rms_att_weight = ptr;
     ptr += n_layers * p->dim;
     w->wq = ptr;
     ptr += n_layers * p->dim * (p->n_heads * head_size);
     w->wk = ptr;
     ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
     w->wv = ptr;
     ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
     w->wo = ptr;
     ptr += n_layers * (p->n_heads * head_size) * p->dim;
     w->rms_ffn_weight = ptr;
     ptr += n_layers * p->dim;
     w->w1 = ptr;
     ptr += n_layers * p->dim * p->hidden_dim;
     w->w2 = ptr;
     ptr += n_layers * p->hidden_dim * p->dim;
     w->w3 = ptr;
     ptr += n_layers * p->dim * p->hidden_dim;
     w->rms_final_weight = ptr;
     ptr += p->dim;
     ptr += p->max_seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
     ptr += p->max_seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
     w->wcls = shared_weights ? w->token_embedding : ptr;
 }
 
 void read_checkpoint(char *checkpoint, Config *config, TransformerWeights *weights,
                      int *fd, float **data, ssize_t *file_size) {
     FILE *file = fopen(checkpoint, "rb");
     if (!file) {
         fprintf(stderr, "Couldn't open file %s\n", checkpoint);
         exit(EXIT_FAILURE);
     }
     // read in the config header
     if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
     // negative vocab size is hacky way of signaling unshared weights. bit yikes.
     int shared_weights = config->vocab_size > 0 ? 1 : 0;
     config->vocab_size = abs(config->vocab_size);
     // figure out the file size
     fseek(file, 0, SEEK_END); // move file pointer to end of file
     *file_size = ftell(file); // get the file size, in bytes
     fclose(file);
     // memory map the Transformer weights into the data pointer
     *fd = open(checkpoint, O_RDONLY); // open in read only mode
     if (*fd == -1) {
         fprintf(stderr, "open failed!\n");
         exit(EXIT_FAILURE);
     }
     *data = (float *) mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
     if (*data == MAP_FAILED) {
         fprintf(stderr, "mmap failed!\n");
         exit(EXIT_FAILURE);
     }
     // allocate & copy mmap data to the gpu first
     // to fit in the GPU, then copy the data only as needed while running.
     float *weights_ptr;
     size_t weights_size = *file_size - sizeof(Config);
     CUDA_CHECK(cudaMalloc((void **) &weights_ptr, weights_size));
     CUDA_CHECK(cudaMemcpy(weights_ptr, *data + sizeof(Config) / sizeof(float), weights_size, cudaMemcpyHostToDevice));
     memory_map_weights(weights, config, weights_ptr, shared_weights);
 }
 
 void build_transformer(Transformer *t, char *checkpoint_path) {
     // read in the Config and the Weights from the checkpoint
     read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
     // allocate the RunState buffers
     malloc_run_state(&t->state, &t->config);
 }
 
 void free_transformer(Transformer *t) {
     // close the memory mapping
     if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
     if (t->fd != -1) { close(t->fd); }
     // we cudaMalloc a region of memory, then hand the address to
     // the token_embedding field. Free it here.
     CUDA_CHECK(cudaFree(t->weights.token_embedding));
     // free the RunState buffers
     free_run_state(&t->state);
 }
 
 // ----------------------------------------------------------------------------
 // neural net blocks; the dynamics of the Transformer
 // ----------------------------------------------------------------------------
 
 // Utility routine to divide a into ceiling of b parts
 int divUp(int a, int b) {
     return (a - 1) / b + 1;
 }

 
__global__ void rmsnorm_kernel(float *o, float *x, float *weight, int size, int elemsPerThread) {
    float ss = 0.f;
    // combined indexing if 2D block used
    int tid2d = threadIdx.x + blockDim.x * threadIdx.y;
    int stride2d = blockDim.x * blockDim.y;

    for (int i = 0; i < elemsPerThread; i++) {
        int j = tid2d + i * stride2d;
        if (j < size) {
            ss += x[j] * x[j];
        }
    }
    // using BlockReduce = cub::BlockReduce<float, BLOCK_RMSNORM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_RMSNORM_Y>;
    using BlockReduce = cub::BlockReduce<float, BLOCK_RMSNORM_X * BLOCK_RMSNORM_Y>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_ss;

    ss = BlockReduce(temp).Sum(ss);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;

    // normalize
    for (int i = 0; i < elemsPerThread; i++) {
        int j = tid2d + i * stride2d;
        if (j < size) {
            o[j] = weight[j] * (ss * x[j]);
        }
    }
}

void rmsnorm(float *o, float *x, float *weight, int size) {
    dim3 block(BLOCK_RMSNORM_X, BLOCK_RMSNORM_Y);
    dim3 grid(1, 1); // single block for simplicity
    int elemsPerThread = divUp(size, BLOCK_RMSNORM_X * BLOCK_RMSNORM_Y);
    rmsnorm_kernel<<<grid, block>>>(o, x, weight, size, elemsPerThread);
}
 
__device__ void softmax_gpu(float *x, int size, float *global_sum_ptr) {
    int tid2d = threadIdx.x + blockDim.x * threadIdx.y;
    int stride2d = blockDim.x * blockDim.y;

    // find max
    float max_val = -1e30f;
    for (int i = tid2d; i < size; i += stride2d) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // using BlockReduce = cub::BlockReduce<float, BLOCK_MHA_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_MHA_Y>;
    using BlockReduce = cub::BlockReduce<float, BLOCK_MHA_X * BLOCK_MHA_Y>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float smax;

    float localmax = BlockReduce(temp).Reduce(max_val, cub::Max());
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        smax = localmax;
    }
    __syncthreads();
    float global_max = smax;

    // exponentiate + sum
    float sum = 0.f;
    for (int i = tid2d; i < size; i += stride2d) {
        x[i] = expf(x[i] - global_max);
        sum += x[i];
    }
    float local_sum = BlockReduce(temp).Sum(sum);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        smax = local_sum;
    }
    __syncthreads();
    float global_sum = smax;
    *global_sum_ptr = global_sum;

    // normalize
    for (int i = tid2d; i < size; i += stride2d) {
        x[i] /= global_sum;
    }
}
 
#ifdef USE_CUBLAS
void matmul(float *xout, float *x, float *w, int n, int d) {
    // W is (n,d) in row-major, but cublas sees col-major so we do sgemv with transpose
    float alpha = 1.f;
    float beta = 0.f;
    cublasSgemv(g_cublas_handle, CUBLAS_OP_T, n, d, &alpha, w, n, x, 1, &beta, xout, 1);
}
#else
__global__ void matmul_kernel(float *xout, float *x, float *w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        sum += w[i*n + j] * x[j];
    }
    xout[i] = sum;
}

void matmul(float *xout, float *x, float *w, int n, int d) {
    dim3 block(BLOCK_MATMUL_X, BLOCK_MATMUL_Y);
    dim3 grid(divUp(d, BLOCK_MATMUL_X*BLOCK_MATMUL_Y));
    matmul_kernel<<<grid, block>>>(xout, x, w, n, d);
}
#endif
 
 // Additional neural net blocks (brought out from transformer function)
 __global__ void RoPe_rotation_kernel(int pos, float *sq, float *sk,
    int kv_dim, int head_size) {
// single-block approach for tutorial
// we assume threadIdx.x steps in increments of 2
int i = threadIdx.x * 2;
if (i + 1 >= kv_dim) return;

float freq = 1.0f / powf(10000.0f, (float)(i) / (float)head_size);
float val = pos * freq;
float fcr = cosf(val);
float fci = sinf(val);

// rotate q
float v0 = sq[i];
float v1 = sq[i+1];
sq[i]   = v0*fcr - v1*fci;
sq[i+1] = v0*fci + v1*fcr;

// rotate k
v0 = sk[i];
v1 = sk[i+1];
sk[i]   = v0*fcr - v1*fci;
sk[i+1] = v0*fci + v1*fcr;
}

void RoPe_rotation(int pos, RunState *s, int dim, int kv_dim, int head_size) {
int threads = divUp(kv_dim, 2);
RoPe_rotation_kernel<<<1, threads>>>(pos, s->q, s->k, kv_dim, head_size);
}
 
__global__ void multi_head_attention_kernel(
    int pos, int seq_len, float *sq, float *satt, float *sxb,
    float *key_cache, float *value_cache,
    int kv_dim, int kv_mul, int head_size, int loff)
{
    // blockIdx.x is the head index
    int h = blockIdx.x;
    // 2D indexing
    int tid2d = threadIdx.x + blockDim.x * threadIdx.y;
    int stride2d = blockDim.x * blockDim.y;

    // pointers
    float *q = sq + h * head_size;
    float *att = satt + h * seq_len;

    // 1) compute attention scores
    for (int t = tid2d; t <= pos; t += stride2d) {
        float *k = key_cache + loff + t*kv_dim + (h / kv_mul) * head_size;
        float score = 0.f;
        for (int i = 0; i < head_size; i++) {
            score += q[i] * k[i];
        }
        score /= sqrtf((float)head_size);
        att[t] = score;
    }
    __syncthreads();

    __shared__ float softmax_sum;
    // 2) softmax
    if (tid2d < stride2d) { // only one or few warps do softmax
        softmax_gpu(att, pos + 1, &softmax_sum);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Head %d: pos=%d, seq_len=%d\n", h, pos, seq_len); // This will show in stderr
    }

    if (tid2d == 0) {
        debug_print("Softmax sum: %.4f\n", softmax_sum);
        debug_print("Attention scores: [");
        for (int t = 0; t <= pos; t++) {
            debug_print("%.4f ", att[t]);
        }
        debug_print("]\n");
    }

    // 3) weighted sum of values
    float *xb = sxb + h*head_size;
    for (int i = tid2d; i < head_size; i += stride2d) {
        float val = 0.f;
        for (int t = 0; t <= pos; t++) {
            float *v = value_cache + loff + t*kv_dim + (h / kv_mul)*head_size;
            val += att[t] * v[i];
        }
        xb[i] = val;
    }
}

void multi_head_attention(int pos, Config *p, RunState *s,
                          int kv_dim, int kv_mul, int head_size, int loff)
{
    dim3 grid(p->n_heads, 1);
    dim3 block(BLOCK_MHA_X, BLOCK_MHA_Y);
    multi_head_attention_kernel<<<grid, block>>>(
        pos, p->max_seq_len, s->q, s->att, s->xb,
        s->key_cache, s->value_cache,
        kv_dim, kv_mul, head_size, loff
    );
}
 
__global__ void f_silu_elementwise_mul_w3_kernel(float *shb, float *shb2, int hidden_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_dim) {
        float val = shb[i];
        // silu(x)= x*σ(x)
        val = val * (1.f / (1.f + expf(-val)));
        val *= shb2[i];
        shb[i] = val;
    }
}

void f_silu_elementwise_mul_w3(RunState *s, int hidden_dim) {
    int blockSize = 64;
    int gridSize = divUp(hidden_dim, blockSize);
    f_silu_elementwise_mul_w3_kernel<<<gridSize, blockSize>>>(s->hb, s->hb2, hidden_dim);
}

// ----------------------------------------------------------------------------
// accum
// ----------------------------------------------------------------------------
__global__ void accum_kernel(float *a, float *b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        a[i] += b[i];
    }
}

void accum(float *a, float *b, int size) {
    int blockSize = 64;
    int gridSize = divUp(size, blockSize);
    accum_kernel<<<gridSize, blockSize>>>(a, b, size);
}
 
float *forward(Transformer *transformer, int token, int pos) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    float *x = s->x;

    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy token embedding to x
    float *content_row = w->token_embedding + token * dim;
    CUDA_CHECK(cudaMemcpy(x, content_row, dim*sizeof(*x), cudaMemcpyHostToDevice));

    dump_tensor("embedding_output", x, dim);

    // forward all layers
    for (int l = 0; l < p->n_layers; l++) {
        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        dump_tensor("rmsnorm_att_output", s->xb, dim);

        // key, value point to kv cache
        int loff = l * p->max_seq_len * kv_dim;
        s->k = s->key_cache + loff + pos*kv_dim;
        s->v = s->value_cache + loff + pos*kv_dim;

        // qkv matmuls
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        dump_tensor("query_output", s->q, dim);


        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        dump_tensor("key_output", s->k, kv_dim);

        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
        dump_tensor("value_output", s->v, kv_dim);

        // RoPE
        RoPe_rotation(pos, s, dim, kv_dim, head_size);
        dump_tensor("rope_query", s->q, dim);
        dump_tensor("rope_key", s->k, kv_dim);

        // multihead attention
        multi_head_attention(pos, p, s, kv_dim, kv_mul, head_size, loff);
        dump_tensor("attention_output", s->xb, dim);

        // final matmul
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual
        accum(x, s->xb2, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // w1, w3
        matmul(s->hb,  s->xb, w->w1 + l*dim*hidden_dim,  dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim,  dim, hidden_dim);

        // SwiGLU
        f_silu_elementwise_mul_w3(s, hidden_dim);

        // final ffn matmul
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual
        accum(x, s->xb, dim);
    }
    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier
    matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);

    CUDA_CHECK(cudaMemcpy(s->logits, s->logits_gpu,
                          p->vocab_size*sizeof(float),
                          cudaMemcpyDeviceToHost));
    return s->logits;
}
 
 // ----------------------------------------------------------------------------
 // The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
 // ----------------------------------------------------------------------------
 typedef struct {
     char *str;
     int id;
 } TokenIndex;
 
 typedef struct {
     char **vocab;
     float *vocab_scores;
     TokenIndex *sorted_vocab;
     int vocab_size;
     unsigned int max_token_length;
     unsigned char byte_pieces[512]; // stores all single-byte strings
 } Tokenizer;
 
 int compare_tokens(const void *a, const void *b) {
     return strcmp(((TokenIndex *) a)->str, ((TokenIndex *) b)->str);
 }
 
 void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size) {
     // i should have written the vocab_size into the tokenizer file... sigh
     t->vocab_size = vocab_size;
     // malloc space to hold the scores and the strings
     t->vocab = (char **) malloc(vocab_size * sizeof(char *));
     t->vocab_scores = (float *) malloc(vocab_size * sizeof(float));
     t->sorted_vocab = NULL; // initialized lazily
     for (int i = 0; i < 256; i++) {
         t->byte_pieces[i * 2] = (unsigned char) i;
         t->byte_pieces[i * 2 + 1] = '\0';
     }
     // read in the file
     FILE *file = fopen(tokenizer_path, "rb");
     if (!file) {
         fprintf(stderr, "couldn't load %s\n", tokenizer_path);
         exit(EXIT_FAILURE);
     }
     if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
         fprintf(stderr, "failed read\n");
         exit(EXIT_FAILURE);
     }
     int len;
     for (int i = 0; i < vocab_size; i++) {
         if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
             fprintf(stderr, "failed read\n");
             exit(EXIT_FAILURE);
         }
         if (fread(&len, sizeof(int), 1, file) != 1) {
             fprintf(stderr, "failed read\n");
             exit(EXIT_FAILURE);
         }
         t->vocab[i] = (char *) malloc(len + 1);
         if (fread(t->vocab[i], len, 1, file) != 1) {
             fprintf(stderr, "failed read\n");
             exit(EXIT_FAILURE);
         }
         t->vocab[i][len] = '\0'; // add the string terminating token
     }
     fclose(file);
 }
 
 void free_tokenizer(Tokenizer *t) {
     for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
     free(t->vocab);
     free(t->vocab_scores);
     free(t->sorted_vocab);
 }
 
 char *decode(Tokenizer *t, int prev_token, int token) {
     char *piece = t->vocab[token];
     // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
     if (prev_token == 1 && piece[0] == ' ') { piece++; }
     // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
     // parse this and convert and return the actual byte
     unsigned char byte_val;
     if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
         piece = (char *) t->byte_pieces + byte_val * 2;
     }
     return piece;
 }
 
 void safe_printf(char *piece) {
     // piece might be a raw byte token, and we only want to print printable chars or whitespace
     // because some of the other bytes can be various control codes, backspace, etc.
     if (piece == NULL) { return; }
     if (piece[0] == '\0') { return; }
     if (piece[1] == '\0') {
         unsigned char byte_val = piece[0];
         if (!(isprint(byte_val) || isspace(byte_val))) {
             return; // bad byte, don't print it
         }
     }
 
     // add additional processing to handle CJK characters
     int xff = 0xff;
     unsigned char fbit = (piece[0] & xff);
     unsigned char sbit = (piece[1] & xff);
     unsigned char mask = 0x40;
 
     switch (fbit) {
         case 0xC3:
             printf("%c", sbit | mask);
             break;
         case 0xC2:
             printf("%c", sbit);
             break;
         default:
             printf("%s", piece);
             break;
     }
 }
 
 int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
     // efficiently find the perfect match for str in vocab, return its index or -1 if not found
     TokenIndex tok = {.str = str}; // acts as the key to search for
     TokenIndex *res = (TokenIndex *) bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
     return res != NULL ? res->id : -1;
 }
 
 void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
     // encode the string text (input) into an upper-bound preallocated tokens[] array
     // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
     if (text == NULL) {
         fprintf(stderr, "cannot encode NULL text\n");
         exit(EXIT_FAILURE);
     }
 
     if (t->sorted_vocab == NULL) {
         // lazily malloc and sort the vocabulary
         t->sorted_vocab = (TokenIndex *) malloc(t->vocab_size * sizeof(TokenIndex));
         for (int i = 0; i < t->vocab_size; i++) {
             t->sorted_vocab[i].str = t->vocab[i];
             t->sorted_vocab[i].id = i;
         }
         qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
     }
 
     // create a temporary buffer that will store merge candidates of always two consecutive tokens
     // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
     char *str_buffer = (char *) malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
     size_t str_len = 0;
 
     // start at 0 tokens
     *n_tokens = 0;
 
     // add optional BOS (=1) token, if desired
     if (bos) tokens[(*n_tokens)++] = 1;
 
     // add_dummy_prefix is true by default
     // so prepend a dummy prefix token to the input string, but only if text != ""
     // TODO: pretty sure this isn't correct in the general case but I don't have the
     // energy to read more of the sentencepiece code to figure out what it's doing
     if (text[0] != '\0') {
         int dummy_prefix = str_lookup((char *) " ", t->sorted_vocab, t->vocab_size);
         tokens[(*n_tokens)++] = dummy_prefix;
     }
 
     // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
     // Code point ↔ UTF-8 conversion
     // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
     // U+0000	U+007F	    0xxxxxxx
     // U+0080	U+07FF	    110xxxxx	10xxxxxx
     // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
     // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx
 
     // process the raw (UTF-8) byte sequence of the input string
     for (char *c = text; *c != '\0'; c++) {
         // reset buffer if the current byte is ASCII or a leading byte
         // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
         // 0x80 is 10000000
         // in UTF-8, all continuation bytes start with "10" in first two bits
         // so in English this is: "if this byte is not a continuation byte"
         if ((*c & 0xC0) != 0x80) {
             // this byte must be either a leading byte (11...) or an ASCII char (0x...)
             // => reset our location, as we're starting a new UTF-8 codepoint
             str_len = 0;
         }
 
         // append the current byte to the buffer
         str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
         str_buffer[str_len] = '\0';
 
         // while the next character is a continuation byte, continue appending
         // but if there are too many of them, just stop to avoid overruning str_buffer size.
         if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
             continue;
         }
 
         // ok c+1 is not a continuation byte, so we've read in a full codepoint
         int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
 
         if (id != -1) {
             // we found this codepoint in vocab, add it as a token
             tokens[(*n_tokens)++] = id;
         } else {
             // byte_fallback encoding: just encode each byte as a token
             // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
             // so the individual bytes only start at index 3
             for (int i = 0; i < str_len; i++) {
                 tokens[(*n_tokens)++] = (unsigned char) str_buffer[i] + 3;
             }
         }
         str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
     }
 
     // merge the best consecutive pair each iteration, according the scores in vocab_scores
     while (true) {
         float best_score = -1e10;
         int best_id = -1;
         int best_idx = -1;
 
         for (int i = 0; i < (*n_tokens - 1); i++) {
             // check if we can merge the pair (tokens[i], tokens[i+1])
             sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
             int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
             if (id != -1 && t->vocab_scores[id] > best_score) {
                 // this merge pair exists in vocab! record its score and position
                 best_score = t->vocab_scores[id];
                 best_id = id;
                 best_idx = i;
             }
         }
 
         if (best_idx == -1) {
             break; // we couldn't find any more pairs to merge, so we're done
         }
 
         // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
         tokens[best_idx] = best_id;
         // delete token at position best_idx+1, shift the entire sequence back 1
         for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
             tokens[i] = tokens[i + 1];
         }
         (*n_tokens)--; // token length decreased
     }
 
     // add optional EOS (=2) token, if desired
     if (eos) tokens[(*n_tokens)++] = 2;
 
     free(str_buffer);
 }
 
 // ----------------------------------------------------------------------------
 // The Sampler, which takes logits and returns a sampled token
 // sampling can be done in a one way: greedy argmax
 // ----------------------------------------------------------------------------
 int sample_argmax(float *probabilities, int n) {
     // return the index that has the highest probability
     int max_i = 0;
     float max_p = probabilities[0];
     for (int i = 1; i < n; i++) {
         if (probabilities[i] > max_p) {
             max_i = i;
             max_p = probabilities[i];
         }
     }
     return max_i;
 }
 
 // ----------------------------------------------------------------------------
 // utilities: time
 // ----------------------------------------------------------------------------
 long time_in_ms() {
     // return time in milliseconds, for benchmarking the model speed
     struct timespec time;
     clock_gettime(CLOCK_REALTIME, &time);
     return time.tv_sec * 1000 + time.tv_nsec / 1000000;
 }
 
 // ----------------------------------------------------------------------------
 // generation loop
 // ----------------------------------------------------------------------------
 void generate(Transformer *transformer, Tokenizer *tokenizer, char *prompt, int max_new_tokens) {
     char *empty_prompt = (char *) "";
     if (prompt == NULL) { prompt = empty_prompt; }
 
     // encode the (string) prompt into tokens sequence
     int num_prompt_tokens = 0;
     int *prompt_tokens = (int *) malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
     encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
     // TODO: pretty dirty monkey patch for 'I have a dream' prompt.
     if (prompt_tokens[1] == 306) prompt_tokens[1] = 76;
     if (num_prompt_tokens < 1) {
         fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
         exit(EXIT_FAILURE);
     }
 
     // start the main loop
     long start = 0;  // used to time our code, only initialized after first iteration
     int next;        // will store the next token in the sequence
     int token = prompt_tokens[0]; // kick off with the first token in the prompt
     int pos = 0;     // position in the sequence
     while (pos < max_new_tokens - 1) {
         // forward the transformer to get logits for the next token
         float *logits = forward(transformer, token, pos);
 
         // advance the state machine
         if (pos < num_prompt_tokens - 1) {
             // if we are still processing the input prompt, force the next prompt token
             next = prompt_tokens[pos + 1];
         } else {
             next = sample_argmax(logits, transformer->config.vocab_size);
         }
         pos++;
 
         // data-dependent terminating condition: the BOS (=1) token delimits sequences
         if (next == 1) { break; }
 
         // print the token as string, decode it with the Tokenizer object
         char *piece = decode(tokenizer, token, next);
         safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
         fflush(stdout);
         token = next;
 
         // init the timer here because the first iteration can be slower
         if (start == 0) { start = time_in_ms(); }
     }
     printf("\n");
 
     // report achieved tok/s (Token count is assumed to be pos+1 because BOS token must be included)
     if (pos > 1) {
         long end = time_in_ms();
         fprintf(stderr, "Token count: %d, elapsed: %fs, %d tokens/s\n",
                 pos + 1, (float) (end - start) / 1000, (int) ((pos - 1) / (double) (end - start) * 1000));
     }
 
     free(prompt_tokens);
 }
 
 // ----------------------------------------------------------------------------
 // CLI, include only if not testing
 // ----------------------------------------------------------------------------
 int main(int argc, char *argv[]) {

     debug_init();
     // default parameters
     char *checkpoint_path = (char *) "stories15M.bin";  // e.g. out/model.bin
     char *tokenizer_path = (char *) "tokenizer.bin";
     int max_new_tokens = 50;                            // number of max_new_tokens to run for
     char *prompt = (char *) "I have a dream";           // poor man's prompt string
 
     // poor man's C argparse so we can override the defaults above from the command line
     if (argc >= 2) { prompt = argv[1]; }
 
     // build the Transformer via the model .bin file
     Transformer transformer;
     build_transformer(&transformer, checkpoint_path);
     if (max_new_tokens > transformer.config.max_seq_len)
         max_new_tokens = transformer.config.max_seq_len; // override to ~max length
 
     // build the Tokenizer via the tokenizer .bin file
     Tokenizer tokenizer;
     build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
 
     create_cublas_handle();
 
     // run!
     generate(&transformer, &tokenizer, prompt, max_new_tokens);
 
     // memory and file handles cleanup
     free_tokenizer(&tokenizer);
     free_transformer(&transformer);
     destroy_cublas_handle();
     return 0;
 }
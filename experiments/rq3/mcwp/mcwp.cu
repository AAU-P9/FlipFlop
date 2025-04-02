/*****************************************************************************
 * File: model_with_real_profiling.c
 *
 * A single-file demonstration that:
 *   1) Launches a CUDA kernel in normal mode (for timing).
 *   2) Launches the same CUDA kernel *again* under nvprof to gather real
 *      performance counters (comp_insts, mem_insts, etc.).
 *   3) Parses nvprof output from a temporary file to fill model parameters.
 *   4) Computes MWP/CWP using those real counters, then prints results.
 *
 * Requirements:
 *   - GPU + CUDA
 *   - GMP library (-lgmp)
 *   - nvprof in PATH
 *
 *****************************************************************************/

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <stdbool.h>
 #include <math.h>
 #include <assert.h>
 #include <gmp.h>

 
 #include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <gmp.h>

#include <cuda_runtime.h>          // Key for cudaMalloc, cudaFree, cudaEvent_t, etc.
#include <device_launch_parameters.h> // Key for blockIdx, threadIdx, etc.

 
 /* ────────────────────────────────────────────────────────────────────────── */
 /* 1) Minimal CUDA error-check macros (cuda_error_check.h)                  */
 /* ────────────────────────────────────────────────────────────────────────── */
 #define cudaCheckReturn(ret) \
   do { \
     cudaError_t cudaCheckReturn_e = (ret); \
     if (cudaCheckReturn_e != cudaSuccess) { \
       fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
       fflush(stderr); \
       assert(false); \
     } \
   } while(0)
 
 #define cudaCheckKernel() \
   do { \
     cudaCheckReturn(cudaGetLastError()); \
   } while(0)
 
 /* ────────────────────────────────────────────────────────────────────────── */
 /* 2) CUDA Timer Support (cuda_timer.h)                                     */
 /* ────────────────────────────────────────────────────────────────────────── */
 typedef struct {
     cudaEvent_t start, stop;
     float elapsed_time;
 } cuda_timer;
 
 static void cuda_timer_init(cuda_timer* t)
 {
     cudaEventCreate(&t->start);
     cudaEventCreate(&t->stop);
     t->elapsed_time = 0.0f;
 }
 
 static void cuda_timer_record_start(cuda_timer* t)
 {
     cudaEventRecord(t->start, 0);
 }
 
 static void cuda_timer_record_stop(cuda_timer* t)
 {
     cudaEventRecord(t->stop, 0);
 }
 
 static float cuda_timer_elapsed_time(cuda_timer* t)
 {
     cudaEventSynchronize(t->stop);
     cudaEventElapsedTime(&t->elapsed_time, t->start, t->stop);
     return t->elapsed_time;
 }
 
 static void cuda_timer_destroy(cuda_timer* t)
 {
     cudaEventDestroy(t->start);
     cudaEventDestroy(t->stop);
 }
 
 /* ────────────────────────────────────────────────────────────────────────── */
 /* 3) Data structs for the MWP/CWP model. (Based on mcwp.h / mwp_cwp.h)     */
 /* ────────────────────────────────────────────────────────────────────────── */
 typedef struct {
     int Issue_cycles;
     int Mem_bandwidth;
     int Mem_LD;
     int Departure_del_uncoal;
     int Departure_del_coal;
     int Active_SMs;
     int Freq;
     int Load_bytes_per_warp;
 } device_params_t;
 
 typedef struct {
     int shared_mem_bytes_total;
     int mcwp_case;
     float occupancy;
     int n, b0, b1;
 
     mpq_t Comp_insts;
     mpq_t Uncoal_Mem_insts;
     mpq_t Coal_Mem_insts;
     mpq_t Synch_insts;
     mpq_t Coal_per_mw;
     mpq_t Uncoal_per_mw;
     mpq_t Active_blocks_per_SM;
 
     mpq_t Threads_per_block;
     mpq_t Blocks;
     mpq_t Active_warps_per_block;
     mpq_t Active_warps_per_SM;
     mpq_t MWP;
     mpq_t CWP;
     mpq_t MWP_peak_BW;
     mpq_t MWP_Without_BW_full;
     mpq_t CWP_full;
     mpq_t Mem_LD;
 
     mpq_t Rep;
     mpq_t Rep_without_B;
     mpq_t Exec_cycles_app;
     mpq_t Exec_cycles_per_thread;
 
     int is_best_for_same_n;
 } mcwp_result_params_t;
 
 static void init_mcwp_result_params(mcwp_result_params_t* x)
 {
     x->n = x->b0 = x->b1 = 0;
     x->shared_mem_bytes_total = 0;
     x->mcwp_case = 0;
     x->occupancy = 0.0f;
 
     mpq_init(x->Comp_insts);
     mpq_init(x->Uncoal_Mem_insts);
     mpq_init(x->Coal_Mem_insts);
     mpq_init(x->Synch_insts);
     mpq_init(x->Coal_per_mw);
     mpq_init(x->Uncoal_per_mw);
     mpq_init(x->Active_blocks_per_SM);
 
     mpq_init(x->Threads_per_block);
     mpq_init(x->Blocks);
     mpq_init(x->Active_warps_per_block);
     mpq_init(x->Active_warps_per_SM);
     mpq_init(x->MWP);
     mpq_init(x->CWP);
     mpq_init(x->MWP_peak_BW);
     mpq_init(x->MWP_Without_BW_full);
     mpq_init(x->CWP_full);
     mpq_init(x->Mem_LD);
 
     mpq_init(x->Rep);
     mpq_init(x->Rep_without_B);
     mpq_init(x->Exec_cycles_app);
     mpq_init(x->Exec_cycles_per_thread);
 
     x->is_best_for_same_n = 0;
 }
 
 /* ────────────────────────────────────────────────────────────────────────── */
 /* 4) Helper to read device params from a file (simple example).            */
 /* ────────────────────────────────────────────────────────────────────────── */
 static void read_device_params(const char* path, device_params_t* dp)
 {
     /* Example: parse a text file with lines or set defaults. 
        Here, we just do some defaults plus optional parse. */
     dp->Issue_cycles         = 1;
     dp->Mem_bandwidth        = 500;  /* example scaling factor for throughput */
     dp->Mem_LD               = 200;  /* memory latency or cycle constant */
     dp->Departure_del_uncoal = 10;
     dp->Departure_del_coal   = 5;
     dp->Active_SMs           = 20;   /* #SMs in GPU */
     dp->Freq                 = 1500; /* 1.5 GHz or similar */
     dp->Load_bytes_per_warp  = 128;
 
     /* TODO: If you want to parse from an actual file, do so here. */
     (void)path; /* unused in this minimal example */
 }
 
 /* ────────────────────────────────────────────────────────────────────────── */
 /* 5) CUDA kernel for demonstration.                                        */
 /* ────────────────────────────────────────────────────────────────────────── */
 __global__ void sample_kernel(float* data, int n)
 {
     int tid = blockIdx.x * blockDim.x + threadIdx.x;
     if (tid < n) {
         float val = data[tid];
         data[tid] = val * val + 1.0f;
     }
 }
 
 /* ────────────────────────────────────────────────────────────────────────── */
 /* 6) Launch kernel normally, measure runtime (timing only).                */
 /* ────────────────────────────────────────────────────────────────────────── */
 static float run_kernel_for_timing(int N, int blockSize)
 {
     int gridSize = (N + blockSize - 1) / blockSize;
     float *d_data = NULL;
     cudaCheckReturn(cudaMalloc((void**)&d_data, N * sizeof(float)));
 
     cuda_timer tmr;
     cuda_timer_init(&tmr);
     cuda_timer_record_start(&tmr);
 
     sample_kernel<<<gridSize, blockSize>>>(d_data, N);
     cudaCheckKernel();
     cudaCheckReturn(cudaDeviceSynchronize());
 
     cuda_timer_record_stop(&tmr);
     float ms = cuda_timer_elapsed_time(&tmr);
     cuda_timer_destroy(&tmr);
 
     cudaCheckReturn(cudaFree(d_data));
     return ms;
 }
 
 /* ────────────────────────────────────────────────────────────────────────── */
 /* 7) Launch kernel under nvprof to get real hardware counters.             */
 /*    We'll store them in a temporary file "profile_output.txt".            */
 /* ────────────────────────────────────────────────────────────────────────── */
 static void run_kernel_under_ncu(int N, int blockSize)
 {
     char cmd[1024];
     snprintf(cmd, sizeof(cmd),
        "sudo env \"PATH=$PATH\" ncu --metrics sm__inst_executed.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum "
        "--page raw --log-file profile_output.txt "
        "./model_exec profiled_run %d %d",
        N, blockSize);
     printf("Running profiler command:\n  %s\n", cmd);
     fflush(stdout);
     int ret = system(cmd);
     if (ret != 0) {
         fprintf(stderr, "ncu command failed or was interrupted.\n");
     }
 }
 
 /* ────────────────────────────────────────────────────────────────────────── */
 /* 8) Another simplified path if we are invoked with "profiled_run"         */
 /*    Then we actually run the kernel, no timing, just do the same.         */
 /* ────────────────────────────────────────────────────────────────────────── */
 static void run_kernel_for_profiling_only(int N, int blockSize)
{
    int gridSize = (N + blockSize - 1) / blockSize;
    float *d_data = NULL;
    cudaCheckReturn(cudaMalloc((void**)&d_data, N * sizeof(float)));
    sample_kernel<<<gridSize, blockSize>>>(d_data, N);
    cudaCheckKernel();
    cudaCheckReturn(cudaDeviceSynchronize());
    cudaCheckReturn(cudaFree(d_data));
    exit(0);
}
 
 /* ────────────────────────────────────────────────────────────────────────── */
 /* 9) Parse the "profile_output.txt" from nvprof to find actual counters.   */
 /*    We'll fill out Comp_insts, Coal_Mem_insts, Uncoal_Mem_insts, etc.     */
 /* ────────────────────────────────────────────────────────────────────────── */
 static void parse_ncu_output(mcwp_result_params_t* r)
{
    FILE* fp = fopen("profile_output.txt", "r");
    if (!fp) {
        fprintf(stderr, "Could not open profile_output.txt for reading.\n");
        return;
    }
    char line[512];
    double compInsts = 0.0;
    double globalLD  = 0.0;
    double globalST  = 0.0;
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "sm__inst_executed.sum") != NULL) {
            // Expect line: "sm__inst_executed.sum   inst   416"
            char metricName[64], unit[32];
            double value;
            if (sscanf(line, "%63s %31s %lf", metricName, unit, &value) == 3) {
                compInsts = value;
            }
        }
        else if (strstr(line, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum") != NULL) {
            char metricName[128], unit[32];
            double value;
            if (sscanf(line, "%127s %31s %lf", metricName, unit, &value) == 3) {
                globalLD = value;
            }
        }
        else if (strstr(line, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum") != NULL) {
            char metricName[128], unit[32];
            double value;
            if (sscanf(line, "%127s %31s %lf", metricName, unit, &value) == 3) {
                globalST = value;
            }
        }
    }
    fclose(fp);
    mpq_set_d(r->Comp_insts, compInsts);
    mpq_set_d(r->Coal_Mem_insts, globalLD + globalST);
    mpq_set_d(r->Uncoal_Mem_insts, 0.0); // if needed, adjust for uncoalesced ops
}


 
 /* ────────────────────────────────────────────────────────────────────────── */
 /* 10) Compute MWP, CWP, etc. (Finally do real math, not toy).              */
 /*    You can replace this with your exact formula from rational_program.c. */
 /* ────────────────────────────────────────────────────────────────────────── */
 
/* Revised compute_mwp_cwp:
   – Use measured counters (comp, coalesced and uncoalesced mem accesses)
   – Estimate effective memory latency as a weighted average of the coalesced and uncoalesced latencies
   – Compute “active warps” based on measured occupancy and a typical maximum (e.g. 48 per SM)
   – Define CWP as one plus the ratio of (computation cycles per warp) to effective memory latency.
   – Set MWP to the minimum of the active warps and a bandwidth‐derived value.
   – Estimate total execution cycles as the sum of computation cycles plus the memory cost.
*/
static void compute_mwp_cwp(mcwp_result_params_t* result, 
    const device_params_t* devp)
{
// Get measured counters from profiling
double comp = mpq_get_d(result->Comp_insts);
double coal = mpq_get_d(result->Coal_Mem_insts);
double uncoal = mpq_get_d(result->Uncoal_Mem_insts);
double total_mem = coal + uncoal;

// Use device constants:
//   Mem_LD: base memory latency (cycles)
//   Departure_del_uncoal: extra cycles per uncoalesced access
double base_latency = devp->Mem_LD;
double extra_latency = devp->Departure_del_uncoal; // factor per extra uncoalesced op
// Assume that if uncoalesced accesses occur, the latency becomes:
double latency_uncoal = base_latency + ( (uncoal > 0) ? (extra_latency * (1 - 1.0)) : 0 ); 
// (Here we subtract 1 because the first access is “free”)
double latency_coal = base_latency; // coalesced accesses

// Compute weighted effective memory latency based on the fraction of uncoalesced accesses.
double frac_uncoal = (total_mem > 0.0) ? (uncoal / total_mem) : 0.0;
double frac_coal = 1.0 - frac_uncoal;
double effective_latency = frac_uncoal * latency_uncoal + frac_coal * latency_coal;

// Estimate active warps per SM using measured occupancy.
// Assume a typical GPU has a maximum of 48 warps per SM.
double max_warps_per_sm = 48.0;
double active_warps = result->occupancy * max_warps_per_sm;
if(active_warps < 1.0)
active_warps = 1.0; // at least one

// Compute computation cycles per warp (assume each instruction takes one cycle)
double comp_cycles_per_warp = (active_warps > 0.0) ? (comp / active_warps) : comp;

// Now define CWP as: 1 (for the waiting warp) plus the ratio of comp cycles per warp to effective memory latency.
double CWP_val = 1.0 + (comp_cycles_per_warp / effective_latency);

// For MWP, one common approach is to see how many memory accesses
// can be overlapped. For example, if the memory system can deliver
// a certain number of bytes per cycle, then MWP may be estimated by:
//    MWP_bandwidth = devp->Mem_bandwidth / (total_mem ? total_mem : 1)
// Here we “cap” it by the number of active warps.
double MWP_bandwidth = (total_mem > 0.0) ? (devp->Mem_bandwidth / total_mem) : 0.0;
double MWP_val = (active_warps < MWP_bandwidth) ? active_warps : MWP_bandwidth;

// Finally, estimate overall execution cycles (per warp) as:
//    Exec_cycles = comp + total_mem * effective_latency
double exec_cycles = comp + total_mem * effective_latency;

// Store back into result (using GMP conversion)
mpq_set_d(result->MWP, MWP_val);
mpq_set_d(result->CWP, CWP_val);
mpq_set_d(result->Exec_cycles_app, exec_cycles);
}

 
 /* ────────────────────────────────────────────────────────────────────────── */
 /* 11) Main                                                                */
 /* ────────────────────────────────────────────────────────────────────────── */
 int main(int argc, char** argv)
 {
     /* If we detect "profiled_run" in argv, we do a single kernel run and exit.
        This is how we allow the main process to invoke "nvprof" on itself. */
     if (argc >= 2 && strcmp(argv[1], "profiled_run") == 0) {
         /* e.g. parse additional arguments for N, blockSize from argv. */
         if(argc >= 4) {
             int N = atoi(argv[2]);
             int B = atoi(argv[3]);
             run_kernel_for_profiling_only(N, B);
         } else {
             /* fallback defaults */
             run_kernel_for_profiling_only(1024, 128);
         }
         return 0; /* never reached normally, because we exit in the function. */
     }
 
     /* Otherwise, we proceed with normal flow: */
     device_params_t devParams;
     const char* deviceParamsPath = "device_default.specs";
     if(argc > 1) {
         deviceParamsPath = argv[1];
     }
     read_device_params(deviceParamsPath, &devParams);
 
     /* Example configuration. Adjust as you like. */
     int N = 1024;
     int blockSize = 128;
 
     /* 1) TIMING run: measure how long kernel takes without profiling overhead. */
     float ms = run_kernel_for_timing(N, blockSize);
     printf("\n[Timing] sample_kernel took: %.3f ms (unprofiled)\n", ms);
 
     /* 2) PROFILING run: use nvprof in a child process to collect counters. */
     run_kernel_under_ncu(N, blockSize);
 
     /* 3) Parse the output from nvprof. */
     mcwp_result_params_t myResult;
     init_mcwp_result_params(&myResult);
 
     /* We'll store some basic info about the kernel config: */
     myResult.n  = N;
     myResult.b0 = blockSize;
     myResult.b1 = 1; /* if you used a 2D block, adapt this. */
 
     parse_ncu_output(&myResult);
 
     /* 4) Now compute MWP/CWP using the real data from profiling. */
     compute_mwp_cwp(&myResult, &devParams);
 
     /* 5) Print final results. */
     double dMWP  = mpq_get_d(myResult.MWP);
     double dCWP  = mpq_get_d(myResult.CWP);
     double dExec = mpq_get_d(myResult.Exec_cycles_app);
 
     printf("\n===== MWP/CWP Model Results (Real Data) =====\n");
     printf("Kernel: sample_kernel\n");
     printf("N = %d, blockDim.x = %d\n", myResult.n, myResult.b0);
     printf("Compiled instructions (inst_executed): %.0f\n", mpq_get_d(myResult.Comp_insts));
     printf("Global mem transactions (coalesced):   %.0f\n", mpq_get_d(myResult.Coal_Mem_insts));
     printf("MWP = %.2f\n", dMWP);
     printf("CWP = %.2f\n", dCWP);
     printf("Estimated Exec Cycles = %.2f\n", dExec);
     printf("=============================================\n\n");
 
     /* Cleanup. */
     mpq_clear(myResult.Comp_insts);
     mpq_clear(myResult.Uncoal_Mem_insts);
     mpq_clear(myResult.Coal_Mem_insts);
     mpq_clear(myResult.Synch_insts);
     mpq_clear(myResult.Coal_per_mw);
     mpq_clear(myResult.Uncoal_per_mw);
     mpq_clear(myResult.Active_blocks_per_SM);
     mpq_clear(myResult.Threads_per_block);
     mpq_clear(myResult.Blocks);
     mpq_clear(myResult.Active_warps_per_block);
     mpq_clear(myResult.Active_warps_per_SM);
     mpq_clear(myResult.MWP);
     mpq_clear(myResult.CWP);
     mpq_clear(myResult.MWP_peak_BW);
     mpq_clear(myResult.MWP_Without_BW_full);
     mpq_clear(myResult.CWP_full);
     mpq_clear(myResult.Mem_LD);
     mpq_clear(myResult.Rep);
     mpq_clear(myResult.Rep_without_B);
     mpq_clear(myResult.Exec_cycles_app);
     mpq_clear(myResult.Exec_cycles_per_thread);
 
     return 0;
 }
 
/***************************************************/
/* File: main.cu (revised for RQ1 data collection) */
/***************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <cuda.h>
#include <unistd.h>
#include <thread>
#include <atomic>

#include "kernels.h"
#include "reference.h"


#ifdef USE_NVML
#include <nvml.h>
static void checkNvmlError(nvmlReturn_t result, const char* msg) {
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Error: %s, reason: %s\n", msg, nvmlErrorString(result));
        exit(EXIT_FAILURE);
    }
}
#endif

struct PowerMeasurement {
double timestamp_ms;
unsigned int power_mW;
double temperature_c;
unsigned int clock_MHz;
};

class PowerMonitor {
private:
    nvmlDevice_t& device;
    std::vector<PowerMeasurement>& measurements;
    std::atomic<bool>& should_continue;
    int sampling_rate_ms;

public:
    PowerMonitor(nvmlDevice_t& dev, std::vector<PowerMeasurement>& meas, 
                std::atomic<bool>& cont, int rate = 1)
        : device(dev), measurements(meas), should_continue(cont), 
          sampling_rate_ms(rate) {}

    void operator()() {
        while (should_continue.load()) {
            PowerMeasurement measurement;
            
            measurement.timestamp_ms = 
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()
                ).count();
                
            unsigned int power;
            checkNvmlError(nvmlDeviceGetPowerUsage(device, &power),
                          "getting GPU power usage failed");
            measurement.power_mW = power;
            
            unsigned int temp;
            checkNvmlError(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp),
                          "getting GPU temperature failed");
            measurement.temperature_c = temp;
            
            unsigned int clock;
            checkNvmlError(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock),
                          "getting GPU clock failed");
            measurement.clock_MHz = clock;
            
            measurements.push_back(measurement);
            std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate_ms));
        }
    }
};

// Function to collect power measurements
void collectPowerData(nvmlDevice_t& device, 
                     std::vector<PowerMeasurement>* measurements,
                     bool* should_continue,
                     int sampling_rate_ms = 1) {
    while (*should_continue) {
        PowerMeasurement measurement;
        
        // Get timestamp
        measurement.timestamp_ms = 
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();
            
        // Get power
        unsigned int power;
        checkNvmlError(nvmlDeviceGetPowerUsage(device, &power),
                      "getting GPU power usage failed");
        measurement.power_mW = power;
        
        // Get temperature
        unsigned int temp;
        checkNvmlError(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp),
                      "getting GPU temperature failed");
        measurement.temperature_c = temp;
        
        // Get clock speed
        unsigned int clock;
        checkNvmlError(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock),
                      "getting GPU clock failed");
        measurement.clock_MHz = clock;
        
        measurements->push_back(measurement);
        std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate_ms));
    }
}

//--------------------------------------------------------------
// A helper function to factorize an integer T into (x,y) pairs
//--------------------------------------------------------------
std::vector<dim3> factorPairs2D(unsigned int T) {
    std::vector<dim3> pairs;
    for (unsigned int x = 1; x <= T; x++) {
        if (T % x == 0) {
            unsigned int y = T / x;
            // We'll keep z=1 for a “2D” block
            // Filter out invalid shapes:
            //   block.x <= 1024, block.y <= 1024, block.x*block.y <= 1024
            if (x <= 1024 && y <= 1024 && (x * y <= 1024)) {
                pairs.push_back(dim3(x, y, 1));
            }
        }
    }
    return pairs;
}

//--------------------------------------------------------------
// The attention_device function runs 1-3 kernels depending on impl
//--------------------------------------------------------------

float* attention_device(const float* key,
                        const float* value,
                        const float* query,
                        int n,
                        int d,
                        int impl_num,
                        int repeat,
                        const dim3 &grid1, const dim3 &block1,
                        const dim3 &grid2, const dim3 &block2,
                        const dim3 &grid3, const dim3 &block3,
                        double &time_ms_out) // we'll pass time back
{
    float *d_key, *d_value, *d_query;
    cudaMalloc((void**)&d_key,   n*d*sizeof(float));
    cudaMalloc((void**)&d_value, n*d*sizeof(float));
    cudaMalloc((void**)&d_query, d   *sizeof(float));

    cudaMemcpy(d_key,   key,   n*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, n*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, d   *sizeof(float), cudaMemcpyHostToDevice);

    float *d_dot_product, *d_exp_sum, *d_output;
    cudaMalloc((void**)&d_dot_product, n*sizeof(float));
    cudaMalloc((void**)&d_exp_sum,     sizeof(float));
    cudaMalloc((void**)&d_output,      d*sizeof(float));

    cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    if (impl_num == 3) {
        for (int i = 0; i < repeat; i++) {
            cudaMemset(d_exp_sum, 0, sizeof(float));
            kernel1_warpReduce<<<grid1, block1>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
            kernel2_blockReduce<<<grid2, block2>>>(d_exp_sum, d_dot_product, d_value, d_output, n, d);
        }
    }
    else if (impl_num == 2) {
        for (int i = 0; i < repeat; i++) {
            cudaMemset(d_exp_sum, 0, sizeof(float));
            kernel1_warpReduce<<<grid1, block1>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
            kernel2_warpReduce<<<grid2, block2>>>(d_exp_sum, d_dot_product, d_value, d_output, n, d);
        }
    }
    else if (impl_num == 1) {
        for (int i = 0; i < repeat; i++) {
            cudaMemset(d_exp_sum, 0, sizeof(float));
            kernel1_blockReduce<<<grid1, block1>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
            kernel2_blockReduce<<<grid2, block2>>>(d_exp_sum, d_dot_product, d_value, d_output, n, d);
        }
    }
    else {
        // naive = 3 separate kernels
        float *d_score;
        cudaMalloc((void**)&d_score, n*sizeof(float));
        for (int i = 0; i < repeat; i++) {
            cudaMemset(d_exp_sum, 0, sizeof(float));
            kernel1<<<grid1, block1>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
            kernel2<<<grid2, block2>>>(d_exp_sum, d_dot_product, d_score, n);
            kernel3<<<grid3, block3>>>(d_score, d_value, d_output, n, d);
        }
        cudaFree(d_score);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    double total_ns  = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double time_ms   = total_ns * 1.0e-6 / repeat;
    time_ms_out = time_ms;

    float* host_output = (float*)malloc(d*sizeof(float));
    cudaMemcpy(host_output, d_output, d*sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_query);
    cudaFree(d_dot_product);
    cudaFree(d_exp_sum);
    cudaFree(d_output);

    return host_output; 
}

//--------------------------------------------------------------
// main
//--------------------------------------------------------------
int main(int argc, char* argv[]) {

    if (argc != 5) {
        printf("Usage: %s <rows> <columns> <implementation> <repeat>\n", argv[0]);
        printf("  implementation 0: naive\n");
        printf("  implementation 1: fused kernels w/ block reduce\n");
        printf("  implementation 2: fused kernels w/ warp reduce\n");
        printf("  implementation 3: fused kernels w/ mixed reduce\n");
        return 1;
    }

    const int n = atoi(argv[1]);
    const int d = atoi(argv[2]);
    const int k = atoi(argv[3]);
    const int r = atoi(argv[4]);

    printf("n=%d, d=%d, impl=%d, repeat=%d\n", n, d, k, r);

    // Prepare host input
    float* key   = (float*) malloc (n*d*sizeof(float));
    float* value = (float*) malloc (n*d*sizeof(float));
    float* query = (float*) malloc (d  *sizeof(float));

    // Random initialization
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    for (int i = 0; i < n*d; i++) {
        key[i]   = dist(gen);
        value[i] = dist(gen);
        if (i < d) {
            query[i] = dist(gen);
        }
    }

    // Host reference
    float* hout = attention_host(key, value, query, n, d);

    char filename[256];
    snprintf(filename, sizeof(filename), 
             "power_data_%dx%d_impl%d_%lu.csv", 
             n, d, k, 
             std::chrono::system_clock::now().time_since_epoch().count());
    
    FILE* fp = fopen(filename, "w");

    // Write detailed data
    fprintf(fp, "timestamp_ms,matrix_size,impl,threads,blockX,blockY,"
    "gridX,gridY,power_mW,temp_c,clock_MHz,time_ms,energy_mJ\n");



#ifdef USE_NVML
    // ---------------------------------------------------------
    // NVML initialization (optional power measurement)
    // ---------------------------------------------------------
    checkNvmlError(nvmlInit_v2(), "Could not initialize NVML");
    nvmlDevice_t device;
    checkNvmlError(nvmlDeviceGetHandleByIndex(0, &device), "Could not get NVML device handle");
#endif

    unsigned int TBsizes[] = {32, 64, 128, 256, 512, 1024};
    int numSizes = sizeof(TBsizes)/sizeof(TBsizes[0]);

    float bestRmse = 1e30f;

    for (int idxSize = 0; idxSize < numSizes; idxSize++) {
        unsigned int T = TBsizes[idxSize];
        printf("\n=== Testing total threads per block = %u ===\n", T);

        std::vector<dim3> blocks2D = factorPairs2D(T); 
        if (blocks2D.empty()) {
            // No valid factor pairs (should not happen for T <= 1024).
            continue;
        }

#ifdef USE_NVML
        // Rough power measurement before each block-size iteration
        unsigned int powerBefore_mW = 0;
        checkNvmlError(nvmlDeviceGetPowerUsage(device, &powerBefore_mW),
                       "getting GPU power usage (before) failed");
#endif

        for (auto &b1 : blocks2D) {

            // For naive (impl=0), 3-kernel approach => gridK1, gridK2 over n, gridK3 over d
            // For fused => 2 kernels => (gridK1 over n), (gridK2 over n or d)
            // We'll keep consistent approach:
            dim3 gridK1( (n + b1.x*b1.y - 1)/(b1.x*b1.y), 1, 1 );
            dim3 gridK2( (n + b1.x*b1.y - 1)/(b1.x*b1.y), 1, 1 );
            dim3 gridK3( (d + b1.x*b1.y - 1)/(b1.x*b1.y), 1, 1 );

            std::vector<PowerMeasurement> measurements;
            std::atomic<bool> continue_measuring{true};
            
            // Create and start power monitoring thread
            PowerMonitor monitor(device, measurements, continue_measuring);
            std::thread power_thread(std::ref(monitor));

            double time_ms = 0.0;
            float* dout = attention_device(key, value, query,
                                           n, d, k, r,
                                           gridK1, b1,
                                           gridK2, b1,
                                           gridK3, b1,
                                           time_ms);

            // Stop measurements and wait for thread
            continue_measuring.store(false);
            power_thread.join();

            // Calculate energy
            double total_energy_mJ = 0;
            double avg_power_mW = 0;
            
            for (size_t i = 1; i < measurements.size(); i++) {
                double dt = measurements[i].timestamp_ms - 
                           measurements[i-1].timestamp_ms;
                total_energy_mJ += measurements[i].power_mW * (dt / 1000.0);
                avg_power_mW += measurements[i].power_mW;
            }
            
            avg_power_mW /= measurements.size();

            // When writing measurements:
            for (const auto& m : measurements) {
                fprintf(fp, "%lf,%dx%d,%d,%d,%d,%d,%d,%d,%u,%.1f,%u,%.6f,%.6f\n",
                        m.timestamp_ms, 
                        n, d,           // matrix_size as n x d
                        k,             // implementation
                        T,             // threads
                        b1.x, b1.y,    // block dimensions
                        gridK1.x, gridK1.y,  // grid dimensions
                        m.power_mW, 
                        m.temperature_c, 
                        m.clock_MHz,
                        time_ms, 
                        total_energy_mJ);
            }
            // Compute RMSE
            float rmse=0.f;
            for (int i=0; i<d; i++){
                float diff = hout[i] - dout[i];
                rmse += diff*diff;
            }
            rmse = sqrtf(rmse / d);

            if (rmse < bestRmse) { 
                bestRmse = rmse; 
            }


            free(dout);
        }

#ifdef USE_NVML
        // optional post block-size measurement if desired
        // ...
#endif

    } // end for each T

#ifdef USE_NVML
    nvmlShutdown();
#endif

    fclose(fp);

    printf("\nBest overall RMSE = %f\n", bestRmse);

    free(hout);
    free(key);
    free(value);
    free(query);

    return 0;
}
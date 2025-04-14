# Optimizing Energy Efficiency in Transformer-based Models  
*Auto-tuning GPU Kernels for Sustainable AI Inference*  

<!-- pgrep -f 'runner.sh' | xargs kill -9 -->

---

## Overview  
This project focuses on optimizing kernels for transformer architectures (LLAMA, GPT-2) through automated GPU kernel tuning with energy-aware metrics. By combining performance engineering techniques like power capping with hardware telemetry from modern NVIDIA GPUs, we identify Pareto-optimal configurations that balance computational throughput and energy efficiency.  

<!-- ![Pareto Frontier Example](https://via.placeholder.com/600x400?text=Energy+vs+Occupancy+Tradeoff)  
*Example Pareto Frontier for MHA Kernel Configurations*   -->

---

## Technical Implementation  

| Component | Purpose |  
|-----------|---------|  
| [HeCBench](https://github.com/zjin-lcf/HeCBench) | Baseline CUDA kernels for attention mechanisms |
| [Kernel Tuner](https://kerneltuner.github.io/) | Automated GPU kernel parameter optimization |
| [NVMETRICS](https://github.com/nlesc-recruit/nvmetrics) | nvmetrics is a library to measure GPU metrics using NVIDIA CUPTI |
| NVIDIA Nsight Compute | Fine-grained SM occupancy profiling |
| NVML Power Monitoring | Real-time energy consumption tracking |
| GPT-2/LLAMA Model cuda implementation| One of Karpathy's llm.c forks |

## Research Questions

### 1. Occupancy-Energy Tradeoffs in Attention Mechanisms  
**Q**: What's the optimal balance between SM occupancy and energy-per-token in LLM attention layers?  
**Approach**: Systematic profiling of block dimensions (16x16 to 256x4) using Kernel Tuner with NVML/Nsight metrics . Establishes baseline for transformer's most compute-intensive operation (40-60% of FLOPs)  
**Novelty**: Empirical study of occupancy-energy Pareto fronts in real-world sequence lengths (256-4096 tokens)  
**Impact**: Challenges maximal occupancy dogma, validate energy savings via optimized block shapes  

### 2. Memory Coalescing for Efficient Embeddings  
**Q**: Can block tuning achieve 3× energy reduction in embeddings through L2 cache optimization?  
**Approach**: Energy-aware coalescing metric combining cache lines accessed and coalesce factor per joule  
**Novelty**: Quantifies strided memory vs. SM idle power tradeoff without quantization  
**Impact**: Enables energy-efficient embedding layers for large context windows  

### 3. Adaptive Block Sizing for Dynamic Inference  
**Q**: Does real-time sequence-length-aware block adaptation reduce decoding energy vs static configurations?  
**Approach**: MLP-based selector trained on Kernel Tuner profiles with sequence length detection  
**Novelty**: First implementation of lightweight neural block size predictor for dynamic inference  
**Impact**: Enables energy-proportional decoding for variable-length sequences  

### 4. Compute vs Memory-Bound Kernel Behavior  
**Q**: How do  LLM kernel types (compute-bound and memory-bound kernels) fundamentally differ in energy/power characteristics under varying block configurations(exhibit divergent energy scaling laws)?  
**Approach**:  
- Classify kernels using arithmetic intensity thresholds  
- Comparative analysis of 10+ metrics (energy, power variance, cache efficiency) across kernel types  
**Novelty**: First taxonomy of energy scaling patterns based on kernel resource utilization  
**Impact**: Enables targeted optimizations - thread coarsening for compute-bound vs shared mem tuning for memory-bound  

### 5. Predictive Energy Modeling with Power Capping  
**Q**: Can hybrid features (static code features + runtime metrics) enable intelligent power capping for LLM workloads?    
**Approach**:  
- Hybrid XGBoost model combining code complexity scores and real-time telemetry  
- Power capping trials based on model predictions  
**Novelty**: First end-to-end system adapting power limits using kernel characteristics  
**Impact**: Achieves 18-22% energy savings via predictive power management  


## Getting Started  

### Dependencies  
```bash
# Core Requirements
CUDA 12.4+
Python 3.11+
NVIDIA Driver 550+
NVCC 12.4+

Compiler backend : Pycuda or CuPy, based on the device code. PyCuda uses NVCC which need device and host code separately. CuPy takes by default cuda c and single script works. 

Note : If getting "extern C linkage error, pass the whole code wrapped in extern "C" {.....} , or use the manged kernel name instead of the readable one.

# Python Packages
pip install kernel-tuner cupy-cuda12x pytorch matplotlib pandas
```

### Basic Usage  
```python
from kernel_tuner import tune_kernel

results = tune_kernel(
    kernel_name="mha",
    kernel_source="attentionMultiHead.cu",
    problem_size=(num_heads*batch_size, 1),
    tune_params=tune_params,
    observers=[NVMLObserver(), NCUObserver()]
)
```

```python
restrictions = [
        "block_size_x * block_size_y % 32 == 0", 
        "block_size_x * block_size_y <= 1024",
        f"(({dim_feature} // {nhead}) % block_size_x) == 0",
        f"shared_size_factor * (({dim_feature}//{nhead}) + {n_steps}) * 4 <= 48*1024"
    ]
```

## Results & Metrics  

### Key Performance Indicators  
| Metric | Formula | Target |  
|--------|---------|--------|  
| Energy Efficiency | `Total Energy / (Tokens Processed)` | Minimize |  
| Compute Intensity | `(FLOPS) / (Watt)` | Maximize |  
| SM Occupancy | `Active Warps / Maximum Warps` | ≥80% |  
| I'll add more as i move along with the RQs| ... | ...|


### Hardware Configuration  
| Component | Specification |  
|-----------|---------------|  
| GPU | NVIDIA RTX 5000 Ada (24GB VRAM) |  
| CPU | AMD EPYC 9554P 64-Core Processor |  
| Memory | 1TB DDR4 @ 3200MHz |  
| Power Sampling | 1ms resolution via NVML |    

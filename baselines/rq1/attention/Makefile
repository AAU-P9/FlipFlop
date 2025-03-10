#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_86
LAUNCHER  = 

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Xcompiler -Wall -arch=$(ARCH) -DUSE_NVML --ptxas-options=-v

# Linker Flags
LDFLAGS = -lnvidia-ml -lcuda

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu reference.h kernels.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)
	rm -f power_data_*.csv

# # Test different matrix sizes and thread configurations
# run: $(program)
# # Small matrices
# 	$(LAUNCHER) CUDA_LAUNCH_BLOCKING=1 ./$(program) 1024 1024 3 100
# # Medium matrices
# 	$(LAUNCHER) CUDA_LAUNCH_BLOCKING=1 ./$(program) 4096 4096 3 100
# # Large matrices
# 	$(LAUNCHER) CUDA_LAUNCH_BLOCKING=1 ./$(program) 8192 8192 3 100
# # Very large matrices
# 	$(LAUNCHER) CUDA_LAUNCH_BLOCKING=1 ./$(program) 16384 16384 3 100


SIZES      := 512 1024 2048 4096 8192 16384
IMPLEMENTATIONS := 0 1 2 3
REPEATS    := 100

#===============================================================================
# Run Targets
#===============================================================================

run: $(program)
	@for impl in $(IMPLEMENTATIONS); do \
		for size in $(SIZES); do \
			echo "Running: n=$$size d=$$size impl=$$impl repeat=$(REPEATS)"; \
			$(LAUNCHER) ./$(program) $$size $$size $$impl $(REPEATS); \
		done; \
	done

## CUDA Runtime APIs

The Runtime APIs are a high-level interface that lets you query and manage the GPU without reading white papers. They handle device management, memory allocation, and kernel execution.

**Key difference from normal functions:** Runtime API functions don't return their result via the return value. Instead, they write results into a parameter you pass in. The return value is an error status code.

```
cudaGetDeviceCount( int* count )
         ^                ^
    returns error      writes result HERE
    status code        (pointer to your variable)
```

### cudaGetDeviceProperties

The most important query function. Returns a struct (`cudaDeviceProp`) filled with GPU info.

```
cudaGetDeviceProperties( &prop, device_id )
                            ^       ^
                        struct    which GPU
                        to fill   (0, 1, 2...)
```

Fields you can read from the prop struct:

```
prop.name                          -> "GeForce RTX 3090"
prop.totalGlobalMem                -> total VRAM in bytes
prop.memoryClockRate               -> memory clock in kHz
prop.memoryBusWidth                -> bus width in bits
prop.multiProcessorCount           -> number of SMs
prop.maxThreadsPerBlock            -> 1024
prop.maxThreadsPerMultiProcessor   -> 1536 (RTX 3090)
prop.maxThreadsDim[0,1,2]          -> max threads per block per dimension
prop.maxGridSize[0,1,2]            -> max blocks per grid per dimension
```

### cudaDeviceGetAttribute (alternative method)

Gets a single attribute instead of the whole struct:

```
cudaDeviceGetAttribute( &value, cudaDevAttrMaxThreadsPerMultiProcessor, device_id )
                           ^           ^                                     ^
                       stores       attribute name                        GPU id
                       result       (from documentation list)
```

Both methods give identical results. Use whichever fits your needs.

### RTX 3090 Key Numbers (from runtime API output)

```
Name:                    GeForce RTX 3090
Total global memory:     ~24 GB
Memory bus width:        384 bits
Compute capability:      8.6
Number of SMs:           82
Max threads/block:       1024
Max warps/block:         1024 / 32 = 32 warps
Max threads/SM:          1536
Max warps/SM:            1536 / 32 = 48 warps
Max threads X/block:     1024
Max threads Y/block:     1024
Max threads Z/block:     64
```

---

## nvidia-smi

A command-line tool for monitoring and managing Nvidia GPUs in real time.

### Three Key Uses

```
nvidia-smi
    |
    +-- Performance Monitoring
    |       GPU utilization %, memory usage, temperature, power
    |
    +-- Settings Management
    |       Control clock speeds, set power limits
    |
    +-- Device Info Querying
            GPU model, driver version, CUDA version, current clocks
```

### Basic Output Structure

Running `nvidia-smi` shows two tables:

```
TABLE 1: Status
+------------------------------------------------------------------+
| Driver Version: 546.12        CUDA Version: 12.3                 |
+------------------------------------------------------------------+
| GPU | Name         | Persistence | Bus-Id    | Memory-Usage       |
| Fan | Temp  | Perf | Pwr:Use/Cap |           | GPU-Util           |
+------------------------------------------------------------------+
|  0  | RTX 3090     |     Off     | 00000...  | 4061MiB / 24576MiB |
| 30% | 53C   |  P8  | 31W / 350W  |           | 42%                |
+------------------------------------------------------------------+

TABLE 2: Running Processes
+------------------------------------------------------------------+
| GPU | PID   | Process Name                         | Memory      |
+------------------------------------------------------------------+
(empty if no CUDA apps running)
```

### Effect of Running a CUDA Application

```
                  IDLE          RUNNING matrix multiply
                  ----          ----------------------
GPU Util:         42%           100%
Memory:           4 GB          8 GB
Power:            31W           318W  (10x increase!)
Temperature:      53C           63C
Fan:              0%            increased
```

The 42% idle utilization in the example was from recording 4K video, not CUDA.

### Useful Commands

```
# Monitor continuously, refresh every 5 seconds
nvidia-smi -l 5

# Query specific metrics as CSV
nvidia-smi --query-gpu=gpu_name,driver_version,temperature.gpu --format=csv

# Set power limit to 200W
nvidia-smi -i 0 -pl 200

# Enable persistence mode (required before changing clocks)
nvidia-smi -pm 1

# Check current clock speeds
nvidia-smi -q -d CLOCK

# List all supported clock values you can set
nvidia-smi -q -d SUPPORTED_CLOCKS

# Set specific clocks (memory_clock,SM_clock)
nvidia-smi --application-clocks=1216,765

# Reset clocks to default
nvidia-smi --reset-gpu-clocks

# Save output to file
nvidia-smi -l 5 > gpu_log.txt

# See all available options
nvidia-smi -h
```

### Clock Behavior

The GPU dynamically adjusts clock speed based on load:

```
No workload:   SM clock ~570 MHz   (saves power)
Under load:    SM clock ~1695 MHz  (boosts to finish faster)
Maximum:       SM clock ~2100 MHz  (full speed)
```

You can lock the clock to a specific supported value using `--application-clocks`, but you must first enable persistence mode (`-pm 1`) and can only use values from the supported clocks list.

---

## Occupancy

### What is Occupancy?

A measure of how well you're utilizing the GPU's compute resources.

```
Two types:
    Theoretical Occupancy  =  ideal/best-case calculation
    Achieved Occupancy     =  what actually happened at runtime
```

### Theoretical Occupancy Formula

```
Theoretical Occupancy = (warps used in kernel per SM) / (max warps per SM)
```

RTX 3090 examples:

```
Max warps/SM = 48

48 warps used -> 48/48 = 100%
12 warps used -> 12/48 = 25%
```

### How Threads/Block Affects Occupancy (RTX 3090)

```
Threads/block    Warps/block    Active blocks/SM    Total warps/SM    Occupancy
-------------    -----------    ----------------    --------------    ---------
32               1              16                  16                16/48 = 33%
64               2              16                  32                32/48 = 67%
96               3              16                  48                48/48 = 100%
128              4              12                  48                48/48 = 100%
```

(The "16 active blocks" comes from the Block Limit constraints, explained in section 4.4.)

To go from 33% to 100%: increase threads per block to 96 (3 warps × 16 blocks = 48 warps = max).

### Achieved Occupancy — Scenario 1: No memory requests

```
4 warps available, no stalls, no memory latency

Cycle:  0  1  2  3  4  5  6  7  8  9  10  11  12  13 ...
Warp:  W0 W1 W2 W3 W0 W1 W2 W3 W0 W1  W2  W3  W0  ...

Every cycle: a warp is executing an instruction
Active cycles / Total cycles = 100%
Achieved occupancy = 100%  (matches theoretical)
```

Instruction types sent to different pipelines:

- FMUL (floating multiply) -> FP units
- ISETP (integer compare) -> INT units
- Load/Store -> LD/ST units

### Achieved Occupancy — Scenario 2: With memory requests

Assume load instruction takes 6 cycles (real GPU: 30–300 cycles depending on cache level):

```
Cycle:  0    1    2    3    4    5    6    7    8    9    10   11   12   13
        W0   W1   W2   W3   W0   W1   W2   W3   W0   W1   W2   W3  ???  ???
       FMUL FMUL FMUL FMUL ISETP ISETP ISETP ISETP LOAD LOAD LOAD LOAD

At cycle 12: W0's load started at cycle 8, needs 6 cycles, finishes at 14
             W1's load started at cycle 9, finishes at 15
             W2: finishes at 16
             W3: finishes at 17

Cycles 12, 13: NO warp is ready -> STALL CYCLES
```

```
If total cycles = 100, active cycles = 85:
Achieved occupancy = 85 / 100 = 85%
Theoretical was 100%, achieved is 85%  <- gap caused by stalls
```

### Memory Latency by Cache Level

```
L1 cache hit:     ~30 cycles
L2 cache hit:     ~190-200 cycles
Global memory:    ~300 cycles
```

### Latency Hiding (why GPUs don't use out-of-order execution)

CPUs handle instruction dependencies with out-of-order execution — reordering independent instructions to fill stall gaps. GPUs don't need this:

```
CPU approach (4 cores):
    If instruction A stalls (dependency)...
    Reorder: find independent instruction B and run it instead
    Complex hardware, few cores

GPU approach (thousands of cores, many warps):
    If Warp 0 stalls (waiting for memory)...
    Switch to Warp 1 (zero cost — all warp states live in registers)
    If Warp 1 stalls... switch to Warp 2
    If Warp 2 stalls... switch to Warp 3
    By the time we cycle back to Warp 0, data may have arrived
```

Zero-cost warp switching is the key. Nothing needs to be saved or restored because all 64 warp register files live on-chip simultaneously.

### Achieved Occupancy is an Average

```
One SM has 4 partitions:
    Each partition computes its own achieved occupancy
    Average across 4 partitions -> achieved occupancy per SM

GPU has 82 SMs (RTX 3090):
    Each SM computes its achieved occupancy
    Average across all SMs -> overall achieved occupancy
```

### Why Occupancy Matters

```
High theoretical occupancy:
    -> many warps available to hide latency
    -> BUT doesn't guarantee high performance (memory bandwidth may still be bottleneck)

Low theoretical occupancy:
    -> bottleneck exists (not enough threads/warps)
    -> scheduler has nothing to switch to when stalls occur

Achieved < Theoretical:
    -> stall cycles occurring (memory latency, instruction dependency)
    -> investigate with Nsight Compute profiler
```

---

## Allocated Active Blocks per SM (AABS)

### What is it?

The number of blocks that can run _concurrently_ on a single SM. You can assign thousands of blocks to an SM, but only a limited number can be active at once.

```
Assigned blocks per SM:  potentially thousands (queued)
Active blocks per SM:    limited by hardware constraints
```

### The Four Hardware Limits (A100 GPU)

```
+---------------------------+----------+
| Limit                     | A100     |
+---------------------------+----------+
| Max thread blocks / SM    | 32       |  <- Block Limit SM
| Max warps / SM            | 64       |  <- Block Limit Warps
| Max 32-bit registers / SM | 65,536   |  <- Block Limit Registers
| Shared memory / SM        | 164 KB   |  <- Block Limit Shared Mem
+---------------------------+----------+
```

The final AABS = **minimum of all four limits** (expressed as blocks).

### How Each Limit Restricts Active Blocks

**1. Warp Limit Example**

```
Max warps/SM = 64
Each block has 4 warps

If we try 32 blocks: 32 × 4 = 128 warps  -> EXCEEDS limit!

Compiler solution: reduce to 16 blocks
16 × 4 = 64 warps  -> within limit ✓

AABS limited to 16 (not 32) because of warp limit
This is called "Block Limit Warps"
```

**2. Register Limit Example**

```
Block size = 512 threads, each thread uses 16 registers
Registers per block = 512 × 16 = 8,192

Max registers/SM = 65,536
Max active blocks = 65,536 / 8,192 = 8 blocks

AABS limited to 8 by register constraint
```

If registers needed exceed the limit:

```
Option 1: Reduce threads per block
Option 2: Do nothing -> register spilling occurs
          (values stored in local memory, which is slow)
          -> performance degradation
```

**3. Final Example: Taking the Minimum**

```
Block Limit SM (hardware max):    32 blocks
Block Limit Warps:                 4 blocks  <- SMALLEST
Block Limit Registers:             8 blocks
Block Limit Shared Memory:        16 blocks

AABS = min(32, 4, 8, 16) = 4 blocks
```

The warp limit wins here — only 4 blocks can run concurrently despite the 32-block hardware ceiling.

### Flow of AABS Calculation

```
You choose: threads per block (e.g. 96)
    |
    v
Convert to warps: 96 / 32 = 3 warps/block
    |
    v
Each limit asks: "how many blocks can I allow?"
    |
    +-- SM limit:          max_blocks_SM
    +-- Warp limit:        floor(max_warps / warps_per_block)
    +-- Register limit:    floor(max_registers / regs_per_block)
    +-- Shared mem limit:  floor(max_shmem / shmem_per_block)
    |
    v
AABS = minimum of the above four values
    |
    v
Theoretical Active Warps = AABS × warps_per_block
    |
    v
Theoretical Occupancy = Theoretical Active Warps / max_warps_per_SM
```

### Seeing This in Nsight Compute (ncu)

The occupancy section output looks like:

```
Section: Occupancy
--------------------------------------------------------------
Metric Name                    Unit    Value
--------------------------------------------------------------
Block Limit SM                 block   16
Block Limit Registers          block   42
Block Limit Shared Mem         block   16
Block Limit Warps              block   16
Theoretical Active Warps/SM    warp    48
Theoretical Occupancy          %       100
Achieved Occupancy             %       74.00
Achieved Active Warps/SM       warp    35.52
--------------------------------------------------------------
```

Minimum of (16, 42, 16, 16) = 16 blocks × 3 warps = 48 warps = 48/48 = 100% theoretical. Achieved is lower (74%) due to runtime stalls.

---

## Code Reference: Vector Addition

```
__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;  // unique global index
    if (i < n) {                                      // bounds check
        C[i] = A[i] + B[i];
    }
}

int threadsPerBlock = 96;
int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, SIZE);
```

With 96 threads/block on RTX 3090:

- 3 warps per block
- AABS = 16 blocks
- Active warps = 48 = max -> 100% theoretical occupancy
- Achieved occupancy ~73% (runtime stalls from memory latency)

---

## Quick Reference: Theoretical Occupancy Calculation Steps

```
Step 1:  Get max warps/SM from runtime API
         (prop.maxThreadsPerMultiProcessor / 32)
         RTX 3090: 1536 / 32 = 48

Step 2:  Choose threads per block (your configuration)
         e.g., 96 threads

Step 3:  Convert to warps per block
         96 / 32 = 3 warps

Step 4:  Get AABS from profiler (or calculate from limits)
         e.g., 16 blocks

Step 5:  Active warps = AABS × warps/block
         16 × 3 = 48

Step 6:  Theoretical occupancy = active warps / max warps
         48 / 48 = 100%
```


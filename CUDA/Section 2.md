## The CUDA Toolkit

Everything you need comes bundled in the CUDA Toolkit:

```
CUDA Toolkit
│
├── NVCC Compiler
│     CUDA code → PTX (like GPU assembly) → machine code
│
├── Libraries (pre-built GPU-accelerated math)
│     cuBLAS  — linear algebra (matrix multiply etc.)
│     cuFFT   — fast fourier transforms
│     cuRAND  — random number generation
│     cuDNN   — deep neural network operations
│
├── Runtime & Driver APIs (how your code talks to the GPU)
│     cudaMalloc()   — allocate GPU memory
│     cudaMemcpy()   — copy data CPU ↔ GPU
│     cudaFree()     — free GPU memory
│
└── Tools
      Nsight Systems  — system-level profiler (timeline view)
      Nsight Compute  — kernel-level profiler (deep GPU metrics)
      CUDA GDB        — debugger
      CUDA Memcheck   — finds memory errors
```

NVCC is not just a compiler — it also separates host code (C++) from device code (CUDA), compiling each with the right toolchain.

---

# PART 2: Host vs Device — Two Separate Worlds

This is the most fundamental concept in CUDA. There are two separate computers working together.

```
HOST                          DEVICE
─────────────────────         ─────────────────────
CPU                           GPU
DRAM (your RAM, e.g. 16GB)    GDRAM / VRAM (e.g. 40GB)

Runs your main program        Runs your kernel functions
Sequential, complex logic     Parallel, simple repeated ops
```

They have **separate memory**. Data does not magically appear on the GPU. You must explicitly copy it.

```
Flow of every CUDA program:

CPU: allocate memory on GPU        (cudaMalloc)
CPU: copy data CPU → GPU           (cudaMemcpy HostToDevice)
CPU: tell GPU to run the kernel    (kernel<<<blocks,threads>>>())
GPU: executes kernel in parallel
CPU: copy results GPU → CPU        (cudaMemcpy DeviceToHost)
CPU: use the results
```

---

# PART 3: GPU Hardware — What It Physically Looks Like

## CPU vs GPU — The Core Philosophy

```
CPU (e.g. Intel i9)
┌──────────────────────────────────┐
│  Core 0  │  Core 1  │  Core 2   │   ← 8-32 powerful cores
│          │          │           │     each handles complex,
│  Large   │  Large   │  Large    │     branching, varied tasks
│  Cache   │  Cache   │  Cache    │     independently
└──────────────────────────────────┘
Good at: OS, game logic, web servers, anything sequential

GPU (e.g. A100)
┌────────────────────────────────────────────────────────┐
│  SM0  │  SM1  │  SM2  │  SM3  │ ... │ SM107  │        │
│                                                        │
│  each SM has 128 tiny cores = 13,824 cores total      │
│                                                        │
│  Global Memory (VRAM) — 40GB — shared by all SMs      │
└────────────────────────────────────────────────────────┘
Good at: doing the exact same simple operation on millions of things at once
```

The GPU does NOT replace the CPU. The CPU is always in charge — it is the manager that sends work to the GPU.

## The SM — Streaming Multiprocessor

The SM is the main worker unit of the GPU. Think of the GPU as a factory, and each SM as one workstation inside it.

```
One SM contains:
┌──────────────────────────────────────┐
│  128 CUDA cores (FP/INT units)       │
│  64 warp slots (max 64 warps active) │
│  L1 Cache (~192KB, private to SM)    │
│  Shared Memory (programmable)        │
│  Registers (per-thread, fastest)     │
│                                      │
│  Warp Schedulers (inside partitions) │
│  — pick which warp runs next         │
└──────────────────────────────────────┘
```

Each SM is further divided into **partitions**, each with its own warp scheduler. This is what allows the SM to manage many warps at once.

## Who Assigns Work to SMs?

The **Gigathread Engine** — hardware built into the GPU. When you launch a kernel, the Gigathread Engine distributes your blocks across available SMs automatically. You do not control this directly.

---

# PART 4: Memory — Where Data Lives and How Fast It Is

Your data lives at different levels. Where it lives completely determines your performance.

```
SPEED HIERARCHY (fastest → slowest)

  Thread
    │
    ▼
  Registers ────── ~1 cycle    per-thread, tiny, on-chip
    │                          (local variables like int i)
    ▼
  L1 Cache  ────── ~20 cycles  per-SM, ~192KB, on-chip
    │                          (recently used data, automatic)
    ▼
  L2 Cache  ────── ~200 cycles shared across all SMs, ~40MB
    │                          (larger working set, automatic)
    ▼
  VRAM ─────────── ~600 cycles off-chip, 40-80GB
  (Global Memory)              (where cudaMalloc puts your arrays)
    │
    ▼
  CPU RAM ──────── ~6000 cycles completely separate, need cudaMemcpy
```

When a thread does `C[i] = A[i] + B[i]`, the GPU checks each level in order. If the data is not in L1, it checks L2, then goes all the way to VRAM.

```
Array size → where it fits → latency

Few KB   → L1 cache  → ~20  cycles  (tiny wait)
~10MB    → L2 cache  → ~200 cycles  (moderate wait)
100MB+   → VRAM      → ~600 cycles  (huge wait, cache miss)
```

For the 32M element vector in the [code](CUDA/cuda-tutorial/1/2-add.cu) (128MB), every access is a cache miss going all the way to VRAM.

---

# PART 5: The Software Hierarchy — Threads, Warps, Blocks, Grid

CUDA maps software concepts onto hardware levels. You need to understand all four levels.

## 5.1 Thread — The Smallest Unit

One thread = one run of your kernel function on one element.

```
Array A:  [ 0,  1,  2,  3,  4, ... 2047 ]
            ↑   ↑   ↑   ↑   ↑         ↑
Thread:     0   1   2   3   4  ...  2047

Each thread handles exactly one element.
```

## 5.2 Warp — The Real Execution Unit

The SM does NOT run threads one at a time. It groups every **32 threads** into a **warp** and runs the whole warp together.

```
Block 0 (1024 threads) → split into warps by hardware:

Warp 0:  threads  0-31    ← all execute same instruction, same cycle
Warp 1:  threads 32-63
Warp 2:  threads 64-95
...
Warp 31: threads 992-1023

Total: 1024 / 32 = 32 warps per block
```

This is called **SIMT** — Single Instruction, Multiple Threads. All 32 threads in a warp are in lockstep.

The warp is what the scheduler actually sees. It does not schedule individual threads — it schedules warps.

## 5.3 Block — Assigned to One SM

A block is a group of threads (and therefore warps). The hardware rule: **one block runs on exactly one SM, never split**.

```
Block 0 (your 1024 threads) ──► SM 0
Block 1 (your 1024 threads) ──► SM 1
Block 2 (your 1024 threads) ──► SM 2
```

Hardware limit on Ampere: **maximum 1024 threads per block**.

## 5.4 Grid — The Whole Launch

The grid is all your blocks together. One kernel launch = one grid.

```
Grid (your entire kernel launch)
├── Block 0  ──► SM 0
├── Block 1  ──► SM 1
├── Block 2  ──► SM 2
├── ...
└── Block N  ──► SM N  (or queued if N > 108)
```

## 5.5 The Complete Mapping

```
SOFTWARE                    HARDWARE                   MANAGED BY
──────────────────────────────────────────────────────────────────
Grid                   →    Entire GPU             →   You (kernel launch)
  Block                →    One SM                 →   Gigathread Engine
    Warp (32 threads)  →    SM Partition           →   Warp Scheduler
      Thread           →    One GPU Core           →   Hardware
```

---

# PART 6: The Indexing Equation — How Threads Know What to Work On

This equation is in every CUDA kernel ever written.

## The Problem

With 2 blocks × 1024 threads, both blocks have threads numbered 0 to 1023. Block 1 needs to work on elements 1024–2047, but its thread IDs start at 0 again.

```
Block 0:  thread IDs 0..1023  →  should touch array[0..1023]    ✓ easy
Block 1:  thread IDs 0..1023  →  should touch array[1024..2047]  ← problem
```

## The Solution

```cuda
int i = threadIdx.x + blockIdx.x * blockDim.x;
```

|Variable|What It Is|Example Value|
|---|---|---|
|`threadIdx.x`|Thread's ID within its block|0 to 1023|
|`blockIdx.x`|Which block this thread is in|0 or 1|
|`blockDim.x`|How many threads per block|1024|
|`i`|Final index into your array|0 to 2047|

## Worked Examples

```
Block 0, Thread 0:    i = 0    + (0 × 1024) = 0       ✓
Block 0, Thread 1023: i = 1023 + (0 × 1024) = 1023    ✓

Block 1, Thread 0:    i = 0    + (1 × 1024) = 1024    ✓
Block 1, Thread 1023: i = 1023 + (1 × 1024) = 2047    ✓
```

No gaps. No overlaps. Every element gets exactly one thread.

## Visual

```
Array: [  0 ...  1023  |  1024 ...  2047  ]
                       |
        Block 0        |   Block 1
        tid + 0×1024   |   tid + 1×1024
```

## Always Add a Bounds Check

Ceiling division launches slightly more threads than elements. Guard against out-of-bounds writes:

```cuda
// Ceiling division — ensures enough blocks for all elements
int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;

// In the kernel
__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {           // ← critical bounds check
        C[i] = A[i] + B[i];
    }
}
```

---

# PART 7: Memory Latency — Why Threads Spend 99% of Time Waiting

## What Actually Happens When a Thread Executes One Addition

```
Thread: C[i] = A[i] + B[i]

Cycle 1:        Request A[i] from VRAM
Cycle 2-601:    WAITING... nothing happening...
Cycle 602:      A[i] arrives
Cycle 603:      Request B[i] from VRAM
Cycle 604-1203: WAITING again...
Cycle 1204:     B[i] arrives
Cycle 1205:     Do the addition  (~2 cycles, nearly instant)
Cycle 1207:     Write result to C[i]

Total:    ~1207 cycles
Math:     ~2 cycles
Waiting:  ~1205 cycles = 99.8% idle
```

The arithmetic is almost free. The memory fetch is the entire cost.

---

# PART 8: Latency Hiding — How Warps Cover the Wait

The GPU's answer: **keep many warps loaded, switch between them for free**.

## Zero-Cost Warp Switching

Unlike a CPU where switching between processes costs thousands of cycles, the SM switches between warps in **zero clock cycles**. All warp states live in registers simultaneously — nothing to save or restore.

## With Multiple Warps (Latency Hidden)

```
SM has 4 warps. Each needs data from VRAM (~600 cycle wait):

Cycle 1:   Warp A requests memory → STALL, switch to Warp B
Cycle 2:   Warp B requests memory → STALL, switch to Warp C
Cycle 3:   Warp C requests memory → STALL, switch to Warp D
Cycle 4:   Warp D requests memory → STALL, nothing left...

Cycle 602: Warp A's data arrived → SM runs Warp A → computes ✓
Cycle 603: Warp B ready → computes ✓
Cycle 604: Warp C done ✓
Cycle 605: Warp D done ✓

All 4 warps done. The 600-cycle stalls were HIDDEN by overlap.
```

## With Only 1 Warp (Latency Not Hidden)

```
SM has 1 warp:

Cycle 1:    Warp A requests memory → STALL
            Nothing to switch to.
Cycle 2-601: SM sits completely idle
Cycle 602:  Data arrives
Cycle 603:  Warp A computes (~2 cycles)
Cycle 605:  Warp A requests next data → STALL again

SM is technically "active" but computing less than 1% of the time.
```

---

# PART 9: Warp Occupancy

**Warp Occupancy** = active warps on SM ÷ max warps SM can hold.

```
A100: max 64 warps per SM

1  warp  → 1/64  = 1.5%   terrible, constant stalls
8  warps → 8/64  = 12.5%  some latency hiding
32 warps → 32/64 = 50%    decent
64 warps → 64/64 = 100%   ideal

Warps come from your block size:
  1024 threads/block → 32 warps → 50% occupancy
  512  threads/block → 16 warps → 25% occupancy
  96   threads/block →  3 warps → 4.7% occupancy  ← the code
  32   threads/block →  1 warp  → 1.5% occupancy  ← minimum
```

---

# PART 10: SM Utilization

**SM Utilization** = active SMs ÷ total SMs.

```
A100: 108 SMs total

2   blocks → 2/108   = 1.9%   almost all SMs idle
64  blocks → 64/108  = 59%    better
108 blocks → 108/108 = 100%   all SMs working
```

More blocks = more SMs working in parallel = faster (up to 108).

---

# PART 11: The Two Dimensions of GPU Performance

GPU performance depends on TWO things simultaneously:

```
Dimension 1: SM Utilization  — how many SMs are doing work
Dimension 2: Warp Occupancy  — how busy each active SM is

You need BOTH to be high for peak performance on large data.
```

## The Four Cases Visualised (A100)

### Case 1 — 2 blocks × 1024 threads

```
SM utilization:  2/108  = 1.9%
Warp occupancy: 32/64   = 50%

[SM 0]  ████████████████░░░░░░░░░░░░░░░░  32 warps — latency hiding OK
[SM 1]  ████████████████░░░░░░░░░░░░░░░░  32 warps — latency hiding OK
[SM 2]  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  IDLE
...
[SM107] ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  IDLE

❌ 106 SMs sitting completely idle.
```

### Case 2 — 64 blocks × 32 threads

```
SM utilization: 64/108 = 59%
Warp occupancy:  1/64  = 1.5%

[SM 0]  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1 warp — stalls, nothing to switch to
[SM 1]  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
...
[SM 63] █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
[SM 64] ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  IDLE

❌ Each active SM stalls 99% of the time.
```

### Case 3 — 32 blocks × 256 threads

```
SM utilization: 32/108 = 30%
Warp occupancy:  8/64  = 12.5%

[SM 0]  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  8 warps — some hiding
...
[SM 31] ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
[SM 32] ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  IDLE

⚠️  Mediocre on both dimensions.
```

### Case 4 — 349,526 blocks × 96 threads (the actual code, 32M elements)

```
SM utilization: 108/108 = 100%  (waves of blocks keep all SMs fed)
Warp occupancy:   3/64  = 4.7%  (96 threads = only 3 warps — not great)

[SM 0]   ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░  waves of blocks
[SM 1]   ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░
...
[SM107]  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░

✅ SM utilization: great
⚠️  Warp occupancy: low — 512 or 1024 threads/block would be much better
```

---

# PART 12: Small vs Large Data — Which Dimension Matters When

This resolves the apparent contradiction in the instructor's example.

He showed that **64 blocks × 32 threads was faster than 2 blocks × 1024 threads**, even though 2 blocks gives more warps per SM. Shouldn't more warps win?

No — because the data was tiny.

## Why Data Size Changes Everything

```
Instructor's example: 2048 elements × 4 bytes = 8KB total
A100 L1 cache per SM = 192KB

8KB fits entirely in L1 cache.
Memory latency with cache hit = ~20 cycles (not 600).
```

With 20-cycle latency, even 1 other warp covers the wait. Warp occupancy stops mattering. The only remaining factor is how many SMs are working.

```
SMALL DATA (fits in cache):
  Latency: ~20 cycles
  Even 1 warp can hide the wait
  → SM Utilization dominates
  → More SMs = faster, warp count barely matters

LARGE DATA (cache miss, goes to VRAM):
  Latency: ~600 cycles
  Need many warps to fill the 600-cycle gap
  → Both SM Utilization AND Warp Occupancy matter
```

## Applied to the Instructor's Example

```
Case A: 2 blocks × 1024 threads
  → 2 SMs active, 32 warps each
  → Only 2 things running in parallel
  → Slow

Case B: 64 blocks × 32 threads
  → 64 SMs active, 1 warp each
  → 64 things running in parallel
  → Cache covers the tiny 20-cycle stall anyway
  → Fast

Winner: Case B — 64 SMs > 2 SMs, and cache makes warp count irrelevant here
```

The instructor is teaching SM utilization. His example is sized specifically so that is the only factor. Not a contradiction — just a different data-size regime.

---

# PART 13: Wave Scheduling — When Blocks Outnumber SMs

With 349,526 blocks and only 108 SMs, the GPU runs them in **waves**.

```
Wave 1:  Blocks 0..107    → all 108 SMs working simultaneously
Wave 2:  As each SM finishes, it immediately grabs the next block
         Blocks 108..215
Wave 3:  Blocks 216..323
...
Wave ~3,236: Last blocks finish

The Gigathread Engine handles all of this automatically.
You launch 349,526 blocks. The GPU queues and assigns them.
```

```
Timeline view:

SM 0:  [Block 0]──[Block 108]──[Block 216]──...──[done]
SM 1:  [Block 1]──[Block 109]──[Block 217]──...
SM 2:  [Block 2]──[Block 110]──...
...
SM107: [Block 107]──[Block 215]──...
```

---

# PART 14: The Challenge of Very Large Vectors — Chunking

When vectors are large enough, a completely different problem appears: **not enough CPU RAM to hold the data**.

## The Problem

```
1 billion elements × 4 bytes (int) = 4GB per vector
3 vectors (A, B, C)                = 12GB total RAM needed

Laptop with 16GB RAM, 5-6GB free → runs out of RAM → crash
```

This is a **runtime error**, not a compile error. The program compiles fine. It fails silently when malloc cannot allocate enough CPU memory.

## Debugging: Printf Breadcrumbs

When you have a runtime error with no compile error, use printf to find the exact line that fails:

```c
printf("hello 00\n");    // if this prints, everything before here is fine
A = malloc(size);
printf("hello 01\n");    // if this prints, A allocation worked
B = malloc(size);
printf("hello 02\n");    // if this does NOT print, B allocation failed ← found it
```

The last printf that appears on screen = the last line that succeeded. The crash happened immediately after.

## The Solution: Chunking

Divide each vector into small pieces. Process one piece at a time, freeing RAM between iterations.

```
Full vector A (4GB): [ chunk 0 | chunk 1 | chunk 2 | ... | chunk 7 ]
Full vector B (4GB): [ chunk 0 | chunk 1 | chunk 2 | ... | chunk 7 ]
Full vector C (4GB): [ chunk 0 | chunk 1 | chunk 2 | ... | chunk 7 ]

Chunk size = 4GB / 8 = 500MB each

At any moment, RAM only holds 3 chunks, not 3 full vectors.
```

## Chunk Processing Flow

```
For each chunk i (0 to 7):

  1. malloc chunk_A, chunk_B, chunk_C       ← allocate RAM for this chunk only
  2. Fill chunk_A and chunk_B with values   ← using offset to get correct values
  3. cudaMemcpy chunk_A → d_A              ← send to GPU
     cudaMemcpy chunk_B → d_B
  4. vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, chunkSize)  ← GPU adds
  5. cudaMemcpy d_C → chunk_C              ← get result back
  6. Use chunk_C (print, save, etc.)
  7. free(chunk_A), free(chunk_B), free(chunk_C)  ← release RAM
  8. advance offset by chunkSize, repeat ──────────────────────────┐
                                                                    │
  ←──────────────────────────────────────────────────────────────── ┘

GPU-side d_A, d_B, d_C are allocated once at chunk size and stay allocated.
Only the CPU-side buffers are freed and reallocated each loop.
```

## The Offset

Each chunk starts at a different position in the full vector:

```
Chunk 0: offset = 0            → elements [0 .. chunkSize-1]
Chunk 1: offset = chunkSize    → elements [chunkSize .. 2*chunkSize-1]
Chunk 2: offset = 2*chunkSize
...

In code:
for (int chunk = 0; chunk < numChunks; chunk++) {
    int offset = chunk * chunkSize;
    for (int j = 0; j < chunkSize; j++) {
        chunk_A[j] = offset + j;         // correct position in full vector
        chunk_B[j] = SIZE - (offset + j);
    }
    // copy to GPU, run kernel, copy back
}
```

## RAM Usage Comparison

```
Without chunking:
  RAM: [full A — 4GB] [full B — 4GB] [full C — 4GB] = 12GB needed → crash

With chunking (8 chunks):
  RAM: [chunk_A — 500MB] [chunk_B — 500MB] [chunk_C — 500MB] = 1.5GB → fine
```

---

# PART 15: The Complete Code Walkthrough

```c
#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 1024*1024*32   // 32 million elements = 128MB per vector

// ── THE KERNEL ────────────────────────────────────────────────────────────────
// __global__ = runs on GPU, called from CPU
__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;  // unique index for this thread
    if (i < n) {            // bounds check
        C[i] = A[i] + B[i];
    }
}

int main() {

    // ── HOST POINTERS (CPU RAM) ───────────────────────────────────────────────
    int *A, *B, *C;

    // ── DEVICE POINTERS (GPU VRAM) ───────────────────────────────────────────
    int *d_A, *d_B, *d_C;

    int size = SIZE * sizeof(int);  // 32M × 4 bytes = 128MB

    // ── TIMING SETUP ─────────────────────────────────────────────────────────
    // cudaEvent_t = GPU timestamp variable
    cudaEvent_t start, stop;
    cudaEventCreate(&start);   // & = pass address so function can modify the variable
    cudaEventCreate(&stop);

    // ── ALLOCATE CPU MEMORY ───────────────────────────────────────────────────
    A = (int *)malloc(size);   // malloc returns an address, cast to int*
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
        B[i] = SIZE - i;       // A[i] + B[i] will always = SIZE
    }

    // ── ALLOCATE GPU MEMORY ───────────────────────────────────────────────────
    // cudaMalloc needs to WRITE an address into d_A
    // So it needs the address OF d_A → &d_A → type is int** (pointer to pointer)
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // ── COPY DATA CPU → GPU ───────────────────────────────────────────────────
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // ── START TIMER ──────────────────────────────────────────────────────────
    cudaEventRecord(start);

    // ── LAUNCH KERNEL ────────────────────────────────────────────────────────
    int threadsPerBlock = 96;
    // Ceiling division: guarantees enough blocks for all 32M elements
    // Without -1 trick: 32M/96 might miss the last partial block
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    // Launches ~349,526 blocks × 96 threads = ~33.5M threads
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, SIZE);

    // ── STOP TIMER ───────────────────────────────────────────────────────────
    cudaEventRecord(stop);

    // ── COPY RESULT GPU → CPU ─────────────────────────────────────────────────
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // ── CALCULATE ELAPSED TIME ───────────────────────────────────────────────
    cudaEventSynchronize(stop);   // wait for GPU to reach stop event before reading
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f milliseconds\n", milliseconds);

    // ── CLEANUP ───────────────────────────────────────────────────────────────
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```

---

# PART 16: End-to-End Flow — What Happens When You Run

```
vectorAdd<<<349526, 96>>>(d_A, d_B, d_C, SIZE)

Step 1: CPU passes the launch to the Gigathread Engine

Step 2: Gigathread Engine sends blocks to SMs in waves
        Wave 1: SM 0 ← Block 0, SM 1 ← Block 1, ... SM 107 ← Block 107

Step 3: Each SM receives its block (96 threads = 3 warps)
        Warp 0: threads 0-31
        Warp 1: threads 32-63
        Warp 2: threads 64-95

Step 4: SM's warp scheduler runs warps:
        Warp 0 requests A[i] from VRAM → STALL (~600 cycles)
        Switch to Warp 1 → requests memory → STALL
        Switch to Warp 2 → requests memory → STALL
        (Only 3 warps available — poor latency hiding — SM often idle)

Step 5: Data arrives, warps compute, write results to d_C

Step 6: Block finishes → SM immediately grabs next block from queue
        (Block 108, then 216, then 324...)

Step 7: All ~3,236 waves complete

Step 8: cudaMemcpy copies d_C back to CPU RAM
```

---

# PART 17: Reference Tables

## A100 Hardware Numbers

|Fact|Value|
|---|---|
|Total SMs|108|
|Cores per SM|128|
|Total CUDA cores|~13,824|
|Max threads per block|1024|
|Warp size|32 threads (hardware fixed)|
|Max warps per SM|64|
|Max threads per SM|2,048|
|L1 cache per SM|~192 KB|
|L2 cache (shared)|~40 MB|
|VRAM|40–80 GB|
|L1 latency|~20 cycles|
|L2 latency|~200 cycles|
|VRAM latency|~600 cycles|

## Key Concepts

|Concept|Definition|
|---|---|
|Thread|Smallest unit — runs kernel once on one element|
|Warp|32 threads in lockstep — what the scheduler actually sees|
|Block|Group of warps assigned to one SM — never split|
|Grid|All blocks in one kernel launch|
|Gigathread Engine|GPU hardware that assigns blocks to SMs|
|SIMT|Single Instruction Multiple Threads — all threads in warp run same instruction|
|SM Utilization|Active SMs ÷ total SMs|
|Warp Occupancy|Active warps on SM ÷ max warps SM can hold (64)|
|Latency Hiding|Filling memory wait cycles with other warps|
|Cache Miss|Data not in L1/L2, must go to VRAM — ~600 cycle penalty|
|Chunking|Splitting large data into pieces to fit in available CPU RAM|

## CUDA Key Functions

|Function|What It Does|
|---|---|
|`cudaMalloc(&ptr, size)`|Allocate memory on GPU VRAM|
|`cudaFree(ptr)`|Free GPU memory|
|`cudaMemcpy(dst, src, size, dir)`|Copy data between CPU and GPU|
|`cudaEventCreate(&event)`|Create a GPU timer event|
|`cudaEventRecord(event)`|Stamp current GPU time|
|`cudaEventSynchronize(event)`|Wait for GPU to reach that event|
|`cudaEventElapsedTime(&ms, start, stop)`|Compute time between two events|
|`cudaEventDestroy(event)`|Clean up timer event|
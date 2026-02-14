## CPU vs GPU Architecture

### The Fundamental Difference

**CPU Philosophy**: Few powerful workers doing complex tasks sequentially **GPU Philosophy**: Thousands of simple workers doing simple tasks in parallel

```
CPU:                          GPU:
┌─────────────┐              ┌──┬──┬──┬──┬──┬──┬──┬──┐
│   Complex   │              │░░│░░│░░│░░│░░│░░│░░│░░│
│   Worker    │              ├──┼──┼──┼──┼──┼──┼──┼──┤
│   (ALU)     │              │░░│░░│░░│░░│░░│░░│░░│░░│
│             │              ├──┼──┼──┼──┼──┼──┼──┼──┤
│   3-5 GHz   │              │░░│░░│░░│░░│░░│░░│░░│░░│  ... thousands more
└─────────────┘              ├──┼──┼──┼──┼──┼──┼──┼──┤
  4-16 cores                 │░░│░░│░░│░░│░░│░░│░░│░░│
                             └──┴──┴──┴──┴──┴──┴──┴──┘
                             Each = 1 CUDA Core
                             2,000-10,000+ cores
                             1.2-2 GHz each
```

### Memory (DRAM)

**DRAM = Dynamic Random Access Memory** (the main working memory)

- **CPU uses**: System RAM (DDR4/DDR5) - typically 8GB, 16GB, 32GB
- **GPU uses**: VRAM/GDDR (Graphics DDR) - typically 8GB, 12GB, 24GB

**Important**: CPU and GPU have **completely separate** memory spaces!

```
┌──────────────┐              ┌──────────────┐
│     CPU      │              │     GPU      │
├──────────────┤              ├──────────────┤
│  DDR4/DDR5   │              │ GDDR6/GDDR6X │
│   (System    │              │   (Video     │
│    RAM)      │              │    RAM)      │
│              │              │              │
│   16 GB      │◄───PCIe─────►│   24 GB      │
└──────────────┘              └──────────────┘
     Separate!                     Separate!
```

### ALUs (Arithmetic Logic Units)

**ALU = The actual calculator that does math operations**

**CPUs**:

- 4-16 powerful ALUs (in consumer CPUs)
- Each ALU is complex: can handle branches, predictions, out-of-order execution
- High clock speed: 3-5 GHz
- Great for: Complex sequential tasks, unpredictable branches

**GPUs**:

- Thousands of simple ALUs called **CUDA Cores** or **Streaming Processors**
- Each core is simpler: just does basic math
- Lower clock speed: 1.2-2 GHz
- Great for: Simple operations repeated thousands of times

**Example Comparison**:

```
Task: Add 1 million pairs of numbers

CPU Approach:                  GPU Approach:
┌──────┐                      ┌─┬─┬─┬─┬─┬─┬─┬─┐
│ Core │ → processes          │ │ │ │ │ │ │ │ │ Each core
│  1   │   all 1 million      ├─┼─┼─┼─┼─┼─┼─┼─┤ processes
└──────┘   one by one         │ │ │ │ │ │ │ │ │ just ONE
                               ├─┼─┼─┼─┼─┼─┼─┼─┤ addition
Time: ~300ms                   │ │ │ │ │ │ │ │ │
                               └─┴─┴─┴─┴─┴─┴─┴─┘
                               (10,000 cores)
                               
                               Time: ~3ms
```

### Control and Cache Architecture

**Cache = Super fast memory close to the processor**

**CPU Cache Structure**:

```
┌────────────────────────────────┐
│         L3 Cache (32-128 MB)   │ ← Shared by all cores
│              LARGE              │
└────────────────────────────────┘
              ▲
    ┌─────────┼─────────┐
    │         │         │
┌───▼───┐ ┌──▼───┐ ┌──▼───┐
│L2 (2MB)│ │L2    │ │L2    │ ← Per core
├───────┤ ├──────┤ ├──────┤
│L1     │ │L1    │ │L1    │ ← Per core
├───────┤ ├──────┤ ├──────┤
│ Core  │ │Core  │ │Core  │
│  #1   │ │ #2   │ │ #3   │
└───────┘ └──────┘ └──────┘

Good for: Sequential code with unpredictable patterns
```

**GPU Cache Structure**:

```
┌────────────────────────────────┐
│      L2 Cache (4-6 MB total)   │ ← Shared by ALL SMs
│            SMALLER              │
└────────────────────────────────┘
              ▲
    ┌─────────┼─────────┐
    │         │         │
┌───▼───┐ ┌──▼───┐ ┌──▼───┐
│L1/Shr │ │L1/Shr│ │L1/Shr│ ← Small per SM
│(128KB)│ │      │ │      │
├───────┤ ├──────┤ ├──────┤
│  SM   │ │  SM  │ │  SM  │ ← Each SM has
│  #1   │ │  #2  │ │  #3  │   many cores
│(128   │ │(128  │ │(128  │
│cores) │ │cores)│ │cores)│
└───────┘ └──────┘ └──────┘

Good for: Parallel code with predictable patterns
```

**Why the difference?**

- CPUs need large caches because code jumps around unpredictably
- GPUs rely on massive parallelism instead of caching - if one thread waits for memory, just switch to another thread!

### Clock Speeds - The Trade-off

```
CPU:                           GPU:
┌──────────────┐              ┌──────────────┐
│   4 cores    │              │ 10,000 cores │
│   @ 4 GHz    │              │  @ 1.5 GHz   │
│              │              │              │
│ Single task: │              │Single task:  │
│   VERY fast  │              │   slower     │
│              │              │              │
│ 10k tasks:   │              │10k tasks:    │
│   slow       │              │  VERY fast   │
└──────────────┘              └──────────────┘

Example times for 10,000 independent calculations:
CPU: 10,000 / 4 = 2,500 cycles
GPU: 10,000 / 10,000 = 1 cycle (in parallel!)
```

---

## Sequential vs Non-Sequential Instructions

### Sequential (Good for CPU)

**Definition**: Instructions where each step depends on the previous one

```
Example: Finding maximum in a list
┌─────┐     ┌─────┐     ┌─────┐
│ max │────►│ max │────►│ max │
│= 5  │     │= 8  │     │= 8  │
└─────┘     └─────┘     └─────┘
   ▲           ▲           ▲
   │           │           │
compare 5   compare 8   compare 3
   
Can't parallelize - need previous result!
```

**Examples of sequential tasks**:

- Parsing text character by character
- Recursive algorithms (Fibonacci, tree traversal)
- If-else chains where each decision depends on previous
- Searching for a specific item sequentially

### Non-Sequential/Parallel (Good for GPU)

**Definition**: Instructions that are independent and can run simultaneously

```
Example: Adding two arrays
Array A: [1, 2, 3, 4, 5, 6, 7, 8]
Array B: [9, 8, 7, 6, 5, 4, 3, 2]
         +  +  +  +  +  +  +  +    ← All independent!
Result:  [10,10,10,10,10,10,10,10]

GPU can do ALL at once:
Thread 1: 1+9  │ Thread 5: 5+5
Thread 2: 2+8  │ Thread 6: 6+4
Thread 3: 3+7  │ Thread 7: 7+3
Thread 4: 4+6  │ Thread 8: 8+2
       ALL PARALLEL!
```

**Examples of parallel tasks**:

- Image processing (each pixel independent)
- Matrix multiplication
- Physics simulations (each particle independent)
- Rendering (each pixel independent)
- Neural network training (many operations on same data)

---

## PCI Interfaces - How CPU and GPU Connect

### PCIe (PCI Express) Connection

```
┌─────────────────────────────────────────────────────────┐
│                    Motherboard                          │
│                                                          │
│  ┌──────────┐                        ┌──────────────┐  │
│  │   CPU    │                        │     GPU      │  │
│  ├──────────┤                        ├──────────────┤  │
│  │  DDR5    │                        │   GDDR6X     │  │
│  │  32GB    │                        │   24GB       │  │
│  └────┬─────┘                        └──────▲───────┘  │
│       │                                     │          │
│       │  ┌──────────────────────────────────┘          │
│       └──┤ PCIe 4.0 x16 Bus                            │
│          │ (32 GB/s bandwidth)                         │
│          └─────────────────────────────────────────────┤
│                                                          │
└─────────────────────────────────────────────────────────┘

Data must travel across PCIe to move between CPU and GPU!
This is often a BOTTLENECK!
```

### Separate DRAM - A Critical Concept

```
CPU Memory (DDR5):              GPU Memory (GDDR6):
┌──────────────────┐            ┌──────────────────┐
│ int x = 5;       │            │                  │
│ float y[1000];   │   COPY     │                  │
│ ....             │───────────►│ float y[1000];   │
│                  │  via PCIe  │                  │
└──────────────────┘            └──────────────────┘
  CPU can access                 GPU can access
      this!                          this!
```

**Key Points**:

- CPU cannot directly access GPU memory
- GPU cannot directly access CPU memory
- You must **explicitly copy** data between them
- Copying takes time - minimize transfers!

**VRAM/GDDR naming**:

- **VRAM** = Video RAM (general term for GPU memory)
- **GDDR** = Graphics DDR (the technology: GDDR5, GDDR6, GDDR6X)
- Sometimes called "GRAM" but GDDR is more common

---

## General GPU Architecture - The Complete Picture

### Understanding the Hierarchy: Core vs SM vs GPU

**This is the most confusing part for beginners! Let's break it down:**

```
HIERARCHY (smallest to largest):

1. CORE          = One tiny calculator (does 1 operation)
2. SM            = A team of cores + shared workspace
3. GPU           = Many SMs + global memory
```

### Level 1: What is a CORE?

**A Core = One processing unit that does ONE operation at a time**

```
┌──────────────┐
│   ONE CORE   │
│              │
│  ┌────────┐  │
│  │  ALU   │  │ ← Does ONE math operation
│  └────────┘  │   (add, multiply, etc.)
│              │
└──────────────┘

Types of cores:
• FP32 Core  → Does 32-bit float math (3.14 + 2.71)
• INT32 Core → Does 32-bit integer math (5 + 3)
• FP64 Core  → Does 64-bit float math (high precision)
• Tensor Core → Does matrix operations (AI specific)
• RT Core → Does ray tracing (graphics specific)
```

### Level 2: What is an SM (Streaming Multiprocessor)?

**An SM = A group of cores that work together + shared resources**

Think of it like a **department in a company**:

- Multiple workers (cores)
- Shared office space (shared memory)
- A manager (scheduler)
- Private desks (registers)

```
┌────────────────────────────────────────────────────────────┐
│              ONE SM (Streaming Multiprocessor)             │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │   SHARED MEMORY / L1 CACHE (48-128 KB)               │ │
│  │   ═══════════════════════════════════════            │ │
│  │   Like a shared whiteboard - all cores in THIS       │ │
│  │   SM can read/write here. FAST access!               │ │
│  └──────────────────────────────────────────────────────┘ │
│                           ▲                                 │
│                           │                                 │
│  ┌────────────────────────┴───────────────────────────┐   │
│  │         REGISTER FILE (65,536 registers)           │   │
│  │         ════════════════════════════════           │   │
│  │   Like private desks - each thread gets some       │   │
│  │   registers. SUPER FAST but private!               │   │
│  └────────────────────────────────────────────────────┘   │
│                           ▲                                 │
│                           │                                 │
│  ┌────────────────────────┴───────────────────────────┐   │
│  │       WARP SCHEDULER & DISPATCH UNITS              │   │
│  │       ════════════════════════════════             │   │
│  │   The "manager" - decides which cores do what      │   │
│  │   Schedules groups of 32 threads ("warps")         │   │
│  └────────────────────────────────────────────────────┘   │
│                           │                                 │
│          ┌────────────────┼────────────────┐               │
│          │                │                │               │
│  ┌───────▼──────┐  ┌──────▼─────┐  ┌──────▼─────┐        │
│  │ FP32 Cores   │  │ INT32 Cores│  │Tensor Cores│        │
│  │ (64 cores)   │  │ (64 cores) │  │ (4 cores)  │        │
│  │              │  │            │  │            │        │
│  │ ████████████ │  │ ████████████│  │ ████       │        │
│  │ ████████████ │  │ ████████████│  │            │        │
│  │ ████████████ │  │ ████████████│  │            │        │
│  └──────────────┘  └────────────┘  └────────────┘        │
│          │                │                │               │
│  ┌───────▼────────────────▼────────────────▼──────┐       │
│  │         LOAD/STORE UNITS (LD/ST)                │       │
│  │         ════════════════════════                │       │
│  │   Special units for reading/writing memory      │       │
│  └─────────────────────────────────────────────────┘       │
│                                                             │
│  Example: NVIDIA RTX 4090 SM has:                          │
│  • 128 FP32 cores                                          │
│  • 128 INT32 cores                                         │
│  • 4 Tensor Cores (4th gen)                                │
│  • 64KB-128KB Shared Memory/L1                             │
│  • 4 Warp Schedulers                                       │
└─────────────────────────────────────────────────────────────┘
```

**Key Point**: When people say "CUDA cores," they mean the individual FP32/INT32 cores. An SM **contains** many cores!

### Level 3: The Complete GPU

```
┌──────────────────────────────────────────────────────────────┐
│                       ENTIRE GPU CHIP                         │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         GLOBAL MEMORY (GDDR6X) - 24 GB                 │ │
│  │         ════════════════════════════════               │ │
│  │  Accessible by ALL SMs, but SLOW (relatively)          │ │
│  │  Like a warehouse - lots of space, far away            │ │
│  └────────────────────────────────────────────────────────┘ │
│                              ▲                                │
│                              │                                │
│  ┌───────────────────────────┴──────────────────────────┐   │
│  │           L2 CACHE (6 MB) - Shared by all            │   │
│  │           ════════════════════════════               │   │
│  │  Faster than global memory, slower than L1           │   │
│  └──────────────────────────────────────────────────────┘   │
│                              ▲                                │
│         ┌────────────────────┼────────────────────┐          │
│         │                    │                    │          │
│    ┌────▼─────┐         ┌───▼──────┐        ┌───▼──────┐   │
│    │   SM 0   │         │   SM 1   │  ...   │  SM 127  │   │
│    │          │         │          │        │          │   │
│    │ 128 FP32 │         │ 128 FP32 │        │ 128 FP32 │   │
│    │ 128 INT32│         │ 128 INT32│        │ 128 INT32│   │
│    │ 4 Tensor │         │ 4 Tensor │        │ 4 Tensor │   │
│    └──────────┘         └──────────┘        └──────────┘   │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           GLOBAL SCHEDULER                             │ │
│  │           Distributes work to available SMs            │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  RTX 4090 Stats:                                             │
│  • 128 SMs × 128 cores/SM = 16,384 CUDA cores total!        │
│  • 24 GB GDDR6X memory                                       │
│  • 512 Tensor Cores (4th gen)                                │
└──────────────────────────────────────────────────────────────┘
```

---

## Memory Hierarchy - Who Can Access What?

This is **critical** for writing fast CUDA code!

```
┌─────────────────────────────────────────────────────────────┐
│                    SPEED vs SIZE                            │
│                                                              │
│  FASTEST ↑                                          SMALLEST│
│          │                                                   │
│  ┌───────┴────────────────────────────────────┐            │
│  │  REGISTERS (per thread)                    │            │
│  │  • 32-bit chunks                            │            │
│  │  • Each thread gets ~20-60 registers       │            │
│  │  • Access time: 1 clock cycle              │            │
│  │  • Who can access: ONLY that specific      │            │
│  │    thread (thread-private)                 │            │
│  │  • Example: int x = 5; // stored in reg   │            │
│  └────────────────────────────────────────────┘            │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────┐           │
│  │  SHARED MEMORY / L1 CACHE (per SM)         │           │
│  │  • 48-128 KB per SM                         │           │
│  │  • Access time: ~5 clock cycles            │           │
│  │  • Who can access: ALL threads within      │           │
│  │    the same thread block on this SM        │           │
│  │  • Programmable - you control what goes    │           │
│  │    here!                                    │           │
│  │  • Example: __shared__ float cache[256];  │           │
│  └─────────────────────────────────────────────┘           │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────┐           │
│  │  L2 CACHE (global, per GPU)                │           │
│  │  • 4-6 MB total                             │           │
│  │  • Access time: ~200 clock cycles          │           │
│  │  • Who can access: ALL SMs                 │           │
│  │  • Automatic - you don't control this      │           │
│  └─────────────────────────────────────────────┘           │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────┐           │
│  │  GLOBAL MEMORY (VRAM)                       │           │
│  │  • 8-24 GB (huge!)                          │           │
│  │  • Access time: ~400-800 clock cycles      │           │
│  │  • Who can access: ALL threads on GPU      │           │
│  │  • Must allocate explicitly                │           │
│  │  • Example: cudaMalloc(&ptr, size);        │           │
│  └─────────────────────────────────────────────┘           │
│                     │                                        │
│  SLOWEST ↓                                         LARGEST  │
└─────────────────────────────────────────────────────────────┘

Takeaway: Use faster memory when possible!
• Registers: Best, but limited per thread
• Shared: Good for threads that need to cooperate
• Global: Use for large data, but minimize access
```

### Visual Example - Thread Accessing Memory

```
Thread #42 wants to compute: result = a + b + c

┌──────────────────────────────────────────────────────┐
│              Thread #42's View                       │
│                                                       │
│  int a = 5;              ┌─────────────┐            │
│  (stored in register) ──►│  Register   │ 1 cycle    │
│                          │   a = 5     │            │
│                          └─────────────┘            │
│                                                       │
│  __shared__ int b;       ┌─────────────┐            │
│  (stored in shared) ────►│  Shared     │ 5 cycles   │
│                          │   b = 10    │            │
│                          └─────────────┘            │
│                                                       │
│  int* c;                 ┌─────────────┐            │
│  (in global memory) ────►│  Global     │ 400 cycles │
│                          │   c = 15    │            │
│                          └─────────────┘            │
│                                                       │
│  result = 5 + 10 + 15 = 30                          │
│  Total time: ~406 cycles (global access dominates!)  │
└──────────────────────────────────────────────────────┘

If all were in registers: ~3 cycles total!
This is why memory optimization matters!
```

---

## Load/Store Units - The Memory Movers

```
┌────────────────────────────────────────────────────┐
│         Why separate Load/Store units?             │
│                                                     │
│  Without LD/ST units:                              │
│  Core must stop computing to fetch data            │
│  ┌──────┐                                          │
│  │ Core │ "I need data... waiting... waiting..."   │
│  └───┬──┘                                          │
│      │ ◄───── fetch from memory (400 cycles)       │
│      ▼                                              │
│  ┌──────┐                                          │
│  │ Core │ "Finally! Now computing..."              │
│  └──────┘                                          │
│  WASTED TIME!                                      │
│                                                     │
│  With LD/ST units:                                 │
│  Core computes while LD/ST fetches data            │
│  ┌──────┐                    ┌──────┐             │
│  │ Core │ "Computing..."     │LD/ST │ "Fetching." │
│  └──────┘                    └───┬──┘             │
│      │                           │                 │
│      ▼                           ▼                 │
│  Still working!           Memory access happens    │
│                           in parallel!             │
│                                                     │
│  BETTER EFFICIENCY!                                │
└────────────────────────────────────────────────────┘
```

**LD/ST Units** (Load/Store Units):

- Specialized hardware for memory operations
- Handle reading (loading) and writing (storing) data
- Allow computation and memory access to overlap
- Typically 32-64 LD/ST units per SM

---

## Architecture Whitepapers

Where to find detailed technical specifications:

**NVIDIA's Official Whitepapers**:

- Published for each major architecture
- Available at: `docs.nvidia.com` or search "NVIDIA [Architecture Name] Whitepaper"

**Examples**:

- "NVIDIA Ampere Architecture Whitepaper"
- "NVIDIA Hopper Architecture Whitepaper"
- "NVIDIA Ada Lovelace Architecture Whitepaper"

**What's inside**:

- SM detailed design
- Memory hierarchy specifications
- New features and improvements
- Performance characteristics
- Use cases and optimizations

**When to read them**: When you need to optimize code for a specific GPU generation

---

## NVIDIA Categories and Generations

### The Two Naming Systems (This Confuses Everyone!)

```
┌──────────────────────────────────────────────────────────┐
│              NVIDIA's DUAL NAMING SYSTEM                  │
│                                                           │
│  ARCHITECTURE               PRODUCT CATEGORY             │
│  (How it's built)           (Who it's for)               │
│                                                           │
│  Named after                Named by purpose             │
│  scientists                                               │
│       │                            │                      │
│       │                            │                      │
│  ┌────▼────────┐             ┌─────▼──────┐             │
│  │  Ampere     │────────────►│ RTX 3090   │ (GeForce)   │
│  │             │    uses      │ RTX A6000  │ (Quadro)    │
│  │  (2020)     │────────────►│ A100       │ (HPC)       │
│  └─────────────┘             └────────────┘             │
│       │                            │                      │
│       │                            │                      │
│  Design/features              Market segment             │
└──────────────────────────────────────────────────────────┘

Same architecture → Different products!
```

### Architecture Names (Scientists)

**What it means**: The underlying technology and design

**Timeline and Key Features**:

```
2016  PASCAL      • First with HBM2 memory
                  • No Tensor Cores yet

2017  VOLTA       • ★ Introduced Tensor Cores (1st gen)
                  • 8x faster for matrix operations
                  • FP64 performance boost

2018  TURING      • ★ Introduced RT Cores (ray tracing) ✓
                  • 2nd gen Tensor Cores
                  • First real-time ray tracing

2020  AMPERE      • 3rd gen Tensor Cores
                  • 2nd gen RT Cores
                  • Improved power efficiency

2022  ADA         • 4th gen Tensor Cores
      LOVELACE    • 3rd gen RT Cores
                  • DLSS 3.0 support

2022  HOPPER      • Transformer Engine
                  • For datacenter AI only
                  • No consumer products
```

**Format**: X.Y (Major.Minor)

- Major (X): Architecture generation
- Minor (Y): Improvements/variants within that architecture

**Examples**:

- 7.0 = Volta (V100)
- 7.5 = Turing (RTX 2080)
- 8.0 = Ampere (A100)
- 8.6 = Ampere (RTX 3090)
- 8.9 = Ada Lovelace (RTX 4090)
- 9.0 = Hopper (H100)

---

### Product Categories (Market Segments)

Now, the **same architecture** gets packaged into different products:

```
┌─────────────────────────────────────────────────────────────┐
│                  PRODUCT CATEGORIES                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  GEFORCE (Gaming/Consumer)                           │  │
│  │  ══════════════════════════                          │  │
│  │  Format: RTX 4090, RTX 3080, GTX 1660               │  │
│  │  Target: Gamers, streamers, content creators         │  │
│  │  Features:                                            │  │
│  │    • Display outputs (HDMI, DisplayPort)            │  │
│  │    • Optimized for gaming                            │  │
│  │    • Consumer pricing                                │  │
│  │  Example: RTX 3090 (Ampere architecture)            │  │
│  │           24GB GDDR6X, $1,499                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  QUADRO / RTX PROFESSIONAL                           │  │
│  │  ══════════════════════════════                      │  │
│  │  Format: RTX A6000, RTX A4000, Quadro RTX 8000      │  │
│  │  Target: Engineers, designers, 3D artists            │  │
│  │  Features:                                            │  │
│  │    • ECC memory (error correction)                   │  │
│  │    • Certified drivers for CAD/professional apps     │  │
│  │    • Longer support lifecycle                        │  │
│  │    • Higher price for reliability                    │  │
│  │  Example: RTX A6000 (Ampere architecture)           │  │
│  │           48GB GDDR6 with ECC, $4,650                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  TESLA / A-SERIES / H-SERIES (HPC/AI)               │  │
│  │  ══════════════════════════════════════              │  │
│  │  Format: A100, H100, V100                            │  │
│  │  Target: Data centers, AI research, supercomputing   │  │
│  │  Features:                                            │  │
│  │    • NO display outputs (compute only!)             │  │
│  │    • Maximum compute performance                     │  │
│  │    • HBM2/HBM3 memory (ultra-high bandwidth)        │  │
│  │    • NVLink for multi-GPU                            │  │
│  │    • Better FP64 performance                         │  │
│  │  Example: A100 (Ampere architecture)                │  │
│  │           40GB/80GB HBM2e, $10,000-15,000           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  TEGRA (Mobile/Embedded)                             │  │
│  │  ══════════════════════                              │  │
│  │  Format: Tegra X1, Xavier, Orin                      │  │
│  │  Target: Autonomous vehicles, robotics, edge AI      │  │
│  │  Features:                                            │  │
│  │    • Integrated CPU + GPU in one chip               │  │
│  │    • Power efficient (5-50W)                         │  │
│  │    • Small form factor                               │  │
│  │  Example: Jetson AGX Orin                           │  │
│  │           (Used in self-driving cars, drones)        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Quick Comparison Table

|Category|Example|Architecture|Price|Best For|
|---|---|---|---|---|
|GeForce|RTX 3090|Ampere|$1,500|Gaming, ML hobbyists|
|Quadro/RTX Pro|RTX A6000|Ampere|$4,650|CAD, rendering|
|HPC|A100|Ampere|$12,000|AI training, research|
|Tegra|Orin|Ampere|$500-2,000|Robotics, automotive|

**Key Insight**: Same Ampere architecture, completely different use cases and features!

---

## Useful Resource: TechPowerUp

**Website**: `techpowerup.com/gpu-specs`

**What it provides**:

- Complete specs for virtually every GPU ever made
- Architecture details
- Core counts, memory specs
- Release dates, prices
- Comparison tools

**Example search**: "RTX 4090 specs" → shows:

- 16,384 CUDA cores
- 128 SMs
- 24GB GDDR6X
- Ada Lovelace architecture
- 450W TDP
- etc.

---

## GPU Chip vs GPU Card

### The Confusion Explained

```
┌────────────────────────────────────────────────────────┐
│                                                         │
│  THE CHIP (Die)              THE CARD (Board)          │
│                                                         │
│  ┌─────────────┐             ┌──────────────────────┐ │
│  │  ███████    │             │ ┌────────────────┐   │ │
│  │  ███████    │             │ │   GPU Chip     │   │ │
│  │  GA102     │◄────────────┤ │   (GA102)      │   │ │
│  │  ███████    │   mounted   │ └────────────────┘   │ │
│  │  ███████    │     on      │                      │ │
│  └─────────────┘             │  ┌──┐ ┌──┐ ┌──┐     │ │
│                               │  │  │ │  │ │  │     │ │
│  Just the silicon            │  └──┘ └──┘ └──┘     │ │
│  Contains:                    │   VRAM chips         │ │
│  • CUDA cores                │                      │ │
│  • SMs                        │  Cooling, power,    │ │
│  • Cache                      │  PCIe connector,    │ │
│  • Control logic             │  display ports      │ │
│                               │                      │ │
│  ~2cm x 2cm                  │  ~28cm x 12cm       │ │
└────────────────────────────────────────────────────────┘
```

**The Chip (GPU Die)**:

- The actual silicon processor
- Contains all the SMs, cores, caches
- Example names: GA102, AD102, GH100

**The GPU Card/Board**:

- Complete product you buy
- Includes:
    - The GPU chip soldered to PCB
    - VRAM chips (GDDR6 modules)
    - Power delivery (VRMs, capacitors)
    - Cooling (heatsink, fans, sometimes liquid)
    - Display outputs (HDMI, DP ports)
    - PCIe connector
- Example names: RTX 4090, RTX 3080

**Analogy**:

- Chip = The CPU die
- Card = The entire graphics card you plug into your motherboard

---

## Architecture Chip Families

Each architecture has multiple chip variants for different market tiers:

```
AMPERE ARCHITECTURE (2020)
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  GA102 (Top tier)          GA104 (Mid tier)             │
│  ┌──────────────┐          ┌──────────────┐            │
│  │ 128 SMs      │          │ 48 SMs       │            │
│  │ 16,384 cores │          │ 6,144 cores  │            │
│  │              │          │              │            │
│  │ Used in:     │          │ Used in:     │            │
│  │ • RTX 3090   │          │ • RTX 3070   │            │
│  │ • RTX A6000  │          │ • RTX 3060 Ti│            │
│  └──────────────┘          └──────────────┘            │
│                                                          │
│  GA100 (Datacenter)        GA106 (Budget)               │
│  ┌──────────────┐          ┌──────────────┐            │
│  │ 128 SMs      │          │ 24 SMs       │            │
│  │ + More FP64  │          │ 3,072 cores  │            │
│  │              │          │              │            │
│  │ Used in:     │          │ Used in:     │            │
│  │ • A100       │          │ • RTX 3060   │            │
│  └──────────────┘          └──────────────┘            │
│                                                          │
│  GA = GeForce Ampere                                    │
└─────────────────────────────────────────────────────────┘

ADA LOVELACE ARCHITECTURE (2022)
┌─────────────────────────────────────────────────────────┐
│  AD102 → RTX 4090, RTX 4080                             │
│  AD103 → RTX 4080 (lower tier)                          │
│  AD104 → RTX 4070 Ti, RTX 4070                          │
│  AD106 → RTX 4060 Ti                                    │
│                                                          │
│  AD = Ada                                               │
└─────────────────────────────────────────────────────────┘
```

**Naming Pattern**:

- **GA** = GeForce Ampere
- **AD** = Ada
- **GH** = GPU Hopper
- **GV** = GeForce Volta

**Higher number = Bigger chip = More cores = Better performance**

---

## Main Parameters for GPU Performance Comparison

When comparing GPUs, focus on these key specs:

### 1. Memory Bandwidth (GB/s)

**What it is**: How much data can move to/from memory per second

**Formula**:

```
Bandwidth = (Memory Speed × Bus Width) / 8

Example: RTX 4090
Memory Speed: 21 Gbps (gigabits per second)
Bus Width: 384-bit
Bandwidth = (21 × 384) / 8 = 1,008 GB/s
```

**Components**:

```
┌──────────────────────────────────────────────────────┐
│         MEMORY BANDWIDTH FACTORS                      │
│                                                       │
│  1. MEMORY SPEED (Clock rate)                        │
│     • Measured in Gbps or MHz                        │
│     • GDDR6: ~14-16 Gbps                            │
│     • GDDR6X: ~19-21 Gbps (faster!)                 │
│     • HBM2e: ~2.4 Gbps per pin (but more pins!)     │
│                                                       │
│  2. BUS WIDTH (How many "lanes")                     │
│     • Measured in bits                               │
│     • 256-bit (mid-range): 32 bytes at once         │
│     • 384-bit (high-end): 48 bytes at once          │
│     • 512-bit (top-end): 64 bytes at once           │
│                                                       │
│  3. MEMORY TECHNOLOGY                                │
│     • DDR4: Basic (rare in GPUs now)                │
│     • GDDR6: Standard for most GPUs                 │
│     • GDDR6X: 30-50% faster than GDDR6              │
│     • HBM2/HBM3: Ultra-wide bus (4096-bit+)         │
│                  Highest bandwidth for datacenter    │
│                                                       │
│  Why it matters:                                     │
│  • More bandwidth = Less waiting for data           │
│  • Critical for large models, high-res textures     │
│  • Can be bottleneck even with fast cores           │
└──────────────────────────────────────────────────────┘
```

**Real Examples**:

|GPU|Memory Type|Bus Width|Bandwidth|
|---|---|---|---|
|RTX 3060|GDDR6|192-bit|360 GB/s|
|RTX 4090|GDDR6X|384-bit|1,008 GB/s|
|A100|HBM2e|5120-bit|1,935 GB/s|
|H100|HBM3|5120-bit|3,350 GB/s|

### 2. Throughput (TFLOPS)

**What it is**: Trillions of floating-point operations per second

**Formula**:

```
TFLOPS = (Core Count × 2 × Clock Speed) / 1000

Example: RTX 4090
Cores: 16,384 CUDA cores
Clock: 2.52 GHz (boost)
FP32 TFLOPS = (16,384 × 2 × 2.52) / 1000 ≈ 82.6 TFLOPS
```

**Why "× 2"?** Each core can do a **fused multiply-add (FMA)** per clock:

- One multiply operation
- One add operation
- = 2 FLOPs per cycle

**Components**:

```
┌───────────────────────────────────────────────────────┐
│           THROUGHPUT COMPONENTS                       │
│                                                        │
│  1. CORE COUNT                                        │
│     • More cores = more parallel work                │
│     • RTX 3060: 3,584 cores                          │
│     • RTX 4090: 16,384 cores                         │
│                                                        │
│  2. CLOCK SPEED                                       │
│     • Base clock: Guaranteed minimum                 │
│     • Boost clock: Maximum under load                │
│     • Higher = faster per core                       │
│     • RTX 4090: 2.23 GHz base, 2.52 GHz boost       │
│                                                        │
│  Different for each precision:                       │
│  • FP32 (single): Standard CUDA cores               │
│  • FP16 (half): 2x throughput of FP32               │
│  • INT8: Even faster (for quantized ML)             │
│  • FP64 (double): Much slower on consumer GPUs      │
│                                                        │
│  Example: RTX 4090                                   │
│  • FP32: 82.6 TFLOPS                                │
│  • FP16: 165.2 TFLOPS (2x)                          │
│  • FP64: 1.29 TFLOPS (64x slower!)                  │
│                                                        │
│  vs A100 (datacenter):                               │
│  • FP32: 19.5 TFLOPS                                │
│  • FP64: 9.7 TFLOPS (only 2x slower - way better!) │
│  • Tensor FP16: 312 TFLOPS (with Tensor Cores)      │
└───────────────────────────────────────────────────────┘
```

### 3. New Features: Tensor Cores & Data Types

**Tensor Cores** = Specialized hardware for matrix operations

```
EVOLUTION OF TENSOR CORES
┌──────────────────────────────────────────────────────┐
│                                                       │
│  PASCAL (2016)                                       │
│  • NO Tensor Cores                                   │
│  • Matrix ops done on regular CUDA cores            │
│                                                       │
│  VOLTA (2017) ★                                      │
│  • ★ FIRST Tensor Cores introduced! ✓               │
│  • 4x4 matrix multiply-accumulate per clock         │
│  • FP16 input, FP32 accumulate                      │
│  • ~8-12x faster than CUDA cores for matrix ops     │
│                                                       │
│  TURING (2018)                                       │
│  • 2nd gen Tensor Cores                             │
│  • Added INT8, INT4 support                         │
│  • Better for inference                              │
│                                                       │
│  AMPERE (2020)                                       │
│  • 3rd gen Tensor Cores                             │
│  • Added TF32 (TensorFloat-32) data type            │
│  • Added BF16 (BFloat16)                            │
│  • Sparsity support (2x faster for sparse)          │
│                                                       │
│  ADA LOVELACE (2022)                                 │
│  • 4th gen Tensor Cores                             │
│  • FP8 support                                       │
│  • Even more efficient                               │
│                                                       │
│  HOPPER (2022)                                       │
│  • Transformer Engine                                │
│  • Dynamic precision switching                       │
│  • Optimized for Large Language Models              │
└──────────────────────────────────────────────────────┘
```

**Impact Example**:

```
Matrix Multiplication (1024×1024 matrices)

Pascal (without Tensor Cores):
Using CUDA cores: ~0.5 TFLOPS
Time: 4.2 ms

Volta (with Tensor Cores):
Using Tensor Cores: ~120 TFLOPS  
Time: 0.02 ms

Speed up: ~210x faster! ✓
(Note: This is for FP16 mixed precision)
```

---

## Data Types and Sizes

### Integer Types

```
┌────────────────────────────────────────────────────┐
│              INTEGER TYPES                         │
│                                                     │
│  INT8 (8-bit)                                      │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┐                       │
│  │  │  │  │  │  │  │  │  │ = 1 byte              │
│  └──┴──┴──┴──┴──┴──┴──┴──┘                       │
│  Range: -128 to 127                                │
│  Use: Quantized neural networks, small counters    │
│                                                     │
│  INT32 (32-bit) - Most common                      │
│  ┌──────────────────────────────┐                 │
│  │                              │ = 4 bytes        │
│  └──────────────────────────────┘                 │
│  Range: -2,147,483,648 to 2,147,483,647           │
│  Use: Indexing, counting, general integer math     │
│                                                     │
│  INT64 (64-bit)                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │                                              │ │
│  └──────────────────────────────────────────────┘ │
│  = 8 bytes                                         │
│  Range: ±9 quintillion                            │
│  Use: Large counters, timestamps                   │
└────────────────────────────────────────────────────┘
```

### Floating-Point Types

```
┌────────────────────────────────────────────────────────┐
│           FLOATING-POINT TYPES                         │
│                                                         │
│  FP16 (Half Precision) - 16 bits = 2 bytes             │
│  ┌───┬────────┬──────────────────┐                    │
│  │ S │ Exp(5) │   Mantissa(10)   │                    │
│  └───┴────────┴──────────────────┘                    │
│  Sign: 1 bit, Exponent: 5 bits, Mantissa: 10 bits     │
│  Range: ±65,504 (limited!)                             │
│  Precision: ~3 decimal digits                          │
│  Use Cases:                                            │
│    • Deep learning (training & inference)              │
│    • Gaming graphics (colors, positions)               │
│    • When speed > accuracy                             │
│  Throughput: 2x faster than FP32                       │
│                                                         │
│  ─────────────────────────────────────────────────    │
│                                                         │
│  FP32 (Single Precision) - 32 bits = 4 bytes          │
│  ┌───┬─────────┬────────────────────────┐             │
│  │ S │ Exp(8)  │    Mantissa(23)        │             │
│  └───┴─────────┴────────────────────────┘             │
│  Sign: 1 bit, Exponent: 8 bits, Mantissa: 23 bits     │
│  Range: ±3.4 × 10³⁸                                    │
│  Precision: ~7 decimal digits                          │
│  Use Cases:                                            │
│    • DEFAULT for most GPU computing                    │
│    • Graphics rendering                                │
│    • General scientific computing                      │
│    • Good balance of speed and accuracy                │
│  Throughput: Standard baseline (100%)                  │
│                                                         │
│  ─────────────────────────────────────────────────    │
│                                                         │
│  FP64 (Double Precision) - 64 bits = 8 bytes          │
│  ┌───┬──────────┬────────────────────────────────┐    │
│  │ S │  Exp(11) │      Mantissa(52)              │    │
│  └───┴──────────┴────────────────────────────────┘    │
│  Sign: 1 bit, Exponent: 11 bits, Mantissa: 52 bits    │
│  Range: ±1.8 × 10³⁰⁸                                   │
│  Precision: ~15-17 decimal digits                      │
│  Use Cases:                                            │
│    • Scientific simulations (weather, physics)         │
│    • Financial calculations                            │
│    • When precision is CRITICAL                        │
│  Throughput:                                           │
│    • Consumer GPUs: 1/32 to 1/64 of FP32 (slow!)      │
│    • HPC GPUs (A100/H100): 1/2 of FP32 (much better!) │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### Specialized Tensor Core Types

```
┌────────────────────────────────────────────────────┐
│       TENSOR CORE DATA TYPES                       │
│       (Ampere and newer)                           │
│                                                     │
│  TF32 (TensorFloat-32) - Ampere+                   │
│  ┌───┬─────────┬───────────┐                      │
│  │ S │ Exp(8)  │ Mant(10)  │ = 19 bits            │
│  └───┴─────────┴───────────┘                      │
│  • FP32 range, FP16 precision                      │
│  • Automatic in Tensor Cores                       │
│  • Best of both worlds for ML                      │
│                                                     │
│  BF16 (BrainFloat16)                               │
│  ┌───┬─────────┬───────────┐                      │
│  │ S │ Exp(8)  │ Mant(7)   │ = 16 bits            │
│  └───┴─────────┴───────────┘                      │
│  • Same range as FP32                              │
│  • Less precision than FP16                        │
│  • Popular in ML frameworks                        │
│                                                     │
│  FP8 (8-bit Float) - Ada/Hopper                    │
│  • Extremely fast                                  │
│  • For inference                                   │
│  • Requires careful scaling                        │
└────────────────────────────────────────────────────┘
```

### Precision vs Speed Trade-off

```
            ACCURACY ↔ SPEED

High Precision               Low Precision
(Slow)                       (Fast)
   │                            │
   ▼                            ▼
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│ FP64 │→ │ FP32 │→ │ FP16 │→ │ INT8 │
└──────┘  └──────┘  └──────┘  └──────┘
   1x       32x       64x      128x
 speed    faster    faster   faster

 15-17      7         3        N/A
 digits   digits   digits   (integer)

Use When:
Scientific  Default   ML       Inference
Computing   Choice   Training  Quantized
```

---

## Compute Capability (CC) Number

### Format: X.Y

**X = Major Version** (Architecture) **Y = Minor Version** (Variant/improvements)

```
┌─────────────────────────────────────────────────────┐
│         COMPUTE CAPABILITY VERSIONS                  │
│                                                      │
│  6.x - PASCAL (2016)                                │
│    6.0: Tesla P100                                  │
│    6.1: GTX 1080, GTX 1070                          │
│                                                      │
│  7.x - VOLTA/TURING (2017-2018)                     │
│    7.0: Tesla V100 (Volta, first Tensor Cores!)     │
│    7.5: RTX 2080, RTX 2070 (Turing, first RT)      │
│                                                      │
│  8.x - AMPERE (2020)                                │
│    8.0: A100 (datacenter)                           │
│    8.6: RTX 3090, RTX 3080 (consumer)               │
│    8.9: Ada Lovelace (actually Ada, not Ampere!)    │
│          RTX 4090, RTX 4080                         │
│                                                      │
│  9.x - HOPPER (2022)                                │
│    9.0: H100 (datacenter only)                      │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### What CC Determines

```
┌──────────────────────────────────────────────────────┐
│     COMPUTE CAPABILITY DETERMINES:                   │
│                                                       │
│  1. COMPATIBLE CUDA VERSION                          │
│     CC 7.5 needs CUDA 10.0+                          │
│     CC 8.6 needs CUDA 11.1+                          │
│     CC 9.0 needs CUDA 11.8+                          │
│                                                       │
│  2. AVAILABLE FEATURES                               │
│     ┌──────────┬──────────┬──────────┐              │
│     │  CC 7.0  │  CC 8.0  │  CC 9.0  │              │
│     ├──────────┼──────────┼──────────┤              │
│     │ Tensor   │ 3rd gen  │ 4th gen  │              │
│     │ Cores v1 │ Tensor   │ Tensor   │              │
│     │          │          │          │              │
│     │ No TF32  │ TF32     │ FP8      │              │
│     │ No async │ Async    │ Trans-   │              │
│     │ copy     │ copy     │ former   │              │
│     │          │          │ Engine   │              │
│     └──────────┴──────────┴──────────┘              │
│                                                       │
│  3. PTX INSTRUCTION SET                              │
│     PTX = Parallel Thread Execution                  │
│     (NVIDIA's intermediate assembly language)        │
│                                                       │
│     Higher CC → More PTX instructions available      │
│                                                       │
│     CUDA C++ ──compile──► PTX ──assemble──► SASS    │
│     (source)             (portable)       (machine)  │
│                                                       │
│  4. MEMORY & THREAD LIMITS                           │
│     Max threads per block                            │
│     Max shared memory per block                      │
│     Max registers per thread                         │
│     (These increase with newer CC)                   │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### How to Find Your GPU's CC

**Method 1: Check NVIDIA docs**

- Visit: `docs.nvidia.com/cuda/cuda-c-programming-guide`
- Look for "Compute Capabilities" table

**Method 2: Run deviceQuery**

```bash
# In CUDA Samples directory
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
make
./deviceQuery

Output:
  Device 0: "NVIDIA GeForce RTX 3090"
  CUDA Capability Major/Minor version number: 8.6
                                              ↑
                                          CC = 8.6
```

**Method 3: In your CUDA code**

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Compute Capability: %d.%d\n", 
       prop.major, prop.minor);
```

### Why CC Matters for Programming

```
Example: Using Tensor Cores

❌ Won't compile on CC < 7.0:
#include <mma.h>  // Tensor Core API
// Error: "mma.h" requires CC 7.0 or higher

✓ Will compile on CC 7.0+:
#include <mma.h>
// Works! Can use wmma operations
```

---

## Summary: Key Takeaways

```
┌──────────────────────────────────────────────────────┐
│              MEMORY HIERARCHY                        │
├──────────────────────────────────────────────────────┤
│  Registers        → Fastest, private to thread       │
│  Shared Memory    → Fast, shared within block        │
│  L2 Cache         → Medium, auto-managed             │
│  Global Memory    → Slowest, accessible to all       │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│           GPU COMPONENT HIERARCHY                    │
├──────────────────────────────────────────────────────┤
│  Core    → Single ALU (one operation)                │
│  SM      → Group of ~128 cores + shared resources    │
│  GPU     → Many SMs (128 SMs in RTX 4090)            │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│         PERFORMANCE FACTORS                          │
├──────────────────────────────────────────────────────┤
│  Bandwidth    → Memory speed × bus width             │
│  Throughput   → Core count × clock × 2               │
│  Features     → Tensor/RT cores, new data types      │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│         NVIDIA NAMING                                │
├──────────────────────────────────────────────────────┤
│  Architecture → How it's built (Ampere, Hopper)      │
│  Product      → Who it's for (GeForce, Tesla)        │
│  CC Number    → Features available (8.6, 9.0)        │
└──────────────────────────────────────────────────────┘
```
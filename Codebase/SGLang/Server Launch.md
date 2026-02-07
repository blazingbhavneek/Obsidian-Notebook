
## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Launch Flow Deep Dive](#launch-flow-deep-dive)
3. [Process Architecture & IPC](#process-architecture--ipc)
4. [Model Loading Pipeline](#model-loading-pipeline)
5. [Parallelism Strategies](#parallelism-strategies)
6. [Memory Management](#memory-management)
7. [Request Processing Flow](#request-processing-flow)

---

## System Architecture Overview

SGLang implements a **multi-process, pipeline-based serving architecture** that separates I/O, scheduling, inference, and text processing into independent processes communicating via ZeroMQ.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Main Process                                │
│                                                                       │
│  ┌──────────────┐    ┌─────────────────────┐    ┌────────────┐     │
│  │   FastAPI    │───►│  TokenizerManager   │───►│   Engine   │     │
│  │ HTTP Server  │◄───│  (text→token IDs)   │◄───│ Controller │     │
│  │  (uvicorn)   │    │  (async I/O loop)   │    │            │     │
│  └──────────────┘    └─────────────────────┘    └────────────┘     │
│         │                      │                        │            │
└─────────┼──────────────────────┼────────────────────────┼───────────┘
          │                      │ ZMQ PUSH/PULL          │
          │                      ▼                        │
          │         ┌─────────────────────────┐           │
          │         │   ZMQ Message Queue     │           │
          │         │  (IPC Unix Sockets)     │           │
          │         └─────────────────────────┘           │
          │                      │                        │
          │                      ▼                        │
┌─────────┼──────────────────────────────────────────────┼───────────┐
│         │         Scheduler Process (per TP/PP/DP rank) │           │
│         │                                                │           │
│  ┌──────▼────────┐  ┌──────────────┐  ┌───────────────▼────────┐  │
│  │   Scheduler   │─►│  TpWorker    │─►│   ModelRunner          │  │
│  │               │  │  (wrapper)   │  │   • Model forward      │  │
│  │ • Batching    │  │              │  │   • KV cache mgmt      │  │
│  │ • KV alloc    │  │              │  │   • Attn backend       │  │
│  │ • Scheduling  │  │              │  │   • Memory profiling   │  │
│  │ • Event loops │  │              │  │                        │  │
│  └───────────────┘  └──────────────┘  └────────────────────────┘  │
│         │                                        │                  │
│         │                                        ▼                  │
│         │                          ┌──────────────────────────┐    │
│         │                          │  PyTorch Model (on GPU)  │    │
│         │                          │  • Transformer layers    │    │
│         │                          │  • Embeddings            │    │
│         │                          │  • KV Cache Tensors      │    │
│         │                          │  • Attention Kernels     │    │
│         │                          └──────────────────────────┘    │
│         │                                                           │
└─────────┼───────────────────────────────────────────────────────────┘
          │ ZMQ PUSH/PULL (output tokens)
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Detokenizer Process                               │
│                                                                       │
│  ┌────────────────────────────────────────────────────────┐         │
│  │  DetokenizerManager                                    │         │
│  │  • Receives token IDs from Scheduler                   │         │
│  │  • Converts token IDs → UTF-8 strings                  │         │
│  │  • Applies chat templates (if needed)                  │         │
│  │  • Handles streaming (incremental detokenization)      │         │
│  │  • Sends results back to TokenizerManager              │         │
│  └────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

**Design Rationale:**

1. **Separation of Concerns**:
    
    - **I/O Layer (FastAPI)**: Handles HTTP, validates requests, manages connections
    - **Tokenization Layer**: CPU-bound text processing doesn't block GPU
    - **Scheduling Layer**: Complex batching logic separated from model execution
    - **Inference Layer**: Pure GPU compute with minimal Python overhead
    - **Detokenization Layer**: CPU-bound text conversion runs in parallel with next batch
2. **Concurrency Model**:
    
    - Main process uses Python's async/await for I/O multiplexing
    - Each process has dedicated CPU cores (via affinity settings)
    - GPU processes use CUDA streams for kernel overlap
    - ZMQ provides lock-free, zero-copy message passing
3. **Scalability**:
    
    - Horizontal: Multiple scheduler processes for DP/TP/PP
    - Vertical: Each process can be pinned to NUMA nodes for memory locality
    - Network: Supports multi-node via NCCL for distributed inference

---

## Launch Flow Deep Dive

### Entry Point: CLI Command

```bash
python -m sglang.cli.serve \
    --model-path meta-llama/Llama-3-8B \
    --tp 4 \
    --port 30000
```

### Code Flow Walkthrough

#### 1. `python/sglang/cli/serve.py::serve()`

```python
def serve(args, extra_argv):
    # Step 1: Help handling
    # If user asks for --help, print all server args and exit
    
    # Step 2: Model type detection
    # Checks model config to determine if it's:
    # - Multimodal (e.g., LLaVA, Qwen-VL)
    # - Diffusion (e.g., Stable Diffusion)
    # - Standard language model
    
    # Step 3: Route to appropriate runtime
    if is_multimodal:
        # Routes to vision-specific server
        from sglang.srt.entrypoints.vision_server import run_server
    elif is_diffusion:
        # Routes to diffusion-specific server
        from sglang.srt.entrypoints.diffusion_server import run_server
    else:
        # Standard LLM path (most common)
        from sglang.launch_server import run_server
        from sglang.srt.server_args import prepare_server_args
        
        # Prepares ServerArgs object from CLI arguments
        server_args = prepare_server_args(extra_argv)
        run_server(server_args)
    
    finally:
        # Cleanup: kill all child processes when parent exits
        kill_process_tree(os.getpid(), include_parent=False)
```

**Key Points:**

- `prepare_server_args()` parses ~100+ CLI flags and validates them
- Model type detection happens by reading config.json from the model directory
- The `finally` block ensures clean shutdown even on crashes

---

#### 2. `python/sglang/launch_server.py::run_server()`

This is a thin wrapper that:

```python
def run_server(server_args: ServerArgs):
    # Routes to the main HTTP server entrypoint
    from sglang.srt.entrypoints.http_server import launch_server
    launch_server(server_args)
```

**Why the indirection?**

- Allows different entry points (HTTP server, offline engine, OpenAI server) to share launch logic
- Makes testing easier (can inject different launch functions)

---

#### 3. `python/sglang/srt/entrypoints/http_server.py::launch_server()`

This is the **main orchestrator**. Let's break down what it does:

```python
def launch_server(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable = init_tokenizer_manager,
    run_scheduler_process_func: Callable = run_scheduler_process,
    run_detokenizer_process_func: Callable = run_detokenizer_process,
    execute_warmup_func: Callable = _execute_server_warmup,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.
    
    Architecture:
    - HTTP Server (FastAPI) routes requests to engine
    - Engine has 3 components:
      1. TokenizerManager: text → token IDs → scheduler
      2. Scheduler (subprocess): batching, scheduling, forward passes
      3. DetokenizerManager (subprocess): token IDs → text → TokenizerManager
    
    All run in main process except Scheduler and Detokenizer.
    IPC via ZMQ sockets.
    """
```

**Step-by-step execution:**

##### Step 3.1: Launch Subprocesses

```python
tokenizer_manager, template_manager, scheduler_infos, port_args = (
    _launch_subprocesses(
        server_args,
        init_tokenizer_manager_func,
        run_scheduler_process_func,
        run_detokenizer_process_func,
    )
)
```

This calls `engine.py::_launch_subprocesses()` which:

1. Spawns scheduler process(es) - one per TP/PP/DP rank
2. Spawns detokenizer process
3. Initializes tokenizer manager in main process
4. Waits for all processes to be ready (via pipe communication)

##### Step 3.2: Store in Global State

```python
_global_state = GlobalState()
_global_state.tokenizer_manager = tokenizer_manager
_global_state.template_manager = template_manager
_global_state.scheduler_infos = scheduler_infos
```

**Why global state?**

- FastAPI routes are stateless functions
- Need access to tokenizer/scheduler from any route handler
- Alternative would be dependency injection (FastAPI supports this too)

##### Step 3.3: Setup FastAPI App

```python
async def lifespan(fast_api_app: FastAPI):
    # Startup: Initialize OpenAI-compatible serving layer
    fast_api_app.state.openai_serving_chat = OpenAIServingChat(
        _global_state.tokenizer_manager, 
        _global_state.template_manager
    )
    
    yield  # Server runs here
    
    # Shutdown: cleanup logic (if needed)

app = FastAPI(
    lifespan=lifespan,
    openapi_url=None if DISABLE_OPENAPI_DOC else "/openapi.json",
)

# Allow cross-origin requests (for web UIs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

##### Step 3.4: Include API Routes

```python
from sglang.srt.entrypoints.v1_loads import router as v1_loads_router
app.include_router(v1_loads_router)
```

This adds routes like:

- `POST /v1/chat/completions` (OpenAI compatible)
- `POST /v1/completions`
- `POST /generate`
- `GET /health`
- `GET /v1/models`

##### Step 3.5: Execute Warmup

```python
execute_warmup_func(server_args, _global_state)
```

Warmup does:

- **CUDA Graph Capture**: Pre-records GPU kernels for common batch sizes
- **Memory Profiling**: Determines max batch size based on available VRAM
- **Cache Initialization**: Pre-allocates KV cache memory pools

**Why warmup is critical:**

- CUDA graphs reduce kernel launch overhead by ~10-30%
- Memory profiling prevents OOM crashes during serving
- First requests would be slow without pre-warming

##### Step 3.6: Start HTTP Server

```python
uvicorn.run(app, host=server_args.host, port=server_args.port)
```

This blocks until server is shut down (Ctrl+C or SIGTERM).

---

## Process Architecture & IPC

### Process Creation: `engine.py::_launch_subprocesses()`

```python
def _launch_subprocesses(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable,
    run_scheduler_process_func: Callable,
    run_detokenizer_process_func: Callable,
    port_args: Optional[PortArgs] = None,
) -> Tuple[TokenizerManager, TemplateManager, Tuple[Dict], PortArgs]:
```

#### Port Allocation

```python
if port_args is None:
    port_args = PortArgs.init_new(server_args)
```

**What `PortArgs` does:**

- Allocates unique ports for each component:
    - `tokenizer_port`: TokenizerManager listens here
    - `scheduler_port`: Scheduler listens here
    - `detokenizer_port`: Detokenizer listens here
    - `nccl_port`: NCCL (NVIDIA Collective Communications) for multi-GPU

**Port allocation strategy:**

```python
# Example for TP=4, PP=2, DP=1
base_port = 30000

tokenizer_port = base_port          # 30000
scheduler_port = base_port + 10     # 30010
detokenizer_port = base_port + 20   # 30020
nccl_port = base_port + 30          # 30030
```

**Why multiple ports?**

- Each process binds to its own ZMQ socket
- Avoids port conflicts in multi-instance deployments
- Allows processes to restart independently

---

### Scheduler Process Launch: `_launch_scheduler_processes()`

This is complex because it handles all parallelism configurations.

```python
def _launch_scheduler_processes(
    server_args: ServerArgs,
    port_args: PortArgs,
    run_scheduler_process_func: Callable,
):
    scheduler_procs = []
    scheduler_pipe_readers = []  # For readiness signaling
    
    if server_args.dp_size == 1:
        # Single data-parallel instance: launch TP/PP processes directly
        
        # Calculate how many PP ranks run on this node
        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
        
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
        )
        
        # Calculate how many TP ranks run on this node
        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )
```

**What's happening here?**

Let's use an example:

```
Configuration:
- nnodes = 2 (2 machines)
- pp_size = 2 (2 pipeline stages)
- tp_size = 4 (4-way tensor parallelism)
- node_rank = 0 (we're on the first machine)

Calculation:
pp_size_per_node = max(2 // 2, 1) = 1  # 1 PP stage per node
nnodes_per_pp_rank = max(2 // 2, 1) = 1  # 1 node per PP stage

pp_rank_range = range(1 * (0 // 1), 1 * (0 // 1 + 1)) = range(0, 1) = [0]
# This node runs PP rank 0

tp_size_per_node = 4 // 1 = 4  # All 4 TP ranks on this node
tp_rank_range = range(4 * (0 % 1), 4 * (0 % 1 + 1)) = range(0, 4) = [0,1,2,3]
# This node runs TP ranks 0, 1, 2, 3
```

**Result:** Node 0 launches 4 processes (PP0-TP0, PP0-TP1, PP0-TP2, PP0-TP3)

---

#### GPU Assignment

```python
for pp_rank in pp_rank_range:
    for tp_rank in tp_rank_range:
        # Create a pipe for readiness signaling
        reader, writer = mp.Pipe(duplex=False)
        
        # Calculate which GPU this rank uses
        gpu_id = (
            server_args.base_gpu_id  # e.g., 0
            + ((pp_rank % pp_size_per_node) * tp_size_per_node)
            + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
        )
        
        # Calculate MoE expert parallelism rank
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
```

**GPU mapping example:**

```
base_gpu_id = 0
gpu_id_step = 1
pp_rank = 0, tp_rank = 0: gpu_id = 0 + 0 + 0 = 0
pp_rank = 0, tp_rank = 1: gpu_id = 0 + 0 + 1 = 1
pp_rank = 0, tp_rank = 2: gpu_id = 0 + 0 + 2 = 2
pp_rank = 0, tp_rank = 3: gpu_id = 0 + 0 + 3 = 3
```

**`gpu_id_step`** allows skipping GPUs (useful if some are broken or reserved).

---

#### Process Spawning

```python
        with maybe_reindex_device_id(gpu_id) as gpu_id:
            proc = mp.Process(
                target=run_scheduler_process_func,
                args=(
                    server_args,
                    port_args,
                    gpu_id,
                    tp_rank,
                    moe_ep_rank,
                    pp_rank,
                    None,  # dp_rank (None for single DP)
                    writer,  # Pipe to signal readiness
                ),
            )
            
            with memory_saver_adapter.configure_subprocess(), \
                 numa_utils.configure_subprocess(server_args, gpu_id):
                proc.start()
        
        scheduler_procs.append(proc)
        scheduler_pipe_readers.append(reader)
```

**Context managers explained:**

1. **`maybe_reindex_device_id(gpu_id)`**:
    - Handles `CUDA_VISIBLE_DEVICES` remapping
    - If `CUDA_VISIBLE_DEVICES=2,3,4,5`, then logical GPU 0 → physical GPU 2
2. **`memory_saver_adapter.configure_subprocess()`**:
    - Sets environment variables for memory tracking
    - Enables CPU weight offloading if configured
3. **`numa_utils.configure_subprocess(server_args, gpu_id)`**:
    - Pins process to NUMA node closest to GPU
    - On dual-socket servers, GPUs 0-3 might be on NUMA node 0, GPUs 4-7 on NUMA node 1
    - Reduces memory access latency by 2-3x

**NUMA = Non-Uniform Memory Access**
It’s a memory architecture in multi-CPU (or multi-socket) systems where *not all memory is equally fast to access from all processors*. Each CPU (or group of cores) has *local memory* that it can reach with lower latency and higher bandwidth than *remote memory* that belongs to a different CPU/socket. This is what “non-uniform” means — access times depend on *where* the memory is physically relative to the processor. ([Wikipedia][1])

**NUMA nodes (hardware)**

* On a dual-socket server, you typically have 2 NUMA nodes — one per socket.
* Each node contains a CPU and its DRAM.
* Accessing that node’s memory is faster than accessing the other node’s memory. ([Wikipedia][1])

**Why that matters for GPU + CPU**
GPUs connect to the system via PCIe. On many servers, each GPU is *closer* to one NUMA node’s CPU and memory than to another’s. If a process that drives a GPU lives on a *different NUMA node* than the GPU, then:

* CPU <-> GPU transfers may go through a longer interconnect path.
* Memory allocations and CPU threads may access memory remotely.
  Both slow down communication and reduce throughput. ([TECHCOMMUNITY.MICROSOFT.COM][2])

**Process pinning to NUMA nodes**
`numa_utils.configure_subprocess(...)` sets *CPU and memory affinity* so the process runs on the cores and memory banks *closest to the GPU’s NUMA node*. That improves locality — the process uses local memory and local CPU resources that are physically proximate to the GPU, reducing latency and boosting memory performance. ([Wikipedia][1])

**Analogy**
Imagine two rooms (NUMA nodes). Each room has its own snacks (memory). A host (CPU) in Room A can grab snacks in Room A quickly, but going to Room B to get snacks takes longer. If your event (process) is in Room B but fetching snacks from Room A, it wastes time. If you pin the event to Room A, serving is faster. ([Wikipedia][1])

[1]: https://en.wikipedia.org/wiki/Non-uniform_memory_access?utm_source=chatgpt.com "Non-uniform memory access"
[2]: https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/optimizing-ai-workloads-on-azure-cpu-pinning-via-nccl-topology-file/4371810?utm_source=chatgpt.com "Optimizing AI Workloads on Azure: CPU Pinning via NCCL Topology file | Microsoft Community Hub"

bind the worker processes to the CPU cores and DRAM memory that are physically closest to the GPU’s PCIe connection. This reduces cross-socket traffic and CPU ↔ GPU latency.

Why that matters

On multi-socket servers, each CPU socket has its own DRAM and its own PCIe controllers.

A GPU plugged into PCIe slots tied to socket A will have faster access to memory and CPU cores on that socket than to resources on socket B, because remote access crosses the inter-CPU interconnect.

If your process runs on the wrong NUMA node, GPU transfers and kernel launches have to traverse that interconnect, hurting throughput and latency.

What the SGLang snippet does
numa_utils.configure_subprocess(server_args, gpu_id) makes the process:

Use CPU cores on the same NUMA node that owns the GPU’s PCIe root.

Allocate memory from the DRAM attached to that node.

That aligns CPU execution and memory with the GPU’s physical locality, minimizing remote memory accesses and PCIe bridging across sockets.

**NUMA nodes are a _hardware architecture thing_** — not just arbitrary OS groups.

- On multi-CPU/socket machines, each socket has its own **memory controllers + local DRAM banks** physically connected to that CPU (or a slice of a CPU die). Processors access their own local memory faster than memory that’s on another socket across the interconnect. That’s what “non-uniform” means. ([Wikipedia](https://en.wikipedia.org/wiki/Non-uniform_memory_access?utm_source=chatgpt.com "Non-uniform memory access"))
    
- The **hardware platform (CPU + motherboard + memory controllers)** defines NUMA topology. For example, a dual-socket server typically has 2 NUMA nodes — one per socket with its own DRAM. ([supermicro.com](https://www.supermicro.com/en/glossary/numa?utm_source=chatgpt.com "What Is Non-Uniform Memory Access (NUMA)? | Supermicro"))
    
- The **OS maps these hardware nodes into software NUMA nodes** so schedulers and memory allocators can optimize locality. Linux, Windows, etc., expose these to userland through APIs and tools (`numactl`, `lscpu`). ([Kernel.org](https://www.kernel.org/doc/html/latest/mm/numa.html?utm_source=chatgpt.com "What is NUMA? — The Linux Kernel documentation"))
    

So NUMA nodes are _physical groupings of CPU cores + DRAM on the machine_, and the OS gives logical identifiers so software can make placement decisions for CPU threads and memory. ([Kernel.org](https://www.kernel.org/doc/html/latest/mm/numa.html?utm_source=chatgpt.com "What is NUMA? — The Linux Kernel documentation"))

**Who defines NUMA nodes?**
◆ **Hardware defines them.** Modern server boards with more than one CPU socket (or CPUs with multiple memory controllers) physically group CPU cores and RAM into *local clusters*. Each cluster is a **NUMA node** — it’s built into how the motherboard/CPU memory controllers are wired. ([Kernel.org][1])

**What exactly a NUMA node is (beginner terms)**

* Think of a **CPU socket + its directly attached RAM** as a *room*. That room is a NUMA node. ([Kernel.org][1])
* You can still use memory from another room, but it’s slower because you must cross hallways (chip interconnect) between rooms. ([Wikipedia][2])
* All the memory is part of the system, but access times vary: local (fast) vs remote (slow). ([supermicro.com][3])

**Where they live**

* On physical servers with **multiple CPU sockets**, each socket + its RAM forms a separate NUMA node. ([Kernel.org][1])
* Even within a single socket, some processors split memory into multiple NUMA units (so there can be more than one NUMA node per socket on newer hardware). ([Red Hat Customer Portal][4])
* The **operating system exposes this hardware topology** to userspace apps and schedulers so they can optimize placement. ([Kernel.org][1])

**Are nodes “free” until used?**

* They’re *always there because of the hardware design.* They don’t run tasks by default. The OS and applications only **schedule work and allocate memory on a node when processes/threads arrive and request resources.** ([Kernel.org][1])
* Until a process runs or memory is allocated, a node has no active workload, but it still owns its CPUs and memory banks. The OS knows about it and may assign threads/memory to it. ([Kernel.org][1])

**Role of the OS**

* The OS maps the hardware NUMA nodes into software abstractions (so apps can query/schedule by node). ([Kernel.org][1])
* By default, memory allocations try to be local to the node that the thread runs on, but you can override that with NUMA APIs or tools. ([Intel][5])

**So, simple summary**

* NUMA nodes are physical groupings on the hardware (CPU socket + local memory). ([supermicro.com][3])
* The OS exposes them so software can place processes/threads and memory near each other. ([Kernel.org][1])
* They’re *not “created” by software* and they *don’t run tasks until the OS schedules something on them*. ([Kernel.org][1])

[1]: https://www.kernel.org/doc/html/latest/mm/numa.html?utm_source=chatgpt.com "What is NUMA? — The Linux Kernel documentation"
[2]: https://en.wikipedia.org/wiki/Non-uniform_memory_access?utm_source=chatgpt.com "Non-uniform memory access"
[3]: https://www.supermicro.com/en/glossary/numa?utm_source=chatgpt.com "What Is Non-Uniform Memory Access (NUMA)? | Supermicro"
[4]: https://access.redhat.com/sites/default/files/attachments/rhel7_numa_perf_brief.pdf?utm_source=chatgpt.com "RED HAT® ENTERPRISE LINUX® 7:"
[5]: https://www.intel.com/content/www/us/en/developer/articles/technical/hardware-and-software-approach-for-using-numa-systems.html?utm_source=chatgpt.com "Hardware and Software Approach for Using NUMA Systems"





---

#### Data Parallelism Mode

```python
    else:
        # dp_size > 1: Launch data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            kwargs=dict(
                server_args=server_args,
                port_args=port_args,
                pipe_writer=writer,
                run_scheduler_process_func=run_scheduler_process_func,
            ),
        )
        proc.start()
        scheduler_procs.append(proc)
```

**What's the DP controller?**

- Manages multiple identical model replicas
- Routes requests using:
    - Round-robin (balanced load)
    - Shortest queue (minimize latency)
    - Custom routing policies

**Why a separate controller?**

- Centralizes routing logic
- Can implement sophisticated load balancing
- Simplifies request tracking across replicas

---

### Scheduler Process Execution: `scheduler.py::run_scheduler_process()`

```python
def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
```

#### Process Setup

```python
    # Generate logger prefix for identification
    prefix = ""
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"
    
    # Set process name (visible in `ps`, `top`, etc.)
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    
    # Enable crash dumps
    faulthandler.enable()
    
    # Ensure this process dies if parent dies (prevents orphans)
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()
```

**Process naming example:**

```
sglang::scheduler_DP0_PP1_TP2_EP0
```

Makes debugging multi-process crashes much easier!

---

#### CPU Affinity

```python
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, 
            server_args.tp_size, 
            server_args.nnodes, 
            gpu_id
        )
```

**What CPU affinity does:**

- Pins process to specific CPU cores
- Prevents context switching between cores (reduces cache misses)
- Example on 64-core server with 8 GPUs:
    
    ```
    GPU 0 → Cores 0-7GPU 1 → Cores 8-15GPU 2 → Cores 16-23...
    ```
    

---

#### NUMA Binding

```python
    if (numa_node := server_args.numa_node) is not None:
        numa_bind_to_node(numa_node[gpu_id])
```

**NUMA (Non-Uniform Memory Access):**

- Dual-socket servers have two memory banks
- Accessing "far" memory is slower than "local" memory
- Example topology:
    
```
Socket 0: CPUs 0-31,  GPUs 0-3,  RAM Bank 0
Socket 1: CPUs 32-63, GPUs 4-7,  RAM Bank 1
```
    
- Binding GPU 2 to NUMA node 0 ensures fast memory access

---

#### Tracing Setup

```python
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        
        thread_label = "Scheduler"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
            
        trace_set_thread_info(thread_label, tp_rank, dp_rank)
```

**Distributed tracing:**

- Uses OpenTelemetry Protocol (OTLP)
- Sends traces to Jaeger/Zipkin for visualization
- Helps debug latency issues in distributed serving

---

#### Scheduler Initialization

```python
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
        )
```

**What `Scheduler.__init__()` does:**

1. Creates `TpWorker` instance
2. `TpWorker` creates `ModelRunner` instance
3. `ModelRunner` loads model weights onto GPU
4. Profiles memory usage
5. Allocates KV cache memory pools
6. Sets up batching queues

---

#### Readiness Signaling

```python
        result_dict = {
            "status": "ready",
            "max_total_num_tokens": scheduler.max_total_num_tokens,
            "max_req_input_len": scheduler.max_req_input_len,
        }
        
        if server_args.remote_instance_weight_loader_use_transfer_engine():
            # Remote weight loading metadata
            (
                remote_instance_transfer_engine_session_id,
                remote_instance_transfer_engine_weights_info_dict,
            ) = scheduler.get_remote_instance_transfer_engine_info()
            result_dict.update({
                "tp_rank": tp_rank,
                "remote_instance_transfer_engine_session_id": ...,
                "remote_instance_transfer_engine_weights_info_dict": ...,
            })
        
        pipe_writer.send(result_dict)
```

**Why send this metadata?**

- Main process needs to know max context length for request validation
- Remote weight loading needs TP rank-specific metadata
- Signals readiness to accept requests

---

#### Event Loop Dispatch

```python
        disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
        
        if disaggregation_mode == DisaggregationMode.NULL:
            # Standard serving (prefill + decode together)
            if scheduler.enable_pdmux:
                scheduler.event_loop_pdmux()
            elif server_args.pp_size > 1:
                scheduler.event_loop_pp()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap()
            else:
                scheduler.event_loop_normal()
                
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            # This instance only does prefill
            if server_args.pp_size > 1:
                scheduler.event_loop_pp_disagg_prefill()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_prefill()
            else:
                scheduler.event_loop_normal_disagg_prefill()
                
        elif disaggregation_mode == DisaggregationMode.DECODE:
            # This instance only does decode
            if server_args.pp_size > 1:
                scheduler.event_loop_pp_disagg_decode()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_decode()
            else:
                scheduler.event_loop_normal_disagg_decode()
```

**Event loop variants explained:**

| Mode               | Description                          | Use Case                              |
| ------------------ | ------------------------------------ | ------------------------------------- |
| **normal**         | Single-threaded, sequential batching | Simplest, good for debugging          |
| **overlap**        | Overlaps compute & communication     | 10-20% better throughput              |
| **pp**             | Pipeline parallelism micro-batching  | Large models split across GPUs        |
| **pdmux**          | Multiplexed prefill/decode           | Separate batches for prefill & decode |
| **disagg_prefill** | Prefill-only instance                | Disaggregated serving architecture    |
| **disagg_decode**  | Decode-only instance                 | Disaggregated serving architecture    |

**Disaggregated serving:**

- Prefill is compute-bound (matrix multiplications)
- Decode is memory-bound (KV cache reads)
- Running them on separate instances allows different tuning:
    - Prefill: Large batch size, high compute utilization
    - Decode: Small batch size, low latency

**Yes — you can separate prefill and decode across different workers/GPU instances.** It doesn’t mean *one request must stay on a single GPU* all the time. There’s a pattern called **prefill-decode disaggregation** where the work for those phases is decoupled and potentially run on separate processes or machines. ([vLLM][1])

**Why that works (key idea)**
LLM inference has two distinct phases:

* **Prefill:** processes the entire prompt and builds the KV cache (big matrix work).
* **Decode:** generates tokens one at a time using the previously created KV cache. ([fltech - 富士通研究所の技術ブログ][2])

These phases have **very different resource profiles** — prefill is *compute-bound*, decode is *memory-bound*. Because of that mismatch, you can **specialize different worker instances** for each phase and scale them independently instead of doing both in one process. ([Ray][3])

**How separation actually happens**

1. The **prefill instance** (or process) runs the prompt, computes the KV cache, and produces the first token. ([vLLM][1])
2. The **decode instance** gets the generated KV cache (via a connector like NIXL or a shared KV service) and continues generating further tokens for the same request. ([vLLM][1])
3. A router or proxy coordinates which instance handles which phase and transfers the KV cache between them by messaging or shared memory. ([Ray][3])

So the request *state* (especially the KV cache) moves between the prefill and decode workers. The actual heavy compute for the prompt and the subsequent decode token generation no longer need to live in one process/GPU if the system manages the state transfer. ([vLLM][1])

**Benefits of disaggregation**

* You can **tune parallelism separately** for prefill vs decode based on different bottlenecks. ([vllm.website.cncfstack.com][4])
* You can **scale more decode workers** when lots of generation is happening while prefill is short. ([Ray][3])
* You can control latency characteristics like time-to-first-token vs inter-token latency. ([vllm.website.cncfstack.com][4])

**So the short answer to your question**

> *Doesn’t each request need prefill and decode on the same GPU?*
> Not necessarily. You can **split them** and *transfer the intermediate KV cache state* from prefill to decode workers. It means the same high-level request is handled in stages across different workers, but the logic ensures continuity by moving the KV cache between them. ([vLLM][1])

[1]: https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment.html?utm_source=chatgpt.com "Expert Parallel Deployment - vLLM"
[2]: https://blog.fltech.dev/entry/2026/01/28/llminference_modeling?utm_source=chatgpt.com "LLM推論性能モデリング - fltech - 富士通研究所の技術ブログ"
[3]: https://docs.ray.io/en/latest/serve/llm/architecture/serving-patterns/prefill-decode.html?utm_source=chatgpt.com "Prefill-decode disaggregation — Ray 2.53.0"
[4]: https://vllm.website.cncfstack.com/features/disagg_prefill/?utm_source=chatgpt.com "vLLM"

---

#### Error Handling

```python
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
```

**Why send `SIGQUIT` to parent?**

- Crashes in subprocess can go unnoticed
- Parent process needs to shut down entire server cleanly
- `SIGQUIT` triggers core dump for debugging

---

### Detokenizer Process Launch

```python
detoken_proc = mp.Process(
    target=run_detokenizer_process_func,
    args=(server_args, port_args,),
)
detoken_proc.start()
```

**What detokenizer does:**

1. Receives token IDs via ZMQ from scheduler
2. Converts to UTF-8 strings using tokenizer
3. Handles streaming (incremental detokenization for `\n`, spaces, etc.)
4. Sends strings back to TokenizerManager

**Why separate process?**

- Detokenization is CPU-bound (HuggingFace tokenizers are slow)
- Doesn't block GPU from processing next batch
- Can run on dedicated CPU cores for lower latency

---

### TokenizerManager Initialization

```python
if server_args.tokenizer_worker_num == 1:
    tokenizer_manager, template_manager = init_tokenizer_manager_func(
        server_args, port_args
    )
else:
    # Multi-tokenizer router for high-throughput scenarios
    tokenizer_manager = MultiTokenizerRouter(server_args, port_args)
    template_manager = None
```

**Single vs Multi-tokenizer:**

|Config|Use Case|Architecture|
|---|---|---|
|**Single**|Most deployments|TokenizerManager in main process|
|**Multi**|Very high QPS (>10k req/s)|Multiple tokenizer workers, router balances load|

**`MultiTokenizerRouter`:**

- Spawns N tokenizer worker processes
- Round-robin request distribution
- Useful when tokenization is the bottleneck (e.g., very long prompts)

---

### Waiting for Readiness

```python
scheduler_infos = _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs)
```

**What this does:**

```python
def _wait_for_scheduler_ready(pipe_readers, procs):
    infos = []
    for reader, proc in zip(pipe_readers, procs):
        if reader.poll(timeout=600):  # 10 minute timeout
            info = reader.recv()
            if info["status"] == "ready":
                infos.append(info)
            else:
                raise RuntimeError(f"Scheduler failed: {info}")
        else:
            raise TimeoutError("Scheduler took too long to start")
    return infos
```

**Why 10 minute timeout?**

- Large models (70B+) can take 5+ minutes to load
- Downloading from HuggingFace Hub can be slow
- Network-based weight loading is also slow

---

### Sync Metadata Back

```python
tokenizer_manager.max_req_input_len = scheduler_infos[0]["max_req_input_len"]
```

**Why sync this?**

- TokenizerManager needs to reject requests longer than model's context window
- Avoids sending invalid requests to scheduler
- Example: Model has 4096 context, reject prompts with 5000 tokens

---

## Model Loading Pipeline

### Overview

The model loading pipeline is responsible for:

1. Discovering weight files (local or remote)
2. Creating an empty model structure with correct architecture
3. Loading weights with TP/PP sharding
4. Post-processing (quantization, RoPE cache, etc.)

---

### Key Classes

#### `ModelRunner` (`srt/model_executor/model_runner.py`)

This is the core inference engine.

```python
class ModelRunner(ModelRunnerKVCacheMixin):
    """
    Manages model execution and memory.
    
    Responsibilities:
    - Load model weights
    - Profile memory usage
    - Allocate KV cache
    - Execute forward passes
    - Manage attention backends
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        draft_model_idx: Optional[int] = None,
    ):
```

**Key initialization steps:**

1. **Parse configuration:**
    
    ```python
    self.model_config = model_config
    self.tp_rank = tp_rank
    self.tp_size = tp_size
    self.pp_rank = pp_rank
    self.pp_size = pp_size
    self.device = f"cuda:{gpu_id}"
    ```
    
2. **Initialize distributed state:**
    
    ```python
    # Sets up process groups for NCCL communication
    init_distributed_environment(
        backend="nccl",
        world_size=tp_size * pp_size,
        rank=tp_rank + pp_rank * tp_size,
        dist_init_method=f"tcp://localhost:{nccl_port}",
    )
    ```
    
3. **Setup memory pools:**
    
    ```python
    if req_to_token_pool is None:
        # Create shared pool for token IDs
        self.req_to_token_pool = ReqToTokenPool(
            size=server_args.max_total_num_tokens,
            max_context_len=model_config.context_len,
        )
    
    if token_to_kv_pool_allocator is None:
        # Create KV cache allocator
        self.token_to_kv_pool_allocator = create_kv_pool_allocator(
            server_args=server_args,
            model_config=model_config,
        )
    ```
    
4. **Load model:**
    
    ```python
    self.load_model()
    ```
    

---

### `load_model()` Deep Dive

```python
def load_model(self):
    tic_total = time.perf_counter()
    before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
    logger.info(
        f"Load weight begin. avail mem={before_avail_memory:.2f} GB"
    )
```

#### Thread Configuration

```python
    # Reduce thread contention during weight loading
    if self.device != "cpu":
        torch.set_num_threads(1)
```

**Why set threads to 1?**

- During parallel loading (TP ranks load simultaneously)
- PyTorch's default thread pool can cause contention
- Each process uses only 1 CPU core → no benefit from multi-threading
- Reduces mutex contention and cache thrashing

---

#### Device Capability Checks

```python
    if self.device == "cuda":
        if torch.cuda.get_device_capability()[0] < 8:
            logger.info(
                "Compute capability below sm80. "
                "Use float16 due to lack of bfloat16 support."
            )
            self.server_args.dtype = "float16"
            self.model_config.dtype = torch.float16
            
            if torch.cuda.get_device_capability()[1] < 5:
                raise RuntimeError("SGLang only supports sm75 and above.")
```

**NVIDIA GPU Compute Capabilities:**

| Arch         | SM  | GPUs           | bfloat16 Support |
| ------------ | --- | -------------- | ---------------- |
| Turing       | 75  | RTX 20xx, T4   | ❌                |
| Ampere       | 80  | A100, RTX 30xx | ✅                |
| Ada Lovelace | 89  | RTX 40xx, L40  | ✅                |
| Hopper       | 90  | H100, H200     | ✅                |

**Why bfloat16 matters:**

- Same range as float32 (8 exponent bits)
- Better for training/inference than float16 (less likely to overflow)
- Natively supported on modern GPUs (faster than float16 on Ampere+)

Here’s a **clear table of common LLM quant formats** and their differences (precision, memory, quality, method):

| Format                         | Type                      | Bits per value | Memory ≈      | What it is                                     | Typical trade-off                                                        |
| ------------------------------ | ------------------------- | -------------: | ------------- | ---------------------------------------------- | ------------------------------------------------------------------------ |
| **FP32**                       | Floating point            |             32 | 1× (baseline) | Full precision floats; highest accuracy        | Slowest, largest memory                                                  |
| **BF16**                       | Floating point            |             16 | ~0.5×         | Brain-Float16; same exponent range as FP32     | Good quality, training/serving baseline (vLLM/SGLang) ([lafzusa.com][1]) |
| **FP16**                       | Floating point            |             16 | ~0.5×         | IEEE half precision                            | Similar to BF16 in size; smaller range than BF16 ([Medium][2])           |
| **FP8**                        | Floating point            |              8 | ~0.25×        | Low-precision floating point (e.g., E4M3/E5M2) | Lower memory, good range; needs hardware support ([nvidia.github.io][3]) |
| **INT8**                       | Integer                   |              8 | ~0.25×        | 8-bit integers for weights/activations         | Minimal quality loss, strong memory + speed win ([BentoML][4])           |
| **INT4**                       | Integer                   |              4 | ~0.125×       | 4-bit integer quantization                     | Very high memory reduction, some quality drop ([lafzusa.com][1])         |
| **NF4**                        | Integer with distribution |              4 | ~0.125×       | NormalFloat 4 (bitsandbytes)                   | Better accuracy than naive INT4 ([lafzusa.com][1])                       |
| **INT2**                       | Integer                   |              2 | ~0.0625×      | Extreme 2-bit                                  | Large compression; experimental/quality fragile ([lafzusa.com][1])       |
| **FP4 / mixed low-bit floats** | Floating point            |              4 | ~0.125×       | FP4 formats supported on very new GPUs         | Experimental/fast on newest hardware ([nvidia.github.io][3])             |

**Quantization *methods*** (how bits are chosen and applied)

| Method                                            | What it does                                   | Key properties                                                          |
| ------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------- |
| **Post-Training Quantization (PTQ)**              | Quantize weights after training                | No retraining; good memory cut, low complexity ([NVIDIA Docs][5])       |
| **Quantization-Aware Training (QAT)**             | Simulates quantization during training         | Better quality but requires retraining ([NVIDIA Docs][5])               |
| **GPTQ (Generalized Post-Training Quantization)** | Calibrated weight quantization                 | Often 4-bit; layer-aware, reduces quantization error ([lafzusa.com][1]) |
| **AWQ (Activation-Aware Weight Quantization)**    | Prioritizes important weights                  | Keeps quality higher at same bit width vs GPTQ ([lafzusa.com][1])       |
| **GGUF & variants**                               | File format with quant types (Q4_K_M, Q5_K_M…) | Used in llama.cpp etc.; multiple quant levels ([lafzusa.com][1])        |
| **bitsandbytes (bnb) dynamic)**                   | Load-time quantization (INT8/NF4)              | Good in training loops (QLoRA) ([lafzusa.com][1])                       |

**What the differences mean (simple)**

* **FP32/BF16/FP16** keep floats; BF16 has big dynamic range like FP32 but lower precision, so often used as “baseline” in LLM serving. FP16 is similar size but slightly different numeric properties. ([lafzusa.com][1])
* **FP8** is emerging low-bit float that retains wider range than INT8 and can be lossless on some tasks if hardware supports it (Tensor cores). ([nvidia.github.io][3])
* **INT8** trades precision for memory and speed; good quality for many models with calibration. ([BentoML][4])
* **INT4 / NF4** go further down, cutting memory a lot. NF4 has better accuracy than plain INT4 by placing quant levels according to weight distribution. ([lafzusa.com][1])
* Methods like **GPTQ, AWQ** define how those bits are chosen; AWQ often gives better quality than basic GPTQ at the same bit size. ([lafzusa.com][1])

**Quality vs compression (rule of thumb)**

* Higher precision (BF16/FP16/FP8) → *best quality* but larger memory. ([nvidia.github.io][3])
* Mid precision (INT8) → *good quality* with significant memory/cost reduction. ([BentoML][4])
* Low precision (INT4/NF4) + smart methods (AWQ/GPTQ) → *huge memory cut* with decent quality if method is good. ([lafzusa.com][1])

**When each is used**

* **FP32/BF16/FP16**: baseline training & high-quality inference. ([lafzusa.com][1])
* **FP8**: on GPUs with support, mix of quality and memory. ([nvidia.github.io][3])
* **INT8**: mainstream efficient inference. ([BentoML][4])
* **INT4/NF4 with GPTQ/AWQ**: very low-memory deployment, good for consumer hardware or dense multi-context serving. ([lafzusa.com][1])

[1]: https://lafzusa.com/llm-masterclass/03-inference/quantization.html?utm_source=chatgpt.com "LLM Quantization Guide - GGUF, GPTQ, AWQ, bitsandbytes | LafzUSA"
[2]: https://medium.com/%40dimpleukothari178/tiny-but-mighty-how-quantization-makes-large-language-models-practical-e4a314dbb764?utm_source=chatgpt.com "“Tiny But Mighty: How Quantization Makes Large Language Models Practical” | by Dimpleukothari | Medium"
[3]: https://nvidia.github.io/TensorRT-LLM/1.2.0rc5/features/quantization.html?utm_source=chatgpt.com "Quantization — TensorRT LLM"
[4]: https://bentoml.com/llm/getting-started/llm-quantization?utm_source=chatgpt.com "LLM quantization | LLM Inference Handbook"
[5]: https://docs.nvidia.com/nemo-framework/user-guide/24.09/model-optimization/quantization/quantization.html?utm_source=chatgpt.com "Quantization — NVIDIA NeMo Framework User Guide"

---

#### CUDA Architecture Detection

```python
    set_cuda_arch()
```

**What this does:**

```python
def set_cuda_arch():
    # Detects GPU architecture and sets environment variables
    # Used by custom CUDA kernels to compile for correct SM version
    
    capability = torch.cuda.get_device_capability()
    arch = f"sm_{capability[0]}{capability[1]}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
```

**Why needed?**

- Custom kernels (FlashAttention, etc.) compile at runtime
- Need to know target architecture for optimal code generation

---

#### Load Configuration

```python
    from sglang.srt.configs.modelopt_config import ModelOptConfig
    
    modelopt_config = ModelOptConfig(
        quant=self.server_args.modelopt_quant,
        checkpoint_restore_path=self.server_args.modelopt_checkpoint_restore_path,
        checkpoint_save_path=self.server_args.modelopt_checkpoint_save_path,
        export_path=self.server_args.modelopt_export_path,
        quantize_and_serve=self.server_args.quantize_and_serve,
    )
    
    self.load_config = LoadConfig(
        load_format=self.server_args.load_format,
        download_dir=self.server_args.download_dir,
        model_loader_extra_config=self.server_args.model_loader_extra_config,
        tp_rank=self.tp_rank,
        remote_instance_weight_loader_seed_instance_ip=...,
        remote_instance_weight_loader_seed_instance_service_port=...,
        remote_instance_weight_loader_send_weights_group_ports=...,
        remote_instance_weight_loader_backend=...,
        remote_instance_weight_loader_transfer_engine=...,
        modelopt_config=modelopt_config,
        rl_quant_profile=self.server_args.rl_quant_profile,
        draft_model_idx=self.draft_model_idx,
    )
```

**Load formats:**

|Format|Description|Use Case|
|---|---|---|
|`auto`|Try safetensors, fallback to .pt|Default|
|`safetensors`|HuggingFace format|Fast, memory-efficient|
|`fastsafetensors`|Optimized safetensors|Even faster loading|
|`pt`|PyTorch checkpoint|Legacy models|
|`mistral`|Mistral consolidated format|Mistral models|
|`dummy`|Random weights|Testing, profiling|
|`remote_instance`|Load from another server|Multi-instance deployment|
|`npcache`|NumPy cache|Custom formats|

---

#### CPU TP Adjustment

```python
    if self.device == "cpu":
        self.model_config = adjust_config_with_unaligned_cpu_tp(
            self.model_config, self.load_config, self.tp_size
        )
```

**What's "unaligned CPU TP"?**

Example:

```
Model: LLaMA 7B
- num_attention_heads = 32
- hidden_size = 4096
- intermediate_size = 11008

TP size = 5
```

Problem: `32 % 5 != 0` (can't evenly split attention heads)

Solution: Pad to 35 heads, use only 32, waste some compute

**Why only for CPU?**

- GPU TP uses more sophisticated sharding (can split individual heads)
- CPU TP is simpler, requires even division

---

#### Remote Instance Weight Loading (NCCL Backend)

```python
    if (
        self.server_args.load_format == LoadFormat.REMOTE_INSTANCE
        and self.server_args.remote_instance_weight_loader_backend
        == RemoteInstanceWeightLoaderBackend.NCCL
    ):
        if self.tp_rank == 0:
            instance_ip = socket.gethostbyname(socket.gethostname())
            t = threading.Thread(
                target=trigger_init_weights_send_group_for_remote_instance_request,
                args=(
                    self.server_args.remote_instance_weight_loader_seed_instance_ip,
                    self.server_args.remote_instance_weight_loader_seed_instance_service_port,
                    self.server_args.remote_instance_weight_loader_send_weights_group_ports,
                    instance_ip,
                ),
            )
            t.start()
```

**Remote weight loading architecture:**

```
┌─────────────────────────────────────┐
│   Seed Instance (has weights)       │
│   - Loaded model from disk          │
│   - Runs NCCL send server           │
└─────────────┬───────────────────────┘
              │ NCCL Send
              ▼
┌─────────────────────────────────────┐
│   Client Instance (no local weights)│
│   - Receives weights over network   │
│   - Uses NCCL recv                  │
│   - Saves memory & disk space       │
└─────────────────────────────────────┘
```

**Use case:**

- 10 GPU servers want to serve same model
- Instead of each downloading 70GB, only one downloads
- Others stream weights from the first server
- Saves 630GB of network bandwidth & disk space

**NCCL vs HTTP backend:**

- NCCL: Faster (GPU Direct RDMA), but requires NCCL-compatible network
- HTTP: Slower, but works on any network

---

#### Model Loading with Memory Tracking

```python
    monkey_patch_vllm_parallel_state()
    
    enable_cpu_backup = self.server_args.enable_weights_cpu_backup or (
        self.is_draft_worker and self.server_args.enable_draft_weights_cpu_backup
    )
    
    with self.memory_saver_adapter.region(
        GPU_MEMORY_TYPE_WEIGHTS,
        enable_cpu_backup=enable_cpu_backup,
    ):
        self.loader = get_model_loader(
            load_config=self.load_config,
            model_config=self.model_config,
        )
        self.model = self.loader.load_model(
            model_config=self.model_config,
            device_config=DeviceConfig(self.device, self.gpu_id),
        )
        
        if hasattr(self.loader, "remote_instance_transfer_engine_weight_info"):
            self.remote_instance_transfer_engine_weight_info = (
                self.loader.remote_instance_transfer_engine_weight_info
            )
    
    monkey_patch_vllm_parallel_state(reverse=True)
```

**Monkey patching vLLM:**

SGLang reuses some vLLM components (e.g., quantization kernels, attention backends), but vLLM's parallel state management assumes a different architecture.

```python
def monkey_patch_vllm_parallel_state():
    # Save original functions
    original_funcs = {}
    original_funcs["get_tensor_model_parallel_world_size"] = (
        vllm.get_tensor_model_parallel_world_size
    )
    
    # Replace with SGLang's versions
    vllm.get_tensor_model_parallel_world_size = sglang_get_tp_world_size
    ...

def monkey_patch_vllm_parallel_state(reverse=True):
    # Restore original functions
    vllm.get_tensor_model_parallel_world_size = (
        original_funcs["get_tensor_model_parallel_world_size"]
    )
```

**Why patch and unpatch?**

- Only needed during model loading
- After loading, SGLang uses its own parallel state
- Unpatching avoids interfering with other code

---

**Memory Saver Adapter:**

```python
class MemorySaverAdapter:
    """
    Tracks GPU memory usage by region.
    Optionally offloads regions to CPU.
    """
    
    def region(self, region_type, enable_cpu_backup=False):
        return MemoryRegion(
            adapter=self,
            region_type=region_type,
            enable_cpu_backup=enable_cpu_backup,
        )
```

**How it works:**

```python
# Before entering region
free_memory_before = torch.cuda.mem_get_info()[0]

with memory_saver_adapter.region(GPU_MEMORY_TYPE_WEIGHTS, enable_cpu_backup=True):
    # Load model weights
    model.load_state_dict(...)
    
    # After loading
    free_memory_after = torch.cuda.mem_get_info()[0]
    weight_usage = free_memory_before - free_memory_after
    
    if enable_cpu_backup:
        # Copy weights to CPU
        cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # Free GPU memory
        model.cpu()
        torch.cuda.empty_cache()
        
        # Reload only when needed
        # (SGLang doesn't actually do this, but the adapter supports it)
```

**Use cases:**

- **Model too large for GPU**: Load, run forward pass, offload
- **Multi-model serving**: Swap models in/out of GPU memory
- **Memory profiling**: Track exactly how much memory each component uses

---

#### FP8 KV Cache Scaling Factors

```python
    if self.server_args.kv_cache_dtype == "fp8_e4m3":
        if self.server_args.quantization_param_path is not None:
            if callable(getattr(self.model, "load_kv_cache_scales", None)):
                self.model.load_kv_cache_scales(
                    self.server_args.quantization_param_path
                )
                logger.info(
                    "Loaded KV cache scaling factors from %s",
                    self.server_args.quantization_param_path,
                )
            else:
                raise RuntimeError(
                    "Using FP8 KV cache and scaling factors provided but "
                    "model %s does not support loading scaling factors.",
                    self.model.__class__,
                )
        else:
            logger.warning(
                "Using FP8 KV cache but no scaling factors "
                "provided. Defaulting to scaling factors of 1.0. "
                "This may lead to less accurate results!"
            )
```

**FP8 quantization for KV cache:**

Normal KV cache:

```
Q: [seq_len, num_heads, head_dim] in bfloat16 = 2 bytes per element
K: [seq_len, num_heads, head_dim] in bfloat16 = 2 bytes per element
V: [seq_len, num_heads, head_dim] in bfloat16 = 2 bytes per element

Example: 2048 tokens, 32 heads, 128 head_dim
= 2048 * 32 * 128 * 2 bytes * 2 (K+V)
= 33.5 MB per request
```

FP8 KV cache:

```
K: [seq_len, num_heads, head_dim] in fp8_e4m3 = 1 byte per element
V: [seq_len, num_heads, head_dim] in fp8_e4m3 = 1 byte per element

Same example:
= 2048 * 32 * 128 * 1 byte * 2 (K+V)
= 16.7 MB per request

50% memory savings!
```

**Why scaling factors?**

FP8 has limited range:

- **e4m3**: 4 exponent bits, 3 mantissa bits
- Max value: ~448
- Min value: ~0.002

If KV values are outside this range, they clip → accuracy loss.

Solution: Per-layer scaling factors

```python
# Calibration (done offline)
for layer in model.layers:
    k_max = torch.abs(k_cache[layer]).max()
    v_max = torch.abs(v_cache[layer]).max()
    
    k_scale = 448.0 / k_max
    v_scale = 448.0 / v_max
    
    save(k_scale, v_scale)

# Inference (runtime)
k_fp8 = (k_fp16 * k_scale).to(torch.float8_e4m3fn)
v_fp8 = (v_fp16 * v_scale).to(torch.float8_e4m3fn)

# Dequantize for attention
k_fp16 = k_fp8.to(torch.bfloat16) / k_scale
v_fp16 = v_fp8.to(torch.bfloat16) / v_scale
```

---

#### Sliding Window Size

```python
    self.sliding_window_size = None
    if hasattr(self.model, "get_attention_sliding_window_size"):
        self.sliding_window_size = self.model.get_attention_sliding_window_size()
    elif (
        self.model_config.is_hybrid_swa
        and self.model_config.sliding_window_size is not None
    ):
        self.sliding_window_size = self.model_config.sliding_window_size
    elif self.model_config.attention_chunk_size is not None:
        self.sliding_window_size = self.model_config.attention_chunk_size
```

**Sliding Window Attention (SWA):**

Normal attention:

```
Token at position 1000 attends to all 1000 previous tokens
→ O(n²) memory and compute
```

Sliding window attention:

```
Token at position 1000 attends to last 512 tokens only
→ O(n * window_size) memory and compute
```

**Models using SWA:**

- Mistral (window size = 4096)
- Mixtral (window size = 4096)
- Some long-context models

**Hybrid SWA:**

- Some layers use full attention
- Other layers use sliding window
- Reduces memory while maintaining quality

---

#### Memory Usage Reporting

```python
    after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
    self.weight_load_mem_usage = before_avail_memory - after_avail_memory
    
    logger.info(
        f"Load weight end. "
        f"elapsed={time.perf_counter() - tic_total:.2f} s, "
        f"type={type(self.model).__name__}, "
        f"dtype={self.dtype}, "
        f"avail mem={after_avail_memory:.2f} GB, "
        f"mem usage={self.weight_load_mem_usage:.2f} GB."
    )
```

**Example output:**

```
Load weight end. elapsed=12.3s, type=LlamaForCausalLM, dtype=torch.bfloat16, 
avail mem=45.2 GB, mem usage=14.8 GB.
```

This helps debug OOM issues and optimize memory allocation.

---

#### Debug Tensor Dumping

```python
    if self.server_args.debug_tensor_dump_output_folder is not None:
        register_forward_hook_for_model(
            self.model,
            self.server_args.debug_tensor_dump_output_folder,
            self.server_args.debug_tensor_dump_layers,
            self.tp_size,
            self.tp_rank,
            self.pp_rank,
        )
```

**What this does:**

- Registers PyTorch forward hooks on every layer
- During inference, saves intermediate activations to disk
- Used for debugging numerical issues, TP/PP correctness

**Example:**

```python
# Save layer 0 output
hook_fn(module, input, output):
    torch.save(output, f"layer_0_tp{tp_rank}_pp{pp_rank}.pt")
```

---

#### RoPE Cache Pre-expansion

```python
    reserve_rope_cache_for_long_sequences(
        self.model,
        self.server_args,
        self.model_config,
        logger,
    )
```

**RoPE (Rotary Position Embeddings):**

Requires precomputed sin/cos values:

```python
# Positions 0 to max_seq_len
pos = torch.arange(max_seq_len)

# Frequencies
freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))

# Precompute sin/cos
rope_cache = pos[:, None] * freqs[None, :]
cos_cache = rope_cache.cos()
sin_cache = rope_cache.sin()
```

**Why pre-expand?**

1. **CUDA Graphs**: Graphs require static memory allocation
2. **Lazy allocation**: PyTorch might allocate RoPE cache during first forward pass
3. **Graph capture**: If allocation happens during capture, it fails

Solution: Expand to `max_seq_len` before graph capture

**RoPE (Rotary Position Embeddings)**
It’s how many LLMs encode *token position* inside attention.

Instead of adding position vectors, RoPE **rotates** query/key vectors using sin/cos based on token index.

**What this code does**

* `pos`: token positions `0..max_seq_len-1`
* `freqs`: per-dimension rotation frequencies
* `rope_cache`: outer product → angle for each (position, dim)
* `sin_cache`, `cos_cache`: lookup tables used in every attention call

So at runtime, attention just does:

```
(q, k) = rotate(q, k, sin[pos], cos[pos])
```

No trig calls during inference.

**Why pre-expand to max_seq_len**

* CUDA Graphs need **fixed memory addresses**
* First-time tensor creation inside a forward = new allocation
* Allocation during graph capture = ❌ graph breaks
* Precomputing sin/cos once makes memory **static and reusable**

**Mental model**

* RoPE cache = read-only table
* CUDA graph = “record once, replay forever”
* Dynamic allocation inside recording = illegal

**Bottom line**
Precompute RoPE sin/cos upfront so:

* no runtime allocation
* graph capture succeeds
* faster, deterministic inference


---

#### Distributed Barrier

```python
    if self.server_args.elastic_ep_backend == "mooncake":
        dist.barrier(group=get_tp_group().cpu_group)
    else:
        try:
            dist.monitored_barrier(
                group=get_tp_group().cpu_group,
                timeout=datetime.timedelta(seconds=1800),  # 30 minutes
                wait_all_ranks=True,
            )
        except RuntimeError:
            raise ValueError(
                f"TP rank {self.tp_rank} could finish the model loading, but "
                "there are other ranks that didn't finish loading. "
                "It is likely due to unexpected failures (e.g., OOM) or a slow node."
            ) from None
```

**Why barrier?**

- All TP ranks must finish loading before serving starts
- If rank 0 finishes but rank 3 OOM crashes, barrier catches it
- Without barrier, rank 0 would start serving with incomplete model → wrong results

**Monitored barrier vs regular barrier:**

- Regular: Waits indefinitely, hangs if one rank dies
- Monitored: Timeout + error reporting, fails fast

**Mooncake backend:**

- Alibaba's custom distributed backend
- Doesn't support monitored barriers (uses regular barrier)

---

### `DefaultModelLoader` Deep Dive

This is where actual weight loading happens.

```python
@dataclasses.dataclass
class Source:
    """Represents a source of model weights."""
    model_or_path: str              # HF model ID or local path
    revision: Optional[str]         # Git revision/tag
    prefix: str = ""                # Prefix for weight names
    fall_back_to_pt: bool           # Allow .pt if .safetensors unavailable
    model_config: ModelConfig       # Model architecture
```

**Multiple sources example:**

```python
# Primary model
primary_source = Source(
    model_or_path="meta-llama/Llama-3-8B",
    revision="main",
    prefix="",
)

# LoRA adapter
lora_source = Source(
    model_or_path="user/llama-3-8b-lora-math",
    revision="main",
    prefix="lora.",
)

# Vision encoder (for multimodal models)
vision_source = Source(
    model_or_path="openai/clip-vit-large-patch14",
    revision="main",
    prefix="vision.",
)
```

---

#### Step 1: `_prepare_weights()` - File Discovery

```python
def _prepare_weights(
    self,
    model_name_or_path: str,
    revision: Optional[str],
    fall_back_to_pt: bool,
) -> Tuple[str, List[str], bool]:
    """
    Download model if needed, find weight files.
    
    Returns:
        hf_folder: Local path to model directory
        hf_weights_files: List of weight file paths
        use_safetensors: Whether using safetensors format
    """
```

**Download logic:**

```python
    # Check if local path or HF model ID
    is_local = os.path.isdir(model_name_or_path)
    
    if not is_local:
        # Download from HuggingFace Hub
        if get_bool_env_var("SGLANG_USE_MODELSCOPE"):
            # Use ModelScope (Alibaba's model hub)
            from modelscope.hub.snapshot_download import snapshot_download
            hf_folder = snapshot_download(
                model_id=model_name_or_path,
                revision=revision,
                cache_dir=self.load_config.download_dir,
            )
        else:
            # Use HuggingFace Hub
            hf_folder = snapshot_download(
                repo_id=model_name_or_path,
                revision=revision,
                cache_dir=self.load_config.download_dir,
                allow_patterns=["*.safetensors", "*.bin", "*.pt", "*.json"],
            )
    else:
        hf_folder = model_name_or_path
```

**File format detection:**

```python
    use_safetensors = False
    
    # Check for safetensors files
    safetensors_files = glob.glob(os.path.join(hf_folder, "*.safetensors"))
    
    if safetensors_files:
        use_safetensors = True
        hf_weights_files = safetensors_files
    elif fall_back_to_pt:
        # Fallback to PyTorch files
        pt_files = glob.glob(os.path.join(hf_folder, "*.pt"))
        bin_files = glob.glob(os.path.join(hf_folder, "*.bin"))
        hf_weights_files = pt_files + bin_files
    else:
        raise ValueError("No weight files found!")
```

**Deduplication:**

```python
    # Remove consolidated files if sharded files exist
    # Example: model.safetensors vs model-00001-of-00005.safetensors
    
    if any("-" in f for f in hf_weights_files):
        # Has sharded files
        hf_weights_files = [
            f for f in hf_weights_files 
            if "-" in f  # Keep only sharded files
        ]
```

**Why deduplicate?**

Some models have both:

- `model.safetensors` (70GB, all weights in one file)
- `model-00001-of-00005.safetensors` to `model-00005-of-00005.safetensors` (14GB each)

Both contain the same weights, so only load one set!

---

#### Step 2: `_get_weights_iterator()` - Create Weight Stream

```python
def _get_weights_iterator(
    self, source: Source
) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    Create an iterator that yields (weight_name, tensor) pairs.
    """
    
    hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
        source.model_or_path,
        source.revision,
        source.fall_back_to_pt,
    )
```

**Choose iterator based on format:**

```python
    if use_safetensors:
        if self.load_config.load_format == LoadFormat.FASTSAFETENSORS:
            # Optimized safetensors (uses memory mapping)
            weights_iter = fastsafetensors_weights_iterator(hf_weights_files)
            
        elif self.server_args.enable_multithread_load:
            # Multi-threaded loading (8 threads default)
            weights_iter = multi_thread_safetensors_weights_iterator(
                hf_weights_files,
                num_threads=8,
            )
            
        else:
            # Single-threaded safetensors
            weights_iter = safetensors_weights_iterator(hf_weights_files)
    else:
        # PyTorch checkpoint
        weights_iter = pt_weights_iterator(hf_weights_files)
```

**Add prefix if needed:**

```python
    if source.prefix:
        weights_iter = (
            (source.prefix + name, tensor)
            for name, tensor in weights_iter
        )
```

**Filter for MTP draft models:**

```python
    if self.load_config.draft_model_idx is not None:
        # Multi-token prediction uses multiple draft models
        # Each draft model only needs specific layers
        
        weights_iter = (
            (name, tensor)
            for name, tensor in weights_iter
            if f"draft_models.{self.load_config.draft_model_idx}" in name
        )
```

---

#### Weight Iterator Implementations

**Safetensors iterator:**

```python
def safetensors_weights_iterator(
    files: List[str]
) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    Load weights from safetensors files.
    Uses memory mapping for efficiency.
    """
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                yield (name, tensor)
```

**Fast safetensors iterator:**

```python
def fastsafetensors_weights_iterator(
    files: List[str]
) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    Optimized safetensors loading.
    Uses mmap and avoids unnecessary copies.
    """
    for file in files:
        # Memory-map the file
        with open(file, "rb") as f:
            data = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            
        # Parse safetensors header
        header_size = int.from_bytes(data[:8], "little")
        header = json.loads(data[8:8+header_size])
        
        # Yield tensors (zero-copy via mmap)
        for name, info in header.items():
            if name == "__metadata__":
                continue
                
            offset = info["data_offsets"][0] + 8 + header_size
            shape = info["shape"]
            dtype = DTYPE_MAP[info["dtype"]]
            
            # Create tensor view (no copy)
            tensor = torch.frombuffer(
                data, 
                dtype=dtype, 
                count=np.prod(shape), 
                offset=offset
            ).reshape(shape)
            
            yield (name, tensor)
```

**Multi-threaded iterator:**

```python
def multi_thread_safetensors_weights_iterator(
    files: List[str],
    num_threads: int = 8,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    Load multiple files in parallel using ThreadPoolExecutor.
    Significantly faster for models with many files.
    """
    from concurrent.futures import ThreadPoolExecutor
    
    def load_file(file):
        weights = []
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                weights.append((name, f.get_tensor(name)))
        return weights
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all files
        futures = [executor.submit(load_file, f) for f in files]
        
        # Yield weights as they become available
        for future in futures:
            for name, tensor in future.result():
                yield (name, tensor)
```

**Speedup comparison:**

```
Model: Llama-3-70B (with 19 safetensors files)

Single-threaded: 180 seconds
Multi-threaded (8 threads): 35 seconds

5x speedup!
```

---

#### Step 3: `_get_all_weights()` - Aggregate Sources

```python
def _get_all_weights(
    self,
    primary_weights: Source,
    secondary_weights: Optional[List[Source]] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    Combine weights from primary and secondary sources.
    """
    
    # Load primary model weights
    yield from self._get_weights_iterator(primary_weights)
    
    # Load secondary weights (LoRA, vision encoder, etc.)
    if secondary_weights:
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)
```

**Example with LoRA:**

```python
# Yields:
("model.layers.0.self_attn.q_proj.weight", tensor_q)
("model.layers.0.self_attn.k_proj.weight", tensor_k)
...
("lora.model.layers.0.self_attn.q_proj.lora_A", tensor_lora_a)
("lora.model.layers.0.self_attn.q_proj.lora_B", tensor_lora_b)
```

---

#### Step 4: `load_model()` - Main Entry Point

```python
def load_model(
    self,
    *,
    model_config: ModelConfig,
    device_config: DeviceConfig,
) -> nn.Module:
    """
    Load model with weights.
    
    Returns:
        Initialized PyTorch model ready for inference.
    """
    
    # Set device and dtype contexts
    target_device = torch.device(device_config.device)
    
    with set_default_torch_dtype(model_config.dtype):
        with torch.device(target_device):
            # Create empty model structure
            model = self._initialize_model(model_config, device_config)
    
    # Get weight iterator
    weights = self._get_all_weights(
        primary_weights=Source(
            model_or_path=self.load_config.model_or_path,
            ...
        ),
        secondary_weights=model_config.secondary_weights,
    )
    
    # Load weights and post-process
    load_weights_and_postprocess(
        model=model,
        weights=weights,
        device_config=device_config,
    )
    
    return model.eval()
```

---

#### `_initialize_model()` - Create Empty Model

```python
def _initialize_model(
    self,
    model_config: ModelConfig,
    device_config: DeviceConfig,
) -> nn.Module:
    """
    Create model structure without loading weights.
    Parameters are initialized with random values.
    """
    
    # Import model class
    model_class = get_model_class(model_config.hf_config.architectures[0])
    
    # Example: LlamaForCausalLM
    
    # Create model
    with torch.device(device_config.device):
        model = model_class(
            config=model_config.hf_config,
            linear_method=get_linear_method(model_config.quantization),
            tp_rank=self.load_config.tp_rank,
            tp_size=model_config.tp_size,
            pp_rank=model_config.pp_rank,
            pp_size=model_config.pp_size,
        )
    
    return model
```

**What happens inside model creation:**

```python
class LlamaForCausalLM(nn.Module):
    def __init__(self, config, linear_method, tp_rank, tp_size, ...):
        super().__init__()
        
        # Embedding layer
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                config=config,
                linear_method=linear_method,
                layer_idx=i,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
            for i in range(config.num_hidden_layers)
            if self._should_include_layer(i, pp_rank, pp_size)
        ])
        
        # Output layer
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            linear_method=linear_method,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
```

**TP Sharding during initialization:**

```python
class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer sharded across TP ranks.
    
    Example: vocab_size=32000, tp_size=4
    Rank 0: embeddings[0:8000]
    Rank 1: embeddings[8000:16000]
    Rank 2: embeddings[16000:24000]
    Rank 3: embeddings[24000:32000]
    """
    
    def __init__(self, num_embeddings, embedding_dim, tp_rank, tp_size):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        
        # Calculate this rank's slice
        vocab_start_index = tp_rank * num_embeddings // tp_size
        vocab_end_index = (tp_rank + 1) * num_embeddings // tp_size
        self.num_embeddings_per_partition = vocab_end_index - vocab_start_index
        
        # Create parameter (random init)
        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                self.embedding_dim,
            )
        )
```

**PP Filtering:**

```python
def _should_include_layer(self, layer_idx, pp_rank, pp_size):
    """
    Determine if this layer belongs to this PP rank.
    
    Example: 32 layers, pp_size=4
    PP0: layers 0-7
    PP1: layers 8-15
    PP2: layers 16-23
    PP3: layers 24-31
    """
    layers_per_rank = self.config.num_hidden_layers // pp_size
    start_layer = pp_rank * layers_per_rank
    end_layer = (pp_rank + 1) * layers_per_rank
    
    return start_layer <= layer_idx < end_layer
```

---

#### `load_weights_and_postprocess()` - Weight Assignment

```python
def load_weights_and_postprocess(
    model: nn.Module,
    weights: Iterator[Tuple[str, torch.Tensor]],
    device_config: DeviceConfig,
):
    """
    Load weights into model and apply post-processing.
    """
    
    # Load weights
    model.load_weights(weights)
    
    # Post-process quantization methods
    for module_name, module in model.named_modules():
        if hasattr(module, "quant_method"):
            quant_method = module.quant_method
            
            # Some quantization methods need post-processing
            # Example: Repacking INT4 weights, computing scales
            if hasattr(quant_method, "process_weights_after_loading"):
                # Temporarily move to target device if CPU offloaded
                original_device = next(module.parameters()).device
                
                if original_device.type == "cpu":
                    module.to(device_config.device)
                
                quant_method.process_weights_after_loading(module)
                
                if original_device.type == "cpu":
                    module.to(original_device)
```

> IMPORTANT: Post load ops of Quantization formats like AWQ, GPTQ happen here

**`model.load_weights()` implementation:**

```python
def load_weights(self, weights_iterator: Iterator):
    """
    Load weights into model parameters.
    """
    
    # Build parameter name → parameter mapping
    params_dict = dict(self.named_parameters())
    
    for name, loaded_weight in weights_iterator:
        # Handle name mismatches (e.g., HF vs SGLang naming)
        param_name = self._convert_name(name)
        
        if param_name not in params_dict:
            # Skip if this parameter not in model (e.g., wrong PP rank)
            continue
        
        param = params_dict[param_name]
        
        # Call parameter's weight loader
        # This handles TP sharding
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
```

**Default weight loader (with TP sharding):**

```python
def default_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
):
    """
    Load weight with TP sharding.
    """
    
    # Get TP shard info from parameter
    tp_rank = getattr(param, "tp_rank", 0)
    tp_size = getattr(param, "tp_size", 1)
    tp_dim = getattr(param, "tp_dim", None)
    
    if tp_size == 1:
        # No TP, load entire weight
        param.data.copy_(loaded_weight)
    else:
        # Calculate this rank's slice
        shard_size = loaded_weight.size(tp_dim) // tp_size
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        
        # Extract shard
        if tp_dim == 0:
            shard = loaded_weight[start_idx:end_idx]
        elif tp_dim == 1:
            shard = loaded_weight[:, start_idx:end_idx]
        else:
            raise ValueError(f"Unsupported tp_dim: {tp_dim}")
        
        # Move to GPU and copy into parameter
        param.data.copy_(shard.to(param.device))
```

**Example weight loading:**

```
Model: Llama-3-8B with TP=4

Parameter: model.layers.0.self_attn.q_proj.weight
Shape: [4096, 4096]
TP dimension: 0 (split output dimension)

Loaded weight shape: [4096, 4096]

TP rank 0: Loads [0:1024, :] → param.data (shape [1024, 4096])
TP rank 1: Loads [1024:2048, :] → param.data (shape [1024, 4096])
TP rank 2: Loads [2048:3072, :] → param.data (shape [1024, 4096])
TP rank 3: Loads [3072:4096, :] → param.data (shape [1024, 4096])
```

---

#### Quantization Post-Processing

**Example: INT4 weight packing (GPTQ/AWQ)**

```python
class GPTQLinearMethod:
    def process_weights_after_loading(self, module):
        """
        Repack weights from INT32 to packed INT4.
        """
        
        # Original: INT4 weights stored as INT32 (wasteful)
        # [out_features, in_features // 8] INT32
        # Each INT32 holds 8 INT4 values
        
        qweight = module.qweight.data
        
        # Repack to actual INT4 storage (custom CUDA kernel)
        packed_qweight = pack_int4_weights(qweight)
        
        # Replace parameter
        module.qweight = nn.Parameter(packed_qweight, requires_grad=False)
        
        # Compute scales if needed
        if not hasattr(module, "scales"):
            module.scales = self.compute_scales(module.qweight, module.qzeros)
```

**Why post-process?**

- Checkpoint files store weights in "easy to load" format
- Runtime wants weights in "fast to compute" format
- Post-processing converts between them

---

## Parallelism Strategies

### Tensor Parallelism (TP)

**What it splits:** Model layers horizontally across GPUs

**How it works:**

```
Single GPU:
┌─────────────────────────────┐
│  Embedding [32k × 4096]     │
│  Layer 0:                   │
│    Q proj [4096 × 4096]     │
│    K proj [4096 × 4096]     │
│    V proj [4096 × 4096]     │
│    O proj [4096 × 4096]     │
│    Gate proj [4096 × 11008] │
│    Up proj [4096 × 11008]   │
│    Down proj [11008 × 4096] │
│  ...                        │
│  Layer 31                   │
│  LM Head [4096 × 32k]       │
└─────────────────────────────┘

TP=4:
GPU 0                GPU 1                GPU 2                GPU 3
┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ Embed[8k×4096] │  │ Embed[8k×4096] │  │ Embed[8k×4096] │  │ Embed[8k×4096] │
│ Q[1024×4096]   │  │ Q[1024×4096]   │  │ Q[1024×4096]   │  │ Q[1024×4096]   │
│ K[1024×4096]   │  │ K[1024×4096]   │  │ K[1024×4096]   │  │ K[1024×4096]   │
│ V[1024×4096]   │  │ V[1024×4096]   │  │ V[1024×4096]   │  │ V[1024×4096]   │
│ O[4096×1024]   │  │ O[4096×1024]   │  │ O[4096×1024]   │  │ O[4096×1024]   │
│ Gate[4096×2752]│  │ Gate[4096×2752]│  │ Gate[4096×2752]│  │ Gate[4096×2752]│
│ Up[4096×2752]  │  │ Up[4096×2752]  │  │ Up[4096×2752]  │  │ Up[4096×2752]  │
│ Down[2752×4096]│  │ Down[2752×4096]│  │ Down[2752×4096]│  │ Down[2752×4096]│
│ ...            │  │ ...            │  │ ...            │  │ ...            │
│ LM[4096×8k]    │  │ LM[4096×8k]    │  │ LM[4096×8k]    │  │ LM[4096×8k]    │
└────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘
```

**Communication pattern:**

```python
# Column-parallel (split output dimension)
# Example: Q projection

# Input: [batch, seq, 4096] (replicated on all ranks)
# Weight: [4096, 1024] per rank (total [4096, 4096])
# Output: [batch, seq, 1024] per rank

# Each rank computes its slice
local_output = F.linear(input, local_weight)  # [batch, seq, 1024]

# No communication needed! (splits are independent)


# Row-parallel (split input dimension)
# Example: O projection

# Input: [batch, seq, 1024] per rank (different on each rank!)
# Weight: [1024, 4096] per rank (total [4096, 4096])
# Output: [batch, seq, 4096] (needs to be same on all ranks)

# Each rank computes partial result
local_output = F.linear(local_input, local_weight)  # [batch, seq, 4096]

# All-reduce to sum across ranks
output = dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
```

**TP Benefits:**

- Splits large layers across GPUs
- Reduces memory per GPU
- Parallelizes computation

**TP Costs:**

- All-reduce communication (but overlapped with compute)
- Requires fast interconnect (NVLink, InfiniBand)

---

### Pipeline Parallelism (PP)

**What it splits:** Model layers vertically across GPUs

```
PP=4:

GPU 0 (PP rank 0):
┌─────────────────┐
│ Embedding       │
│ Layers 0-7      │
└─────────────────┘
        ↓ activations
GPU 1 (PP rank 1):
┌─────────────────┐
│ Layers 8-15     │
└─────────────────┘
        ↓ activations
GPU 2 (PP rank 2):
┌─────────────────┐
│ Layers 16-23    │
└─────────────────┘
        ↓ activations
GPU 3 (PP rank 3):
┌─────────────────┐
│ Layers 24-31    │
│ LM Head         │
└─────────────────┘
```

**Micro-batching for efficiency:**

```
Naive PP (sequential):
Time: |GPU0|    |GPU1|    |GPU2|    |GPU3|
      [====]    [====]    [====]    [====]
                ^^^^ GPUs idle ^^^^

PP with micro-batching:
Batch split into 4 micro-batches: [A, B, C, D]

Time: GPU0    GPU1    GPU2    GPU3
      [A  ]   
      [B  ]   [A  ]   
      [C  ]   [B  ]   [A  ]   
      [D  ]   [C  ]   [B  ]   [A  ]
              [D  ]   [C  ]   [B  ]
                      [D  ]   [C  ]
                              [D  ]
                              
Much better GPU utilization!
```

**PP Benefits:**

- Fits very large models (each GPU only holds ~1/4 of layers)
- Less communication than TP (only activations, not gradients)

**PP Costs:**

- Pipeline bubbles (idle time)
- Requires large batch sizes for efficiency
- More complex scheduling

---

### Data Parallelism (DP)

**What it splits:** Different requests to different model replicas

```
DP=3:

Replica 0:         Replica 1:         Replica 2:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Full Model  │    │ Full Model  │    │ Full Model  │
│             │    │             │    │             │
│ Request A   │    │ Request B   │    │ Request C   │
└─────────────┘    └─────────────┘    └─────────────┘
```

**DP Benefits:**

- Scales throughput linearly
- No communication during inference
- Simple to implement

**DP Costs:**

- Memory usage scales linearly (3 replicas = 3x memory)
- Load balancing needed

---

### Expert Parallelism (EP) - for MoE models

**What it splits:** Different experts across GPUs

```
MoE Layer (8 experts):

EP=4:
GPU 0: Experts 0-1
GPU 1: Experts 2-3
GPU 2: Experts 4-5
GPU 3: Experts 6-7

Forward pass:
1. Router decides which experts to use for each token
2. Tokens are routed to appropriate GPUs
3. Experts compute
4. Results are gathered back
```

**EP Communication:**

```python
# All-to-all exchange
# Each GPU sends tokens to other GPUs based on routing

tokens_to_send = {
    0: [tokens routed to experts 0-1],
    1: [tokens routed to experts 2-3],
    2: [tokens routed to experts 4-5],
    3: [tokens routed to experts 6-7],
}

tokens_to_compute = dist.all_to_all(tokens_to_send)

# Compute
expert_outputs = []
for expert in local_experts:
    expert_outputs.append(expert(tokens_to_compute[expert.id]))

# All-to-all gather
final_outputs = dist.all_to_all(expert_outputs)
```

---

### Combined Parallelism Example

**Configuration:** TP=4, PP=2, DP=2 (16 GPUs total)

```
Data Parallel Replica 0:          Data Parallel Replica 1:

PP Stage 0:                       PP Stage 0:
GPU 0  GPU 1  GPU 2  GPU 3        GPU 8  GPU 9  GPU 10 GPU 11
[E   ] [E   ] [E   ] [E   ]       [E   ] [E   ] [E   ] [E   ]
[L0-15] sharded with TP=4         [L0-15] sharded with TP=4
        ↓                                  ↓
PP Stage 1:                       PP Stage 1:
GPU 4  GPU 5  GPU 6  GPU 7        GPU 12 GPU 13 GPU 14 GPU 15
[L16-31] sharded with TP=4        [L16-31] sharded with TP=4
[LM  ] [LM  ] [LM  ] [LM  ]       [LM  ] [LM  ] [LM  ] [LM  ]
```

**Serving flow:**

```
Request A → Replica 0
Request B → Replica 1

Within each replica:
1. PP stage 0 computes on all TP ranks in parallel
2. Send activations to PP stage 1
3. PP stage 1 computes on all TP ranks in parallel
4. Return output
```

---

## Memory Management

### KV Cache

**What is it?**

During autoregressive generation:

```
Input: "The capital of France is"

Step 1: Compute attention for "The capital of France is"
- Store keys and values for all tokens

Step 2: Generate " Paris"
- Reuse stored keys/values from step 1
- Only compute new key/value for " Paris"

Step 3: Generate " is"  
- Reuse all previous keys/values
- Only compute new key/value for " is"

...
```

**Memory usage:**

```python
# Per request, per layer
seq_len = 2048  # tokens
num_heads = 32
head_dim = 128
dtype_size = 2  # bfloat16

k_cache_size = seq_len * num_heads * head_dim * dtype_size
v_cache_size = seq_len * num_heads * head_dim * dtype_size

per_layer = (k_cache_size + v_cache_size) / (1024**2)  # MB
= (2048 * 32 * 128 * 2 * 2) / (1024**2)
= 33.5 MB per layer

For 32 layers:
= 33.5 * 32
= 1.07 GB per request!

For 100 concurrent requests:
= 107 GB just for KV cache!
```

---

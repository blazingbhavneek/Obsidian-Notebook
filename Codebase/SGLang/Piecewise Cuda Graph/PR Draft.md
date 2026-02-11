# Piecewise CUDA Graph System Design and Optimization

This document provides a comprehensive overview of SGLang's Piecewise CUDA Graph system, covering its architecture, workflow, and key components. It details the compilation pipeline, memory optimization techniques, and configuration parameters, serving as a complete reference for developers to understand and work with Piecewise CUDA Graph for efficient LLM inference.

## Why and What is Piecewise CUDA Graph?

In large language model inference, the forward pass involves launching hundreds of small GPU kernels sequentially—attention projections, layer normalizations, MLPs, residual connections, and more. Each kernel launch incurs CPU-GPU communication overhead, which becomes a significant bottleneck when processing many requests.

Standard CUDA graphs address this by capturing the entire forward pass as a monolithic graph and replaying it with minimal overhead. However, this approach creates severe memory pressure: the entire execution must use fixed-size buffers allocated upfront, and all intermediate tensors remain in memory until the graph completes. For LLM serving with variable batch sizes and sequence lengths, this wastes GPU memory and limits throughput.

**Piecewise CUDA Graph** takes a fundamentally different approach: it decomposes the forward pass into layer-granular pieces, captures each piece as an independent CUDA graph, and orchestrates their execution through a lightweight stitching mechanism. This enables:

- **Fine-grained memory management**: Pieces share a single memory pool, and intermediate outputs are released immediately after consumption via weak tensor references
- **Shape flexibility**: Each piece can handle multiple pre-captured token counts, with runtime binary search to select the appropriate graph
- **Compilation optimization**: Pieces can be compiled with different backends (Inductor for performance, Eager for compatibility)

The result is significantly reduced kernel launch overhead while maintaining memory efficiency comparable to non-graph execution.

## System Design

[IMAGE: System Design Overview - Four-panel diagram showing: (1) Three-phase execution model (Compile → Warmup → Capture → Replay), (2) Compilation pipeline with trampoline pattern (original forward → trampoline → torch.compile → SGLangBackend), (3) Graph splitting (full model → split at layer boundaries → independent pieces → stitching graph), (4) Runtime shape resolution (request → binary search → graph selection → replay with padding/slicing)]

### Overall Architecture

Piecewise CUDA Graph follows a three-phase execution model:

1. **Compile Phase**: Each graph piece is compiled using `torch.compile` with the selected backend (Inductor or Eager). The compilation transforms PyTorch operations into optimized kernel calls.

2. **Warmup Phase**: Compiled pieces execute once to ensure all CUDA kernels are loaded and GPU driver state is initialized. This prevents compilation-related overhead during graph capture.

3. **Capture and Replay Phase**: A `torch.cuda.CUDAGraph` captures all kernel launches within each piece. Subsequent executions replay the captured graph directly, bypassing kernel launch overhead entirely.

The system uses a **trampoline pattern** to dynamically switch between compiled and uncompiled execution. The model's forward method is replaced with a dispatcher function that checks a global compilation flag and routes execution accordingly. This enables seamless debugging and profiling by toggling compilation on/off without code changes.

### Compilation Pipeline

The compilation pipeline transforms a standard PyTorch model into a piecewise-compiled version through several key mechanisms:

**Dynamic Forward Replacement**: The `install_torch_compiled` function intercepts the model's forward method and replaces it with a trampoline. This trampoline:
- Checks the `_COMPILE_ENABLED` context variable
- Triggers lazy compilation on first use (with compilation enabled)
- Dispatches to either the compiled or original forward path

**Lazy Compilation Strategy**: Compilation doesn't occur during initialization. Instead, when the trampoline first executes with `_COMPILE_ENABLED=True`, it:
- Marks dynamic tensor dimensions using `torch._dynamo.mark_dynamic`
- Invokes `torch.compile` with the SGLang backend
- Caches the compiled callable for subsequent use

**Bytecode Hooking**: TorchDynamo's bytecode transformation system is leveraged to track compilation. A registered hook captures the mapping between original and compiled bytecode, enabling cache management and debugging.

### Graph Splitting Strategy

The model graph is decomposed at layer boundaries using PyTorch FX's graph manipulation capabilities:

1. **Split Operation Selection**: The `CompilationConfig` defines which operations trigger splits (typically attention and MoE layer boundaries). These operations cannot be compiled themselves but serve as natural separation points.

2. **Submodule Isolation**: The `split_graph` function analyzes the FX graph, identifies split operation nodes, and partitions the graph into independent submodules. Each submodule represents one transformer layer or computation block.

3. **Stitching Graph Generation**: A top-level "stitching graph" is created that calls submodules sequentially, preserving data flow dependencies. This stitching graph handles:
   - Input distribution to the first piece
   - Intermediate tensor passing between pieces
   - Output aggregation from the final piece

**PiecewiseCompileInterpreter** executes this stitching graph and intercepts calls to targeted submodules. When a submodule execution is triggered, the interpreter:
- Compiles the submodule if needed
- Replaces the original submodule with a `CUDAPiecewiseBackend` instance
- Routes subsequent calls through the backend's optimized path

### Multi-Shape Support

Piecewise CUDA Graph pre-captures graphs for a predefined set of token counts. At runtime, variable-size requests are handled through a binning strategy:

**ConcreteSizeEntry Tracking**: Each pre-captured size has an associated entry that tracks:
- Whether compilation is needed for this size
- Whether a CUDA graph should be captured
- Compilation status and the compiled callable
- The captured CUDA graph object
- Stored output tensors (as weak references)

**Runtime Shape Resolution**: When a request arrives:
1. Extract the token count from the batch
2. Use binary search to find the smallest captured size ≥ token count
3. Select the corresponding CUDA graph
4. Pad inputs to match the captured size
5. Execute the graph and slice outputs back to the original size

**Fallback Mechanism**: If the token count exceeds the largest captured size, execution falls back to the standard (non-graph) forward path. This ensures correctness at the cost of higher latency for very large batches.

## Workflow

[IMAGE: Complete Workflow - Three-panel diagram showing: (1) Initialization Phase (capability checks → config setup → buffer allocation → memory pool → torch.compile warmup for each size), (2) Capture Phase (large-to-small size iteration → per-size: enter graph_capture context → dual-run mechanism → first run compiles/executes → second run captures graph), (3) Runtime Replay (request arrives → replay_prepare: shape resolution + buffer zeroing + data copying + static batch creation → replay: set contexts → model.forward triggers cudagraph.replay() → slice output to original size)]

### Initialization Phase

The initialization phase prepares the system for piecewise execution and occurs during `ModelRunner.__init__`:

1. **Capability Checks**: The system verifies compatibility by checking:
   - Pipeline parallelism (PP) size must be 1
   - Backend must not be Mooncake or DeepEP (which have compilation issues)
   - Global `torch.compile` must be disabled (fundamental conflicts)
   - All transformer layers must have standard attention modules

2. **Compilation Config Setup**: A `CompilationConfig` object is created with:
   - Capture sizes (auto-generated or user-specified)
   - Compiler backend selection (Inductor or Eager)
   - Split operation definitions
   - Debug mode flags

3. **Buffer Allocation**: Fixed-size tensors are pre-allocated for the maximum batch size and token count:
   - `input_ids`: token ID buffer
   - `positions`: position embedding indices
   - `out_cache_loc`: KV cache location pointers
   - Additional buffers for Mamba models, multimodal inputs, etc.

4. **Memory Pool Setup**: A global CUDA memory pool is initialized and shared across all graph pieces. This pool persists throughout the server lifetime and is reused across all capture sizes.

5. **Torch.Compile Warmup**: For each capture size (in reverse order), the system:
   - Enters the compilation context (`set_compiled(True)`)
   - Creates a dummy `ForwardBatch` with the target size
   - Calls `model.forward` to trigger torch.compile tracing
   - Caches the compiled callable for later use

This warmup ensures all compilation occurs upfront, before CUDA graph capture. Without it, capture would trigger compilation mid-stream, causing failures.

### Capture Phase

After warmup completes, the system enters the capture phase to record CUDA graphs:

**Capture Ordering**: Sizes are processed from largest to smallest. This ordering is critical for memory efficiency:
- Large graphs allocate memory in the pool first
- Smaller graphs reuse the same pool with minimal additional allocation
- Total memory usage is dominated by the largest graph, not the sum of all graphs

**Per-Size Capture Process**:
1. Enter the `graph_capture()` context, which creates a dedicated CUDA stream
2. Set the global capture stream so all pieces know which stream to use
3. Call `capture_one_batch_size(num_tokens)` within the capture context

**Dual-Run Mechanism**: Each size executes `run_once()` twice:
- **First run**: Executes the compiled piece normally. For sizes requiring compilation, this triggers the `CUDAPiecewiseBackend` to compile the piece for this specific shape.
- **Second run**: Captures the CUDA graph. The backend detects the capture context and wraps execution in `torch.cuda.graph()`, recording all kernel launches.

**Stream and Context Management**: The capture process carefully manages CUDA streams and Python contexts:
- Garbage collection is frozen to prevent memory address changes during capture
- The capture stream is set globally via `set_pcg_capture_stream`
- MultiPlatformOp layers are switched to their compile-compatible implementations
- For non-first pieces, `gc.collect` and `torch.cuda.empty_cache` are patched out to avoid slowdown

### Runtime Replay

During inference, when a request can be served with piecewise CUDA graph:

**Replay Preparation** (`replay_prepare`):
1. **Shape Resolution**: Find the smallest captured size ≥ actual token count using binary search
2. **Buffer Zeroing**: If the request is smaller than the captured size, zero out unused slots in `out_cache_loc` buffers to prevent stale data
3. **Data Copying**: Copy actual request data (input_ids, positions, etc.) into the fixed-size internal buffers
4. **Static Batch Creation**: Construct a `ForwardBatch` with views of the internal buffers truncated to the captured size

**Graph Execution** (`replay`):
1. Enter the piecewise CUDA graph context
2. Initialize attention backend metadata for the batch
3. Prepare the static batch via `replay_prepare`
4. Set the forward context with the static batch
5. Call `model.forward` with the static batch

When `model.forward` executes, each `CUDAPiecewiseBackend` instance:
- Detects that a CUDA graph exists for the current shape
- Calls `cudagraph.replay()` instead of executing PyTorch operations
- Returns the stored output tensors (whose underlying memory was just overwritten by replay)

**Output Processing**: The replay function slices the output to match the original request size, discarding padding and returning only the relevant results.

## Memory Optimization Techniques

[IMAGE: Memory Optimization Strategies - Four-panel diagram showing: (1) Shared Memory Pool (single global pool → all pieces and sizes share → eliminates fragmentation), (2) Weak Tensor References (piece output → convert to weak ref → GC reclaims memory immediately), (3) Capture Order (largest size first allocates pool → smaller sizes reuse without new allocation), (4) GC Control (freeze GC during entire capture → patch gc.collect/empty_cache for non-first pieces)]

### Shared Memory Pool

All graph pieces across all capture sizes share a single global CUDA memory pool. This pool is allocated once during initialization and managed via `graph_pool_handle()`.

**Benefits**:
- Eliminates memory fragmentation from per-graph allocations
- Reduces total memory footprint (peak is dominated by the largest graph)
- Enables memory reuse across sizes

**Lifecycle**: The pool persists throughout the server's lifetime. When a graph is captured, CUDA allocates from this pool. When the graph is destroyed (never, in normal operation), memory returns to the pool.

### Weak Tensor References

After a graph piece completes execution, its output tensors are no longer needed—the next piece will overwrite them. Standard tensor references would keep this memory alive unnecessarily.

**Solution**: The `weak_ref_tensors` function converts output tensors to weak references. Python's garbage collector can reclaim the underlying memory immediately, even though the weak reference object persists.

**Implementation**: For the last graph piece, `weak_ref_tensors` is called within the CUDA graph capture context. This ensures the weak reference conversion is baked into the graph itself.

### Capture Order Optimization

The large-to-small capture order is a deliberate memory optimization:

1. **First capture (largest size)**: Allocates memory in the pool for the maximum required tensors
2. **Subsequent captures (smaller sizes)**: Reuse the existing pool memory without additional allocation

If capture order were reversed (small-to-large), each larger graph would trigger new allocations, fragmenting the pool and increasing total memory usage.

### Garbage Collection Control

**During Initialization**: GC is frozen with `freeze_gc()` context manager during the entire capture phase. This prevents Python's garbage collector from running mid-capture, which could:
- Trigger CUDA memory reallocations
- Change tensor memory addresses
- Invalidate the captured graph

**During Non-First Piece Capture**: For pieces after the first, `gc.collect()` and `torch.cuda.empty_cache()` are explicitly patched to no-op functions. This avoids repeated garbage collection overhead when capturing many pieces sequentially.

## Integration Points

### Model Runner Integration

The `ModelRunner` class serves as the orchestration layer:

**Initialization Hook**: `init_piecewise_cuda_graphs()` is called during `ModelRunner.__init__` if piecewise is enabled. This method:
- Collects attention and MoE layers from the model
- Validates layer structure (all layers must have standard attention modules)
- Instantiates `PiecewiseCudaGraphRunner`

**Forward Extend Hook**: In `forward_extend()`, the runner checks if piecewise can be used and delegates to the piecewise runner's replay method if conditions are met.

**Can-Run Conditions**: Piecewise is disabled if:
- Logprobs are requested with a non-zero start position (mid-sequence logprob calculation)
- Token count exceeds the maximum captured size

**Fallback Path**: When piecewise cannot run, execution proceeds through the standard attention backend path.

### Compiler Backend Integration

The **SGLangBackend** serves as the entry point for torch.compile:

**Architecture**: When `torch.compile` is invoked with `backend=SGLangBackend`, the backend receives the FX graph and example inputs. It then:
1. Splits the graph at configured split operations
2. Identifies submodules to compile
3. Delegates compilation to `PiecewiseCompileInterpreter`

**PiecewiseCompileInterpreter**: This interpreter traverses the stitching graph and intercepts submodule calls. For each targeted submodule:
1. Retrieves the submodule's FX graph
2. Calls the compiler manager to compile it
3. Replaces the submodule with a `CUDAPiecewiseBackend` instance

**Inductor Adapter**: For Inductor backend compilation, the `InductorAdaptor` handles:
- Cache directory setup and hash computation
- Configuration of autotuning (enabled only for static shapes)
- Deep copying of graphs (Inductor mutates graphs in-place)
- Version-specific monkey-patching for PyTorch compatibility

**Eager Adapter**: The `EagerAdapter` is a no-op compiler that returns the graph unchanged. It's used for maximum compatibility when Inductor compilation fails or is disabled.

### Multi-Platform Support

**CUDA Backend** (`CUDAPiecewiseBackend`): The primary backend for NVIDIA GPUs. Implements:
- Per-size compilation and CUDA graph capture
- Input address validation in debug mode
- Weak reference optimization for last graph

**NPU Backend** (`NPUPiecewiseBackend`): A parallel implementation for Huawei NPUs. Key differences:
- Uses `torch.npu` instead of `torch.cuda`
- Uses `int32` for cache location dtype (NPU limitation)
- Platform-specific stream and graph management

Both backends share the same interface and are selected automatically based on device type.

## Multi-Rank Synchronization

In multi-GPU tensor parallel (TP) configurations, all ranks must maintain consistent state during compilation and capture:

**Initialization Synchronization**: After torch.compile warmup and before capture, the system synchronizes all devices and tensor parallel workers to ensure all ranks complete warmup before any rank begins capture.

**Capture Synchronization**: CUDA graph capture is inherently local (each rank captures its own graph), but the *order* must be consistent. The barrier ensures ranks don't drift out of sync during multi-size capture.

**Replay Synchronization**: No explicit synchronization is needed during replay—each rank replays its own graph independently. Data dependencies between ranks are handled by the model's existing TP communication kernels (captured within the graph).

## Debug and Profiling Support

### Graph Dumping

When debug mode is enabled in `CompilationConfig`, the system dumps FX graphs at multiple stages:

- **Pre-Split Graph**: The original model's FX graph before splitting
- **Post-Split Pieces**: Each submodule graph after splitting
- **Compiled Graphs**: The final compiled graphs from Inductor

Graphs are dumped using `torch._dynamo.utils.lazy_format_graph_code()`, which produces human-readable Python code representation.

### Address Validation

In debug mode, `CUDAPiecewiseBackend` records the `data_ptr()` addresses of all input tensors during graph capture. During replay, it asserts these addresses match. This catches bugs where input tensors are reallocated between capture and replay, which would cause incorrect results.

### Compilation Counter Instrumentation

The `CompilationCounter` class tracks system-wide statistics:

- `num_graphs_seen`: Total FX graphs encountered (including split ops)
- `num_piecewise_graphs_seen`: Submodule graphs (excluding split ops)
- `num_cudagraph_captured`: Successfully captured CUDA graphs
- `num_inductor_compiles`: Inductor backend compilations
- `num_backend_compilations`: Total backend invocations

These counters are accessible for monitoring and debugging compilation behavior.

## Configuration Parameters

### Core Flags

- **`--enable-piecewise-cuda-graph`**: Boolean flag to enable the feature. Must be explicitly set (default: disabled).

- **`--piecewise-cuda-graph-compiler {eager|inductor}`**: Selects the compiler backend. Inductor provides better performance but has compatibility limitations. Eager is a fallback for maximum compatibility (default: `eager`).

- **`--piecewise-cuda-graph-max-tokens`**: Maximum token count for which graphs are captured. Requests exceeding this fall back to non-graph execution. Default is model-dependent:
  - MLA models (DeepSeek V3/R1, Qwen MLA): `2048`
  - Other models: `chunked_prefill_size`

### Custom Capture Sizes

- **`--piecewise-cuda-graph-tokens`**: Space-separated list of token counts to capture. Must be in ascending order. Example: `--piecewise-cuda-graph-tokens 128 256 512 1024`

If not specified, sizes are auto-generated with increasing granularity:

| Token Range | Step Size |
|-------------|-----------|
| 4 – 32      | 4         |
| 48 – 256    | 16        |
| 288 – 512   | 32        |
| 576 – 1024  | 64        |
| 1280 – 4096 | 256       |
| 4608+       | 512       |

Sizes are capped at `--piecewise-cuda-graph-max-tokens`.

### Model-Specific Defaults

When piecewise is enabled on MLA backend models (DeepSeek V3/R1/V3.1, Qwen MLA variants):
- Max tokens automatically defaults to `2048` (prevents kernel dispatch differences)
- DeepSeek attention method switches to MLA for prefill paths

## Limitations and Constraints

### Prefill-Only Scope

Piecewise CUDA graph applies **only to extend (prefill) operations**, not decode. Decode requests always use the standard forward path.

Rationale: Decode operates on single tokens with small computational overhead. CUDA graph benefits are minimal, and the memory cost of capturing per-batch-size decode graphs would be prohibitive.

### Incompatibility Matrix

Piecewise CUDA graph is automatically disabled when:

- **Draft workers**: Disabled on all draft workers (in speculative decoding setups)
- **Global `torch.compile`**: Fundamental conflicts with whole-model torch.compile
- **Pipeline parallelism (PP > 1)**: Not yet supported (inter-stage communication not captured)
- **DeepEP or Mooncake MoE A2A backends**: Compilation errors prevent usage
- **Non-standard GQA layers**: All transformer layers must use standard Grouped Query Attention

### Shape Constraints

- **Static buffer allocation**: All captured sizes must fit within the maximum allocated buffer size
- **Padding overhead**: Small requests rounded up to the next captured size waste computation
- **Fallback latency**: Requests exceeding max captured size incur higher latency (no graph acceleration)

## Code Responsibility Table

| Component | File Path | Responsibilities |
|-----------|-----------|------------------|
| **Model Runner** | `python/sglang/srt/model_executor/model_runner.py` | - Entry point for piecewise initialization<br>- Capability checks (PP size, backend compatibility)<br>- Forward extend hook and can-run logic<br>- Fallback path handling |
| **Piecewise Runner** | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` | - Orchestrates entire capture and replay workflow<br>- Buffer allocation and management<br>- Torch.compile warmup execution<br>- Capture phase coordination<br>- Runtime replay and output processing |
| **Compilation Entry Point** | `python/sglang/srt/compilation/compile.py` | - Dynamic forward method replacement (trampoline)<br>- Lazy compilation triggering<br>- Bytecode hook registration<br>- Compilation context management |
| **Backend Orchestration** | `python/sglang/srt/compilation/backend.py` | - SGLangBackend implementation<br>- Graph splitting coordination<br>- PiecewiseCompileInterpreter execution<br>- Submodule compilation delegation |
| **CUDA Backend** | `python/sglang/srt/compilation/cuda_piecewise_backend.py` | - Per-piece CUDA graph capture<br>- Multi-shape compilation management<br>- Runtime graph replay<br>- Debug address validation |
| **NPU Backend** | `python/sglang/srt/compilation/npu_piecewise_backend.py` | - NPU-specific graph capture<br>- Platform-specific stream management<br>- NPU-compatible dtype handling |
| **Compilation Config** | `python/sglang/srt/compilation/compilation_config.py` | - Capture size management<br>- Compiler backend selection<br>- Split operation definitions<br>- Debug mode configuration |
| **Compiler Interface** | `python/sglang/srt/compilation/compiler_interface.py` | - Abstract compiler contract<br>- InductorAdaptor implementation<br>- EagerAdapter implementation<br>- Cache management |
| **Context Management** | `python/sglang/srt/compilation/piecewise_context_manager.py` | - Global compilation state tracking<br>- Capture/replay mode flags<br>- Thread-safe context variables |
| **Pass Manager** | `python/sglang/srt/compilation/pass_manager.py` | - Post-grad compiler pass orchestration<br>- Pass configuration and execution<br>- Functionalization fixing |
| **Inductor Passes** | `python/sglang/srt/compilation/inductor_pass.py` | - Custom compiler pass framework<br>- Graph transformation utilities<br>- Pass context management |
| **Compilation Counter** | `python/sglang/srt/compilation/compilation_counter.py` | - System-wide compilation statistics<br>- Counter instrumentation<br>- Monitoring and debugging support |
| **Memory Utilities** | `python/sglang/srt/compilation/weak_ref_tensor.py` | - Weak tensor reference creation<br>- Early memory release utilities<br>- Platform-specific implementations |

## Related Parameters

All configuration parameters are detailed in the [Configuration Parameters](#configuration-parameters) section above. For the complete list of server arguments and their interactions with other features, refer to the main SGLang server arguments documentation.

Key parameters to tune for performance:
- `--piecewise-cuda-graph-max-tokens`: Higher values enable more graph hits but increase memory usage
- `--piecewise-cuda-graph-compiler`: Inductor for best performance, Eager for compatibility
- `--mem-fraction-static`: Lower values (e.g., 0.7-0.8) leave more room for graph capture

## Summary

Piecewise CUDA Graph represents a novel approach to reducing kernel launch overhead in LLM inference. By decomposing the forward pass into layer-granular pieces, capturing each independently, and leveraging aggressive memory optimization techniques, it achieves the latency benefits of CUDA graphs without the memory pressure of monolithic capture.

The system's careful orchestration of compilation, capture, and replay—combined with robust multi-shape support and fallback mechanisms—makes it a production-ready optimization for extend/prefill workloads. Understanding its workflow and integration points enables developers to extend, debug, and optimize the system for new models and deployment scenarios.
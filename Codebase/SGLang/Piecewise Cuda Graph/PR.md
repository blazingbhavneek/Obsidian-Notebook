# Piecewise CUDA Graph System Design and Optimization

This document provides a comprehensive overview of SGLang's Piecewise CUDA Graph, covering its system architecture, compilation pipeline, and runtime execution. It details the motivation, graph splitting strategy, memory management, configuration parameters, and integration with the model execution framework, serving as a complete reference for users and developers to understand and optimize Piecewise CUDA Graph for efficient LLM inference.

## Why and What is Piecewise CUDA Graph?

In large language model inference, the forward pass involves launching many small GPU kernels sequentially—attention projections, layer norms, MLPs, residual connections, and more. The overhead of launching each kernel individually can become a significant performance bottleneck, especially during the prefill/extend phase where processing long sequences requires executing thousands of operations.

**Standard CUDA Graph** addresses this by capturing the entire forward pass as a single monolithic graph and replaying it. This eliminates per-kernel launch overhead by recording all kernel launches once and replaying them as a unit. However, this approach creates severe memory pressure: the entire execution must use fixed-size buffers, and all intermediate tensors remain allocated throughout the graph's lifetime. For LLM serving with variable batch sizes and sequence lengths, this memory rigidity becomes problematic.

**Piecewise CUDA Graph** takes a fundamentally different approach: it decomposes the forward pass into per-layer pieces and captures each piece as an independent CUDA graph. This enables fine-grained memory management—pieces share a single memory pool, intermediate outputs are released immediately via weak tensor references, and memory can be reused across graph boundaries. The result is significantly reduced memory footprint while maintaining the performance benefits of CUDA graph execution.

## System Design

### Overall Architecture

Piecewise CUDA Graph follows a **three-phase execution model**:

1. **Compile Phase**: Each graph piece is compiled using `torch.compile` with Inductor backend, optimizing kernel generation and enabling static shape specialization.

2. **Warmup Phase**: Compiled pieces run once to ensure CUDA kernels are loaded into GPU memory and ready for graph capture, establishing stable execution patterns.

3. **Capture Phase**: A `torch.cuda.CUDAGraph` is created for each piece at each target batch size. During capture, the piece executes within a special CUDA context that records all kernel launches. Subsequent forward passes replay these captured graphs directly, bypassing normal kernel dispatch overhead.

The system maintains a **stitching graph** (the top-level `fx.GraphModule` returned after splitting) that orchestrates execution of all pieces, preserving data flow dependencies and ensuring correct execution order.

### Graph Splitting Strategy

The forward pass is decomposed at **layer boundaries** using PyTorch FX's `split_module` API. The splitting process works as follows:

1. **Operation Identification**: The system identifies splitting operations defined in `CompilationConfig.split_ops`. These are typically high-level operations that mark natural layer boundaries 

2. **Subgraph Assignment**: Each node in the FX graph is assigned to a subgraph ID. When a splitting operation is encountered, the subgraph ID increments, creating a boundary. The splitting operation itself gets its own subgraph (marked as `is_splitting_graph=True`) which is not compiled—these operations typically have dynamic behavior incompatible with CUDA graph capture.

3. **Module Creation**: `split_module` creates independent `fx.GraphModule` objects for each subgraph, each representing one transformer layer or attention/MoE block.

4. **Stitching Graph**: A parent graph module is created that orchestrates execution of all submodules in the correct order, handling intermediate tensor passing between pieces.

The key insight is that **splitting operations are excluded from compilation**—they execute normally during replay, while the heavy computational kernels in each layer piece benefit from CUDA graph optimization.

### Memory Management

Piecewise CUDA Graph employs several sophisticated memory optimization techniques:

#### Shared Memory Pool

All graph pieces across all capture sizes share a **single global memory pool**, allocated once during initialization via `graph_pool_handle()`. This pool persists throughout the model's lifetime and is managed by CUDA's memory pool API. Key benefits:

- **Reduced fragmentation**: A single allocation domain prevents memory from fragmenting across multiple pools.
- **Cross-size reuse**: Memory allocated for a large capture size (e.g., 4096 tokens) can be reused by smaller sizes (e.g., 512 tokens) without additional allocation.
- **Total memory reduction**: Compared to per-size or per-piece pools, a shared pool dramatically reduces total memory consumption.

#### Capture Size Scheduling

The system pre-captures CUDA graphs for a **schedule of token counts**, allowing efficient execution across varying input sizes. The default schedule uses increasing granularity:

| Token Range | Step Size |
|-------------|-----------|
| 4 – 32      | 4         |
| 48 – 256    | 16        |
| 288 – 512   | 32        |
| 576 – 1024  | 64        |
| 1280 – 4096 | 256       |
| 4608+       | 512       |

Sizes are capped at the maximum token configuration parameter.

At runtime, incoming token counts are **rounded up** to the nearest captured size using binary search. If the count exceeds the maximum captured size, the system falls back to normal (non-graph) execution.

#### Large-to-Small Capture Order

CUDA graphs are captured in **descending size order** (e.g., 4096 → 2048 → 1024 → ... → 4). This ordering is critical:

- **Memory allocation strategy**: CUDA's memory pool allocates from a single contiguous region. Larger graphs allocate first, establishing the pool's maximum size.
- **Reuse for smaller sizes**: When smaller graphs are captured, they reuse memory already allocated by larger graphs, requiring minimal additional allocation.
- **Minimized total memory**: This strategy ensures total memory usage is dominated by the largest graph size, rather than the sum of all sizes.

#### Weak Tensor References

After capturing a CUDA graph piece, the system converts intermediate output tensors to **weak references** via `weak_ref_tensors()`. This optimization works in two phases:

- **Strong references during capture**: During CUDA graph capture, output tensors hold strong references to ensure memory remains valid. For the last graph piece, outputs are converted to weak references within the capture context to allow immediate release of the original tensor.
- **Weak conversion post-capture**: After capture completes for all graph pieces, outputs are stored as weak references using custom CUDA/NPU kernels that wrap tensor storage without increasing reference counts.
- **Early memory release**: When the tensor is no longer needed (e.g., after being passed to the next piece), memory is released immediately rather than waiting for the entire forward pass to complete.
- **Memory reuse**: Released memory becomes available for subsequent graph pieces, enabling in-place style memory patterns.

This technique significantly reduces peak memory usage for non-first graph pieces compared to holding strong references throughout the forward pass.

#### Garbage Collection Control

Python's garbage collector can interfere with CUDA graph capture by freeing memory at unpredictable times. To prevent this:

- **GC frozen during capture**: The `freeze_gc()` context manager disables Python's garbage collector during CUDA graph recording.
- **First piece only**: GC runs normally for the first piece to clean up warmup artifacts.
- **Subsequent pieces**: For all pieces after the first, `gc.collect()` and `torch.cuda.empty_cache()` are patched to no-ops, preventing memory churn.

This ensures captured graphs reference stable memory addresses and prevents performance degradation from repeated GC cycles.

### Compilation Pipeline

The compilation pipeline transforms computation graphs into optimized, executable backends through several stages:

#### Stage 1: Graph Splitting

When `torch.compile` invokes the SGLang backend, the first step is graph decomposition. The system calls `split_graph()` which produces:
- `split_gm`: The stitching graph that orchestrates all pieces
- `piecewise_graphs`: List of `SplitItem` objects, each containing a submodule graph

#### Stage 2: Submodule Compilation

The `PiecewiseCompileInterpreter` traverses the stitching graph and compiles targeted submodules (excluding splitting operations).

For each targeted submodule:

1. **Dynamic Shape Compilation**: First, the submodule is compiled for a general (dynamic) shape using the selected compiler backend (Eager or Inductor). This compilation happens through the `CompilerManager`, which coordinates compilation requests and manages caching.

2. **Symbolic Shape Tracking**: The interpreter identifies which input arguments contain symbolic (variable) shapes by detecting `torch.SymInt` objects in the inputs.

3. **Backend Substitution**: The original submodule is replaced with a `CUDAPiecewiseBackend` instance. This backend acts as a smart dispatcher that compiles specific shapes lazily and manages CUDA graph capture/replay.

#### Stage 3: Shape-Specific Compilation

The `CUDAPiecewiseBackend` maintains a `ConcreteSizeEntry` for each capture size, tracking:
- `runtime_shape`: The specific batch size (num_tokens)
- `need_to_compile`: Whether this size needs compilation
- `use_cudagraph`: Whether this size will use CUDA graphs
- `compiled`: Compilation status
- `runnable`: The compiled callable
- `num_finished_warmup`: Warmup run count
- `cudagraph`: The captured CUDA graph (if any)
- `output`: Weak-referenced outputs
- `input_addresses`: Debug information (optional)

When the backend is called with a specific shape for the first time, and that shape is in the compilation schedule, the system compiles that specific shape through the `CompilerManager`. This **lazy shape-specific compilation** happens only for shapes in the capture schedule, optimizing compile time.

### CUDA Graph Capture and Replay

#### Capture Process

CUDA graph capture happens in the `PiecewiseCudaGraphRunner.capture()` method. The process:

1. Freezes garbage collection using `freeze_gc()`
2. Creates a graph capture context with a dedicated stream
3. Sets the global capture stream for all pieces via `set_pcg_capture_stream()`
4. Iterates through capture sizes in descending order
5. For each size, captures one batch via `capture_one_batch_size()`

For each size, the capture method runs the forward pass **twice**:

1. **First run (warmup)**: Executes normally, incrementing `num_finished_warmup` for each piece.
2. **Second run (actual capture)**: When `CUDAPiecewiseBackend.__call__` detects sufficient warmup and no existing CUDA graph, it triggers capture by:
   - Creating a `torch.cuda.CUDAGraph` object
   - Running the compiled piece within a graph capture context
   - Recording all kernel launches into the graph
   - For the last graph only, converting outputs to weak references within the capture context
   - Storing the output (which gets converted to weak references after capture via `weak_ref_tensors()`)

The `torch.cuda.graph` context manager records all kernel launches. After capture completes, all graph outputs are stored as weak references (`entry.output = weak_ref_tensors(output)`) to enable memory reuse across replay cycles.

#### Stream Management

The capture stream is set globally so all pieces capture on the same stream. This ensures:

- **Sequential capture**: Pieces capture in order, maintaining dependencies.
- **Consistent stream context**: All kernel launches within a piece are recorded on the same stream.
- **Isolation from default stream**: Background operations on the default stream don't interfere with capture.

#### Replay Process

At runtime, when a forward pass is dispatched to the `PiecewiseCudaGraphRunner.replay()` method:

1. **Eligibility Check**: `can_run(forward_batch)` verifies the batch can use CUDA graphs (no mid-sequence logprob requests, size within limits).

2. **Buffer Preparation**: `replay_prepare(forward_batch)` maps dynamic batch data to static capture buffers:
   - Uses `bisect.bisect_left()` to find the smallest captured size that fits the actual batch
   - Copies input data to pre-allocated fixed-size buffers
   - Zeros out unused buffer slots
   - Creates a static `ForwardBatch` with truncated tensor views

3. **Graph Execution**: With the static batch prepared, the model's `forward()` method executes. Each piece's backend checks if a CUDA graph exists for this size, and if so, replays it instead of executing the compiled code normally.

4. **Piece Replay**: When replaying, `entry.cudagraph.replay()` is called, which overwrites the static buffers in-place with new results. The weak-referenced outputs (`entry.output`) now point to the freshly computed data.

5. **Output Slicing**: The final output is sliced to the actual batch size (`output.next_token_logits[:raw_num_tokens]`) to remove padding.

This workflow achieves zero-copy replay by overwriting static buffers in-place.

### Shape Specialization

#### Runtime Shape Extraction

For graph pieces with dynamic dimensions (tracked via `sym_shape_indices`), the runtime shape is extracted from input arguments:

```python
runtime_shape = args[self.sym_shape_indices[0]]
```

This is typically the number of tokens (`num_tokens`) in the forward batch.

#### Binary Search Lookup

At runtime, the actual token count is matched to a captured size. If the runtime shape is not in the set of compiled entries (`self.concrete_size_entries`), the system falls back to the general compiled version (`compiled_graph_for_general_shape`) for dynamic shapes.

The `replay_prepare()` method uses `bisect.bisect_left()` to find the smallest captured size greater than or equal to the actual size, enabling efficient shape rounding.

#### Fallback Mechanism

If the runtime shape exceeds all captured sizes, or if other conditions prevent CUDA graph usage (e.g., mid-sequence logprob requests), the system falls back to:
- **General compiled graph**: Uses the dynamic shape compilation from Stage 2.
- **Normal forward path**: If compilation is disabled, executes the original uncompiled forward method.

This ensures correctness even when CUDA graph requirements aren't met.

### Multi-Rank Synchronization

In distributed settings (e.g., tensor parallelism), all ranks must capture and replay graphs in lockstep:

#### Warmup Synchronization

After `torch.compile` warmup, before capture:

```python
self.device_module.synchronize()
self.model_runner.tp_group.barrier()
```

This ensures all ranks have completed compilation before proceeding to capture.

#### Capture Coordination

The same capture sizes and ordering are used across all ranks, determined by `CompilationConfig.get_capture_sizes()`. This guarantees:
- **Identical graph structure**: All ranks capture the same computational patterns.
- **Consistent memory layout**: Memory addresses across ranks remain synchronized.
- **Collective operation compatibility**: Operations like all-reduce have matching graph nodes across ranks.

#### Replay Synchronization

During replay, collective operations are embedded within the CUDA graph. The graph replays these operations at the exact same points across all ranks, maintaining synchronization automatically.

## Compilation System Components

### Compilation Configuration

The `CompilationConfig` class centralizes all compilation settings:

- **traced_files**: Set of file paths to trace for debugging
- **capture_sizes**: List of token counts to capture
- **compiler**: Backend to use ("eager" or "inductor")
- **enable_debug_mode**: Enable debug logging
- **split_ops**: List of operations to split graph at

Key methods:
- `get_capture_sizes()`: Returns the capture schedule, either user-provided or auto-generated
- `add_split_op()`: Dynamically add splitting operations
- `add_traced_file()`: Mark files for tracing

The configuration is initialized in `ModelRunner.__init__` and passed throughout the compilation pipeline.

### Backend Architecture

The `SGLangBackend` serves as the entry point for `torch.compile`. The workflow:

1. **Cache Setup**: Initializes cache directory based on compiler hash. The hash incorporates system state (PyTorch version, GPU driver, etc.) to ensure cache invalidation when environment changes.

2. **Graph Splitting**: Decomposes the monolithic graph into pieces using `split_graph()`.

3. **Compilation Orchestration**: Runs `PiecewiseCompileInterpreter` to compile submodules, substituting `CUDAPiecewiseBackend` instances.

4. **Graph Saving**: On rank 0, saves the stitching graph for debugging.

5. **Return**: Returns the stitching graph module (`split_gm`), now with compiled backends substituted.

### Compiler Interfaces

The `CompilerInterface` abstract base class defines the contract for all compiler backends:

- **name**: String identifier (e.g., "inductor", "eager")
- **initialize_cache()**: Sets up caching directories with proper prefixes
- **compute_hash()**: For cache invalidation - if compiler config changes, hash changes
- **compile()**: Takes a graph and returns a compiled callable plus a handle for caching
- **load()**: Reconstructs a callable from a cached handle

#### Inductor Adapter

The `InductorAdaptor` integrates PyTorch's Inductor compiler:

- **Hash Computation**: Uses `get_inductor_factors()` to create cache keys from system and PyTorch state
- **Autotuning Control**: Only enables `max_autotune` and `coordinate_descent_tuning` for specific shapes (not dynamic), since autotuning dynamic shapes is wasteful
- **Shape Environment Patching**: Patches Inductor's shape guard system with `AlwaysHitShapeEnv` to bypass checks when compiling outside Dynamo's tracing context
- **Caching**: Leverages Inductor's `FxGraphCache` to avoid recompilation across runs

Key optimization: When compiling for a specific shape (not dynamic), `max_autotune` and `coordinate_descent_tuning` are enabled. For dynamic shapes, these are disabled to avoid wasting compilation time on optimizations that may not apply at runtime.

#### Eager Adapter

A no-op compiler for debugging that returns the unmodified graph. Useful for isolating compilation issues or comparing performance.

### Pass Management System

The `PostGradPassManager` orchestrates custom optimization passes:

- **Pass Registration**: Maintains a list of `SGLangInductorPass` objects
- **Pass Execution**: For each pass, checks if it applies to the current shape via `is_applicable_for_shape()`, then executes it
- **Defunctionalization**: After custom passes, runs `FixFunctionalizationPass` to convert functional operations back to in-place mutations, eliminating unnecessary tensor copies

Custom passes can:
- **Track timing**: Each pass logs execution time via `begin()` and `end_and_log()`
- **Filter by shape**: Passes can target specific shapes or skip dynamic shapes
- **Dump graphs**: Provide detailed debugging output via `dump_graph()`

The defunctionalization pass converts operations like `out = torch.add(x, y)` into `x.add_(y); out = x`, eliminating tensor copies that are critical for memory efficiency in CUDA graphs.

## Runtime Execution

### Piecewise CUDA Graph Runner

The `PiecewiseCudaGraphRunner` manages CUDA graph lifecycle from initialization through replay:

#### Initialization

During initialization, the runner:

1. **Configuration Setup**: Creates `CompilationConfig` from server parameters
2. **Buffer Allocation**: Pre-allocates fixed-size buffers for the largest capture size (`input_ids`, `positions`, `out_cache_loc`, etc.)
3. **Plugin Initialization**: Sets up `TboCudaGraphRunnerPlugin` for Two-Batch Overlap optimization
4. **Memory Pool Setup**: Initializes the shared CUDA memory pool via `get_global_graph_memory_pool()`
5. **Model Patching**: Temporarily puts model in compile mode via `patch_model()`
6. **Torch Compile Installation**: Replaces model forward methods with compile-aware trampolines via `install_torch_compiled()`
7. **Warmup**: Triggers `torch.compile` for each capture size via `warmup_torch_compile()` to prime the compilation cache
8. **Synchronization**: Syncs GPUs and tensor parallel workers
9. **Capture**: Records CUDA graphs for all sizes via `capture()`

#### Buffer Management

Fixed-size buffers are essential for CUDA graph compatibility. The system pre-allocates tensors for maximum sizes:
- Input token IDs
- Token positions
- Cache locations
- Sequence lengths
- And many more model-specific buffers

At replay time, actual batch data is copied into slices of these buffers, ensuring memory addresses remain stable across replays.

#### Forward Pass Orchestration

**Eligibility Check**: Before using CUDA graphs, `can_run(forward_batch)` verifies batch compatibility by checking:
- Logprob constraints: Mid-sequence logprob requests cannot be handled efficiently by CUDA graphs
- Size limits: Batch must fit within the maximum captured size

**Replay Preparation**: `replay_prepare(forward_batch)` maps dynamic batch data to static capture buffers by:
- Finding the nearest capture size using `bisect.bisect_left()` (rounded up)
- Storing the actual size for output slicing
- Zeroing unused buffer slots
- Copying batch data to static buffers
- Creating a static `ForwardBatch` with truncated views

**Graph Execution**: Orchestrates the entire replay process by:
- Initializing attention backend metadata
- Preparing static buffers
- Executing `model.forward()` (triggers CUDA graph replay in backends)
- Slicing output to actual batch size

Context managers signal the execution mode (`enable_piecewise_cuda_graph()`, `set_compiled()`, etc.) throughout the call stack.

### Integration with Model Runner

The `ModelRunner` is the top-level executor that dispatches forward passes:

During initialization:
1. `ModelRunner.__init__` calls `can_run_piecewise_cuda_graph()` to check eligibility
2. If eligible, calls `init_piecewise_cuda_graphs()` to create the runner
3. Runner compiles and captures all graphs during startup

During runtime:
1. `forward_extend()` method checks if CUDA graph can run via `can_run(forward_batch)`
2. If yes, dispatches to `piecewise_cuda_graph_runner.replay()`
3. If no, falls back to normal attention backend execution

## Configuration and Parameters

### Core Parameters

- **`--enable-piecewise-cuda-graph`**: Enable piecewise CUDA graph functionality for extend/prefill operations. This is required to use piecewise CUDA graph.

- **`--piecewise-cuda-graph-tokens TOKENS [TOKENS ...]`**: Explicitly specify which token counts to capture CUDA graphs for. Sizes must be in ascending order. Default is auto-generated schedule (see [Capture Size Scheduling](#capture-size-scheduling)). When to use: predictable workload sizes, minimize compilation time, or memory constraints. Example: `--piecewise-cuda-graph-tokens 128 256 512 1024 2048 4096`

- **`--piecewise-cuda-graph-compiler {eager,inductor}`**: Selects the compiler backend for graph piece compilation. Default: `eager`.
  - `eager`: No compilation, faster startup, no autotuning overhead (recommended for most cases)
  - `inductor`: PyTorch's optimizing compiler, longer startup, potentially higher throughput for compute-bound workloads

- **`--piecewise-cuda-graph-max-tokens MAX_TOKENS`**: Maximum token count for CUDA graph capture. Sizes beyond this limit fall back to normal execution.
  - Lower values (e.g., 512): Reduce startup time and memory usage, suitable for short-context workloads
  - Higher values (e.g., 8192): Support longer sequences, increase memory and startup cost
  - MLA models: Keep at 2048 to avoid kernel dispatch mismatches

### Default Capture Schedule

The auto-generated schedule balances coverage and granularity. It uses fine granularity at small sizes (short sequences need tight rounding) and coarse granularity at large sizes (long sequences are less sensitive to rounding, so wider steps reduce capture count).

### Advanced Configuration

- **`--mem-fraction-static MEM_FRACTION`**: Fraction of GPU memory reserved for static allocations (including CUDA graph memory pool). Default: `0.9`. Reduce if CUDA graph capture fails with OOM (e.g., `0.8` or `0.7`).

- **`--chunked-prefill-size CHUNK_SIZE`**: Maximum prefill chunk size. Default: Model-dependent. For non-MLA models, this becomes the default `piecewise_cuda_graph_max_tokens`.

## Compatibility and Constraints

### Compatible Features

#### ViT CUDA Graph (Multimodal Models)

Piecewise CUDA graph can be combined with ViT (Vision Transformer) CUDA graph for multimodal models. Example:

```bash
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
python -m sglang.launch_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --enable-piecewise-cuda-graph \
  --piecewise-cuda-graph-max-tokens 4096 \
  --piecewise-cuda-graph-compiler eager
```

Both mechanisms operate independently:
- **ViT CUDA graph**: Captures image encoder forward pass
- **Piecewise CUDA graph**: Captures language model forward pass

### Model-Specific Behavior

#### MLA (Multi-Latent Attention) Models

Models using MLA backend (DeepSeek V3/R1/V3.1, Qwen MLA variants) have special handling:
- **Max tokens default**: Automatically set to 2048 to avoid kernel dispatch differences between CUDA graph capture and normal execution
- **Attention method switch**: DeepSeek models switch to MLA for prefill paths when piecewise CUDA graph is enabled
- **Rationale**: MLA models use different attention kernels for different sequence lengths, so limiting max tokens ensures consistency

### Automatic Disabling Conditions

Piecewise CUDA graph is automatically disabled under these conditions:

1. **Draft Workers (Speculative Decoding)**: Draft models have different execution patterns (generating multiple token candidates) incompatible with deterministic CUDA graph replay.

2. **Global torch.compile Enabled**: The `--enable-torch-compile` feature conflicts with piecewise CUDA graph's custom compilation approach. Do not use `--enable-torch-compile` with `--enable-piecewise-cuda-graph`. Piecewise CUDA graph already provides compilation via the `--piecewise-cuda-graph-compiler` option.

3. **Pipeline Parallelism (PP > 1)**: Not yet supported. Planned for future releases.

4. **DeepEP or Mooncake MoE Backends**: These custom communication backends trigger compilation errors during CUDA graph capture.

5. **Non-Standard Architectures**: If transformer layers don't have recognizable attention modules (standard GQA), piecewise CUDA graph is disabled.

### Scope Limitations

**Extend/Prefill Only**

Piecewise CUDA graph applies only to extend (prefill) operations, not decode:
- Extend/Prefill: Processing input sequences to generate KV cache (significantly accelerated)
- Decode: Generating one token at a time with cached KV (already uses standard CUDA graphs)

Rationale: Decode operates on very small batch sizes with highly variable batch dimensions. Piecewise splitting provides no benefit; standard CUDA graphs with dynamic batching are optimal.

**Experimental Feature Status**

Piecewise CUDA graph is marked as experimental:
- API stability: Configuration parameters and behavior may change
- Compatibility: New model architectures may not immediately support piecewise CUDA graph
- Testing coverage: While extensively tested on mainstream models, edge cases may exist

## Optimization Techniques

### Memory Optimizations

**Large-to-Small Capture Order**

CUDA graphs are captured in descending size order. The algorithm iterates through sizes from largest to smallest. Memory behavior:
- Largest size (e.g., 4096): Allocates ~8GB from the memory pool
- Next size (e.g., 2048): Reuses most of the 8GB, allocates ~500MB additional
- Smaller sizes: Continue reusing, adding minimal allocation
- Total: ~8.7GB instead of 14GB+ if captured in ascending order

**Pool-Based Allocation Strategy**

The shared memory pool is initialized once via `graph_pool_handle()` and all CUDA graph captures pass the same pool handle:

```python
graph_pool_id = graph_pool_handle()
set_graph_pool_id(graph_pool_id)

# During capture
with torch.cuda.graph(cudagraph, pool=graph_pool_id, stream=stream):
    # Graph capture
```

Benefits:
- Single allocation domain: All graphs allocate from the same pool
- Reduced fragmentation: No per-graph pools fragmenting memory
- Persistent across captures: Pool persists from first to last capture

**Intermediate Tensor Lifecycle Management**

The weak reference optimization works in two phases:

1. **During Capture**: The compiled piece runs and produces output tensors. For the last graph only, these are converted to weak references within the capture context.

2. **Post-Capture**: All graph outputs are stored as weak references. These weak references wrap the tensor storage without increasing reference counts.

3. **During Replay**: When the CUDA graph replays, it overwrites memory in-place. The weak-referenced outputs now point to fresh data.

For large models with many layers, weak references substantially reduce peak memory usage compared to holding strong references for all intermediate outputs.

### Compilation Optimizations

**Cache Hit Maximization**

The `CompilerManager` implements two-level caching:

**Level 1: In-Memory Cache**: Stores compiled callables keyed by `(runtime_shape, graph_index, compiler_name)`

**Level 2: Disk Cache**: Sets environment variables for Inductor and Triton cache directories:

```python
os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir + "/inductor"
os.environ["TRITON_CACHE_DIR"] = cache_dir + "/triton"
```

Copying one directory migrates all caches.

Cache invalidation is based on a hash of system state:

```python
cache_hash = md5(get_inductor_factors())
cache_dir = f"~/.cache/sglang/torch_compile_cache/{cache_hash[:10]}/"
```

If PyTorch version, GPU driver, or compiler flags change, the hash changes and old cache is ignored.

**Shape Environment Patching**

When compiling outside Dynamo's tracing context, Inductor's cache lookup fails without a shape environment. The system patches in an `AlwaysHitShapeEnv` that bypasses guard checks:

```python
class AlwaysHitShapeEnv:
    def evaluate_guards_expression(self, *args, **kwargs):
        return True  # Always pass guards
    
    def get_pruned_guards(self, *args, **kwargs):
        return []  # No guards to check

# Patched into cache lookup
with patch.object(cache, "get_shape_env", return_value=AlwaysHitShapeEnv()):
    compiled_graph = cache._lookup_graph(...)
```

This is safe because shape tracking is done explicitly through the shape specialization system.

**Incremental Compilation Tracking**

The `CompilationCounter` tracks progress across the system:

```python
compilation_counter.num_graphs_seen += 1
compilation_counter.num_piecewise_graphs_seen += len(pieces)
compilation_counter.num_backend_compilations += 1
compilation_counter.num_cudagraph_captured += 1
```

Logging at milestones indicates progress:

```
[INFO] Compiling a graph for shape 1024 takes 3.45 s
[INFO] Capture piecewise CUDA graph end. Time elapsed: 125.23 s. mem usage=12.34 GB
```

**Autotuning Strategy**

Inductor's autotuning optimizes kernel parameters via benchmarking. The system enables aggressive optimization only for specific shapes:

```python
def set_inductor_config(config, runtime_shape):
    if runtime_shape is None:
        # Dynamic shape: disable autotuning
        config["max_autotune"] = False
        config["coordinate_descent_tuning"] = False
    else:
        # Specific shape: enable aggressive optimization
        config["max_autotune"] = True
        config["coordinate_descent_tuning"] = True
```

Rationale: Autotuning for dynamic shapes wastes time because the benchmarked configuration may not apply to actual runtime shapes. For specific shapes, autotuning optimizes for the exact captured size.

## Code Organization

The piecewise CUDA graph system spans multiple modules, each with specific responsibilities:

| Module | Responsibility |
|--------|---------------|
| model_runner.py | Top-level execution orchestration, eligibility checks, runner initialization, forward dispatch |
| piecewise_cuda_graph_runner.py | CUDA graph lifecycle management, capture/replay orchestration, buffer management, TBO plugin integration |
| compile.py | Torch.compile integration, trampoline pattern for compiled/uncompiled switching, dynamic shape marking |
| compilation_config.py | Configuration management, capture size scheduling, split operation tracking |
| compilation_counter.py | Compilation statistics tracking across the system |
| cuda_piecewise_backend.py | Per-piece backend implementation, shape-specific lazy compilation, CUDA graph capture/replay per piece |
| weak_ref_tensor.py | Weak reference utilities for early memory release |
| inductor_pass.py | Custom compiler passes, graph transformation utilities |
| fix_functionalization.py | Defunctionalization pass to eliminate tensor copies |
| pass_manager.py | Orchestration of custom optimization passes |
| compiler_interface.py | Abstract compiler interface, Inductor adapter, Eager adapter, cache management |

## Troubleshooting

### Out-of-Memory During Capture

**Symptoms**: CUDA out of memory errors during graph capture

**Solutions**:

1. **Lower max tokens**: Reduces the largest graph size, decreasing peak memory
2. **Lower memory fraction static**: Reserves less memory for static allocations
3. **Use fewer capture sizes**: Reduces total number of graphs
4. **Disable the feature**: Falls back to normal execution

### Slow Startup (Long Compilation Time)

**Symptoms**: Very long time spent compiling graphs

**Solutions**:

1. **Switch to eager compiler**: 5-10x faster than Inductor with autotuning
2. **Reduce capture sizes**: Fewer large sizes to compile
3. **Enable disk cache**: Second run should be much faster due to cache hits
4. **Check for cache invalidation**: Freeze your environment to benefit from caching

### CUDA Graph Replay Failures

**Symptoms**: CUDA graph replay errors or invalid memory access

**Debugging**:

1. **Enable debug mode**: Logs detailed graph information and validates memory addresses
2. **Check for dynamic shapes**: Verify workload compatibility with captured sizes
3. **Disable weak references**: Temporarily disable to isolate the issue

### Incorrect Results

**Symptoms**: Model outputs differ between CUDA graph replay and normal execution

**Checks**:

1. **Verify input data copying**: Ensure replay preparation copies all necessary tensors
2. **Check attention backend compatibility**: Verify backend initializes metadata correctly
3. **Compare with fallback**: Force fallback for all requests to isolate CUDA graph logic

### Performance Degradation

**Symptoms**: CUDA graph replay is slower than expected

**Optimizations**:

1. **Ensure TBO is enabled**: Overlaps computation to hide memory latency
2. **Check capture size granularity**: Adjust sizes to better match workload
3. **Profile kernel execution**: Use profiling tools to identify bottlenecks
4. **Try Inductor compiler**: May generate more efficient kernels for compute-bound models
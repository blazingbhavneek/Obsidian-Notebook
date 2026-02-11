# Piecewise CUDA Graph System Design and Optimization

This document provides a comprehensive overview of SGLang's Piecewise CUDA Graph, covering its system architecture, compilation pipeline, and runtime execution. It details the motivation, graph splitting strategy, memory management, configuration parameters, and integration with the model execution framework, serving as a complete reference for users and developers to understand and optimize Piecewise CUDA Graph for efficient LLM inference.

## Why and What is Piecewise CUDA Graph?

In large language model inference, the forward pass involves launching many small GPU kernels sequentially—attention projections, layer norms, MLPs, residual connections, and more. The overhead of launching each kernel individually can become a significant performance bottleneck, especially during the prefill/extend phase where processing long sequences requires executing thousands of operations.

**Standard CUDA Graph** addresses this by capturing the entire forward pass as a single monolithic graph and replaying it. This eliminates per-kernel launch overhead by recording all kernel launches once and replaying them as a unit. However, this approach creates severe memory pressure: the entire execution must use fixed-size buffers, and all intermediate tensors remain allocated throughout the graph's lifetime. For LLM serving with variable batch sizes and sequence lengths, this memory rigidity becomes problematic.

**Piecewise CUDA Graph** takes a fundamentally different approach: it decomposes the forward pass into per-layer pieces and captures each piece as an independent CUDA graph. This enables fine-grained memory management—pieces share a single memory pool, intermediate outputs are released immediately via weak tensor references, and memory can be reused across graph boundaries. The result is significantly reduced memory footprint while maintaining the performance benefits of CUDA graph execution.

### Overall Architecture

Piecewise CUDA Graph follows a **three-phase execution model**:

1. **Compile Phase**: Each graph piece is compiled using `torch.compile` with Inductor backend, optimizing kernel generation and enabling static shape specialization.

2. **Warmup Phase**: Compiled pieces run once to ensure CUDA kernels are loaded into GPU memory and ready for graph capture, establishing stable execution patterns.

3. **Capture Phase**: A `torch.cuda.CUDAGraph` is created for each piece at each target batch size. During capture, the piece executes within a special CUDA context that records all kernel launches. Subsequent forward passes replay these captured graphs directly, bypassing normal kernel dispatch overhead.

The system maintains a **stitching graph** (the top-level `fx.GraphModule` returned after splitting) that orchestrates execution of all pieces, preserving data flow dependencies and ensuring correct execution order.

### Memory Management

Piecewise CUDA Graph employs several memory optimization steps:

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

## Compilation and Execution Pipeline

### Graph Splitting Strategy

The system decomposes the transformer's forward pass at natural layer boundaries using PyTorch FX graph transformations. During splitting:

1. **Boundary Detection**: The compiler identifies high-level operations that mark layer transitions (e.g., attention projections or MoE routing kernels) as split points, configured via `CompilationConfig.split_ops`.

2. **Piece Isolation**: Each transformer layer or attention block becomes an independent computational piece. Critically, the splitting operations themselves are isolated into separate uncompiled subgraphs—these typically contain dynamic control flow incompatible with CUDA graph capture.

3. **Orchestration Layer**: A top-level "stitching graph" is generated to sequence execution across all pieces, managing tensor handoffs while preserving data dependencies. This enables per-layer CUDA graph capture without breaking the overall computation flow.

### Three-Stage Compilation Pipeline

**Stage 1: Graph Decomposition**  
When `torch.compile` triggers the SGLang backend, the monolithic forward graph is split into layer-wise pieces. This produces both the stitching graph (orchestrator) and independent subgraphs ready for optimization.

**Stage 2: Dynamic Shape Compilation**  
Each computational piece undergoes initial compilation for dynamic shapes using the selected backend (Eager or Inductor). During this phase, the system tracks which inputs contain symbolic dimensions (typically sequence length) to enable later shape specialization. Original submodules are replaced with smart dispatcher backends that manage shape-specific compilation on demand.

**Stage 3: Lazy Shape-Specific Compilation**  
Rather than compiling all possible shapes upfront, the system employs lazy compilation triggered at first use. When a piece executes with a token count matching the capture schedule, it compiles that exact shape with aggressive optimizations (including kernel autotuning for Inductor). This minimizes startup time while ensuring optimal performance for frequently encountered batch sizes.

### CUDA Graph Capture and Replay

**Capture Workflow**  
Graph capture occurs during model initialization in descending size order (largest to smallest). For each target size:

1. A warmup pass executes to stabilize memory patterns and load kernels into GPU cache.
2. A second pass records all kernel launches within a dedicated CUDA stream context, producing a replayable graph object.
3. Intermediate outputs are converted to weak references immediately after capture, enabling memory reuse across pieces.
4. Python's garbage collector is frozen during capture to prevent memory address instability.

This descending-size ordering is critical: larger graphs establish the memory pool's maximum footprint, and smaller graphs reuse this allocation space, minimizing total memory consumption.

**Runtime Replay**  
At inference time:

1. Incoming token counts are rounded up to the nearest pre-captured size via binary search.
2. Input data is copied into pre-allocated static buffers matching that size.
3. The stitching graph executes, with each piece's backend detecting the matching CUDA graph and replaying it directly—overwriting static buffers in-place rather than launching kernels individually.
4. Final outputs are sliced to the actual batch size to remove padding.

This zero-copy replay eliminates kernel launch overhead while maintaining compatibility with dynamic request patterns.

### Shape Specialization and Fallback

The system extracts the runtime sequence length from input arguments and matches it against pre-captured sizes. If the token count falls within the capture schedule, the corresponding CUDA graph replays. For sizes exceeding the maximum captured threshold—or for requests with special constraints like mid-sequence logprob sampling—the system seamlessly falls back to the dynamically compiled graph or standard execution path. This hybrid approach guarantees correctness across all workloads while maximizing performance for common patterns.

### Multi-Rank Synchronization

In distributed tensor-parallel settings, all ranks coordinate capture and replay through explicit synchronization barriers. After warmup compilation completes, ranks synchronize before beginning capture to ensure identical graph structures across devices. The same capture sizes and ordering are used universally, guaranteeing consistent memory layouts and compatible collective operations (like all-reduce) embedded within graphs. During replay, these collective operations execute at precisely synchronized points across ranks, maintaining correctness without additional coordination overhead. This design enables piecewise CUDA graphs to scale efficiently across multi-GPU deployments.

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

## Applicability and Limitations
 
 - Piecewise CUDA Graph is designed specifically for the **extend phase** of LLM inference, where kernel launch overhead dominates and sequence lengths are sufficiently large to benefit from graph replay. It does **not apply to decode steps**, which already use standard CUDA graphs with dynamic batching.
 - The feature is **automatically disabled** in incompatible scenarios: speculative decoding (draft workers), pipeline parallelism (PP > 1), or models with non-standard layer structures (e.g., missing recognizable attention modules). It also conflicts with global `torch.compile` and certain custom backends (e.g., DeepEP, Mooncake MoE).
 - For **MLA-based models** (e.g., DeepSeek-V3, Qwen-MLA), the maximum captured token count is capped at 2048 to ensure consistent kernel dispatch between capture and replay.
 - The system can **coexist with ViT CUDA Graph** in multimodal models: ViT CUDA Graph optimizes the vision encoder, while Piecewise CUDA Graph handles the language model—operating independently on separate subgraphs.
 - As an **experimental feature**, its interface and behavior may evolve; users should expect changes in future releases.


### Troubleshooting

#### CUDA Graph Capture Failure

If CUDA graph capture follow the recommended actions emitted by system:

> **Possible solutions:**  
> 1. Set `--mem-fraction-static` to a smaller value (e.g., `0.8` or `0.7`)  
> 2. Set `--cuda-graph-max-bs` to a smaller value (e.g., `16`)  
> 3. Disable `torch.compile` by not using `--enable-torch-compile`  
> 4. Disable CUDA graph entirely via `--disable-cuda-graph` *(not recommended—causes significant performance loss)*  
>   
> If the issue persists, open an issue at: https://github.com/sgl-project/sglang/issues/new/choose


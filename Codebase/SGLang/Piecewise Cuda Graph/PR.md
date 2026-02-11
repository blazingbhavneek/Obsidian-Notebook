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



\<INSERT HERE>

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
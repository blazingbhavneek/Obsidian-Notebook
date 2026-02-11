
## System Design


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

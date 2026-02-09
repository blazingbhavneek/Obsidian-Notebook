
- python/sglang/srt/model_executor/model_runner.py: 
	- Start from Model runner class, that loads and runs the models in sglang
		- can_run_piecewise_cuda_graph: that checks whether a model can run piewise cuda graph by checking
			- Input: self, needs self params lik self.pp size and backend info
			- Output: bool
			- enable_torch_compile
			- pp_size > 1
			- mooncake or deepep backends
		- init_piecewise_cuda_graphs: then inits the piecewise cuda graphs
			- input: self, from self it takes self.model
			- output: None, sets self.piecewise_cuda_graph_runner
			- Collects attention and MOE layers
			- check ensures ALL transformer layers have recognizable attention modules, standard transformers architecture
			- inits PiecewiseCudaGraphRunner as self.piecewise_cuda_graph_runner
		- forward_extend: calls the self.piecewise_cuda_graph_runner in forward extend
			- calls the self.piecewise_cuda_graph_runner.replay method if we piecewise cuda graph runner can run using can_run(forward_batch)
			- else give it to attention backend
- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
	- docstring """Run the model with cuda graph and torch.compile."""
		- cuda graphs are lower level than torch compile, and are replays of manually captured operations
	- freeze_gc function: (contextmanager)
		- Input: enable_cudagraph_gc bool, this allows to keep garbage collection on
		- Ouput: None
		- Python's GC can trigger at unpredictable times
		- If GC runs during capture, freed memory addresses get baked into the graph
		- When the graph replays, those addresses may point to invalid/different data
		- So it freezes python garbage collector
	- \_to_torch function:
		- Input: torch nn module, reverse (bool), number of tokens 
		- function Recursively travels all submodules of nn module and tells MultiPlatformOp layers to switch between their torch.compile-compatible and regular implementations
	- patch_model function: (context manager)
		- input: torch nn module, compiler string
		- Context manager that temporarily puts model in "compile mode"
		- MultiPlatformOp layers may have different implementations for eager vs compiled execution
		- Calls \_to_torch function to make the model enter compile mode?
	-  global_graph_memory_pool (global var) = None
		- Its getter and setter function are just below it
	- set_torch_compile_config:
		- input/ouput: None
		- torch.compile caches compiled versions. When input shapes change too often, it triggers recompilation. 
		- Increasing cache limits prevents crashes when capturing many different batch sizes/token counts
	- Class PiecewiseCudaGraphRunner:
		- method is_mamba_track_enabled:
			- This checks if Mamba state tracking is needed. Mamba models have recurrent state that needs to be tracked across tokens. 
			- The extra buffer stores intermediate Mamba states for cache management with radix trees
			- enable_mamba_extra_buffer + Radix cache enabled + no speculative decoding
		- \__init__
			- input: Model runner, so its parent class, thats why input self above
			- inits a self.compile_config using its parents params (refer to below for compilaton config)
			- self.capture_num_tokens = self.compile_config.get_capture_sizes()
			- self.capture_forward_mode = ForwardMode.EXTEND
			- self.capture_hidden_mode = CaptureHiddenMode.NULL
				- Whether to return hidden states or not
				- NULL means no, other options are NULL, LAST and FULL
			- Max batch size: self.max_bs = model_runner.req_to_token_pool.size
				- limits how many separate sequences can be processed
			- with torch.device(self.device):
				- self.input_ids = torch.zeros((self.max_num_tokens,), dtype=torch.int64)
					- This is a pre-allocated buffer of token IDs. 
					- CUDA graphs require fixed memory addresses, so we allocate the maximum size upfront and copy actual data into slices of this buffer
				- Similar buffer allocation for out_cache_loc, out_cache_loc_swa, mamba_track_indices, mamba_track_mask, mamba_track_seqlens, self.positions
			- self.tbo_plugin = TboCudaGraphRunnerPlugin()
				- TBO (Two Batch Overlap) is an optimization technique that overlaps the computation of two batches. 
				- While one batch is doing attention (memory-bound), another can do MLP computation (compute-bound).
				- This maximizes GPU utilization by hiding memory latency with computation.
				- The plugin prepares the necessary buffers and state for this overlap.
			- if self.is_multimodal
				- skipping this for now
			- self.attention_layers = self.model_runner.attention_layers
				- Getting list of attention layers and MoE layers (next line) from parent
			- set_graph_pool_id(get_global_graph_memory_pool())
				- CUDA graphs need contiguous memory allocations that persist across replays. 
				- graph_pool_handle() (one line above, where graph_pool_handle sets the graph pool) it returns a handle to a GPU memory pool that can be shared across multiple graph captures, reducing fragmentation and total memory usage.
			- with enable_piecewise_cuda_graph(): 
				- (context manager so all process can know what stage we are in right now, when this block is over)
				- with patch_model(): (see above function def for patch model)
					- calls install_torch_compiled (Entry point to backend module)
						- Inputs
							- patched_model
							- fullgraph=True
							- dynamic_arg_dims=None (which args of the model are dynamic and will change at runtime)
							- compile_config=self.compile_config (see below)
						- JIT Compilation for torch modules with SGLang backend
						- This is the main entry point for enabling torch.compile on a model.
						- It wraps the module's forward method with a "trampoline" that can switch between compiled and uncompiled execution based on the \_COMPILE_ENABLED flag
						- Remeber, earlier we just entered the compile mode in \_to_torch function, not actually compiled it
					- with set_compiled(True), enable_piecewise_cuda_graph_compile (setting contexts again)
						- set compile_range (for self.capture_num_tokens set from server args)
						- for each num_tokens in compile_range:
							- self.warmup_torch_compile(num_tokens=num_tokens)
								- We are doing this torch.compile is lazy - it only traces/compiles on first actual execution
								- We need to "prime" the compiled code path for each token count BEFORE capturing CUDA graphs, because the compilation happens during warmup. 
								- Without warmup, the capture phase would trigger compilation mid-capture, which can cause issues with CUDA graph recording
								- This is a method in this class and it creates a minimal ForwardBatch with dummy data to trigger torch.compile tracing for the given num_tokens. After this, the compiled code is cached.
					- set_graph_pool_id(get_global_graph_memory_pool())
						- After torch.compile warmup, the device module may have been reconfigured or the pool handle may have changed. 
						- This ensures we have a fresh, valid pool handle for the actual CUDA graph capture phase.
					- Sync GPUs and TP workers with this:
						- self.device_module.synchronize()
						- self.model_runner.tp_group.barrier()
					- CRITICAL: NOW CAPTURE CUDA GRAPH
		- \_cache_loc_dtype
			- return torch.int64 if not is_npu() else torch.int32
			- We used above, to set self.out_cache_loc
		- can_run (whether piecewise runner can run this forward batch), used in Model runner 
			- input: forward batch
			- output boolean
			- If logprobs are needed for tokens in the middle of a sequence (not just the last token), we can't use CUDA graphs because the logprob computation path is different and not captured
		- capture (IMPORTANT!!)
			- The last step of init, which actually captures the cuda graph
			- with freeze_gc(): (freezes the gc collection, take arguement for this from model runner) AND graph_capture() as graph_capture_context:
				- Docstring from graph_capture():
					- `graph_capture` is a context manager which should surround the code that is capturing the CUDA graph. Its main purpose is to ensure that the some operations will be run after the graph is captured, before the graph is replayed. It returns a `GraphCaptureContext` object which contains the necessary data for the graph capture. Currently, it only contains the stream that the graph capture is running on. This stream is set to the current CUDA stream when the context manager is entered and reset to the default stream when the context manager is exited. This is to ensure that the graph capture is running on a separate stream from the default stream, in order to explicitly distinguish the kernels to capture from other kernels possibly launched on background in the default stream.
				- with set_pcg_capture_stream(stream): (setting context)
					- capture_range = (getting capture range again just like init)
					- for i, num_tokens in enumerate(capture_range)
						- with set_compiled(True): (context management)
							- self.capture_one_batch_size(num_tokens)
								- This is another method in class just for batch size
		- capture_one_batch_size
			- input: num tokens
			- bs = 1  (but variable token sizes)
			- get all dummy input_ids, input_embeds, out_cache_loc, out_cache_loc_swa ... we defined earlier in init
			- if self.model_runner.server_args.enable_lora: lora_ids = [None] * bs
			- with torch.device(self.device):
				- forward_batch = ForwardBatch()
				- self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)
			- if lora_ids is not None:
				- self.model_runner.lora_manager.prepare_lora_batch(forward_batch)
			- def run_once(): (defined locally here)
				- with set_forward_context(
					- forward_batch, self.attention_layers, self.quant_config, 
					  self.moe_layers):
						  self.model_runner.model.forward(
							  forward_batch.input_ids,
							  forward_batch.positions,
							  forward_batch,
							  \**kwargs,)
						  return
			-  for _ in range(2):
				- (run twice for warmup at the first time and cuda graph capture at the second time)
				- (detail lies in sglang/python/sglang/srt/compilation/cuda_piecewise_backend.py)
				- run_once()
		- replay (main method thats called in model runner)
			- This is the method that will run the CUDA graph once its compiled
			- with enable_piecewise_cuda_graph():
				- with set_forward_context(
					- with set_compiled(True):
						- output = self.model_runner.model.forward
						- CRITICAL: It seems like we are using original models forward, but as you would see later, we have dynamically replaced models forward with CUDA graph wherever we could compile the code
- python/sglang/srt/compilation/compile.py
	- _COMPILE_ENABLED = contextvars.ContextVar("_COMPILE_ENABLED", default=False)
		- contextvars.ContextVar is used instead of a simple global variable because:
			- It's thread-safe and async-safe (each async task/thread gets its own copy)
			- It supports proper scoping with tokens for nested contexts
			- Essential when multiple requests might be processed concurrently
	- def set_compiled(enabled: bool = True):
		- token = _COMPILE_ENABLED.set(enabled)
	- class IntermediateTensors
		- For all pipeline stages except the last, we need to return the hidden states and residuals to be sent to the next stage. 
		- This data structure contains the hidden states and residuals for a request
	- def _normalize_dims(dims, ndim: int):
		- Converts negative dimension indices to positive ones
	- class \_MaybeIntermediateTensors:
		- """Duck-typed check to support your IntermediateTensors without importing."""
		- Duck typing avoids circular imports and allows this to work with any object
		- that has a `tensors` dict attribute, not just IntermediateTensors specifically
	- def \_mark_dynamic_on_value
		- torch.\_dynamo.mark_dynamic tells TorchDynamo that certain tensor dimensions can vary at runtime. 
		- Without this, Dynamo assumes shapes are static and would recompile for every new shape (very slow).  
		- For LLMs, the batch dimension (dim 0) is typically dynamic because Different numbers of requests in a batch and Different sequence lengths during prefill
		- This uses the \_MaybeIntermediateTensors 
		- prime syntax: torch.\_dynamo.mark_dynamic(val, \_normalize_dims(dims, val.ndim))
	- def \_infer_dynamic_arg_dims_from_annotations
		- This function inspects the type annotations of a forward() method to automatically determine which arguments have dynamic dimensions.
	- install_torch_compiled (IMPORTANT!!)
		- JIT Compilation for torch modules with SGLang backend
		- This is the main entry point for enabling torch.compile on a model, It wraps the module's forward method with a "trampoline" that can switch between compiled and uncompiled execution based on the _COMPILE_ENABLED flag
		- This was used piecwise cuda runner to replace forward code with compiled code
		- Input
			- torch nn module
			- Dynamic arg dimension (will use above defined function if not passed)
			- backend factory (Sglang backend used for compilation, CUDA or NPU)
			- compile_config: CompilationConfig, recieved from Model runner 
			- fullgraph (bool) = True 
			- graph_pool: Any
		- unbound_fwd = module.__class__.forward
			- Non compiled fwd method. 
			- We get the unbound method from the class (not instance) so we can properly wrap it while keeping access to both the original and compiled versions
		- original_code = unbound_fwd.\_\_code__
		- dyn_map = dynamic_arg_dims or \_infer_dynamic_arg_dims_from_annotations(unbound_fwd)
		- if backend_factory is None:
			- backend_factory = lambda gm, ex: SGLangBackend
		- compiled_codes: list[type(original_code)] = []
		- state = {"compiled": False, "compiled_callable": None}
		- bytecode_hook
			- Bytecode hook
			  ```
			    ### EXPLANATION: This registers the hook with TorchDynamo's bytecode transformation
			    ### system. Dynamo uses Python's frame evaluation hooks (PEP 523) to intercept
			    ### function calls. When it decides to compile a function:
			    ### 1. It traces the bytecode to build an FX graph
			    ### 2. Compiles the graph with the backend
			    ### 3. Generates new bytecode that uses the compiled version
			    ### 4. Calls all registered bytecode_hooks with (old_code, new_code)
			    ### 
			    ### The hook is global - it will be called for ALL Dynamo compilations,
			    ### which is why we filter to only care about our specific module.
			    torch._dynamo.convert_frame.register_bytecode_hook(bytecode_hook)
			
			    def _ensure_compiled(self, *args, **kwargs):
			        """Compile on first use (with flag ON)."""
			        ### This implements LAZY compilation - we don't compile until the model
			        ### is actually used. This is important because:
			        ### 1. We need real tensor shapes to properly mark dynamic dimensions
			        ### 2. Some model configurations might not be known until runtime
			        ### 3. Avoids compiling code paths that are never used
			        if state["compiled"]:
			            return
			        # Mark dynamic dims only when we are about to compile
			        sig = inspect.signature(unbound_fwd)
			        ba = sig.bind(self, *args, **kwargs)
			        ba.apply_defaults()
			        ### bind() matches args/kwargs to parameter names
			        ### apply_defaults() fills in default values for unspecified params
			        for name, dims in (dyn_map or {}).items():
			            if name in ba.arguments:
			                val = ba.arguments[name]
			                if val is not None:
			                    _mark_dynamic_on_value(val, dims)
			                    ### Mark tensors as having dynamic dimensions BEFORE tracing
			
			        # Avoid cross-instance cache reuse
			        ### Dynamo caches compiled code based on the code object. If we have
			        ### multiple model instances (e.g., draft and target models in speculative
			        ### decoding), we don't want them sharing cached compilations since they
			        ### might have different configurations. This clears any existing cache.
			        torch._dynamo.eval_frame.remove_from_cache(unbound_fwd.__code__)
			
			        bound = types.MethodType(unbound_fwd, self)
			        ### Create a bound method (method + instance) for torch.compile
			        compiled_callable = torch.compile(
			            bound, fullgraph=fullgraph, backend=backend_factory
			        )
			
			        # Trigger Dynamo so bytecode hook can capture
			        ### The first call to compiled_callable actually triggers compilation.
			        ### torch.compile is lazy - it doesn't compile until first execution.
			        ### This call traces the forward pass and invokes our bytecode_hook.
			        compiled_callable(*args, **kwargs)
			
			        state["compiled"] = True
			        state["compiled_callable"] = compiled_callable
			    
			    ## This is where it checks if compiled, the runs the compiled version else compiles it first
			    ## If we dont want compiled then return the original code (unbound forward)
			    ### CORRECT: The "trampoline" pattern provides a single entry point that can
			    ### dispatch to either compiled or uncompiled execution.
			    ### 
			    ### Why "trampoline"? In programming, a trampoline is a function that bounces
			    ### control to different implementations based on runtime conditions.
			    ### 
			    ### Benefits:
			    ### 1. Seamless switching between compiled/uncompiled (for debugging, profiling)
			    ### 2. Lazy compilation (compile only when _COMPILE_ENABLED is True)
			    ### 3. Same API regardless of compilation state
			    def trampoline(self, *args, **kwargs):
			        use_compiled = _COMPILE_ENABLED.get()
			        if use_compiled:
			            if not state["compiled"]:
			                _ensure_compiled(self, *args, **kwargs)
			
			            compiled_callable = state["compiled_callable"]
			            return compiled_callable(*args, **kwargs)
			        else:
			            # Explicitly run the original uncompiled forward
			            return unbound_fwd(self, *args, **kwargs)
			    
			    # CRITICAL! This is why we could run model.forward in model runner because its forward was replaced here
			    module.forward = types.MethodType(trampoline, module)
			    ### Replace the module's forward method with our trampoline.
			    ### types.MethodType binds the trampoline function to the module instance,
			    ### so `self` in trampoline refers to the module.
			    return module
```
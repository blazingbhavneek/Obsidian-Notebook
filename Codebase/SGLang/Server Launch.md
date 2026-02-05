
Serve commands start at:

python/sglang/cli/serve.py

the function is 

```python

def serve(args, extra_argv):

# Detects if user is asking for help and prints cli args
# Tries to detect if its a multimodal/diffusion model, if yes routes to its runtime
# Otherwise starts normal server

        else:
            # Logic for Standard Language Models
            from sglang.launch_server import run_server
            from sglang.srt.server_args import prepare_server_args

            # Add a dummy argument for the program name, expected by prepare_server_args
            # as it typically processes sys.argv
            server_args = prepare_server_args(extra_argv)

            run_server(server_args)
    
    finally:
	    kill_process_tree(os.getpid(), include_parent=False)

```

python/sglang/launch_server.py -> python/sglang/srt/entrypoints/http_server.py

Starts http server

Later: For encoder models, this is that launch server (python/sglang/srt/disaggregation/encode_server.py) 
```python


app = FastAPI()
encoder: Optional[MMEncoder] = None
send_sockets: List[zmq.Socket] = []


def launch_server(server_args: ServerArgs):
    global encoder                          # use/update the module-level 'encoder' variable
    ctx = mp.get_context("spawn")           # get a multiprocessing context that spawns new processes
    zmq_ctx = zmq.Context(10)               # create a ZeroMQ context with 10 I/O threads
    ipc_path_prefix = random_uuid()         # generate a unique prefix for IPC socket paths
    port_args = PortArgs.init_new(server_args)  # allocate/init network ports (e.g., NCCL port) based on server args
    if server_args.dist_init_addr:
        dist_init_method = f"tcp://{server_args.dist_init_addr}"  # use provided distributed init address
    else:
        dist_init_method = f"tcp://127.0.0.1:{port_args.nccl_port}"  # fallback to localhost + allocated NCCL port
    for rank in range(1, server_args.tp_size):   # spawn one worker process per tensor-parallel rank (starting at 1)
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"  # per-rank IPC path for scheduling messages
        send_sockets.append(
            get_zmq_socket(zmq_ctx, zmq.PUSH, schedule_path, bind=False)
        )  # create and store a PUSH ZeroMQ socket for sending schedules (not binding here)
        ctx.Process(
            target=launch_encoder,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        ).start()  # start the child process that runs launch_encoder for this rank (daemonized)
    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)  # instantiate the main encoder in master process
    uvicorn.run(app, host=server_args.host, port=server_args.port)      # run the ASGI app (uvicorn) to serve requests


```

python/sglang/srt/entrypoints/http_server.py

```python

async def lifespan(fast_api_app: FastAPI):
	# ...
    fast_api_app.state.openai_serving_chat = OpenAIServingChat(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    # ...
    
# Fast API
app = FastAPI(
    lifespan=lifespan,
    openapi_url=None if get_bool_env_var("DISABLE_OPENAPI_DOC") else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from sglang.srt.entrypoints.v1_loads import router as v1_loads_router

app.include_router(v1_loads_router)



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

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    
        # Launch subprocesses
    tokenizer_manager, template_manager, scheduler_infos, port_args = (
        _launch_subprocesses(
        
```

python/sglang/srt/entrypoints/engine.py

```python
def _launch_subprocesses(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable,
    run_scheduler_process_func: Callable,
    run_detokenizer_process_func: Callable,
    port_args: Optional[PortArgs] = None,
) -> Tuple[TokenizerManager, TemplateManager, Tuple[Dict], PortArgs]:
    """
    Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
    """
    # Launch scheduler processes
    scheduler_procs, scheduler_pipe_readers = _launch_scheduler_processes(
        server_args=server_args,
        port_args=port_args,
        run_scheduler_process_func=run_scheduler_process_func,
    )
    # ....
    # ....
    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process_func,
        args=(
            server_args,
            port_args,
        ),
    )
    detoken_proc.start()
    
    # Init tokenizer manager first, as the bootstrap server is initialized here
    if server_args.tokenizer_worker_num == 1:
        tokenizer_manager, template_manager = init_tokenizer_manager_func(
            server_args, port_args
        )
    else:
        # Launch multi-tokenizer router
        tokenizer_manager = MultiTokenizerRouter(server_args, port_args)
        template_manager = None
        
    # Wait for the model to finish loading
    scheduler_infos = _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs)

    # Get back some info from scheduler to tokenizer_manager
    tokenizer_manager.max_req_input_len = scheduler_infos[0]["max_req_input_len"]

    return tokenizer_manager, template_manager, scheduler_infos, port_args
    
```

Pydantic models for Server IO? Are at: python/sglang/srt/managers/io_struct.py

This files seems like engine + http server but this is for verl not main serving:
python/sglang/srt/entrypoints/http_server_engine.py 

Entrypoints will be handled by this: python/sglang/srt/entrypoints/openai/serving_chat.py

For now we will focus on launching server only, for now we seems to have covered the http server part, bu we are lacking the model loading etc

python/sglang/srt/entrypoints/engine.py
```python
class Engine(EngineBase):
    """
    The entry point to the inference engine.

    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
```

Here is the offline engine of sglang without the fastapi server it seems, have similar code to fastapi init but is written seperately

> Scheduler -> TP Worker -> Model runner (this loads configs, makes attn backend and weight ig )


[] AI this what is this doing:

```python

def _launch_scheduler_processes(
    server_args: ServerArgs,
    port_args: PortArgs,
    run_scheduler_process_func: Callable,
):
    scheduler_procs = []

    if server_args.dp_size == 1:
        # Launch tensor parallel scheduler processes
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        scheduler_pipe_readers = []

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
        )

        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

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
                            None,
                            writer,
                        ),
                    )
                    with memory_saver_adapter.configure_subprocess(), numa_utils.configure_subprocess(
                        server_args, gpu_id
                    ):
                        proc.start()

                scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)
    else:
        # Launch the data parallel controller
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

    return scheduler_procs, scheduler_pipe_readers


```

the run scheduler callable comes from 

python/sglang/srt/managers/scheduler.py

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
    # Generate the logger prefix
    prefix = ""
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
        dp_rank = int(os.environ["SGLANG_DP_RANK"])
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
        )
    if (
        numa_node := server_args.numa_node
    ) is not None and not envs.SGLANG_NUMA_BIND_V2.get():
        numa_bind_to_node(numa_node[gpu_id])

    # Set up tracing
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "Scheduler"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank)

    # Create a scheduler and run the event loop
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
        result_dict = {
            "status": "ready",
            "max_total_num_tokens": scheduler.max_total_num_tokens,
            "max_req_input_len": scheduler.max_req_input_len,
        }
        if server_args.remote_instance_weight_loader_use_transfer_engine():
            (
                remote_instance_transfer_engine_session_id,
                remote_instance_transfer_engine_weights_info_dict,
            ) = scheduler.get_remote_instance_transfer_engine_info()
            result_dict.update(
                {
                    "tp_rank": tp_rank,
                    "remote_instance_transfer_engine_session_id": remote_instance_transfer_engine_session_id,
                    "remote_instance_transfer_engine_weights_info_dict": remote_instance_transfer_engine_weights_info_dict,
                }
            )

        pipe_writer.send(result_dict)

        # Dispatch to the appropriate event loop based on the disaggregation mode
        disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
        if disaggregation_mode == DisaggregationMode.NULL:
            if scheduler.enable_pdmux:
                scheduler.event_loop_pdmux()
            elif server_args.pp_size > 1:
                scheduler.event_loop_pp()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap()
            else:
                scheduler.event_loop_normal()
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            if server_args.pp_size > 1:
                scheduler.event_loop_pp_disagg_prefill()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_prefill()
            else:
                scheduler.event_loop_normal_disagg_prefill()

        elif disaggregation_mode == DisaggregationMode.DECODE:
            if server_args.pp_size > 1:
                scheduler.event_loop_pp_disagg_decode()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_decode()
            else:
                scheduler.event_loop_normal_disagg_decode()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)

```


The scheduler, it needs it own deep dive, it handles model loading (via TP workers) + all peripheri (constrained decoding,  speculative decoding, disaggregation (of inference to two stages, prefill and decode), lora adapters, request aborting), for now I will focus on model loading, will deal with scheduler in seperate file

python/sglang/srt/managers/scheduler.py

```python
class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerUpdateWeightsMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,
    SchedulerRuntimeCheckerMixin,
    SchedulerPPMixin,
    SchedulerDPAttnMixin,
):
    """A scheduler that manages a tensor parallel GPU worker."""

    def init_tp_model_worker(self):
        from sglang.srt.managers.tp_worker import TpModelWorker

        self.tp_worker = TpModelWorker(
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=self.moe_ep_rank,
            pp_rank=self.pp_rank,
            dp_rank=self.dp_rank,
            nccl_port=self.nccl_port,
        )
        
```

python/sglang/srt/managers/tp_worker.py

On overview it seems like  TpWorker (Base Tp worker) is handling model weight loading but it actually just calls methods with same name in model runner instead

```python
class TpModelWorker(BaseTpWorker):
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        is_multi_layer_eagle: bool = False,
    ):
        # Parse args
        self.server_args = server_args
        self.tp_size = server_args.tp_size
        self.ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self.is_draft_worker = is_draft_worker
        self.is_multi_layer_eagle = is_multi_layer_eagle
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

        # MTP model runners
        self.model_runner_list = []

        self._init_model_config()
        self._init_model_runner()

        if is_multi_layer_eagle:
            self._init_multi_layer_eagle_model_runners()

        self._init_dllm_algorithm()

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
        self.device = self.model_runner.device

        # Init nccl groups
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = self.model_runner.max_running_requests
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_queued_requests = server_args.max_queued_requests
        assert (
            self.max_queued_requests is None or self.max_queued_requests >= 1
        ), "If configured, max_queued_requests must be at least 1 for any work to be scheduled."
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.model_runner.max_token_pool_size - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_size * self.pp_rank + tp_rank,
            self.world_group.cpu_group,
            src=self.world_group.ranks[0],
        )[0]
        set_random_seed(self.random_seed)

        self.enable_overlap = not server_args.disable_overlap_schedule
        self.enable_spec = server_args.speculative_algorithm is not None
        self.hicache_layer_transfer_counter = None

    def _init_model_config(self):
        from sglang.srt.configs.model_config import ModelConfig

        self.model_config = ModelConfig.from_server_args(
            self.server_args,
            model_path=(
                self.server_args.model_path
                if not self.is_draft_worker
                else self.server_args.speculative_draft_model_path
            ),
            model_revision=(
                self.server_args.revision
                if not self.is_draft_worker
                else self.server_args.speculative_draft_model_revision
            ),
            is_draft_model=self.is_draft_worker,
        )

    def _init_model_runner(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        self._model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.ep_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            draft_model_idx=0 if self.is_multi_layer_eagle else None,
        )

```

python/sglang/srt/model_executor/model_runner.py

```python
class ModelRunner(ModelRunnerKVCacheMixin):
    """ModelRunner runs the forward passes of the models."""

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


    def load_model(self):
        tic_total = time.perf_counter()
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                self.model_config.dtype = torch.float16
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()

        # Prepare the model config
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
            remote_instance_weight_loader_seed_instance_ip=self.server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=self.server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=self.server_args.remote_instance_weight_loader_send_weights_group_ports,
            remote_instance_weight_loader_backend=self.server_args.remote_instance_weight_loader_backend,
            remote_instance_weight_loader_transfer_engine=self.remote_instance_transfer_engine,
            modelopt_config=modelopt_config,
            rl_quant_profile=self.server_args.rl_quant_profile,
            draft_model_idx=self.draft_model_idx,
        )
        if self.device == "cpu":
            self.model_config = adjust_config_with_unaligned_cpu_tp(
                self.model_config, self.load_config, self.tp_size
            )

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

        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
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

        get_offloader().post_init()

        # Register model for layerwise NVTX profiling if enabled
        if self.server_args.enable_layerwise_nvtx_marker:
            self.pyt_hooks = PytHooks()
            self.pyt_hooks.register_hooks(self.model, module_prefix="model")

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

        # Parse other args
        self.sliding_window_size = None
        if hasattr(self.model, "get_attention_sliding_window_size"):
            self.sliding_window_size = self.model.get_attention_sliding_window_size()
        elif (
            self.model_config.is_hybrid_swa
            and self.model_config.sliding_window_size is not None
        ):
            # sliding window field in model config may have different meaning for different kinds of models (e.g., dllm), here we only consider the sliding window in SWA model
            self.sliding_window_size = self.model_config.sliding_window_size
        elif self.model_config.attention_chunk_size is not None:
            self.sliding_window_size = self.model_config.attention_chunk_size
            logger.info(
                f"Setting sliding_window_size to be attention_chunk_size: {self.sliding_window_size}"
            )

        self.dtype = self.model_config.dtype

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
        if self.server_args.debug_tensor_dump_output_folder is not None:
            register_forward_hook_for_model(
                self.model,
                self.server_args.debug_tensor_dump_output_folder,
                self.server_args.debug_tensor_dump_layers,
                self.tp_size,
                self.tp_rank,
                self.pp_rank,
            )

        # Pre-expand RoPE cache before CUDA Graph capture
        reserve_rope_cache_for_long_sequences(
            self.model,
            self.server_args,
            self.model_config,
            logger,
        )

        if self.server_args.elastic_ep_backend == "mooncake":
            # Mooncake does not support `monitored_barrier`
            dist.barrier(group=get_tp_group().cpu_group)
        else:
            # Handle the case where some ranks do not finish loading.
            try:
                dist.monitored_barrier(
                    group=get_tp_group().cpu_group,
                    timeout=datetime.timedelta(
                        seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S
                    ),
                    wait_all_ranks=True,
                )
            except RuntimeError:
                raise ValueError(
                    f"TP rank {self.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
                ) from None


```



`ModelRunner` is the core component that manages and executes model inference. It inherits from `ModelRunnerKVCacheMixin`, which provides KV cache management functionality.

---

## **Initialization Parameters**

Let me explain the key `__init__` parameters:

### **Parallelism Settings:**
- **`tp_rank/tp_size`**: Tensor Parallelism - splits model layers across GPUs horizontally
- **`pp_rank/pp_size`**: Pipeline Parallelism - splits model layers vertically (different layers on different GPUs)
- **`dp_rank`**: Data Parallelism - replicates the model across GPUs
- **`moe_ep_rank/moe_ep_size`**: Expert Parallelism for MoE (Mixture of Experts) models

### **Memory Management:**
- **`mem_fraction_static`**: Fraction of GPU memory reserved for static allocations
- **`req_to_token_pool`**: Pool mapping requests to token IDs
- **`token_to_kv_pool_allocator`**: Manages KV cache memory allocation

### **Configuration:**
- **`model_config`**: Model architecture details (layers, hidden size, etc.)
- **`server_args`**: Runtime arguments (dtype, quantization, etc.)
- **`is_draft_worker`**: Whether this is for speculative decoding (draft model)

---

## **The `load_model()` Method - Step by Step**

### **1. Memory Tracking & Thread Setup**
```python
before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
torch.set_num_threads(1)
```
- Records memory before loading to calculate weight size
- Sets threads to 1 to avoid conflicts during parallel loading

### **2. Device Capability Checks**
```python
if torch.cuda.get_device_capability()[0] < 8:
    self.server_args.dtype = "float16"
```
- **SM80** = NVIDIA Ampere (A100, RTX 30xx)
- Older GPUs don't support bfloat16, so it falls back to float16
- **SM75** (Turing) is the minimum supported

### **3. Configure Load Strategy**
```python
self.load_config = LoadConfig(
    load_format=self.server_args.load_format,
    download_dir=self.server_args.download_dir,
    ...
)
```

**Load formats include:**
- **`auto`**: Automatically detect format
- **`pt`**: PyTorch checkpoint
- **`safetensors`**: Hugging Face format (safer, faster)
- **`dummy`**: Random weights for testing
- **`REMOTE_INSTANCE`**: Load weights from another instance (distributed serving)

### **4. Remote Instance Loading (Optional)**
```python
if self.server_args.load_format == LoadFormat.REMOTE_INSTANCE:
```
This allows multiple GPU instances to share weights over network using NCCL (NVIDIA's communication library):
- One "seed instance" has the weights
- Other instances download weights via network
- Saves memory and disk space in multi-node deployments

### **5. Actual Model Loading**
```python
monkey_patch_vllm_parallel_state()
```
**Why monkey patch?** SGLang uses some vLLM components but needs to override parallel state management for its own parallelism strategy.

```python
with self.memory_saver_adapter.region(
    GPU_MEMORY_TYPE_WEIGHTS,
    enable_cpu_backup=enable_cpu_backup,
):
    self.loader = get_model_loader(load_config=self.load_config, ...)
    self.model = self.loader.load_model(...)
```

**Memory saver adapter**: Context manager that:
- Tracks memory usage by region (weights vs KV cache vs activation)
- Optionally backs up weights to CPU if `enable_cpu_backup=True`
- Enables weight offloading for memory-constrained scenarios

**Loader types** (from `get_model_loader`):
- `DefaultModelLoader`: Standard loading from disk
- `RemoteInstanceModelLoader`: Network-based loading
- `DummyModelLoader`: Random weights for testing

### **6. FP8 KV Cache Configuration**
```python
if self.server_args.kv_cache_dtype == "fp8_e4m3":
```
- **FP8**: 8-bit floating point for KV cache (saves memory)
- **e4m3**: 4 exponent bits, 3 mantissa bits
- Requires scaling factors to maintain accuracy
- Loads from `quantization_param_path` if provided

### **7. Sliding Window Attention**
```python
self.sliding_window_size = self.model.get_attention_sliding_window_size()
```
- Used by models like Mistral that only attend to recent tokens
- Reduces memory usage for long sequences
- Falls back to `attention_chunk_size` if not defined

### **8. Memory Usage Reporting**
```python
self.weight_load_mem_usage = before_avail_memory - after_avail_memory
```
Calculates actual GPU memory consumed by weights.

### **9. RoPE Cache Pre-expansion**
```python
reserve_rope_cache_for_long_sequences(self.model, ...)
```
**RoPE (Rotary Position Embeddings)**: 
- Requires precomputed sin/cos values
- Pre-allocates before CUDA Graph capture (graphs require static memory)
- Avoids memory allocation during inference

### **10. Distributed Synchronization**
```python
dist.monitored_barrier(group=get_tp_group().cpu_group, ...)
```
- Ensures all TP ranks finish loading before proceeding
- **Timeout**: 1800s (30 min) for slow nodes
- Catches OOM errors or hardware failures early

---

## **Key Concepts Summary**

| Concept | Purpose |
|---------|---------|
| **Tensor Parallelism** | Split layers horizontally across GPUs |
| **Memory Saver Adapter** | Track and manage different memory regions |
| **Remote Instance Loading** | Share weights across network to save memory |
| **Monkey Patching** | Override vLLM components for SGLang's needs |
| **CUDA Graph** | Pre-record GPU operations for faster inference |
| **KV Cache** | Store attention keys/values to avoid recomputation |
| **Weight Offloading** | Move weights to CPU when GPU memory is tight |

---

## **Common Loading Patterns**

**Single GPU:**
```python
tp_size=1, pp_size=1, moe_ep_size=1
```

**Multi-GPU (4x A100):**
```python
tp_size=4  # Split model across 4 GPUs
```

**Multi-Node with Remote Loading:**
```python
load_format="REMOTE_INSTANCE"
remote_instance_weight_loader_seed_instance_ip="192.168.1.100"
```

This design allows SGLang to efficiently serve large models across diverse hardware configurations while managing GPU memory carefully. The complexity handles distributed loading, memory pressure, different hardware capabilities, and performance optimization.


python/sglang/srt/model_loader/loader.py

def get_model_loader -> class DefaultModelLoader


`DefaultModelLoader` is the standard model loader in SGLang that loads model weights from disk into the initialized model structure. It handles downloading, file format detection, and weight distribution to the appropriate model parameters.

---

## **Key Components**

### **1. Source Dataclass**
```python
@dataclasses.dataclass
class Source:
    model_or_path: str          # Model ID or local path
    revision: Optional[str]      # Git revision/tag
    prefix: str = ""            # Prefix for weight names
    fall_back_to_pt: bool       # Allow .pt files if .safetensors unavailable
    model_config: ModelConfig   # Model configuration
```
Represents a weight source (primary model or secondary weights like adapters).

---

## **Main Loading Pipeline**

### **Step 1: `_prepare_weights()` - File Discovery**
**What it does:**
- Downloads model from HuggingFace/ModelScope if not local
- Detects file format (safetensors, .pt, mistral format, etc.)
- Finds all weight files matching the format
- Filters out duplicate/unnecessary files
- Optionally verifies checksums

**Supported formats:**
- `AUTO`: Try safetensors first, fallback to .pt
- `SAFETENSORS` / `FASTSAFETENSORS`: Use .safetensors only
- `PT`: PyTorch .pt files
- `MISTRAL`: Mistral's consolidated format
- `NPCACHE`: NumPy cache format

**Output:** `(hf_folder, hf_weights_files, use_safetensors)`

---

### **Step 2: `_get_weights_iterator()` - Create Weight Stream**
**What it does:**
- Creates an iterator that yields `(weight_name, tensor)` pairs
- Chooses iterator based on format and config:
  - `safetensors_weights_iterator`: Standard safetensors loading
  - `fastsafetensors_weights_iterator`: Optimized safetensors loading
  - `multi_thread_safetensors_weights_iterator`: Parallel loading (8 threads default)
  - `pt_weights_iterator`: PyTorch checkpoint loading
  - `np_cache_weights_iterator`: NumPy cache format

**Key features:**
- Applies prefix to weight names (for multi-source models)
- Handles MTP (Multi-Token Prediction) draft models by filtering layers
- Supports multi-threaded loading for faster initialization

---

### **Step 3: `_get_all_weights()` - Aggregate All Sources**
**What it does:**
- Loads primary model weights
- Loads secondary weights (e.g., LoRA adapters, vision encoders)
- Yields all `(name, tensor)` pairs in sequence

```python
# Primary weights (main model)
yield from self._get_weights_iterator(primary_weights)

# Secondary weights (from model.secondary_weights attribute)
for source in secondary_weights:
    yield from self._get_weights_iterator(source)
```

---

### **Step 4: `load_model()` - Main Entry Point**
**What it does:**

**Standard path:**
```python
1. Set dtype context (fp16/bf16/fp32)
2. Set device context (cuda:0, cuda:1, etc.)
3. _initialize_model() → Create empty model structure
4. _get_all_weights() → Get weight iterator
5. model.load_weights(weights) → Fill parameters with actual values
6. Post-process quantization methods
7. Return model.eval()
```

**ModelOpt quantization path:**
```python
1. Load full HuggingFace model with accelerate
2. Use device_map="auto" for multi-GPU distribution
3. Apply quantization (handled separately)
4. Return model.eval()
```

---

### **Step 5: `load_weights_and_postprocess()` - Weight Assignment**
**What it does:**
```python
# 1. Load all weights into model
model.load_weights(weights)
# This calls each parameter's weight_loader() method
# which handles TP/PP sharding

# 2. Post-process quantization
for module in model.named_modules():
    if module has quant_method:
        # Temporarily move to target device if CPU offloaded
        quant_method.process_weights_after_loading(module)
        # Examples: repack INT4, compute scales, etc.
```

---

## **Key Features**

| Feature | Description |
|---------|-------------|
| **Multi-format support** | safetensors, PyTorch .pt, Mistral format, NumPy cache |
| **Multi-threaded loading** | 8 threads by default with `enable_multithread_load=True` |
| **Checksum verification** | Optional verification via `model_checksum` |
| **ModelScope support** | Download from ModelScope hub if `SGLANG_USE_MODELSCOPE=True` |
| **Duplicate filtering** | Removes redundant consolidated files |
| **MTP draft models** | Loads specific draft model layers |
| **Secondary weights** | Supports multi-source models (adapters, vision encoders) |
| **Memory-mapped loading** | Efficient loading without copying to RAM |
| **Quantization post-processing** | Repacks weights after loading for quant methods |

---

## **Weight Loading Flow Diagram**

```
DefaultModelLoader.load_model()
    ↓
_initialize_model() → Empty model structure created
    ↓                 (Parameters allocated but random)
_prepare_weights() → Find weight files on disk/HF
    ↓
_get_weights_iterator() → Stream (name, tensor) pairs
    ↓
_get_all_weights() → Primary + secondary sources
    ↓
model.load_weights(weights)
    ↓
For each (name, tensor):
    ├─ Find matching parameter in model
    ├─ Call param.weight_loader(tensor)
    │   ↓
    │   ├─ Extract TP shard: tensor[start:end]
    │   ├─ Move to GPU: shard.to(device)
    │   └─ Copy into param: param.data.copy_(shard)
    └─ Next weight
    ↓
load_weights_and_postprocess()
    ↓
For each module with quant_method:
    └─ quant_method.process_weights_after_loading()
        (Repack, compute scales, etc.)
    ↓
Return model.eval()
```

---

## **Critical Details**

1. **Weights are streamed, not loaded all at once**: Reduces CPU RAM usage
2. **Each TP rank only loads its slice**: Happens in `param.weight_loader()`
3. **PP ranks skip layers they don't own**: Filtered during `_initialize_model()`
4. **Multi-threaded loading**: Multiple files loaded in parallel, significant speedup
5. **Post-quantization processing**: Some methods need full weights on device before repacking
6. **CPU offloading support**: Temporarily moves weights to GPU for processing, then back to CPU

---

## **Example Usage**

```python
loader = DefaultModelLoader(load_config)

# Download only (doesn't load into model)
loader.download_model(model_config)

# Full load
model = loader.load_model(
    model_config=model_config,
    device_config=DeviceConfig("cuda", 0)
)
```

This loader is the workhorse that bridges the gap between checkpoint files on disk and the distributed model structure with proper TP/PP sharding!

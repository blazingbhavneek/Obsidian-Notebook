## Motivation

Add Support for [ibm-granite/granite-4.0-h-micro](https://huggingface.co/ibm-granite/granite-4.0-h-micro) and its [Dense variant](https://huggingface.co/ibm-granite/granite-4.0-micro)

If I try to run the model, i get the message:

```bash
❯ python3 -m sglang.bench_one_batch --correct --model ibm-granite/granite-4.0-h-micro

...
...

[rank0]:   File "/run/media/blazingbhavneek/Common/Code/sglang/python/sglang/srt/model_loader/utils.py", line 74, in resolve_transformers_arch
[rank0]:     raise ValueError(
[rank0]: ValueError: GraniteMoeHybridForCausalLM has no SGlang implementation and the Transformers implementation is not compatible with SGLang.

```

This PR ports IBM Granite model from its vllm implementation

## Benchmarks

Command for SGLang Server:
`python -m sglang.launch_server --model-path ibm-granite/granite-4.0-h-micro --port 30000`

Command for vLLM Server: 
`python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model ibm-granite/granite-4.0-h-micro --disable-log-requests --port 21000`


| Benchmark | Model               | Engine | Accuracy |
| --------- | ------------------- | ------ | -------- |
| MMLU      | granite-4.0-h-micro | SGLang | 0.636    |
|           |                     | vLLM   | 0.636    |
|           | granite-4.0-micro   | SGLang | 0.618    |
|           |                     | vLLM   | 0.625    |
| GSM8K     | granite-4.0-h-micro | SGLang | 0.805    |
|           |                     | vLLM   | 0.790    |
|           | granite-4.0-micro   | SGLang | 0.800    |
|           |                     | vLLM   | 0.815    |

### MMLU

#### granite-4.0-h-micro

##### vLLM

```bash
❯ python3 bench_other.py --nsub 10 --backend vllm

  0%|                                                                                                                                  | 0/10 [00:00<?, ?it/s]Average accuracy 0.510, latency 69.94, #q: 100 - abstract_algebra
 10%|████████████▏                                                                                                             | 1/10 [01:09<10:29, 69.96s/it]Average accuracy 0.578, latency 7.04, #q: 135 - anatomy
 20%|████████████████████████▍                                                                                                 | 2/10 [01:17<04:23, 32.96s/it]Average accuracy 0.770, latency 13.12, #q: 152 - astronomy
 30%|████████████████████████████████████▌                                                                                     | 3/10 [01:30<02:47, 23.91s/it]Average accuracy 0.620, latency 8.53, #q: 100 - business_ethics
 40%|████████████████████████████████████████████████▊                                                                         | 4/10 [01:38<01:47, 17.85s/it]Average accuracy 0.702, latency 15.63, #q: 265 - clinical_knowledge
 50%|█████████████████████████████████████████████████████████████                                                             | 5/10 [01:54<01:25, 17.06s/it]Average accuracy 0.757, latency 10.11, #q: 144 - college_biology
 60%|█████████████████████████████████████████████████████████████████████████▏                                                | 6/10 [02:04<00:58, 14.70s/it]Average accuracy 0.450, latency 8.17, #q: 100 - college_chemistry
 70%|█████████████████████████████████████████████████████████████████████████████████████▍                                    | 7/10 [02:12<00:37, 12.58s/it]Average accuracy 0.610, latency 12.45, #q: 100 - college_computer_science
 80%|█████████████████████████████████████████████████████████████████████████████████████████████████▌                        | 8/10 [02:25<00:25, 12.54s/it]Average accuracy 0.530, latency 8.92, #q: 100 - college_mathematics
 90%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▊            | 9/10 [02:34<00:11, 11.42s/it]Average accuracy 0.630, latency 12.90, #q: 173 - college_medicine
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [02:47<00:00, 16.71s/it]
Total latency: 166.797
Average accuracy: 0.636

```


##### SGLang

```bash
❯ python3 bench_sglang.py --nsub 10
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1369/1369 [01:07<00:00, 20.26it/s]
subject: abstract_algebra, #q:100, acc: 0.480
subject: anatomy, #q:135, acc: 0.593
subject: astronomy, #q:152, acc: 0.757
subject: business_ethics, #q:100, acc: 0.620
subject: clinical_knowledge, #q:265, acc: 0.717
subject: college_biology, #q:144, acc: 0.729
subject: college_chemistry, #q:100, acc: 0.480
subject: college_computer_science, #q:100, acc: 0.570
subject: college_mathematics, #q:100, acc: 0.520
subject: college_medicine, #q:173, acc: 0.653
Total latency: 67.571
Average accuracy: 0.636

```


#### granite-4.0-micro

##### vLLM

```bash
❯ python3 bench_other.py --nsub 10 --backend vllm
  0%|                                                                                                                                  | 0/10 [00:00<?, ?it/s]Average accuracy 0.380, latency 1.23, #q: 100 - abstract_algebra
 10%|████████████▏                                                                                                             | 1/10 [00:01<00:11,  1.24s/it]Average accuracy 0.541, latency 1.45, #q: 135 - anatomy
 20%|████████████████████████▍                                                                                                 | 2/10 [00:02<00:11,  1.38s/it]Average accuracy 0.770, latency 2.03, #q: 152 - astronomy
 30%|████████████████████████████████████▌                                                                                     | 3/10 [00:04<00:11,  1.69s/it]Average accuracy 0.680, latency 1.41, #q: 100 - business_ethics
 40%|████████████████████████████████████████████████▊                                                                         | 4/10 [00:06<00:09,  1.59s/it]Average accuracy 0.672, latency 2.70, #q: 265 - clinical_knowledge
 50%|█████████████████████████████████████████████████████████████                                                             | 5/10 [00:08<00:09,  2.00s/it]Average accuracy 0.778, latency 2.06, #q: 144 - college_biology
 60%|█████████████████████████████████████████████████████████████████████████▏                                                | 6/10 [00:11<00:08,  2.03s/it]Average accuracy 0.430, latency 1.40, #q: 100 - college_chemistry
 70%|█████████████████████████████████████████████████████████████████████████████████████▍                                    | 7/10 [00:12<00:05,  1.83s/it]Average accuracy 0.630, latency 1.80, #q: 100 - college_computer_science
 80%|█████████████████████████████████████████████████████████████████████████████████████████████████▌                        | 8/10 [00:14<00:03,  1.83s/it]Average accuracy 0.460, latency 1.39, #q: 100 - college_mathematics
 90%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▊            | 9/10 [00:15<00:01,  1.69s/it]Average accuracy 0.676, latency 2.73, #q: 173 - college_medicine
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:18<00:00,  1.84s/it]
Total latency: 18.226
Average accuracy: 0.625

```

##### SGLang

```bash
❯ python3 bench_sglang.py --nsub 10
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1369/1369 [00:19<00:00, 71.07it/s]
subject: abstract_algebra, #q:100, acc: 0.340
subject: anatomy, #q:135, acc: 0.541
subject: astronomy, #q:152, acc: 0.757
subject: business_ethics, #q:100, acc: 0.680
subject: clinical_knowledge, #q:265, acc: 0.668
subject: college_biology, #q:144, acc: 0.771
subject: college_chemistry, #q:100, acc: 0.430
subject: college_computer_science, #q:100, acc: 0.610
subject: college_mathematics, #q:100, acc: 0.470
subject: college_medicine, #q:173, acc: 0.676
Total latency: 19.266
Average accuracy: 0.618

```
### GSM8K

#### granite-4.0-h-micro

##### vLLM

```bash
❯ python3 bench_other.py --num-questions 200 --backend vllm
Downloading from https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl to /tmp/test.jsonl
/tmp/test.jsonl: 732kB [00:00, 2.65MB/s]                                                                                                                      
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:46<00:00,  4.33it/s]
Accuracy: 0.790
Invalid: 0.000
Latency: 46.257 s

```


##### SGLang

```bash
❯ python3 bench_sglang.py --num-questions 200
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:42<00:00,  4.65it/s]
Accuracy: 0.805
Invalid: 0.000
Latency: 43.329 s
Output throughput: 431.677 token/s

```

#### granite-4.0-h-micro

##### vllm

```bash
❯ python3 bench_other.py --num-questions 200 --backend vllm
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:18<00:00, 10.76it/s]
Accuracy: 0.815
Invalid: 0.000
Latency: 18.629 s

```

##### SGLang

```bash
❯ python3 bench_sglang.py --num-questions 200
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:17<00:00, 11.49it/s]
Accuracy: 0.800

```




### Benchmark

```bash
❯ python -m sglang.bench_one_batch --model-path ibm-granite/granite-4.0-h-micro --batch 4
<frozen importlib._bootstrap_external>:1241: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
<frozen importlib._bootstrap_external>:1241: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
[2026-02-01 00:05:01 TP0] Init torch distributed begin.
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[2026-02-01 00:05:01 TP0] Init torch distributed ends. elapsed=0.12 s, mem usage=0.02 GB
[2026-02-01 00:05:01 TP0] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
[2026-02-01 00:05:01 TP0] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
[2026-02-01 00:05:01 TP0] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/home/blazingbhavneek/miniconda3/envs/sglang/lib/python3.11/site-packages/transformers/__init__.py)
[2026-02-01 00:05:02 TP0] Load weight begin. avail mem=15.24 GB
[2026-02-01 00:05:02 TP0] Beginning to load weights
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:00<00:00,  1.28it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.84it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.72it/s]

[2026-02-01 00:05:03 TP0] Loading weights took 1.21 seconds
[2026-02-01 00:05:03 TP0] Load weight end. elapsed=1.25 s, type=GraniteMoeHybridForCausalLM, dtype=torch.bfloat16, avail mem=9.26 GB, mem usage=5.98 GB.
[2026-02-01 00:05:03 TP0] Using KV cache dtype: torch.bfloat16
[2026-02-01 00:05:03 TP0] Mamba Cache is allocated. max_mamba_cache_size: 38, conv_state size: 0.03GB, ssm_state size: 2.74GB 
[2026-02-01 00:05:03 TP0] KV Cache is allocated. #tokens: 399740, K size: 1.52 GB, V size: 1.52 GB
[2026-02-01 00:05:03 TP0] Memory pool end. avail mem=3.42 GB
[2026-02-01 00:05:03 TP0] Init attention backend begin.
[2026-02-01 00:05:03 TP0] Init attention backend end. elapsed=0.02 s
[2026-02-01 00:05:03 TP0] Capture cuda graph begin. This can take up to several minutes. avail mem=2.98 GB
[2026-02-01 00:05:03 TP0] Capture cuda graph bs [1, 2, 4, 8]
Capturing batches (bs=1 avail_mem=2.91 GB): 100%|███████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  5.37it/s]
[2026-02-01 00:05:04 TP0] Capture cuda graph end. Time elapsed: 1.24 s. mem usage=0.09 GB. avail mem=2.89 GB.
max_total_num_tokens=399740
Warmup ...
[2026-02-01 00:05:04 TP0] Reset HybridReqToTokenPool
Prefill. latency: 1.02554 s, throughput:   3994.01 token/s
Decode 0. Batch size: 4, latency: 0.19385 s, throughput:     20.63 token/s
Decode 1. Batch size: 4, latency: 0.01851 s, throughput:    216.12 token/s
Decode 2. Batch size: 4, latency: 0.01837 s, throughput:    217.70 token/s
Decode 3. Batch size: 4, latency: 0.01846 s, throughput:    216.68 token/s
Decode 4. Batch size: 4, latency: 0.01839 s, throughput:    217.48 token/s
Decode.  median latency: 0.01837 s, median throughput:    217.70 token/s
Total. latency:  1.477 s, throughput:   2817.07 token/s
Benchmark ...
[2026-02-01 00:05:06 TP0] Reset HybridReqToTokenPool
Prefill. latency: 0.72352 s, throughput:   5661.19 token/s
Decode 0. Batch size: 4, latency: 0.01841 s, throughput:    217.27 token/s
Decode 1. Batch size: 4, latency: 0.01824 s, throughput:    219.30 token/s
Decode 2. Batch size: 4, latency: 0.01823 s, throughput:    219.43 token/s
Decode 3. Batch size: 4, latency: 0.01800 s, throughput:    222.25 token/s
Decode 4. Batch size: 4, latency: 0.01815 s, throughput:    220.42 token/s
Decode.  median latency: 0.01821 s, median throughput:    219.63 token/s
Total. latency:  0.996 s, throughput:   4177.08 token/s
[rank0]:[W201 00:05:08.318920119 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

```



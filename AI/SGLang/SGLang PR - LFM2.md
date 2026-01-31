
# Support for LFM2 Dense

Command for SGLang Server:
`python -m sglang.launch_server --model-path LiquidAI/LFM2-700M --port 30000 --attention-backend triton`

Command for vLLM Server: 
`python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model LiquidAI/LFM2-700M --disable-log-requests --port 21000`


| Benchmark | Model     | Engine | Accuracy |
| --------- | --------- | ------ | -------- |
| MMLU      | LFM2-700M | SGLang | 0.491    |
|           |           | vLLM   | 0.495    |
|           | LFM2-2.6B | SGLang | 0.641    |
|           |           | vLLM   | 0.641    |
| GSM8K     | LFM2-700M | SGLang | 0.425    |
|           |           | vLLM   | 0.430    |
|           | LFM2-2.6B | SGLang | 0.820    |
|           |           | vLLM   | 0.790    |

### MMLU:

#### LFM2-700M
##### SGLang
```bash
❯ python3 bench_sglang.py --nsub 10
100%|████████████████████████████████████| 1369/1369 [00:07<00:00, 187.57it/s]
subject: abstract_algebra, #q:100, acc: 0.340
subject: anatomy, #q:135, acc: 0.444
subject: astronomy, #q:152, acc: 0.638
subject: business_ethics, #q:100, acc: 0.520
subject: clinical_knowledge, #q:265, acc: 0.570
subject: college_biology, #q:144, acc: 0.590
subject: college_chemistry, #q:100, acc: 0.330
subject: college_computer_science, #q:100, acc: 0.370
subject: college_mathematics, #q:100, acc: 0.310
subject: college_medicine, #q:173, acc: 0.532
Total latency: 7.303
Average accuracy: 0.491
```

##### vLLM
```bash
❯ python3 bench_other.py --nsub 10 --backend vllm
  0%|                                                  | 0/10 [00:00<?, ?it/s]Average accuracy 0.340, latency 1.60, #q: 100 - abstract_algebra
 10%|████▏                                     | 1/10 [00:01<00:14,  1.60s/it]Average accuracy 0.437, latency 1.46, #q: 135 - anatomy
 20%|████████▍                                 | 2/10 [00:03<00:12,  1.53s/it]Average accuracy 0.638, latency 2.83, #q: 152 - astronomy
 30%|████████████▌                             | 3/10 [00:05<00:14,  2.13s/it]Average accuracy 0.520, latency 1.87, #q: 100 - business_ethics
 40%|████████████████▊                         | 4/10 [00:07<00:12,  2.03s/it]Average accuracy 0.585, latency 3.37, #q: 265 - clinical_knowledge
 50%|█████████████████████                     | 5/10 [00:11<00:12,  2.52s/it]Average accuracy 0.597, latency 2.14, #q: 144 - college_biology
 60%|█████████████████████████▏                | 6/10 [00:13<00:09,  2.39s/it]Average accuracy 0.340, latency 1.79, #q: 100 - college_chemistry
 70%|█████████████████████████████▍            | 7/10 [00:15<00:06,  2.20s/it]Average accuracy 0.400, latency 2.76, #q: 100 - college_computer_science
 80%|█████████████████████████████████▌        | 8/10 [00:17<00:04,  2.38s/it]Average accuracy 0.290, latency 1.93, #q: 100 - college_mathematics
 90%|█████████████████████████████████████▊    | 9/10 [00:19<00:02,  2.24s/it]Average accuracy 0.526, latency 2.77, #q: 173 - college_medicine
100%|█████████████████████████████████████████| 10/10 [00:22<00:00,  2.26s/it]
Total latency: 22.521
Average accuracy: 0.495
```

#### LFM2-2.6B

##### SGLang
```bash
❯ python3 bench_sglang.py --nsub 10
100%|█████████████████████████████████████| 1369/1369 [00:23<00:00, 58.48it/s]
subject: abstract_algebra, #q:100, acc: 0.400
subject: anatomy, #q:135, acc: 0.615
subject: astronomy, #q:152, acc: 0.770
subject: business_ethics, #q:100, acc: 0.680
subject: clinical_knowledge, #q:265, acc: 0.713
subject: college_biology, #q:144, acc: 0.785
subject: college_chemistry, #q:100, acc: 0.510
subject: college_computer_science, #q:100, acc: 0.540
subject: college_mathematics, #q:100, acc: 0.460
subject: college_medicine, #q:173, acc: 0.676
Total latency: 23.416
Average accuracy: 0.641
```

##### vLLM
```bash
❯ python3 bench_other.py --nsub 10 --backend vllm
  0%|                                                  | 0/10 [00:00<?, ?it/s]Average accuracy 0.400, latency 4.71, #q: 100 - abstract_algebra
 10%|████▏                                     | 1/10 [00:04<00:42,  4.72s/it]Average accuracy 0.593, latency 5.37, #q: 135 - anatomy
 20%|████████▍                                 | 2/10 [00:10<00:40,  5.10s/it]Average accuracy 0.763, latency 10.07, #q: 152 - astronomy
 30%|████████████▌                             | 3/10 [00:20<00:51,  7.38s/it]Average accuracy 0.680, latency 6.63, #q: 100 - business_ethics
 40%|████████████████▊                         | 4/10 [00:26<00:42,  7.09s/it]Average accuracy 0.721, latency 12.00, #q: 265 - clinical_knowledge
 50%|█████████████████████                     | 5/10 [00:38<00:44,  8.86s/it]Average accuracy 0.771, latency 7.64, #q: 144 - college_biology
 60%|█████████████████████████▏                | 6/10 [00:46<00:33,  8.45s/it]Average accuracy 0.510, latency 6.36, #q: 100 - college_chemistry
 70%|█████████████████████████████▍            | 7/10 [00:52<00:23,  7.77s/it]Average accuracy 0.560, latency 9.84, #q: 100 - college_computer_science
 80%|█████████████████████████████████▌        | 8/10 [01:02<00:16,  8.43s/it]Average accuracy 0.480, latency 6.90, #q: 100 - college_mathematics
 90%|█████████████████████████████████████▊    | 9/10 [01:09<00:07,  7.95s/it]Average accuracy 0.676, latency 9.90, #q: 173 - college_medicine
100%|█████████████████████████████████████████| 10/10 [01:19<00:00,  7.95s/it]
Total latency: 79.415
Average accuracy: 0.641
```

### GSM8K

#### LFM2-700M

##### SGLang

```bash
python3 bench_sglang.py --num-questions 200
100%|███████████████████████████████████████| 200/200 [00:07<00:00, 25.62it/s]
Accuracy: 0.425
Invalid: 0.010
Latency: 7.844 s
Output throughput: 2107.954 token/s
```

##### vLLM

```bash
python3 bench_other.py --num-questions 200 --backend vllm
100%|███████████████████████████████████████| 200/200 [00:08<00:00, 23.34it/s]
Accuracy: 0.430
Invalid: 0.010
Latency: 8.609 s
```

#### LFM2-2.6B
##### SGLang
```bash
python3 bench_sglang.py --num-questions 200
100%|███████████████████████████████████████| 200/200 [00:12<00:00, 16.14it/s]
Accuracy: 0.820
Invalid: 0.000
Latency: 12.497 s
Output throughput: 1445.970 token/s
```

##### vLLM
```bash
python3 bench_other.py --num-questions 200 --backend vllm
100%|███████████████████████████████████████| 200/200 [00:25<00:00,  7.98it/s]
Accuracy: 0.790
Invalid: 0.000
Latency: 25.114 s
```



## Hellaswag (Diff too much)

### LFM2-2.6B

#### vLLM
```bash
python3 bench_other.py --num-questions 200 --backend vllm
100%|███████████████████████████████████████| 200/200 [01:35<00:00,  2.10it/s]
Latency: 95.380
Accuracy: 0.265
```

#### SGLang
```bash
python3 bench_sglang.py --num-questions 200
100%|███████████████████████████████████████| 200/200 [00:06<00:00, 31.49it/s]
Latency: 6.481
Accuracy: 0.665
```


### Benchmark

```bash
python -m sglang.bench_one_batch --model-path LiquidAI/LFM2-700M --batch 4 -
-input-len 2048 --output-len 1024 --attention-backend triton
WARNING:sglang.srt.server_args:Disabling overlap schedule since mamba no_buffer is not compatible with overlap schedule, try to use --disable-radix-cache if overlap schedule is necessary
[2025-12-29 18:33:02 TP0] Init torch distributed begin.
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[2025-12-29 18:33:02 TP0] Init torch distributed ends. mem usage=0.00 GB
[2025-12-29 18:33:03 TP0] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/home/blazingbhavneek/miniconda3/envs/sglang/lib/python3.11/site-packages/transformers/__init__.py)
[2025-12-29 18:33:03 TP0] Load weight begin. avail mem=15.33 GB
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.32it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  4.32it/s]

[2025-12-29 18:33:03 TP0] Load weight end. type=Lfm2ForCausalLM, dtype=torch.bfloat16, avail mem=13.81 GB, mem usage=1.52 GB.
[2025-12-29 18:33:03 TP0] Using KV cache dtype: torch.bfloat16
[2025-12-29 18:33:03 TP0] Mamba Cache is allocated. max_mamba_cache_size: 85292, conv_state size: 4.88GB, ssm_state size: 0.00GB 
[2025-12-29 18:33:03 TP0] KV Cache is allocated. #tokens: 473854, K size: 2.71 GB, V size: 2.71 GB
[2025-12-29 18:33:03 TP0] Memory pool end. avail mem=2.43 GB
[2025-12-29 18:33:03 TP0] Using hybrid linear attention backend for hybrid GDN models.
[2025-12-29 18:33:03 TP0] Capture cuda graph begin. This can take up to several minutes. avail mem=2.42 GB
[2025-12-29 18:33:03 TP0] Capture cuda graph bs [1, 2, 4, 8]
Capturing batches (bs=1 avail_mem=2.39 GB): 100%|█| 4/4 [00:00<00:00,  7.80it/
[2025-12-29 18:33:04 TP0] Capture cuda graph end. Time elapsed: 1.01 s. mem usage=0.03 GB. avail mem=2.39 GB.
max_total_num_tokens=473854
Warmup ...
[2025-12-29 18:33:04 TP0] Reset HybridReqToTokenPool
Prefill. latency: 0.30282 s, throughput:  27052.23 token/s
Decode 0. Batch size: 4, latency: 0.17757 s, throughput:     22.53 token/s
Decode 1. Batch size: 4, latency: 0.00518 s, throughput:    771.52 token/s
Decode 2. Batch size: 4, latency: 0.00538 s, throughput:    743.27 token/s
Decode 3. Batch size: 4, latency: 0.00761 s, throughput:    525.59 token/s
Decode 4. Batch size: 4, latency: 0.00516 s, throughput:    775.31 token/s
Decode.  median latency: 0.00507 s, median throughput:    788.46 token/s
Total. latency:  0.636 s, throughput:  13084.74 token/s
Benchmark ...
[2025-12-29 18:33:05 TP0] Reset HybridReqToTokenPool
Prefill. latency: 0.27105 s, throughput:  30222.82 token/s
Decode 0. Batch size: 4, latency: 0.00520 s, throughput:    768.89 token/s
Decode 1. Batch size: 4, latency: 0.00498 s, throughput:    802.91 token/s
Decode 2. Batch size: 4, latency: 0.00717 s, throughput:    557.95 token/s
Decode 3. Batch size: 4, latency: 0.00514 s, throughput:    777.59 token/s
Decode 4. Batch size: 4, latency: 0.00503 s, throughput:    795.59 token/s
Decode.  median latency: 0.00482 s, median throughput:    830.67 token/s
Total. latency:  5.225 s, throughput:   2351.56 token/s

```

# multi_chain_reasoning Test fix

### env

```bash
# OS
Linux fedora 6.17.10-100.fc41.x86_64 #1 SMP PREEMPT_DYNAMIC Mon Dec  1 16:10:21 UTC 2025 x86_64 GNU/Linux

# Python
Python 3.11.14

# pip show sglang
Name: sglang
Version: 0.5.6.post2
```

### Before Fix

```bash
sglang/benchmark/multi_chain_reasoning on  main via 🐍 v3.11.14 via 🅒 sglang took 2s 
❯ python3 bench_other.py --num-questions 64 --backend vllm
Traceback (most recent call last):
  File "sglang/benchmark/multi_chain_reasoning/bench_other.py", line 186, in <module>
    main(args)
  File "sglang/benchmark/multi_chain_reasoning/bench_other.py", line 99, in main
    for i in range(len(lines[: args.num_questions])):
                       ~~~~~^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'generator' object is not subscriptable

sglang/benchmark/multi_chain_reasoning on  main via 🐍 v3.11.14 via 🅒 sglang took 4s 
❯ python3 bench_sglang.py --num-questions 64
Traceback (most recent call last):
  File "sglang/benchmark/multi_chain_reasoning/bench_sglang.py", line 140, in <module>
    main(args)
  File "sglang/benchmark/multi_chain_reasoning/bench_sglang.py", line 48, in main
    for i in range(len(lines[: args.num_questions])):
                       ~~~~~^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'generator' object is not subscriptable
```

### After Fix

```bash
sglang/benchmark/multi_chain_reasoning on  bug/multi_chain_reasoning_test_fix [!] via 🐍 v3.11.14 via 🅒 sglang 
❯ python3 bench_sglang.py --num-questions 64
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [01:10<00:00,  1.11s/it]
Latency: 70.991
Invalid: 0.000
Accuracy: 0.797

sglang/benchmark/multi_chain_reasoning on  bug/multi_chain_reasoning_test_fix [!] via 🐍 v3.11.14 via 🅒 sglang took 1m15s 
❯ python3 bench_other.py --num-questions 64 --backend vllm
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [01:43<00:00,  1.61s/it]
Latency: 103.293
Invalid: 0.000
Accuracy: 0.812
```


- [x] asdfnasdf
- [ ] 
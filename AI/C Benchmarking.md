
SGLang Server

```bash
python -m sglang.launch_server   --model-path Qwen/Qwen3-8B-AWQ   --served-model-name nemotron-nano   --host 0.0.0.0   --port 8000
```

LiteLLM Config:

litellm.yaml
```yaml
model_list:
  - model_name: qwen
    litellm_params:
      model: hosted_vllm/Qwen/Qwen3-8B-AWQ
      api_base: http://localhost:8000/v1
```

## SWE Agent

./config/default.yaml

```yaml
agent:
  model:
    name: hosted_vllm/qwen
    api_base: http://0.0.0.0:4000
    per_instance_cost_limit: 0   
    total_cost_limit: 0
    per_instance_call_limit: 20
    max_input_tokens: 0  

```

cli

```bash
sweagent run-batch     --config ./config/default.yaml     --instances.type swe_bench     --instances.subset lite     --instances.split dev      --instances.slice :10 --instances.shuffle=True
```

## MultiPL-E

```bash
export OPENAI_API_KEY="sk-dummy-key"
export OPENAI_API_BASE="http://localhost:4000"

python3 chat_completions.py bench --name hosted_vllm/qwen --lang cpp --temperature 0.7 --num-concurrent 3 --max-completions 10 --name-override qwen --max-tokens 10000
```


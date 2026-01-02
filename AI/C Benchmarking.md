
SGLang Server

```bash
python -m sglang.launch_server   --model-path nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4   --served-model-name nemotron-nano   --host 0.0.0.0   --port 8000
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


## Ag-LiveCodeBench-X

```bash
python mod.py     completions     --model-name openai/nemotron-nano     --completions-path completions.jsonl     --temperature 0.4     --num-concurrent 5     --max-tokens 20000     --language "C"

python mod.py executions     --container-name agnostics:c     --timeout-seconds 15     --generations-path completions.jsonl     --executions-path executions.jsonl     --num-concurrent 50

python mod.py pass1 executions.jsonl

python mod.py refinements --executions-path executions.jsonl --refinements-path training.jsonl --completions-path completions2.jsonl --model-name openai/nemotron-nano --temperature 0.2     --num-concurrent 6   --max-tokens 10000     --language "C"

python mod.py pass1 executions2.jsonl


python mod.py iterative \
    --model-name openai/nemotron-nano \
    --language "C" \
    --container-name agnostics:c \
    --timeout-seconds 15 \
    --output-dir ./output \
    --temperature 0.4 \
    --num-concurrent 5 \
    --max-tokens 6000 \
    --use-thinking-budget \
    --tokenizer-name-or-path "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4" \
    --max-thinking-budget 2000 \
    --num-problems 20 \
    --max-refinement-iterations 3 \
    --num-completions 1 \
    --max-agent-iterations 0
```

# SWE-Smith


### **1. Environment Setup**
```bash
# Check cmake (required for building C repos)
cmake --version

# Activate conda environment
conda activate cBench

# Install SWE-smith in editable mode
cd ./SWE-smith/
pip install -e .

# Remove .git if it interferes with operations
rm -rf ./.git/
```

---

### **2. Test Repository Installation**
```bash
# Test cloning and building a C repository
python ./swesmith/build_repo/test_install_c.py DaveGamble/cJSON --commit latest
```

---

### **3. Docker Image Creation**
```bash
# List available environments
python swesmith/build_repo/create_images.py --list-envs

# Create Docker image for a repository (use the swesmith/ format)
python swesmith/build_repo/create_images.py -r swesmith/DaveGamble__cJSON.c859b25d

# If Docker socket permission issues occur:
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
# Then add to .bashrc for persistence
echo 'export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock' >> ~/.bashrc
source ~/.bashrc

# Test the image
docker run -it --rm swebench/swesmith.x86_64.davegamble_1776_cjson.c859b25d:latest
```

---

### **4. Bug Generation (LLM-based)**
```bash
# Set API credentials for LLM model
export OPENAI_API_KEY="sk-dummy"
export OPENAI_BASE_URL="http://localhost:4000/v1"  # For hosted models

# Generate bugs (key working command)
python ./swesmith/bug_gen/llm/modify.py swesmith/DaveGamble__cJSON.c859b25d \
    --config_file configs/bug_gen/lm_modify.yml \
    --model openai/nemotron-nano \
    --max_bugs 10 \
    --n_workers 2

# For larger batches:
python ./swesmith/bug_gen/llm/modify.py swesmith/DaveGamble__cJSON.c859b25d \
    --config_file configs/bug_gen/lm_modify.yml \
    --model openai/nemotron-nano \
    --max_bugs 100 \
    --n_workers 5
```

---

### **5. Patch Collection & Validation**
```bash
# Collect generated patches
python swesmith/bug_gen/collect_patches.py logs/bug_gen/swesmith/DaveGamble__cJSON.c859b25d

# Validate patches (first run)
python swesmith/harness/valid.py logs/bug_gen/swesmith/DaveGamble__cJSON.c859b25d_all_patches.json

# Re-run validation for failed/incomplete patches
python swesmith/harness/valid.py logs/bug_gen/swesmith/DaveGamble__cJSON.c859b25d_all_patches.json --redo_existing

# Clean up validation containers if stuck
docker rm -f $(docker ps -aq --filter "name=swesmith.val")
```

---

### **6. Results Gathering**
```bash
# Gather validation results
python swesmith/harness/gather.py logs/run_validation/DaveGamble__cJSON.c859b25d/
```

---

### **7. Issue Generation**
```bash
# Generate issue descriptions (remove --model flag if using default)
python swesmith/issue_gen/generate.py \
    --dataset logs/task_insts/DaveGamble__cJSON.c859b25d.json \
    --config configs/issue_gen/ig_v2.yaml \
    --workers 2 \
    --redo_existing
```


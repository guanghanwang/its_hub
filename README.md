# `its-hub`: A Python library for inference-time scaling

[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub/graph/badge.svg?token=6WD8NB9YPN)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub)
[![PyPI version](https://badge.fury.io/py/its-hub.svg)](https://badge.fury.io/py/its-hub)

**its_hub** is a Python library for inference-time scaling of LLMs, focusing on mathematical reasoning tasks.

## 📚 Documentation

For comprehensive documentation, including installation guides, tutorials, and API reference, visit:

**[https://ai-innovation.team/its_hub](https://ai-innovation.team/its_hub)**

## Quick Start

```python
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT
from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.algorithms import ParticleFiltering
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

# Initialize language model (requires vLLM server)
lm = OpenAICompatibleLanguageModel(
    endpoint="http://localhost:8000/v1", 
    api_key="NO_API_KEY", 
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct", 
    system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT, 
)

# Set up inference-time scaling
sg = StepGeneration("\n\n", 32, r"\boxed")
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B", 
    device="cuda:0", 
    aggregation_method="prod"
)
scaling_alg = ParticleFiltering(sg, prm)

# Solve with inference-time scaling
result = scaling_alg.infer(lm, "Solve x^2 + 5x + 6 = 0", budget=8)
```

## Installation

```bash
# Production
pip install its_hub

# Development
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
```

## Key Features

- 🔬 **Multiple Algorithms**: Particle Filtering, Best-of-N, Beam Search, Self-Consistency
- 🚀 **OpenAI-Compatible API**: Easy integration with existing applications  
- 🧮 **Math-Optimized**: Built for mathematical reasoning with specialized prompts
- 📊 **Benchmarking Tools**: Compare algorithms on MATH500 and AIME-2024 datasets
- ⚡ **Async Support**: Concurrent generation with limits and error handling

## Development

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
pytest tests
```

For detailed documentation, visit: [https://ai-innovation.team/its_hub](https://ai-innovation.team/its_hub)

# Benchmark

* Create a virtual environment
```bash
conda create -n its-hub python=3.11
conda activate its-hub
pip install -e ".[dev]"
```

* Launch vLLM server
```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dtype float16 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.5 \
    --max-num-seqs 128 \
    --tensor-parallel-size 1
```

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/its_hub/checkpoints/gsm8k_qwen_grpo_empo \
    --dtype float16 \
    --port 8001 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.5 \
    --max-num-seqs 128 \
    --tensor-parallel-size 1
```

* Benchmark models
```bash
for n in 1 2 4 8 16 32 64
do
    CUDA_VISIBLE_DEVICES=0 \
    python scripts/benchmark.py \
        --benchmark gsm8k \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --alg particle-filtering \
        --rm_device cuda:0 \
        --endpoint http://0.0.0.0:8000/v1 \
        --shuffle_seed 1110 \
        --does_eval \
        --budgets $n \
        --rm_agg_method model > logs/gsm8k_qwen_lambda-1_N-${n}.log
done
```

```bash
for n in 1 2 4 8 16 32 64
do
    CUDA_VISIBLE_DEVICES=1 \
    python scripts/benchmark.py \
        --benchmark gsm8k \
        --model_name /home/ubuntu/its_hub/checkpoints/gsm8k_qwen_grpo_empo \
        --alg particle-filtering \
        --rm_device cuda:0 \
        --endpoint http://0.0.0.0:8001/v1 \
        --shuffle_seed 1110 \
        --does_eval \
        --budgets $n \
        --rm_agg_method model > logs/gsm8k_qwen_grpo_empo_lambda-1_N-${n}.log
done
```
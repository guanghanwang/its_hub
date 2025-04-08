# A Python library for inference-time scaling LLMs

Example of using `[1]` for inference-time scaling

```python
from inference_time_scaling import ParticleFiltering

llm = ...
prm = ...
prompt = ...
scaling_method = ParticleFiltering(...)
budget = 16

scaling_method.inference(llm, prm, prompt, budget) # => gives output
```

`[1]`: Isha Puri, Shivchander Sudalairaj, Guangxuan Xu, Kai Xu, Akash Srivastava. “A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods”, 2025.

## Installation

```sh
pip install git+https://github.com/Red-Hat-AI-Innovation-Team/inference_time_scaling.git
```

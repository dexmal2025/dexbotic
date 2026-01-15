<!-- # Dexbotic-SimpleVLA-RL -->

Dexbotic extends Vision-Language-Action (VLA) models with SimpleVLA-RL algorithm for RL post-training.

## Installation

### 🐳 Docker (Recommended)

We strongly recommend using Docker as a unified, consistent, and reproducible environment for training and deployment. This approach not only ensures reliability across workflows but also minimizes potential issues arising from CUDA version differences and Python dependency conflicts.

> See [`dockerfile/Dockerfile.RL`](dockerfile/Dockerfile.RL) for more details.

0. Prerequisites

+ Ubuntu 20.04 or 22.04

+ NVIDIA GPU: RTX H20 (8 GPUs recommended for training; 1 GPU for deployment)

+ NVIDIA Docker installed

1. Step 1: Clone the Repository

```bash
git clone git@gitlab.dexmal.com:robotics/dexbotic.git
```

2. Step 2: Start Docker

```bash
docker run -it --rm --gpus all \
  -v /path/to/dexbotic:/dexbotic \
  dexmal/dexbotic:rl \
  bash
```

3. Step 3: Activate Dexbotic Environment

```bash
cd /dexbotic
conda activate dexbotic-rl
pip install -e .
```

## Launch RL Post-Training

```bash
deepspeed playground/benchmarks/libero/libero_simplevla_rl.py \
    --task=train \
    --sft_model_path=/path/to/sft-checkpoint \
    --dataset_name=libero_10
```

> **Note:** The rollout process in RL post-training may take **some** time to collect enough trajectories for per-step updates. Please be patient.
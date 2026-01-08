# Magima Kids - RunPod AI Pipeline

Custom video generation pipeline using LTX-Video and other models on RunPod.

## Quick Start

### On RunPod (any GPU, any region):

```bash
cd /workspace
git clone https://github.com/magimakids/magima-kids-runpod.git
cd magima-kids-runpod
./scripts/setup-runpod.sh
```

### Generate video:

```bash
cd /workspace/magima-kids-runpod
source .venv/bin/activate
python -m src.pipelines.generate --prompt "your prompt here"
```

## Local Development

```bash
git clone https://github.com/magimakids/magima-kids-runpod.git
cd magima-kids-runpod
uv venv && uv sync
```

## Structure

```
├── src/
│   ├── api/          # FastAPI server
│   ├── models/       # Model wrappers
│   └── pipelines/    # Generation pipelines
├── scripts/          # Setup & utility scripts
└── outputs/          # Generated videos (gitignored)
```

## Pull outputs locally

```bash
./scripts/pull-outputs.sh <RUNPOD_IP> <RUNPOD_PORT>
```

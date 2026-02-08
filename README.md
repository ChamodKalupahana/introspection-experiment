# Introspection Experiment

A simple setup for loading and testing Meta-Llama-3.1-8B-Instruct from Hugging Face.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Access to Llama 3.1

Meta's Llama models are "gated" and require you to accept their license:

1. Go to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Click "Request access" and accept Meta's license agreement
3. Wait for approval (usually instant)

### 3. Login to Hugging Face

Get an access token from https://huggingface.co/settings/tokens, then:

```bash
huggingface-cli login
```

Paste your token when prompted.

## Running the Test

```bash
python3 test_llama.py
```

This will:
- Check CUDA/GPU availability
- Load the Llama 3.1 8B Instruct model
- Run a simple generation test ("What is 2 + 2?")
- Verify the model works correctly

## Hardware Requirements

- **GPU**: ~16GB VRAM recommended (the model loads in bfloat16)
- **RAM**: ~32GB system RAM recommended for loading
- **Storage**: ~16GB for model weights
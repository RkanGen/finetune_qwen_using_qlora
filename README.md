# Qwen2.5-7B QLoRA Fine-tuning Guide

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.36+-yellow.svg)
![PEFT](https://img.shields.io/badge/PEFT-0.7+-green.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

**Memory-Efficient Fine-tuning of Qwen2.5-7B using QLoRA**

[Overview](#overview) •
[Features](#features) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Configuration](#configuration) •
[Usage](#usage) •
[Troubleshooting](#troubleshooting)

</div>

---

##  Overview

This repository provides a complete, production-ready implementation for fine-tuning **Qwen2.5-7B** using **QLoRA (Quantized Low-Rank Adaptation)**. QLoRA enables efficient fine-tuning of large language models on consumer-grade GPUs by using 4-bit quantization.

### What is QLoRA?

QLoRA combines:
- **4-bit Quantization**: Reduces model memory footprint by 75%
- **LoRA Adapters**: Only trains small adapter matrices (~1% of parameters)
- **NormalFloat4 (NF4)**: Special 4-bit format optimized for neural networks
- **Double Quantization**: Further compresses quantization constants

**Result**: Fine-tune 7B parameter models on GPUs with as little as 16GB VRAM!

---

##  Features

### 🚀 Core Capabilities
- ✅ **4-bit Quantization** via BitsAndBytes
- ✅ **QLoRA Configuration** (r=64, α=16, dropout=0.05)
- ✅ **OpenOrca Dataset** integration (5K+ high-quality samples)
- ✅ **Custom Chat Templates** with system/user/assistant formatting
- ✅ **Mixed Precision Training** (BF16/FP16)
- ✅ **Gradient Checkpointing** for memory efficiency
- ✅ **Automatic Evaluation** during training
- ✅ **Inference Examples** with generation utilities

###  Memory Efficiency
| Component | Memory Usage |
|-----------|--------------|
| Base Model (4-bit) | ~7 GB |
| LoRA Adapters | ~250 MB |
| Training Overhead | ~8-10 GB |
| **Total** | **~15-17 GB** |

✅ **Works on Google Colab Free Tier (T4 GPU)**

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.8+ (for GPU training)
- 16GB+ GPU VRAM (T4, V100, A100, or similar)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/RkanGen/finetune_qwen_using_qlora
cd finetune_qwen_using_qlora

# Install required packages
pip install -U transformers datasets peft bitsandbytes accelerate trl
pip install -U sentencepiece protobuf torch
```

### Google Colab Setup

```python
# Run in a Colab cell
!pip install -q -U transformers datasets peft bitsandbytes accelerate trl
!pip install -q -U sentencepiece protobuf

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

---

## 🚀 Quick Start

### 1. Basic Usage

```python
# Run the complete training script
python qwen_qlora_finetuning.py
```

That's it! The script will:
1. Load Qwen2.5-7B in 4-bit
2. Apply QLoRA adapters
3. Train on OpenOrca dataset
4. Save the adapter weights
5. Run inference examples

### 2. Training Time Estimates

| GPU | Batch Size | 5K Samples | Full Dataset |
|-----|------------|------------|--------------|
| T4 | 4 | ~2-3 hours | ~20-30 hours |
| V100 | 4 | ~1-2 hours | ~10-15 hours |
| A100 | 8 | ~45 min | ~5-8 hours |

### 3. Expected Results

After training, you should see:
- ✅ Training loss: ~1.5-2.0 (depending on dataset)
- ✅ Eval loss: ~1.8-2.2
- ✅ Perplexity: ~6-9
- ✅ Coherent, context-aware responses

---

## ⚙️ Configuration

### Model Configuration

```python
MODEL_NAME = "Qwen/Qwen2.5-7B"
OUTPUT_DIR = "./qwen2.5-7b-qlora-finetuned"
ADAPTER_DIR = "./qwen2.5-7b-qlora-adapter"
```

### QLoRA Parameters

```python
# LoRA rank (higher = more parameters, better quality)
LORA_R = 64              # Recommended: 8, 16, 32, 64, 128

# LoRA alpha (scaling factor)
LORA_ALPHA = 16          # Typically: r/4 or r/2

# LoRA dropout (regularization)
LORA_DROPOUT = 0.05      # Range: 0.0 - 0.1

# Target modules (which layers to adapt)
TARGET_MODULES = ["q_proj", "v_proj"]  # Can add: "k_proj", "o_proj"
```

### Training Hyperparameters

```python
MAX_LENGTH = 512                    # Maximum sequence length
BATCH_SIZE = 4                      # Per-device batch size
GRADIENT_ACCUMULATION_STEPS = 4     # Effective batch = 16
LEARNING_RATE = 2e-4                # Peak learning rate
NUM_EPOCHS = 3                      # Training epochs
```

### Dataset Options

```python
# Choose one:
DATASET_NAME = "Open-Orca/OpenOrca"              # 1M samples
DATASET_NAME = "Open-Orca/OpenOrca-Slim"         # Curated subset
DATASET_NAME = "Open-Orca/OpenOrca-Platypus2"    # High-quality filtered

MAX_SAMPLES = 5000  # Limit for faster training
```

---

## 📖 Usage

### Training from Scratch

```python
# Default configuration (recommended)
python qwen_qlora_finetuning.py

# Custom configuration
python qwen_qlora_finetuning.py \
    --lora_r 128 \
    --learning_rate 1e-4 \
    --num_epochs 5
```

### Loading and Using the Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model in 4-bit
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load QLoRA adapter
model = PeftModel.from_pretrained(base_model, "./qwen2.5-7b-qlora-adapter")
tokenizer = AutoTokenizer.from_pretrained("./qwen2.5-7b-qlora-adapter")

# Set to evaluation mode
model.eval()
```

### Inference Example

```python
def generate_response(system_prompt, user_prompt):
    # Format input
    input_text = f"""<|system|>{system_prompt}</|system|>
<|user|>{user_prompt}</|user|>
<|assistant|>"""
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|assistant|>")[-1].strip()

# Example usage
response = generate_response(
    system_prompt="You are a helpful AI assistant.",
    user_prompt="Explain quantum computing in simple terms."
)
print(response)
```

### Custom Dataset Format

To use your own dataset, format it as JSONL:

```jsonl
{"system_prompt": "You are a helpful assistant.", "question": "What is AI?", "response": "AI stands for..."}
{"system_prompt": "You are a coding expert.", "question": "Write a function...", "response": "Here's the code..."}
```

Then modify the script:

```python
# Load custom dataset
dataset = load_dataset('json', data_files='your_data.jsonl', split='train')
```

---

##  Advanced Usage

### Merging Adapter with Base Model

For deployment, you can merge the adapter into the base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
model = PeftModel.from_pretrained(base_model, "./qwen2.5-7b-qlora-adapter")

# Merge adapter weights
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./qwen2.5-7b-qlora-merged")
tokenizer.save_pretrained("./qwen2.5-7b-qlora-merged")
```

**Note**: Merging requires loading the full model in memory (~14GB).

### Multi-GPU Training

```python
# Automatic with accelerate
accelerate launch qwen_qlora_finetuning.py

# Or configure in TrainingArguments
training_args = TrainingArguments(
    ...,
    ddp_find_unused_parameters=False,
)
```

### Monitoring Training

```bash
# Enable Weights & Biases logging
# In TrainingArguments:
report_to="wandb"

# Or TensorBoard
report_to="tensorboard"
tensorboard --logdir ./qwen2.5-7b-qlora-finetuned/logs
```

---

##  Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1: Reduce batch size**
```python
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8  # Keep effective batch size
```

**Solution 2: Reduce sequence length**
```python
MAX_LENGTH = 256  # Instead of 512
```

**Solution 3: Use gradient checkpointing** (already enabled)

**Solution 4: Clear CUDA cache**
```python
import torch
torch.cuda.empty_cache()
```

### Slow Training Speed

**Solution 1: Increase batch size** (if memory allows)
```python
BATCH_SIZE = 8
```

**Solution 2: Use fewer samples**
```python
MAX_SAMPLES = 1000  # For testing
```

**Solution 3: Enable Flash Attention**
```python
# Install flash-attn
pip install flash-attn --no-build-isolation

# Set in model loading
attn_implementation="flash_attention_2"
```

### CUDA Version Mismatch

```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### BitsAndBytes Issues

```bash
# Reinstall bitsandbytes
pip uninstall bitsandbytes
pip install bitsandbytes --no-cache-dir

# For Windows, use pre-compiled wheels
pip install bitsandbytes-windows
```

### Model Not Loading

**Issue**: `OSError: Can't load tokenizer`

**Solution**:
```python
# Add trust_remote_code
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
```

### Poor Generation Quality

**Solution 1: Adjust generation parameters**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=512,      # Longer responses
    temperature=0.8,         # More creative
    top_p=0.95,             # Nucleus sampling
    repetition_penalty=1.2,  # Reduce repetition
)
```

**Solution 2: Train longer**
```python
NUM_EPOCHS = 5  # Instead of 3
```

**Solution 3: Use more data**
```python
MAX_SAMPLES = 10000  # Instead of 5000
```

---

## 📊 Performance Benchmarks

### Memory Usage (Google Colab)

| Configuration | GPU Memory | Training |
|--------------|------------|----------|
| Batch=2, r=32 | 13 GB | ✅ T4 |
| Batch=4, r=64 | 15 GB | ✅ T4 |
| Batch=8, r=128 | 18 GB | ❌ T4 |

### Training Speed

| GPU | Samples/sec | Time for 5K samples |
|-----|-------------|---------------------|
| T4 | ~2-3 | 2-3 hours |
| V100 | ~4-6 | 1-2 hours |
| A100 | ~8-12 | 45-60 minutes |

---

## 🔬 Understanding QLoRA

### How It Works

1. **Base Model Frozen**: Original weights stay fixed (4-bit)
2. **LoRA Adapters**: Small trainable matrices added (FP16/BF16)
3. **Forward Pass**: Activations computed in high precision
4. **Backward Pass**: Only adapter gradients computed
5. **Memory Efficient**: Optimizer states only for adapters

### QLoRA vs Full Fine-tuning

| Aspect | QLoRA | Full Fine-tuning |
|--------|-------|------------------|
| Memory | 15-17 GB | 60-80 GB |
| Speed | Fast | Faster |
| Quality | 95-99% | 100% |
| Trainable Params | ~1% | 100% |
| GPU Required | T4+ | A100+ |

### When to Use QLoRA

✅ **Use QLoRA when:**
- Limited GPU memory (< 24GB)
- Fine-tuning for specific tasks
- Need fast iteration cycles
- Training on consumer hardware

❌ **Don't use QLoRA when:**
- Have abundant GPU memory (80GB+)
- Need absolute best quality
- Training from scratch
- Production deployment (merge first)

---

## 📚 Additional Resources

### Documentation
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [Qwen Documentation](https://github.com/QwenLM/Qwen)

### Papers
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)

### Community
- [Hugging Face Forums](https://discuss.huggingface.co/)


---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---


<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

Made with ❤️ by the open-source community

</div>

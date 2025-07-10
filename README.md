# Memory-Efficient LLM Finetuning

This project explores techniques to **reduce GPU memory usage by up to 16×** during fine-tuning of large language models, using **DeepSeek-R1-Distill-Qwen-1.5B** as the base model. These techniques include:

* Gradient Checkpointing
* 4-bit / 8-bit Quantization
* LoRA (Low-Rank Adaptation)
* sequence_len

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Navigate to the experiment directory

```bash
cd experiments
```

### 3. Run fine-tuning

```bash
bash finetune.sh
```

### 4. Run evaluation

```bash
bash evaluate.sh
```

---

## Introduction

* **Dataset**: HuggingFace IMDB sentiment classification
  * Train/Eval/Test = 10k/1k/10k
* **Model**: [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

  * Default `sequence_length = 16384` (you can reduce this in tokenizer or training arguments to save memory)
  * Batch size = 1

## Techniques Used

### Gradient Checkpoint
Reduces memory usage by recomputing intermediate activations during backpropagation.

`model.gradient_checkpointing_enable()`

### Quantization 
Quantization reduces the precision of model weights to 4-bit or 8-bit.

Libaray: `transformers + bitsandbytes`

#### Configs

```python
BitsAndBytesConfig(
    load_in_4bit=True,                       # use 4-bit weight
    bnb_4bit_use_double_quant=True,          # double quantization
    bnb_4bit_quant_type="nf4",               # or "fp4"
    bnb_4bit_compute_dtype=torch.float16     # computation in float16
)
```

#### Memory Flow
```
VRAM
 ↓
[1] Weights stored as 4-bit (e.g., NF4)
 ↓
[2] Decoded by GPU thread to 16-bit (float16 or bfloat16)
 ↓
[3] Computation (Forward Pass)
 ↓
[4] (Optional) Backward Pass
 ↓
[5] Gradients in float16 / float32
 ↓
[6] Optimizer updates float16 weights
 ↓
[7] Re-quantize to 4-bit
 ↓
VRAM
```

###  LoRA (Low-Rank Adaptation)
LoRA reduces trainable parameters, leading to lower memory usage for gradients and optimizer states.
* optimizer states typically consume 4× the size of model parameters 

Libaray: `peft`

#### Configs
* `r`: rank of the low-rank matrices
* `lora_alpha`: a scaling factor
* `lora_dropout`: reduce overfitting

#### Formula

$$
W_{\text{adapted}} = W + \frac{\alpha}{r} \cdot (A \cdot B)
$$

Where:

* $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times k}$
* $W$ is the frozen pre-trained weight
* $A$ and $B$ are trainable low-rank matrices

## Results

Note: Always open gradient checkpointing to avoid OOM

### GPU Memory Usage Summary

#### 1. Overall Memory Usage

| Setting               | GPU Memory |
| --------------------- | -------- |
| All optimizations off | 23.5 GB  |
| All optimizations on  | 1.2 GB   |

> All optimizations off: `seq_len=16384`, `bit=16`, no LoRA
>
> All optimizations on: `seq_len=1024`, `quantization=4bit`, `LoRA rank=8`

#### 2. Sequence Length Comparison (Fixed: 4-bit, Rank=8)

| Sequence Length | GPU Memory |
| --------------- | ---------- |
| 1024            | 1.21 GB     |
| 4096            | 1.42 GB     |
| 16384           | 2.73 GB    |

---

#### 3. Quantization Comparison (Fixed: Rank=8, seq\_len=16384)

| Quantization | GPU Memory |
| ------------ | ---------- |
| 4-bit        | 2.73 GB    |
| 8-bit        | 3.49 GB    |
| float16      | 8.80 GB     |


---

#### 4. LoRA Rank Comparison (Fixed: 4-bit, seq\_len=16384)

| LoRA Rank | Trainable Params | GPU Memory |
| --------- | ---------------- | ---------- |
| 1         | 0.13M            | 2.72 GB    |
| 4         | 0.53M            | 2.73 GB    |
| 8         | 1.09M            | 2.73 GB    |

---

#### 5. Overall Memory Trend

| Component           | Impact              |
| ------------------- | ------------------- |
| ↑ `sequence_length` | ↑ Activation memory |
| ↑ `LoRA rank`       | ↑ Trainable params  |
| ↑ `bit precision`   | ↑ Weight size       |

### Accuracy

#### Overall Performance

| Setting               | Accuracy |
| --------------------- | -------- |
| All optimizations off | 100% |
| All optimizations on  | 100% |
| Without FT  | 12%   |

> All optimizations off: `seq_len=16384`, `bit=16`, no LoRA
>
> All optimizations on: `seq_len=1024`, `quantization=4bit`, `LoRA rank=8`


---

## Reference

* Model: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
* Discussion on `sequence_length`: [Issue #22](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/discussions/22)

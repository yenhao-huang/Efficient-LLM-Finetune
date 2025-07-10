# Memory-Efficient LLM Finetuning with LoRA, Quantization, and Gradient Checkpointing

This project explores techniques to **reduce GPU memory usage** while finetuning large language models using **DeepSeek-R1-Distill-Qwen-1.5B**, including:

* Gradient Checkpointing
* 4-bit / 8-bit Quantization
* LoRA (Low-Rank Adaptation)


TODO
* experiment
    * accuracy


Sure! Here's the English version of your `README` intro and instructions, revised for clarity and professionalism:

---

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

## 📘 Introduction

* **Dataset**: HuggingFace IMDB sentiment classification
* **Model**: [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

  * Default `sequence_length = 16384` (you can reduce this in tokenizer or training arguments to save memory)


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

Open gradient checkpointing to avoid OOM

### 💾 GPU Memory Usage Summary

#### 1. Sequence Length Comparison (Fixed: 4-bit, Rank=8)

| Sequence Length | GPU Memory |
| --------------- | ---------- |
| 1024            | 1.2 GB     |
| 4096            | 1.4 GB     |
| 16384           | 2.73 GB    |

---

#### 2. LoRA Rank Comparison (Fixed: 4-bit, seq\_len=16384)

| LoRA Rank | Trainable Params | GPU Memory |
| --------- | ---------------- | ---------- |
| 1         | 0.13M            | 2.72 GB    |
| 4         | 0.53M            | 2.73 GB    |
| 8         | 1.09M            | 2.73 GB    |

---

#### 3. Quantization Comparison (Fixed: Rank=8, seq\_len=16384)

| Quantization | GPU Memory |
| ------------ | ---------- |
| 4-bit        | 2.73 GB    |
| 8-bit        | 3.49 GB    |
| float16      | 8.8 GB     |

---

#### 4. Overall Memory Trend

| Component           | Impact              |
| ------------------- | ------------------- |
| ↑ `sequence_length` | ↑ Activation memory |
| ↑ `LoRA rank`       | ↑ Trainable params  |
| ↑ `bit precision`   | ↑ Weight size       |

---

### ✅ Gradient Checkpointing

| Setting  | Result    |
| -------- | --------- |
| Disabled | ❌ OOM     |
| Enabled  | ✅ Success |

---

### Accuracy Summary

#### 1. Overall Finetune(FT) Performance

| Setting               | Accuracy |
| --------------------- | -------- |
| All optimizations off | x     |
| All optimizations on  | ↑        |
| Without FT  | 0.12   |

> ✅ `seq_len=1024`, `quantization=4bit`, `LoRA rank=8`
> ❌ `seq_len=16384`, `bit=16`, no LoRA

---

#### 2. LoRA Comparison (Fixed: 4bit, seq\_len=1024)

| LoRA Setting       | Accuracy |
| ------------------ | -------- |
| With LoRA (Rank=8) | ↑        |
| Without LoRA       | ↓        |

---

#### 3. Quantization Comparison (Fixed: LoRA Rank=8, seq\_len=1024)

| Quantization | Accuracy |
| ------------ | -------- |
| 4-bit        | ↑        |
| None         | ↓        |

> ❌ "None" quantization = LoRA only
> ✅ "4-bit" quantization + LoRA

---

#### 4. Sequence Length Comparison (Fixed: LoRA Rank=8, 4-bit)

| Seq Length | Accuracy |
| ---------- | -------- |
| 1024       | ↑        |
| 16384      | ↓        |


---

## Reference

* Model: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
* Discussion on `sequence_length`: [Issue #22](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/discussions/22)

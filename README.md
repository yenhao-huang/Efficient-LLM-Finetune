TODO
* experiment
    * time
    * memory
        * inference
        * non-inference

    
## Requirements
python=3.12

## Used Tool
transformers + bitsandbytes

## Methodologies

### Quantization 

Configs 
* **`load_in_4bit=True`**
  → Store model parameters in **4-bit format** (NF4 or FP4), greatly reducing VRAM usage.

* **`bnb_4bit_use_double_quant=True`**
  → Enable **double quantization**, where weights are first quantized to 8-bit, then to 4-bit.
  **Improves accuracy** and reduces information loss.

* **`bnb_4bit_compute_dtype=torch.float16`**
  → Set the **computation precision** for forward and backward passes.
  Common values: `float16`, `bfloat16`, or `float32`.

Overview
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

### LoRa
**Configs**
* `r`: rank of the low-rank matrices
* `lora_alpha`: a scaling factor
* `lora_dropout`: reduce overfitting

**Formula**

$$
W_{\text{adapted}} = W + \frac{\alpha}{r} \cdot (A \cdot B)
$$

Where:

* $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times k}$
* $W$ is the frozen pre-trained weight
* $A$ and $B$ are trainable low-rank matrices


## Results
### Memory

(PEFT)
(Quantization)
(Activation)

### Accuracy
finetune: 
without: 0.12

### Time 
without: 16m/1000


gradient checkpoint: 不能跑 -> 可以跑

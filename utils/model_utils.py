from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, BitsAndBytesConfig
import torch
from peft import get_peft_model, LoraConfig, TaskType
from utils.utils import CUDAMemoryTrackerCallback

def set_eval_agent(model_name, num_labels, metrics_strategy, output_dir, eval_batch):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)

    args = TrainingArguments(
        output_dir=output_dir, 
        per_device_eval_batch_size=eval_batch,
        fp16=True,
        )

    eval_agent = Trainer(
        model=model,
        args=args,
        compute_metrics=metrics_strategy,
    )
    return eval_agent

def set_train_agent_PEFT(
    model_name, 
    train_dataset, 
    eval_dataset, 
    tokenizer, 
    num_labels, 
    metrics_strategy, 
    output_dir, 
    label_col, 
    n_rank,
    bit_precision,
    ):
    
    if bit_precision == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif bit_precision == 8:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quant_config = None
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels, 
        ignore_mismatched_sizes=True,
        quantization_config=quant_config,
    )

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # 任務類型：分類
        r=n_rank,                         # bottleneck 維度
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print(model.print_trainable_parameters())

    use_bf16 = bit_precision is None
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        label_names=[label_col],
        load_best_model_at_end=False,
        bf16=use_bf16,
    )
    
    train_agent = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=metrics_strategy,
        callbacks=[CUDAMemoryTrackerCallback(log_every=50)],
    )
    return train_agent
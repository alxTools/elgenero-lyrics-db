#!/usr/bin/env python3

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# ✅ Force GPU Optimization
torch.backends.cudnn.benchmark = True  # Optimizes for performance
torch.backends.cuda.matmul.allow_tf32 = True  # Uses TF32 for faster training

# ✅ Load Dataset
dataset = load_dataset("json", data_files="datasets/dataset_lyrics_fixed.jsonl", split="train")

# ✅ Load Tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

# ✅ Tokenize Dataset
def tokenize_function(examples):
    inputs = tokenizer(examples["input"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")  # ✅ Reduced max_length to 256
    outputs = tokenizer(examples["output"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"]  # Model expects `labels`
    }

dataset = dataset.map(tokenize_function, batched=True, remove_columns=["input", "output"])

# ✅ Enable 4-bit Quantization (Using BitsAndBytesConfig)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16  # Use BF16 for lower memory & better precision
)

# ✅ Load Model with QLoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # 🔹 Correct way to enable 4-bit
    torch_dtype=torch.float16,
    device_map="auto",
    use_cache=False  # ✅ Explicitly disable cache to silence warning
)

# ✅ Apply QLoRA (8-bit or 4-bit training)
model = prepare_model_for_kbit_training(model)

# ✅ Manually Enable/Disable Gradient Checkpointing
model.gradient_checkpointing_enable()  # ✅ If memory is a problem, keep this. Otherwise, disable.

# ✅ Apply LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1, bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ✅ Training Arguments (Optimized)
# training_args = TrainingArguments(
#     output_dir="./finetuned_reggaeton",
#     per_device_train_batch_size=12,  # ⬆️ Increase to maximize GPU
#     gradient_accumulation_steps=1,  # ✅ Keep it at 1 for speed
#     num_train_epochs=3,  # ✅ Adjust if needed
#     save_steps=500,
#     save_total_limit=2,
#     logging_dir="./logs",
#     logging_steps=10,
#     eval_strategy="no",
#     fp16=True  # ✅ Ensures FP16 for better speed
# )

training_args = TrainingArguments(
    output_dir="./finetuned_reggaeton",
    per_device_train_batch_size=8,  # ⬆️ Adjust based on VRAM usage
    gradient_accumulation_steps=2,  # ✅ Helps with stability
    num_train_epochs=5,  # ⬆️ Train longer for better results
    learning_rate=2e-5,  # ✅ Lower LR for better fine-tuning
    save_steps=1000,  # ⬆️ Save less often to reduce overhead
    save_total_limit=1,  # ✅ Keep last model checkpoint
    logging_dir="./logs",
    logging_steps=50,  # ✅ Reduce log spam
    evaluation_strategy="no",
    fp16=True,  # ✅ Ensures FP16 for speed
    optim="adamw_bnb_8bit",  # ✅ Faster & memory-efficient optimizer
    report_to="none"  # ✅ Disable reporting (if using WANDB)
)


# ✅ Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# ✅ Start Training
trainer.train()

# ✅ Save Adapter
model.save_pretrained("./finetuned_reggaeton")

print("✅ Training completed! Adapter saved in './finetuned_reggaeton'")

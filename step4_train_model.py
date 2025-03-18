#!/usr/bin/env python3

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# ✅ Optimización para GPU
torch.backends.cudnn.benchmark = True  # Optimiza el rendimiento
torch.backends.cuda.matmul.allow_tf32 = True  # Usa TF32 para acelerar cálculos

# ✅ Cargar el dataset
dataset = load_dataset("json", data_files="datasets/dataset_lyrics_fixed.jsonl", split="train")

# ✅ Cargar el tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Soluciona problema de padding

# ✅ Función de tokenización actualizada
def tokenize_function(examples):
    combined_inputs = []
    for i in range(len(examples["input"])):
        # Combina "instruction" e "input" (si existe "instruction", se separa con un salto de línea)
        inst = examples["instruction"][i] if "instruction" in examples else ""
        inp = examples["input"][i]
        combined = f"{inst}\n{inp}" if inst else inp
        combined_inputs.append(combined)

    inputs = tokenizer(combined_inputs, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    outputs = tokenizer(examples["output"], padding="max_length", truncation=True, max_length=256, return_tensors="pt")

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"]
    }

dataset = dataset.map(tokenize_function, batched=True, remove_columns=["instruction", "input", "output"])

# ✅ Configuración para cuantización a 4-bit (usando BitsAndBytesConfig)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16  # Usa BF16 para menor memoria y mejor precisión
)

# ✅ Cargar el modelo con QLoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # Habilita 4-bit
    torch_dtype=torch.float16,
    device_map="auto",
    use_cache=False  # Desactiva cache para evitar warnings
)

# ✅ Preparar el modelo para entrenamiento en k-bit
model = prepare_model_for_kbit_training(model)

# ✅ Habilitar o deshabilitar el gradient checkpointing según convenga
model.gradient_checkpointing_enable()  # Actívalo si tienes problemas de memoria

# ✅ Aplicar LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ✅ Configuración de entrenamiento (ajustada para estabilidad y eficiencia)
training_args = TrainingArguments(
    output_dir="./models/reggaeton_lyrics",
    per_device_train_batch_size=8,         # Ajusta según VRAM
    gradient_accumulation_steps=2,           # Para mayor estabilidad
    num_train_epochs=5,                      # Entrena por más épocas para mejores resultados
    learning_rate=2e-5,                      # LR bajo para fine-tuning
    save_steps=1000,                         # Menos checkpoints para reducir overhead
    save_total_limit=1,                      # Solo se guarda el último checkpoint
    logging_dir="./logs",
    logging_steps=50,                        # Menos logs para evitar spam
    evaluation_strategy="no",
    fp16=True,                               # Usa FP16 para velocidad
    optim="adamw_bnb_8bit",                  # Optimizador rápido y eficiente en memoria
    report_to="none"                         # Desactiva reportes (por ejemplo, WANDB)
)

# ✅ Definir el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# ✅ Iniciar el entrenamiento
trainer.train()

# ✅ Guardar el adaptador LoRA
model.save_pretrained("./models/reggaeton_lyrics")

print("✅ ¡Entrenamiento completado! Adaptador guardado en './models/reggaeton_lyrics'")

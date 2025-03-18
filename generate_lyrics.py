from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_path = "./finetuned_reggaeton"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Escribe una canción sobre un party en la playa con una gata."
output = generator(prompt, max_length=200, do_sample=True, temperature=0.8)

print(output[0]["generated_text"])

#!/usr/bin/env python3

import os
import json

DATA_DIR = "./cleaned_data"  # Directorio con los JSON limpios
OUTPUT_FILE = "datasets/dataset_lyrics.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".json"):
            file_path = os.path.join(DATA_DIR, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for song in data:
                    out_file.write(f"Artista: {song['artist']}\n")
                    out_file.write(f"Título: {song['title']}\n")
                    out_file.write(f"Descripción: {song['desc']}\n")
                    out_file.write(f"Letras:\n{song['lyrics']}\n\n")
                    out_file.write("="*50 + "\n\n")

print(f"✅ Dataset generado en {OUTPUT_FILE}")

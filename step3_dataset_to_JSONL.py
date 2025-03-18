#!/usr/bin/env python3

import json

INPUT_FILE = "datasets/dataset_lyrics.txt"
OUTPUT_FILE = "datasets/dataset_lyrics_fixed.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as f, open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    data = f.read().strip().split("="*50 + "\n\n")  # Separa cada canción usando el delimitador
    for song in data:
        lines = song.strip().split("\n")
        if len(lines) < 4:
            continue  # Omite entradas vacías o mal formadas

        artist = lines[0].replace("Artista: ", "").strip()
        title = lines[1].replace("Título: ", "").strip()
        desc = lines[2].replace("Descripción: ", "").strip()
        # Se toma la letra a partir de la cuarta línea
        lyrics = "\n".join(lines[4:]).strip()

        # Se arma un prompt más robusto
        prompt_instruction = "Genera una letra de reggaetón siguiendo las siguientes indicaciones."
        prompt_input = (
            f"Artista: {artist}\n"
            f"Título: {title}\n"
            f"Descripción: {desc}\n"
            "Formato sugerido: [Intro] (Opcional), [Pre-Coro] (Opcional), [Coro], [Verso 1], "
            "[Coro], [Verso 2], [Coro], [Outro]. Evita repeticiones innecesarias y usa jerga auténtica del género."
        )

        json_line = json.dumps({
            "instruction": prompt_instruction,
            "input": prompt_input,
            "output": lyrics
        }, ensure_ascii=False)
        out_f.write(json_line + "\n")

print(f"✅ Dataset convertido a JSONL en {OUTPUT_FILE}")

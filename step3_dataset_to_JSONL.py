#!/usr/bin/env python3

import json

INPUT_FILE = "datasets/dataset_lyrics.txt"
OUTPUT_FILE = "datasets/dataset_lyrics_fixed.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as f, open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    data = f.read().strip().split("="*50 + "\n\n")  # Split each song
    for song in data:
        lines = song.strip().split("\n")
        if len(lines) < 4:
            continue  # Skip empty or malformed entries

        artist = lines[0].replace("Artista: ", "").strip()
        title = lines[1].replace("Título: ", "").strip()
        desc = lines[2].replace("Descripción: ", "").strip()
        lyrics = "\n".join(lines[4:]).strip()

        json_line = json.dumps({
            "input": f"Escribe una canción titulada '{title}' de {artist}. {desc}",
            "output": lyrics
        }, ensure_ascii=False)

        out_f.write(json_line + "\n")

print(f"✅ Dataset convertido a JSONL en {OUTPUT_FILE}")

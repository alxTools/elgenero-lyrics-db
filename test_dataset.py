#!/usr/bin/env python3
import json

with open("datasets/dataset_lyrics_fixed.jsonl", "r", encoding="utf-8") as f:
    for _ in range(5):  # Muestra 5 líneas
        print(json.loads(f.readline()))

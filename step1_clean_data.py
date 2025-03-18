#!/usr/bin/env python3

import os
import json
import argparse
import re

def extract_desc_and_clean_lyrics(lyrics, title):
    """
    Extracts 'Letra de' part to 'desc' and cleans 'lyrics' to keep only structured song text.
    """
    # Extract "Letra de" description
    desc_match = re.search(r'(Letra de\s*\".*?\")', lyrics, flags=re.IGNORECASE)
    desc = desc_match.group(1) if desc_match else ""

    # Clean the 'desc' field properly by removing extra escape characters
    desc = desc.replace('\\"', '"').strip()

    # Remove "Letra de" and "Contributors" from lyrics
    lyrics = re.sub(r'Letra de\s*\".*?\"', '', lyrics, flags=re.IGNORECASE).strip()
    lyrics = re.sub(r'\d+\s*Contributors.*?Lyrics', '', lyrics, flags=re.IGNORECASE).strip()
    lyrics = re.sub(r'LyricsFeat\..*', '', lyrics, flags=re.IGNORECASE).strip()

    # Ensure lyrics start at the first song section like [Intro:], [Verso:], etc.
    lyrics = re.sub(r'^.*?(?=\[)', '', lyrics, flags=re.DOTALL).strip()

    return desc, lyrics

def clean_json_file(file_path, output_dir):
    """Processes a JSON file, extracting relevant song data and cleaning lyrics."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "songs" not in data:
        print(f"⚠️ No songs found in {file_path}")
        return

    cleaned_songs = []
    for song in data["songs"]:
        if "title" in song and "artist" in song and "lyrics" in song:
            desc, lyrics = extract_desc_and_clean_lyrics(song["lyrics"], song["title"])
            cleaned_songs.append({
                "artist": song["artist"],
                "title": song["title"],
                "desc": desc,  # Fixed description field
                "lyrics": lyrics  # Strictly structured lyrics
            })

    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save cleaned JSON
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_songs, f, ensure_ascii=False, indent=4)

    print(f"✅ Processed and saved: {output_path}")

def process_directory(directory):
    """Processes all JSON files in the specified directory."""
    output_dir = os.path.join(directory, "datasets/cleaned_data")
    for file_name in os.listdir(directory):
        if file_name.endswith(".json"):
            file_path = os.path.join(directory, file_name)
            clean_json_file(file_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean JSON lyrics and extract descriptions")
    parser.add_argument("-d", "--directory", required=True, help="Directory containing JSON files")
    parser.add_argument("--mode", required=True, choices=["step1"], help="Processing mode")

    args = parser.parse_args()

    if args.mode == "step1":
        process_directory(args.directory)

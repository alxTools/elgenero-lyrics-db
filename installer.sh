#!/usr/bin/env bash

./step1_clean_data.py -d datasets/raw_data/ --mode step1
./step2_dataset_lyrics.py
./step3_dataset_to_JSONL.py
./step4_train_model.py

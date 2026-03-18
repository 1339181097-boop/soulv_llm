#!/usr/bin/env bash
set -euo pipefail

python src/data_pipeline/handlers/handler_itinerary.py --input data/raw/itinerary.jsonl --output data/processed/sft_itinerary.json
python src/data_pipeline/handlers/handler_intent.py --input data/raw/intent.jsonl --output data/processed/sft_intent.json
python src/data_pipeline/handlers/handler_roleplay_safety.py --input data/raw/roleplay_safety.jsonl --output data/processed/sft_roleplay_safety.json
python src/data_pipeline/data_mixer.py --output data/final/soulv_mixed_sft.json --total-samples 1344

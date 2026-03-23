#!/usr/bin/env bash
set -euo pipefail

python src/data_pipeline/handlers/handler_guide_generation.py --mode all --source-raw data/raw/guide_generation_raw.jsonl --prepared-raw data/raw/guide_generation_raw_expanded.jsonl --output data/processed/sft_guide_generation.json --target-raw 1500 --target-cleanable 1500
python src/data_pipeline/handlers/handler_dialogue.py --input data/raw/dialogue_data.jsonl --output data/processed/sft_dialogue.json
python src/data_pipeline/handlers/handler_roleplay_safety.py --input data/raw/roleplay_safety.jsonl --output data/processed/sft_roleplay_safety.json
python src/data_pipeline/data_mixer.py --output data/final/stage1_general_sft.json --total-samples 960 --spec sft_guide_generation.json=0.50 --spec sft_dialogue.json=0.30 --spec sft_roleplay_safety.json=0.20

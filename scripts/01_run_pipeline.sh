#!/usr/bin/env bash
set -euo pipefail

python src/data_pipeline/handlers/handler_guide_generation.py --mode all --source-raw data/raw/guide_generation_raw.jsonl --prepared-raw data/raw/guide_generation_raw_expanded.jsonl --output data/processed/sft_guide_generation.json --target-raw 1500 --target-cleanable 1500
python src/data_pipeline/handlers/handler_travel_qa.py --input data/raw/travel_qa_raw_3_23.jsonl --output data/processed/sft_travel_qa.json --total-samples 1250 --city-cap 80 --answer-cap 5
python src/data_pipeline/handlers/handler_hotel_recommendation.py --input data/raw/hotel_recommendation_0330.jsonl --output data/processed/sft_hotel_recommendation.json
python src/data_pipeline/handlers/handler_traffic_planning.py --output data/processed/sft_traffic_planning.json
python src/data_pipeline/handlers/handler_multiturn.py --output data/processed/sft_multi_turn_dialogue.json

# persona_understanding processed bucket pending.
# persona_understanding 尚未接入时，stage1 会先按五类候选版 stage recipe 输出。
python src/data_pipeline/data_mixer.py --stage current --report data/final/stage_mix_report.json

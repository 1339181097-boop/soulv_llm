# Train

Stage1 and stage2 training entrypoints both live in the repo root `scripts/` and `configs/`.

## Stage1

- Train config: `configs/llamafactory_stage1_sft.yaml`
- Entry: `bash scripts/02_run_sft.sh stage1`

## Stage2 AMap Tool-Use MVP

- Merge stage1 base: `bash scripts/04_merge_stage1_for_stage2.sh`
- Build data: `bash scripts/05_build_stage2_amap_data.sh`
- Train config: `configs/llamafactory_stage2_amap_tool_use_sft.yaml`
- Dataset registration: `configs/llamafactory_dataset_info_stage2_amap_tool.json`
- Entry: `bash scripts/02_run_sft.sh stage2_amap`

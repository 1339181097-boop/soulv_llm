# Train

Stage1 and stage2 training entrypoints both live in the repo root `scripts/` and `configs/`.

For the Aliyun dual-environment bootstrap (`lf` for training, `qwen` for vLLM serving), see
`docs/aliyun_qwen3_32b_env_setup.md`.

## Stage1 8B

- Train config: `configs/llamafactory_stage1_sft.yaml`
- Entry: `bash scripts/02_run_sft.sh stage1`

## Stage2 8B AMap Tool-Use MVP

- Merge stage1 base: `bash scripts/04_merge_stage1_for_stage2.sh`
- Build data: `bash scripts/05_build_stage2_amap_data.sh`
- Train config: `configs/llamafactory_stage2_amap_tool_use_sft.yaml`
- Dataset registration: `configs/llamafactory_dataset_info_stage2_amap_tool.json`
- Entry: `bash scripts/02_run_sft.sh stage2_amap`

## Stage1 32B

- Dataset registration: `configs/llamafactory_dataset_info_stage1_general_sft.json`
- Smoke config: `configs/llamafactory_stage1_32b_smoke_sft.yaml`
- Formal config: `configs/llamafactory_stage1_32b_formal_sft.yaml`
- Merge config: `configs/llamafactory_stage1_32b_merge_for_stage2.yaml`
- Smoke entry: `bash scripts/02_run_sft.sh stage1_32b smoke`
- Formal entry: `bash scripts/02_run_sft.sh stage1_32b formal`
- Merge stage1 base: `bash scripts/04_merge_stage1_for_stage2.sh stage1_32b`

## Stage2 32B AMap Tool-Use

- Smoke config: `configs/llamafactory_stage2_32b_amap_tool_use_smoke_sft.yaml`
- Formal config: `configs/llamafactory_stage2_32b_amap_tool_use_formal_sft.yaml`
- Merge config: `configs/llamafactory_stage2_32b_merge_for_deploy.yaml`
- Smoke entry: `bash scripts/02_run_sft.sh stage2_amap_32b smoke`
- Formal entry: `bash scripts/02_run_sft.sh stage2_amap_32b formal`
- Merge stage2 deploy model: `bash scripts/06_merge_stage2_for_deploy.sh stage2_amap_32b`

## Notes

- 32B training defaults to dual-card distributed launch through `FORCE_TORCHRUN=1` and `NPROC_PER_NODE=2`.
- Stage1 and stage2 32B configs both read datasets from `/root/llama-factory/data`.
- Keep `max_samples` only on the smoke configs. The formal 32B configs intentionally do not cap samples.

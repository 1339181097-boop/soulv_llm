# Deploy

Stage2 AMap tool execution is wired through:

- `src/tool_use/amap_client.py`: real AMap Web Service client
- `src/tool_use/orchestrator.py`: OpenAI-compatible tool orchestration loop

For stage2 deployment, the serving environment needs:

- an OpenAI-compatible chat endpoint for the model
- `AMAP_API_KEY` for real tool execution

The existing `scripts/03_run_vllm_api.sh` remains the place to put the concrete vLLM startup command.

Concrete deploy helpers now live in:

- `scripts/03_run_vllm_api.sh`: vLLM startup script with explicit tokenizer, served model name, generation config, Qwen3 reasoning parser, and auto tool-choice knobs
- `configs/llamafactory_stage2_merge_for_deploy.yaml`: merge config for stage2 deploy artifacts
- `configs/llamafactory_stage2_32b_merge_for_deploy.yaml`: merge config for the 32B stage2 deploy artifacts
- `scripts/06_merge_stage2_for_deploy.sh`: merge the stage2 LoRA into a deployable model directory
- `src/deploy/frontend_server.py`: same-origin gateway that serves the frontend, proxies `/v1/*`, and exposes `/api/tool-orchestrate`
- `src/deploy/web/index.html`: browser UI for chat, first-round tool-call checks, and end-to-end tool execution tests
- `scripts/07_run_frontend_gateway.sh`: run the public frontend gateway

The serving default is now the `32B` rollout:

1. `MODEL_VARIANT=32b`
2. `SERVED_MODEL_NAME=qwen3_32b_stage2_amap_tool_use`
3. `TENSOR_PARALLEL_SIZE=2`

If you want to keep using the old 8B merged model, override it with:

```bash
MODEL_VARIANT=8b bash scripts/03_run_vllm_api.sh
```

Recommended public topology:

1. Run vLLM on `127.0.0.1:8000`
2. Run the gateway on `0.0.0.0:7860`
3. Let other users visit `http://<server-ip>:7860`

See `docs/vllm_frontend_deployment.md` for the full step-by-step deploy guide, and `docs/qwen3_32b_stage1_stage2_rollout_guide.md` for the 32B train/eval/deploy rollout.

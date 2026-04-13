# Deploy

Stage2 AMap tool execution is wired through:

- `src/tool_use/amap_client.py`: real AMap Web Service client
- `src/tool_use/orchestrator.py`: OpenAI-compatible tool orchestration loop

For stage2 deployment, the serving environment needs:

- an OpenAI-compatible chat endpoint for the model
- `AMAP_API_KEY` for real tool execution

The existing `scripts/03_run_vllm_api.sh` remains the place to put the concrete vLLM startup command.

# Stage2 Tool Eval

`src/tool_eval/` is the dedicated evaluation track for the stage2 AMap tool-use MVP.

It is intentionally separate from `src/eval/`, which remains the stage1 natural-language regression path.

## Scope

- Native `Qwen/Qwen3-32B` 0-shot tool baseline through direct vLLM/OpenAI-compatible `tools`
- Optional Qwen-Agent baseline for comparison, not as the training data format
- Stage2 AMap tool-use golden eval
- Tool selection / argument filling / clarify / no-tool / fallback checks
- Real AMap execution through `AMAP_API_KEY`

## Workflow

1. Run native baseline:

```bash
python src/tool_eval/scripts/run_native_tool_baseline.py \
  --base-url http://<server>:8000/v1 \
  --api-key EMPTY \
  --model <served-model-name>
```

2. Optionally run the Qwen-Agent baseline:

```bash
python src/tool_eval/scripts/run_qwen_agent_baseline.py \
  --base-url http://<server>:8000/v1 \
  --api-key EMPTY \
  --model <served-model-name>
```

3. Summarize direct baseline alignment:

```bash
python src/tool_eval/scripts/analyze_native_tool_baseline.py
```

4. Run stage2 tool eval:

```bash
python src/tool_eval/scripts/run_tool_eval.py \
  --base-url http://<server>:8000/v1 \
  --api-key EMPTY \
  --model <served-model-name>
```

For a thinking-mode canary, add `--enable-thinking` and point `--dataset` at
`src/tool_eval/datasets/stage2_amap_thinking_canary.json`.

5. Score stage2 tool eval:

```bash
python src/tool_eval/scripts/score_tool_eval.py
```

6. Re-run stage1 regression using the existing runner in [run_eval.py](/d:/soulv_llm/src/eval/scripts/run_eval.py).

## Release Gates

- Tool selection accuracy `>= 0.90`
- Argument accuracy `>= 0.85`
- No-tool accuracy `>= 0.90`
- Fallback accuracy `>= 0.85`
- Execution success rate `>= 0.90`
- Re-run stage1 eval and ensure the overall pass rate does not drop by more than `5` percentage points

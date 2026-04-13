# Stage2 Tool Eval

`src/tool_eval/` is the dedicated evaluation track for the stage2 AMap tool-use MVP.

It is intentionally separate from `src/eval/`, which remains the stage1 natural-language regression path.

## Scope

- Native `Qwen3-8B-Instruct` 0-shot tool baseline
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

2. Summarize baseline alignment:

```bash
python src/tool_eval/scripts/analyze_native_tool_baseline.py
```

3. Run stage2 tool eval:

```bash
python src/tool_eval/scripts/run_tool_eval.py \
  --base-url http://<server>:8000/v1 \
  --api-key EMPTY \
  --model <served-model-name>
```

4. Score stage2 tool eval:

```bash
python src/tool_eval/scripts/score_tool_eval.py
```

5. Re-run stage1 regression using the existing runner in [run_eval.py](/d:/soulv_llm/src/eval/scripts/run_eval.py).

## Release Gates

- Tool selection accuracy `>= 0.85`
- Argument accuracy `>= 0.80`
- Execution success rate `>= 0.90`
- Re-run stage1 eval and ensure the overall pass rate does not drop by more than `5` percentage points

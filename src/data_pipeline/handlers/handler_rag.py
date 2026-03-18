from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.handlers.handler_intent import (
    DEFAULT_INPUT_PATH as DEFAULT_INTENT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_INTENT_OUTPUT_PATH,
    build_intent_sample,
    main as main,
    process_intent_data,
)

DEFAULT_INPUT_PATH = DEFAULT_INTENT_INPUT_PATH
DEFAULT_OUTPUT_PATH = DEFAULT_INTENT_OUTPUT_PATH
process_rag_data = process_intent_data
build_rag_sample = build_intent_sample

__all__ = [
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "process_rag_data",
    "build_rag_sample",
]


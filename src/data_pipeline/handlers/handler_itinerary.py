from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.handlers.handler_guide_generation import (  # noqa: F401
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PREPARED_RAW_PATH,
    DEFAULT_SOURCE_RAW_PATH,
    build_guide_generation_sample,
    build_itinerary_sample,
    main,
    prepare_guide_generation_raw,
    process_guide_generation_data,
    process_itinerary_data,
)

__all__ = [
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_PREPARED_RAW_PATH",
    "DEFAULT_SOURCE_RAW_PATH",
    "build_guide_generation_sample",
    "build_itinerary_sample",
    "prepare_guide_generation_raw",
    "process_guide_generation_data",
    "process_itinerary_data",
    "main",
]


if __name__ == "__main__":
    main()

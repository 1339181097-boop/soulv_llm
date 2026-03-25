from __future__ import annotations

from pathlib import Path


DOC_SYS_PROMPT_PATH = Path(__file__).resolve().parents[2] / "docs" / "sys_prompt.md"


def load_system_prompt(section_name: str, fallback: str) -> str:
    try:
        text = DOC_SYS_PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        return fallback

    current_section: str | None = None
    buffer: list[str] = []
    sections: dict[str, str] = {}

    for line in text.splitlines():
        if line.startswith("## "):
            if current_section is not None:
                sections[current_section] = "\n".join(buffer).strip()
            current_section = line[3:].strip()
            buffer = []
            continue
        buffer.append(line)

    if current_section is not None:
        sections[current_section] = "\n".join(buffer).strip()

    return sections.get(section_name, fallback)

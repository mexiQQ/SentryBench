"""Lightweight IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    rows = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Write a JSON object to disk with indentation."""
    Path(path).write_text(json.dumps(payload, indent=2))


from __future__ import annotations

from pathlib import Path
import json
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML config files.") from exc
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    raise ValueError("Unsupported config format. Use YAML or JSON.")


def resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


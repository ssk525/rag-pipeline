"""
Persistent pipeline configuration for retrieval tuning and self-improvement.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "pipeline_config.json")
HISTORY_PATH = os.path.join(CONFIG_DIR, "improvement_history.json")
FEEDBACK_PATH = os.path.join(CONFIG_DIR, "feedback.jsonl")


@dataclass
class PipelineConfig:
    top_k: int = 5
    use_hybrid: bool = True
    use_reranker: bool = True
    chunking_strategy: str = "semantic"
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_eval_score: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


def load_config() -> PipelineConfig:
    if not os.path.exists(CONFIG_PATH):
        return PipelineConfig()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return PipelineConfig.from_dict(json.load(f))


def save_config(config: PipelineConfig) -> PipelineConfig:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    config.updated_at = datetime.now(timezone.utc).isoformat()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    return config


def append_improvement_record(record: Dict[str, Any]) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    history: List[Dict[str, Any]] = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append(record)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history[-50:], f, indent=2)


def append_feedback(record: Dict[str, Any]) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

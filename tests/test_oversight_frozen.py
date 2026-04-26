from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from oversight.contracts import compiler_artifact_hash, compiler_artifact_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = REPO_ROOT / "configs" / "system"
THRESHOLD_PATH = REPO_ROOT / "configs" / "shopping" / "oversight_thresholds.yaml"
EXPECTED_THRESHOLD_HASH = "5534b9477a27736f"
EXPECTED_COMPILER_ARTIFACT_HASH = "2f0c015e26f9a335"
FROZEN_PROMPT_VERSION = "c2-lite-v1.4-frozen"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def test_frozen_threshold_file_hash_matches_expected():
    payload = _load_yaml(THRESHOLD_PATH)

    assert payload["loop_similarity_threshold"] == 0.92
    assert payload["loop_window"] == 5
    assert payload["loop_repeat_count"] == 3
    assert payload["coverage_threshold"] == 0.50
    assert payload["frozen_at"] == "2026-04-25"
    assert _canonical_hash(payload) == EXPECTED_THRESHOLD_HASH


def test_compiler_artifact_hash_matches_expected():
    payload = compiler_artifact_payload()

    assert "footwear collection" not in payload["P1_SYSTEM_PROMPT"]
    assert "summer outfit" not in payload["P1_SYSTEM_PROMPT"]
    assert "running gear" not in payload["P1_SYSTEM_PROMPT"]
    assert compiler_artifact_hash() == EXPECTED_COMPILER_ARTIFACT_HASH


def test_all_oversight_enabled_systems_use_frozen_prompt_version():
    for path in sorted(SYSTEM_ROOT.glob("*.yaml")):
        payload = _load_yaml(path)
        if not payload.get("oversight_enabled", False):
            continue
        assert payload.get("overseer_prompt_version") == FROZEN_PROMPT_VERSION


def test_all_system_configs_set_oversight_domains_to_shopping():
    for path in sorted(SYSTEM_ROOT.glob("*.yaml")):
        payload = _load_yaml(path)
        assert payload.get("oversight_domains") == ["shopping"]

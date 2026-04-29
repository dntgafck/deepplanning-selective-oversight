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
EXPECTED_THRESHOLD_HASH = "a839c711d247c542"
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
    assert payload["coverage_threshold"] == 0.60
    assert payload["frozen_at"] == "2026-04-29"
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


def test_c2_family_system_thresholds_match_frozen_threshold_artifact():
    thresholds = _load_yaml(THRESHOLD_PATH)

    for system_name in ("C2", "C2-noretry", "C2-nt"):
        payload = _load_yaml(SYSTEM_ROOT / f"{system_name}.yaml")
        assert (
            payload["loop_similarity_threshold"]
            == thresholds["loop_similarity_threshold"]
        )
        assert payload["loop_window"] == thresholds["loop_window"]
        assert payload["loop_repeat_count"] == thresholds["loop_repeat_count"]
        assert payload["coverage_threshold"] == thresholds["coverage_threshold"]


def test_calibration_log_records_reproducible_freeze_state():
    log_path = REPO_ROOT / "docs" / "calibration_log.md"
    text = log_path.read_text(encoding="utf-8")

    assert "2026-04-29" in text
    assert "9ab07096c194ce68913a2defca96e8feccb40217" in text
    assert "Dirty worktree at freeze documentation time: yes" in text
    assert "configs/shopping/splits.yaml" in text
    assert "5b7aa44e4381f153" in text
    assert "c2-lite-v1.4-frozen" in text
    assert "pixi run python scripts/run_shopping_test_split.py --dry-run" in text

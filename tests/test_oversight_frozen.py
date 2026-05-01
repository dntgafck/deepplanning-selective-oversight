from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from oversight.contracts import compiler_artifact_hash, compiler_artifact_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = REPO_ROOT / "configs" / "system"
EXPECTED_COMPILER_ARTIFACT_HASH = "2f0c015e26f9a335"
FROZEN_PROMPT_VERSION = "c2-lite-v1.4-frozen"
EXPECTED_SYSTEM_THRESHOLDS = {
    "loop_similarity_threshold": 0.92,
    "loop_window": 5,
    "loop_repeat_count": 3,
    "coverage_threshold": 0.60,
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


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


def test_system_threshold_defaults_match_expected_values():
    for path in sorted(SYSTEM_ROOT.glob("*.yaml")):
        payload = _load_yaml(path)
        for key, expected_value in EXPECTED_SYSTEM_THRESHOLDS.items():
            assert payload[key] == expected_value


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

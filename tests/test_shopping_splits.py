from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO_ROOT / "configs" / "shopping" / "splits.yaml"
SHOPPING_DATA_ROOT = (
    REPO_ROOT
    / "external"
    / "qwen-agent"
    / "benchmark"
    / "deepplanning"
    / "shoppingplanning"
    / "data"
)
EXPECTED_SPLIT_HASH = "5b7aa44e4381f153"
EXPECTED_COUNTS = {
    "level_1": {"tune": 13, "test": 37, "total": 50},
    "level_2": {"tune": 12, "test": 38, "total": 50},
    "level_3": {"tune": 5, "test": 15, "total": 20},
}


def _load_split() -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(SPLIT_PATH), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _metadata_ids(level_key: str) -> set[str]:
    level = level_key.removeprefix("level_")
    path = SHOPPING_DATA_ROOT / f"level_{level}_query_meta.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    return {str(record["id"]) for record in records}


def test_shopping_split_file_is_disjoint_and_exhaustive():
    split = _load_split()
    seen_pairs: set[tuple[str, str]] = set()

    for level_key in EXPECTED_COUNTS:
        tune_ids = {str(item) for item in split["tune_split"][level_key]}
        test_ids = {str(item) for item in split["test_split"][level_key]}
        metadata_ids = _metadata_ids(level_key)

        assert tune_ids.isdisjoint(test_ids)
        assert tune_ids | test_ids == metadata_ids
        for sample_id in tune_ids | test_ids:
            pair = (level_key, sample_id)
            assert pair not in seen_pairs
            seen_pairs.add(pair)

    assert sum(len(split["tune_split"][level]) for level in EXPECTED_COUNTS) == 30
    assert sum(len(split["test_split"][level]) for level in EXPECTED_COUNTS) == 90
    assert _canonical_hash(split) == EXPECTED_SPLIT_HASH


def test_shopping_split_has_expected_stratified_counts():
    split = _load_split()

    for level_key, counts in EXPECTED_COUNTS.items():
        tune_ids = split["tune_split"][level_key]
        test_ids = split["test_split"][level_key]
        assert len(tune_ids) == counts["tune"]
        assert len(test_ids) == counts["test"]
        assert len(tune_ids) + len(test_ids) == counts["total"]
        assert all(isinstance(sample_id, str) for sample_id in tune_ids)
        assert all(isinstance(sample_id, str) for sample_id in test_ids)


def test_shopping_split_ids_exist_in_benchmark_metadata():
    split = _load_split()

    for level_key in EXPECTED_COUNTS:
        metadata_ids = _metadata_ids(level_key)
        for split_key in ("tune_split", "test_split"):
            split_ids = {str(item) for item in split[split_key][level_key]}
            assert split_ids <= metadata_ids

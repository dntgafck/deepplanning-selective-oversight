from __future__ import annotations

import pytest

from experiment import build_system_config
from oversight.contracts import parse_task_checklist_json
from oversight.domain_config import load_product_type_hints


def _checklist_payload(description: str = "running shoe from Nike"):
    return {
        "checklist_id": "checklist-1",
        "items": [
            {
                "key": "shoe_item",
                "category": "required_product",
                "description": description,
                "value": {"name": "Nike runner", "brand": "Nike"},
                "required": True,
                "explicit": True,
                "coverage_relevant": True,
                "final_verify_only": False,
                "aliases": [],
                "source_text": "Need a running shoe from Nike.",
            }
        ],
        "coverage_targets": [],
        "final_verification_only_keys": [],
        "ambiguities": [],
        "compiler_signature": "sig",
    }


def test_checklist_validation_works_with_product_type_hints_enabled():
    checklist = parse_task_checklist_json(
        _checklist_payload(),
        task_query="Find a running shoe from Nike.",
        product_type_hints=load_product_type_hints("shopping"),
    )

    item = checklist.items[0]
    assert item["value"].get("product_type") in (None, "")
    assert "running shoe" in item["value"].get("product_type_hints_soft", [])


def test_checklist_validation_works_with_product_type_hints_disabled():
    checklist = parse_task_checklist_json(
        _checklist_payload(),
        task_query="Find a running shoe from Nike.",
        product_type_hints=(),
    )

    item = checklist.items[0]
    assert item["value"].get("product_type") in (None, "")
    assert item["value"].get("product_type_hints_soft", []) == []


def test_current_shopping_footwear_normalization_uses_enabled_hint_file():
    hints = load_product_type_hints("shopping")

    assert ("running shoe", "running shoe") in hints

    checklist = parse_task_checklist_json(
        _checklist_payload("Trail running shoe for hiking"),
        task_query="Need a trail running shoe.",
        product_type_hints=hints,
    )

    aliases = [alias.lower() for alias in checklist.items[0].get("aliases", [])]
    assert "running shoe" in aliases


def test_system_config_can_disable_product_type_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "experiment.config.load_system_defaults",
        lambda _system_name: {
            "name": "C2-no-hints",
            "oversight_enabled": True,
            "oversight_profile": "adaptive_risk",
            "oversight_domains": ["shopping"],
            "product_type_hints_enabled": False,
        },
    )

    config = build_system_config("C2-no-hints", executor_model="qwen3.5-9b")

    assert config.product_type_hints_enabled is False
    assert config.product_type_hints == ()

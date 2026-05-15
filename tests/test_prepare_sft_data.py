from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import prepare_sft_data as prep


class FakeTokenizer:
    def __init__(self) -> None:
        self.enable_thinking_values: list[bool] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        self.enable_thinking_values.append(enable_thinking)
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text.split())))}


def _pair(
    pair_id: str,
    *,
    task_key: str = "level_1:1",
    target_text: str = '{"action":"approve"}',
    user_text: str = "short prompt",
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "input_messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": user_text},
        ],
        "target_text": target_text,
        "decision_label": "approve",
        "source_system": "C2-nt",
        "level": 1,
        "hook": "pre_tool",
        "task_key": task_key,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_input_dir(tmp_path: Path) -> Path:
    input_dir = tmp_path / "lora_sft"
    input_dir.mkdir()
    (input_dir / "task_split_v1.json").write_text(
        json.dumps({"splits": {"held_out": {"level_1": ["99"]}}}),
        encoding="utf-8",
    )
    (input_dir / "held_out.jsonl").write_text("not opened\n", encoding="utf-8")
    return input_dir


def test_prepare_sft_data_converts_train_val_and_never_opens_held_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = _make_input_dir(tmp_path)
    out_dir = tmp_path / "out"
    fake_tokenizer = FakeTokenizer()
    monkeypatch.setattr(prep, "_load_tokenizer", lambda base_model: fake_tokenizer)

    original_open = Path.open

    def guarded_open(self: Path, *args: object, **kwargs: object):
        if self.name == "held_out.jsonl":
            raise AssertionError("held_out.jsonl must not be opened")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    _write_jsonl(
        input_dir / "train.jsonl",
        [
            _pair("kept-train"),
            _pair("too-long", user_text=" ".join(["long"] * 30)),
        ],
    )
    _write_jsonl(input_dir / "val.jsonl", [_pair("kept-val", task_key="level_1:2")])

    prep.main(
        in_dir=str(input_dir),
        out_dir=str(out_dir),
        max_seq_len=12,
        base_model="Qwen/Qwen3.5-9B",
    )

    train_rows = [
        json.loads(line)
        for line in (out_dir / "train_swift.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    val_rows = [
        json.loads(line)
        for line in (out_dir / "val_swift.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    report = json.loads(
        (out_dir / "stratification_report.json").read_text(encoding="utf-8")
    )
    length_report = json.loads(
        (out_dir / "length_filter_report.json").read_text(encoding="utf-8")
    )

    assert [row["pair_id"] for row in train_rows] == ["kept-train"]
    assert [row["pair_id"] for row in val_rows] == ["kept-val"]
    assert train_rows[0]["messages"][-1] == {
        "role": "assistant",
        "content": '{"action":"approve"}',
    }
    assert report["train"]["in"] == 2
    assert report["train"]["out"] == 1
    assert report["train"]["dropped_too_long"] == 1
    assert report["train"]["label_dist"] == {"approve": 1}
    assert length_report["train"]["dropped_too_long"] == 1
    assert fake_tokenizer.enable_thinking_values
    assert set(fake_tokenizer.enable_thinking_values) == {False}


def test_prepare_sft_data_rejects_held_out_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = _make_input_dir(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(prep, "_load_tokenizer", lambda base_model: FakeTokenizer())
    _write_jsonl(input_dir / "train.jsonl", [_pair("leak", task_key="level_1:99")])
    _write_jsonl(input_dir / "val.jsonl", [_pair("kept-val", task_key="level_1:2")])

    with pytest.raises(RuntimeError, match="held-out task_keys leaked"):
        prep.main(in_dir=str(input_dir), out_dir=str(out_dir), max_seq_len=100)


def test_prepare_sft_data_fire_help_does_not_load_transformers() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/prepare_sft_data.py", "--help"],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    help_text = result.stdout + result.stderr
    assert "IN_DIR" in help_text
    assert "OUT_DIR" in help_text
    assert "--max_seq_len" in help_text

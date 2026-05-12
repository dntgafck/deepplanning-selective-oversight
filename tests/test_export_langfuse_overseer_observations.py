from __future__ import annotations

import json
from pathlib import Path

import httpx

from scripts import export_langfuse_overseer_observations as exporter


def _write_session(root: Path) -> Path:
    session_root = root / "shopping-c2-nt" / "2026-04-30_13-35-38"
    session_root.mkdir(parents=True)
    metadata = {
        "experiment": {"name": "shopping-c2-nt"},
        "timestamp": "2026-04-30_13-35-38",
        "parameters": {
            "name": "shopping-c2-nt",
            "domains": ["shopping"],
            "models": {
                "executor": "qwen3.5-9b",
                "overseer": "deepseek-v4-flash-nt",
            },
            "system": {"name": "C2-nt"},
            "runtime": {"runs": 4, "workers": 50, "max_llm_calls": 400},
            "shopping": {"split": "all"},
        },
    }
    (session_root / "experiment_session.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return session_root


def _observation(
    observation_id: str,
    *,
    name: str,
    hook: str,
    trigger_type: str,
    actor: str = "overseer",
) -> dict[str, object]:
    return {
        "id": observation_id,
        "traceId": f"trace-{observation_id}",
        "sessionId": "2026-04-30_13-35-38",
        "name": name,
        "model": "deepseek-v4-flash",
        "metadata": {
            "actor": actor,
            "task_id": "1",
            "level": 1,
            "run_id": 0,
            "phase": "initial",
            "step_index": 1,
            "tool_index": None,
            "hook": hook,
            "trigger_type": trigger_type,
        },
        "usageDetails": {"input": 10, "output": 3},
        "input": {"messages": [{"role": "system", "content": "prompt"}]},
        "output": {
            "choices": [
                {"message": {"content": '{"action":"approve","decision_summary":""}'}}
            ]
        },
    }


def test_filter_includes_runtime_final_and_excludes_setup():
    runtime = _observation(
        "runtime",
        name="overseer.pre_tool.step_001.mutating_action",
        hook="pre_tool",
        trigger_type="mutating_action",
    )
    setup = _observation(
        "setup",
        name="overseer.compile_contract",
        hook="setup",
        trigger_type="compile_contract",
    )
    executor = _observation(
        "executor",
        name="executor.initial.step_001",
        hook="",
        trigger_type="",
        actor="executor",
    )

    assert exporter.is_runtime_or_final_overseer_observation(runtime)
    assert not exporter.is_runtime_or_final_overseer_observation(setup)
    assert not exporter.is_runtime_or_final_overseer_observation(executor)


def test_extract_input_messages_accepts_langfuse_json_string():
    messages = [{"role": "system", "content": "prompt"}]

    assert exporter.extract_input_messages({"input": json.dumps(messages)}) == messages


def test_export_sessions_filters_and_writes_manifest(tmp_path):
    session_root = _write_session(tmp_path / "outputs")
    observations = [
        _observation(
            "runtime",
            name="overseer.pre_tool.step_001.mutating_action",
            hook="pre_tool",
            trigger_type="mutating_action",
        ),
        _observation(
            "setup",
            name="overseer.compile_checklist",
            hook="setup",
            trigger_type="compile_checklist",
        ),
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"data": observations, "meta": {"cursor": None}},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        manifest = exporter.export_sessions(
            session_roots=[session_root],
            output_dir=tmp_path / "exports",
            host="https://cloud.langfuse.com/",
            public_key="pk-test",
            secret_key="sk-test",
            client=client,
        )

    assert len(requests) == 1
    params = requests[0].url.params
    assert params["fields"] == exporter.DEFAULT_FIELDS
    assert json.loads(params["filter"])[0]["value"] == "2026-04-30_13-35-38"

    session = manifest["sessions"][0]
    assert session["source_system"] == "C2-nt"
    assert session["raw_observation_count"] == 2
    assert session["exported_observation_count"] == 1
    assert session["excluded_oversize_count"] == 0
    export_rows = [
        json.loads(line)
        for line in Path(session["jsonl_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert export_rows[0]["observation_id"] == "runtime"
    assert export_rows[0]["input_messages"] == [{"role": "system", "content": "prompt"}]
    assert "metadata_inventory_path" not in manifest


def test_fetch_observations_paginates_with_session_filter():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        cursor = request.url.params.get("cursor")
        if cursor == "next-page":
            return httpx.Response(
                200,
                json={"data": [{"id": "second"}], "meta": {"cursor": None}},
                request=request,
            )
        return httpx.Response(
            200,
            json={"data": [{"id": "first"}], "meta": {"cursor": "next-page"}},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        observations = exporter.fetch_observations(
            session_id="bench-session",
            host="https://cloud.langfuse.com/",
            public_key="pk-test",
            secret_key="sk-test",
            limit=1,
            client=client,
        )

    assert observations == [{"id": "first"}, {"id": "second"}]
    assert len(requests) == 2
    assert requests[0].url.params["fields"] == exporter.DEFAULT_FIELDS
    assert json.loads(requests[0].url.params["filter"])[0]["value"] == "bench-session"
    assert requests[1].url.params["cursor"] == "next-page"

from __future__ import annotations

import json

import httpx

from scripts import fetch_langfuse_session_usage as langfuse_usage


def test_usage_row_from_observation_extracts_cached_tokens():
    row = langfuse_usage.usage_row_from_observation(
        {
            "providedModelName": "deepseek-v4-flash",
            "usageDetails": {
                "input": 100,
                "cache_read_input_tokens": 35,
                "output": 20,
                "total": 120,
            },
        }
    )

    assert row == {
        "model": "deepseek-v4-flash",
        "input": 100,
        "input_cached_tokens": 35,
        "output": 20,
        "total": 120,
    }


def test_summarize_usage_groups_by_model():
    summary = langfuse_usage.summarize_usage(
        [
            {
                "model": "qwen3.5-9b",
                "input": 10,
                "input_cached_tokens": 0,
                "output": 5,
                "total": 15,
            },
            {
                "model": "qwen3.5-9b",
                "input": 20,
                "input_cached_tokens": 4,
                "output": 6,
                "total": 26,
            },
            {
                "model": "deepseek-v4-flash",
                "input": 30,
                "input_cached_tokens": 12,
                "output": 9,
                "total": 39,
            },
        ]
    )

    assert summary.to_dict("records") == [
        {
            "model": "deepseek-v4-flash",
            "input": 30,
            "input_cached_tokens": 12,
            "output": 9,
            "total": 39,
        },
        {
            "model": "qwen3.5-9b",
            "input": 30,
            "input_cached_tokens": 4,
            "output": 11,
            "total": 41,
        },
    ]


def test_fetch_session_observations_filters_and_paginates():
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

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        observations = langfuse_usage.fetch_session_observations(
            session_id="bench-session",
            host="https://cloud.langfuse.com/",
            public_key="pk-test",
            secret_key="sk-test",
            limit=2,
            client=client,
        )

    assert observations == [{"id": "first"}, {"id": "second"}]
    assert len(requests) == 2

    first_params = requests[0].url.params
    assert first_params["fields"] == "core,basic,model,usage"
    assert first_params["limit"] == "2"
    assert "cursor" not in first_params
    assert json.loads(first_params["filter"]) == [
        {
            "column": "sessionId",
            "operator": "=",
            "value": "bench-session",
            "type": "string",
        }
    ]

    assert requests[0].url.path == "/api/public/v2/observations"
    assert requests[1].url.params["cursor"] == "next-page"

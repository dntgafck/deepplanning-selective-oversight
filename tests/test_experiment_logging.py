from __future__ import annotations

from types import SimpleNamespace

from experiment.logging import serialize_response


class FakeDumpResponse:
    def model_dump(self, *, exclude_none: bool = False):
        assert exclude_none is True
        return {
            "id": "resp_1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "Hello",
                                "logprob": -0.001,
                                "top_logprobs": [
                                    {"token": "Hello", "logprob": -0.001},
                                    {"token": "Hi", "logprob": -8.0},
                                ],
                            }
                        ]
                    },
                }
            ],
        }


def test_serialize_response_preserves_model_dump_payload():
    payload = serialize_response(FakeDumpResponse())

    assert payload["choices"][0]["logprobs"]["content"][0]["token"] == "Hello"
    assert (
        payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["token"]
        == "Hi"
    )


class FakeUsage:
    def __init__(self) -> None:
        self.prompt_tokens = 2506
        self.completion_tokens = 80
        self.total_tokens = 2586
        self.prompt_cache_hit_tokens = 2496
        self.prompt_cache_miss_tokens = 10
        self.prompt_tokens_details = SimpleNamespace(cached_tokens=2496)
        self.completion_tokens_details = SimpleNamespace(reasoning_tokens=80)


class FakeMessageResponse:
    def __init__(self) -> None:
        self.role = "assistant"
        self.content = "done"


class FakeChoiceResponse:
    def __init__(self) -> None:
        self.index = 0
        self.finish_reason = "stop"
        self.message = FakeMessageResponse()


class FakeResponse:
    def __init__(self) -> None:
        self.id = "resp_1"
        self.model = "deepseek-reasoner"
        self.system_fingerprint = "fp_test"
        self.choices = [FakeChoiceResponse()]
        self.usage = FakeUsage()


def test_serialize_response_includes_cache_related_usage_fields():
    payload = serialize_response(FakeResponse())

    assert payload["usage"] == {
        "prompt_tokens": 2506,
        "completion_tokens": 80,
        "total_tokens": 2586,
        "prompt_cache_hit_tokens": 2496,
        "prompt_cache_miss_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 2496},
        "completion_tokens_details": {"reasoning_tokens": 80},
    }

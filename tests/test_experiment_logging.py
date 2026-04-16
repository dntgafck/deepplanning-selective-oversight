from __future__ import annotations

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
    assert payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1][
        "token"
    ] == "Hi"

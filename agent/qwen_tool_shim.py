from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

TOOL_REGISTRY: dict[str, type] = {}


def register_tool(name: str, allow_overwrite: bool = False):
    def decorator(cls: type) -> type:
        if name in TOOL_REGISTRY and not allow_overwrite:
            raise ValueError(f"Tool `{name}` already exists")
        cls.name = name
        TOOL_REGISTRY[name] = cls
        return cls

    return decorator


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: list[dict[str, Any]] | dict[str, Any] = []

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}
        if not self.name:
            raise ValueError(
                f"{self.__class__.__name__} must define a tool name before initialization"
            )

    @abstractmethod
    def call(self, params: str | dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError

    def _verify_json_format_args(
        self, params: str | dict[str, Any], strict_json: bool = False
    ) -> dict[str, Any]:
        if isinstance(params, str):
            try:
                params_json = json.loads(params)
            except json.JSONDecodeError as exc:
                raise ValueError("Parameters must be valid JSON") from exc
        else:
            params_json = params

        if isinstance(self.parameters, list):
            for param in self.parameters:
                if param.get("required") and param.get("name") not in params_json:
                    raise ValueError(f"Parameter {param['name']} is required")
        elif isinstance(self.parameters, dict):
            for required_key in self.parameters.get("required", []):
                if required_key not in params_json:
                    raise ValueError(f"Parameter {required_key} is required")

        return params_json

    @property
    def function(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

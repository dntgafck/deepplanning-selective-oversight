from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType

from scripts.deepplanning_common import BENCHMARK_ROOT, REPO_ROOT

SHOPPING_ROOT = BENCHMARK_ROOT / "shoppingplanning"
TRAVEL_ROOT = BENCHMARK_ROOT / "travelplanning"
QWEN_AGENT_ROOT = REPO_ROOT / "external" / "qwen-agent"

_LOAD_LOCK = threading.Lock()


def _ensure_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def clear_vendored_tool_module_cache() -> None:
    """Drop ambiguous top-level tool modules before switching benchmark domains."""
    prefixes = ("tools", "base_shopping_tool", "base_travel_tool")
    for module_name in list(sys.modules):
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in prefixes
        ):
            sys.modules.pop(module_name, None)

    qwen_tool_registry = getattr(
        sys.modules.get("qwen_agent.tools.base"), "TOOL_REGISTRY", None
    )
    if isinstance(qwen_tool_registry, dict):
        qwen_tool_registry.clear()


def _install_qwen_tool_shim() -> None:
    if "qwen_agent.tools.base" in sys.modules:
        return

    from . import qwen_tool_shim

    qwen_agent_module = sys.modules.get("qwen_agent")
    if qwen_agent_module is None:
        qwen_agent_module = ModuleType("qwen_agent")
        qwen_agent_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["qwen_agent"] = qwen_agent_module

    qwen_tools_module = sys.modules.get("qwen_agent.tools")
    if qwen_tools_module is None:
        qwen_tools_module = ModuleType("qwen_agent.tools")
        qwen_tools_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["qwen_agent.tools"] = qwen_tools_module

    base_module = ModuleType("qwen_agent.tools.base")
    base_module.BaseTool = qwen_tool_shim.BaseTool
    base_module.TOOL_REGISTRY = qwen_tool_shim.TOOL_REGISTRY
    base_module.register_tool = qwen_tool_shim.register_tool

    sys.modules["qwen_agent.tools.base"] = base_module
    qwen_agent_module.tools = qwen_tools_module
    qwen_tools_module.base = base_module
    qwen_tools_module.BaseTool = qwen_tool_shim.BaseTool
    qwen_tools_module.TOOL_REGISTRY = qwen_tool_shim.TOOL_REGISTRY


def _load_module(
    module_name: str, module_path: Path, extra_paths: tuple[Path, ...] = ()
) -> ModuleType:
    with _LOAD_LOCK:
        for path in extra_paths:
            _ensure_path(path)

        cached = sys.modules.get(module_name)
        if cached is not None:
            return cached

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module {module_name} from {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


def load_shopping_agent_class() -> type:
    module = _load_module(
        "deepplanning_vendor_shopping_agent",
        SHOPPING_ROOT / "agent" / "shopping_agent.py",
        extra_paths=(SHOPPING_ROOT / "agent", SHOPPING_ROOT / "tools"),
    )
    return module.ShoppingFnAgent


def load_travel_agent_class() -> type:
    _install_qwen_tool_shim()
    module = _load_module(
        "deepplanning_vendor_travel_agent",
        TRAVEL_ROOT / "agent" / "tools_fn_agent.py",
        extra_paths=(QWEN_AGENT_ROOT, TRAVEL_ROOT / "agent", TRAVEL_ROOT / "tools"),
    )
    return module.ToolsFnAgent


def load_shopping_prompt(level: int) -> str:
    module = _load_module(
        "deepplanning_vendor_shopping_prompts",
        SHOPPING_ROOT / "agent" / "prompts.py",
    )
    return getattr(module.prompt_lib, f"SYSTEM_PROMPT_level{level}")


def load_travel_prompt(language: str) -> str:
    module = _load_module(
        "deepplanning_vendor_travel_prompts",
        TRAVEL_ROOT / "agent" / "prompts.py",
    )
    return module.get_system_prompt(language)

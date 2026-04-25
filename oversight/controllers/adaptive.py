from __future__ import annotations

from ..base import OversightContext, OversightController
from .review import (
    _evaluate_final,
    _evaluate_midpoint,
    _evaluate_post_tool,
    _evaluate_pre_tool,
)


class AdaptiveRiskOversight(OversightController):
    profile = "adaptive_risk"
    controller_name = "AdaptiveRiskOversight"
    active_hooks = frozenset({"pre_tool", "post_tool", "midpoint", "final"})

    async def evaluate(self, context: OversightContext):
        if not self.is_active_for_hook(
            state=context.state,
            system_config=context.system_config,
            hook=context.hook,
        ):
            return self.noop_action(system_config=context.system_config)
        if context.hook == "pre_tool":
            return await _evaluate_pre_tool(context, always_on=False)
        if context.hook == "post_tool":
            return await _evaluate_post_tool(context, always_on=False)
        if context.hook == "midpoint":
            return await _evaluate_midpoint(context)
        if context.hook == "final":
            return await _evaluate_final(context)
        return self.noop_action(system_config=context.system_config)

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import OversightContext, OversightController

if TYPE_CHECKING:
    from .. import OversightAction


class ExecutorOnlyOversight(OversightController):
    profile = "executor_only"
    controller_name = "ExecutorOnlyOversight"

    async def evaluate(self, context: OversightContext) -> OversightAction:
        return self.noop_action(system_config=context.system_config)

from __future__ import annotations

from types import SimpleNamespace

from oversight import ConversationState, H1Outcome, OversightAction, compute_h1_outcome


def _state(*, consecutive_blocks: int = 0) -> ConversationState:
    state = ConversationState(
        task_id="T",
        domain="shopping",
        complexity=1,
        system_config_name="streak-cap-test",
    )
    state.consecutive_pre_tool_blocks = consecutive_blocks
    return state


def _config(
    *,
    block_on_mutation_mode: str = "auto",
    max_hard_blocks_per_args: int = 2,
    max_consecutive_pre_tool_blocks: int = 5,
) -> SimpleNamespace:
    return SimpleNamespace(
        mutating_tools=("add_product_to_cart",),
        irreversible_tools=(),
        block_on_mutation_mode=block_on_mutation_mode,
        max_hard_blocks_per_args=max_hard_blocks_per_args,
        max_consecutive_pre_tool_blocks=max_consecutive_pre_tool_blocks,
        require_cited_violation_for_block=True,
    )


def _blocking_action() -> OversightAction:
    return OversightAction(
        intervention_type="provide_guidance",
        violated_contract_ids=["rule-1"],
        unmet_checklist_keys=[],
        violation_confidence="high",
    )


def test_streak_cap_does_not_trip_below_threshold():
    outcome = compute_h1_outcome(
        action=_blocking_action(),
        tool_name="add_product_to_cart",
        arguments={"product_id": "p1"},
        state=_state(consecutive_blocks=4),
        system_config=_config(block_on_mutation_mode="always"),
    )

    assert outcome == H1Outcome.HARD_BLOCK


def test_streak_cap_trips_at_threshold_in_auto_mode():
    outcome = compute_h1_outcome(
        action=_blocking_action(),
        tool_name="add_product_to_cart",
        arguments={"product_id": "p1"},
        state=_state(consecutive_blocks=5),
        system_config=_config(block_on_mutation_mode="auto"),
    )

    assert outcome == H1Outcome.FORCED_APPROVE


def test_streak_cap_trips_in_always_mode_before_other_logic():
    outcome = compute_h1_outcome(
        action=_blocking_action(),
        tool_name="add_product_to_cart",
        arguments={"product_id": "p1"},
        state=_state(consecutive_blocks=5),
        system_config=_config(block_on_mutation_mode="always"),
    )

    assert outcome == H1Outcome.FORCED_APPROVE


def test_streak_cap_disabled_by_zero_threshold():
    outcome = compute_h1_outcome(
        action=_blocking_action(),
        tool_name="add_product_to_cart",
        arguments={"product_id": "p1"},
        state=_state(consecutive_blocks=100),
        system_config=_config(
            block_on_mutation_mode="always",
            max_consecutive_pre_tool_blocks=0,
        ),
    )

    assert outcome == H1Outcome.HARD_BLOCK


def test_streak_cap_only_applies_to_blocking_guidance():
    outcome = compute_h1_outcome(
        action=OversightAction(intervention_type="approve"),
        tool_name="add_product_to_cart",
        arguments={"product_id": "p1"},
        state=_state(consecutive_blocks=10),
        system_config=_config(block_on_mutation_mode="always"),
    )

    assert outcome == H1Outcome.APPROVE_CONTINUE

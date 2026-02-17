from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from game_logic import Card, Suit, TieRevealEvent
from poker_ml import PolicyNet, build_draw_action_mask, build_obs_draw, build_obs_trick, sample_categorical


def draw_action_to_mask_and_reject(action: int) -> Tuple[int, bool]:
    """Decode draw action bit-field into discard mask and reject flag."""
    discard_mask = action & 0b1_1111
    reject_single_draw = ((action >> 5) & 1) == 1
    return discard_mask, reject_single_draw


def choose_bot_draw_action(
    model: PolicyNet,
    hand: Sequence[Card],
    phase: int,
    deck_cards_left: int,
    device: torch.device,
    scores: Sequence[int],
    last_round_scores: Sequence[int],
    n_players: int,
    self_player: int,
    target_score: int,
    showdown_starter: Optional[int],
    table_history: Sequence[Sequence[Optional[Card]]],
    discard_history: Sequence[Sequence[Sequence[Card]]],
    shown_history: Sequence[Optional[Tuple[Card, bool]]],
    announced_points: Sequence[Optional[int]],
    passed: Sequence[bool],
    tie_reveal_events: Sequence[TieRevealEvent],
) -> int:
    """Sample one legal draw-phase action from the policy."""
    obs = build_obs_draw(
        hand,
        phase=phase,
        scores=scores,
        last_round_scores=last_round_scores,
        n_players=n_players,
        self_player=self_player,
        target_score=target_score,
        showdown_starter=showdown_starter,
        table_history=table_history,
        discard_history=discard_history,
        shown_history=shown_history,
        announced_points=announced_points,
        passed=passed,
        tie_reveal_events=tie_reveal_events,
    ).to(device)
    with torch.no_grad():
        draw_logits, _, _ = model(obs)
    mask = build_draw_action_mask(deck_cards_left, phase=phase).to(device)
    return sample_categorical(draw_logits, mask=mask)


def choose_bot_trick_action(
    model: PolicyNet,
    hand: Sequence[Card],
    led_suit: Optional[Suit],
    trick_no: int,
    legal: Sequence[int],
    device: torch.device,
    scores: Sequence[int],
    last_round_scores: Sequence[int],
    n_players: int,
    self_player: int,
    target_score: int,
    table_history: Sequence[Sequence[Optional[Card]]],
    discard_history: Sequence[Sequence[Sequence[Card]]],
    shown_history: Sequence[Optional[Tuple[Card, bool]]],
    announced_points: Sequence[Optional[int]],
    passed: Sequence[bool],
    tie_reveal_events: Sequence[TieRevealEvent],
    showdown_starter: Optional[int],
) -> int:
    """Sample one legal trick-phase card index from the policy."""
    obs = build_obs_trick(
        hand,
        led_suit,
        trick_no,
        scores=scores,
        last_round_scores=last_round_scores,
        n_players=n_players,
        self_player=self_player,
        target_score=target_score,
        table_history=table_history,
        discard_history=discard_history,
        shown_history=shown_history,
        announced_points=announced_points,
        passed=passed,
        tie_reveal_events=tie_reveal_events,
        showdown_starter=showdown_starter,
    ).to(device)
    with torch.no_grad():
        _, trick_logits, _ = model(obs)

    mask = torch.zeros(5, dtype=torch.float32, device=device)
    for i in legal:
        mask[i] = 1.0
    action = sample_categorical(trick_logits, mask=mask)
    if action not in legal:
        return legal[0]
    return action

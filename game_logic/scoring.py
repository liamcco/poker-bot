from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from .cards import Card
from .hand_eval import evaluate_hand

TieRevealOutcome = Literal["reveal", "higher", "pass"]


@dataclass(frozen=True)
class TieRevealEvent:
    component_idx: int
    player: int
    # reveal: exposes an exact tie component value
    # higher: beats current best without exposing exact value
    # pass: cannot beat current best for this component
    outcome: TieRevealOutcome
    revealed_value: Optional[int]


@dataclass(frozen=True)
class AnnouncementResult:
    announced_points: List[Optional[int]]
    passed: List[bool]
    first_announcer: Optional[int]
    scoring_winner: Optional[int]
    winning_points: int
    showdown_starter: int
    tie_reveal_events: List[TieRevealEvent]


def resolve_first_scoring_announcements(
    hands: Sequence[Sequence[Card]],
    start_player: int,
) -> AnnouncementResult:
    """
    Resolve phase-1 announcement scoring, including tie-break reveals.

    The flow matches game rules:
    1. Clockwise announcements from start seat.
    2. Only players tied at the top point level become contenders.
    3. Contenders are resolved by incremental tie-key components.
    """
    n_players = len(hands)
    announced_points: List[Optional[int]] = [None for _ in range(n_players)]
    passed: List[bool] = [False for _ in range(n_players)]

    hand_values = [evaluate_hand(hand) for hand in hands]

    best_points = 0
    first_announcer: Optional[int] = None
    scoring_winner: Optional[int] = None
    contenders: List[int] = []
    tie_reveal_events: List[TieRevealEvent] = []

    for offset in range(n_players):
        player = (start_player + offset) % n_players
        points = hand_values[player].points

        if points > best_points:
            announced_points[player] = points
            best_points = points
            scoring_winner = player
            contenders = [player]
            if first_announcer is None:
                first_announcer = player
        elif points == best_points and points > 0:
            announced_points[player] = points
            contenders.append(player)
        else:
            passed[player] = True

    showdown_starter = start_player
    if len(contenders) > 1:
        ordered_contenders = sorted(contenders, key=lambda player: (player - showdown_starter) % n_players)
        tie_keys: Dict[int, Tuple[int, ...]] = {player: hand_values[player].key[1:] for player in contenders}

        active_contenders = set(contenders)
        max_components = max((len(tie_keys[player]) for player in contenders), default=0)

        for component_idx in range(max_components):
            best_component_value: Optional[int] = None
            component_by_player: Dict[int, int] = {}

            for player in ordered_contenders:
                if player not in active_contenders:
                    continue

                component_value = tie_keys[player][component_idx] if component_idx < len(tie_keys[player]) else -1
                component_by_player[player] = component_value

                if best_component_value is None:
                    tie_reveal_events.append(
                        TieRevealEvent(
                            component_idx=component_idx,
                            player=player,
                            outcome="reveal",
                            revealed_value=component_value,
                        )
                    )
                    best_component_value = component_value
                elif component_value > best_component_value:
                    tie_reveal_events.append(
                        TieRevealEvent(
                            component_idx=component_idx,
                            player=player,
                            outcome="higher",
                            revealed_value=None,
                        )
                    )
                    best_component_value = component_value
                elif component_value == best_component_value:
                    tie_reveal_events.append(
                        TieRevealEvent(
                            component_idx=component_idx,
                            player=player,
                            outcome="reveal",
                            revealed_value=component_value,
                        )
                    )
                else:
                    tie_reveal_events.append(
                        TieRevealEvent(
                            component_idx=component_idx,
                            player=player,
                            outcome="pass",
                            revealed_value=None,
                        )
                    )

            if best_component_value is None:
                continue

            active_contenders = {
                player
                for player in active_contenders
                if component_by_player.get(player, -1) == best_component_value
            }

            if len(active_contenders) <= 1:
                break

        if active_contenders:
            scoring_winner = sorted(active_contenders, key=lambda player: (player - showdown_starter) % n_players)[0]

    return AnnouncementResult(
        announced_points=announced_points,
        passed=passed,
        first_announcer=first_announcer,
        scoring_winner=scoring_winner,
        winning_points=best_points,
        showdown_starter=showdown_starter,
        tie_reveal_events=tie_reveal_events,
    )

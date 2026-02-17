from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from game_logic import (
    Card,
    Category,
    Deck,
    Suit,
    TieRevealEvent,
    apply_draw,
    best_hand_index,
    card_to_id,
    legal_indices,
    resolve_first_scoring_announcements,
)

MAX_PLAYERS = 6
MAX_TRICKS = 5
DRAW_PHASES = 2
ANNOUNCE_POINT_BINS = 8
MAX_TIE_COMPONENTS = 10
MAX_TIE_REVEAL_EVENTS = 64


def one_hot(n: int, idx: int) -> torch.Tensor:
    v = torch.zeros(n, dtype=torch.float32)
    v[idx] = 1.0
    return v


def encode_hand_fixed5(cards: Sequence[Card]) -> torch.Tensor:
    x = torch.zeros(5 * 52, dtype=torch.float32)
    for i, c in enumerate(cards):
        x[i * 52 + card_to_id(c)] = 1.0
    return x


def encode_hand_padded(cards: Sequence[Card]) -> torch.Tensor:
    x = torch.zeros(5 * 52, dtype=torch.float32)
    for i, c in enumerate(cards):
        x[i * 52 + card_to_id(c)] = 1.0
    return x


def encode_score_context(
    scores: Sequence[int],
    n_players: int,
    self_player: int,
    target_score: int,
    showdown_starter: Optional[int] = None,
) -> torch.Tensor:
    x_scores = torch.zeros(MAX_PLAYERS, dtype=torch.float32)
    x_active = torch.zeros(MAX_PLAYERS, dtype=torch.float32)
    for i in range(min(n_players, MAX_PLAYERS)):
        x_scores[i] = float(scores[i]) / float(max(1, target_score))
        x_active[i] = 1.0
    # Reuse active-player slots to also mark the current round's key seat
    # without increasing observation dimensionality:
    # - Draw 1: announcement-order start seat
    # - Draw 2 / Trick: showdown starter seat
    if showdown_starter is not None and 0 <= showdown_starter < min(n_players, MAX_PLAYERS):
        x_active[showdown_starter] = 2.0
    x_self = one_hot(MAX_PLAYERS, self_player if 0 <= self_player < MAX_PLAYERS else 0)
    return torch.cat([x_scores, x_active, x_self], dim=0)


def encode_last_round_scores(
    last_round_scores: Sequence[int],
    n_players: int,
    target_score: int,
) -> torch.Tensor:
    x = torch.zeros(MAX_PLAYERS, dtype=torch.float32)
    for i in range(min(len(last_round_scores), n_players, MAX_PLAYERS)):
        x[i] = float(last_round_scores[i]) / float(max(1, target_score))
    return x


def encode_table_history(table_history: Sequence[Sequence[Optional[Card]]], n_players: int) -> torch.Tensor:
    x = torch.zeros(MAX_TRICKS * MAX_PLAYERS * 52, dtype=torch.float32)
    for trick_no, row in enumerate(table_history[:MAX_TRICKS]):
        for p in range(min(len(row), n_players, MAX_PLAYERS)):
            card = row[p]
            if card is None:
                continue
            slot = trick_no * MAX_PLAYERS + p
            x[slot * 52 + card_to_id(card)] = 1.0
    return x


def encode_discard_history(
    discard_history: Sequence[Sequence[Sequence[Card]]],
    n_players: int,
) -> torch.Tensor:
    x = torch.zeros(DRAW_PHASES * MAX_PLAYERS * 52, dtype=torch.float32)
    for phase in range(min(len(discard_history), DRAW_PHASES)):
        phase_rows = discard_history[phase]
        for p in range(min(len(phase_rows), n_players, MAX_PLAYERS)):
            for card in phase_rows[p]:
                slot = phase * MAX_PLAYERS + p
                x[slot * 52 + card_to_id(card)] = 1.0
    return x


def encode_shown_history(
    shown_history: Sequence[Optional[Tuple[Card, bool]]],
    n_players: int,
) -> torch.Tensor:
    # Per player: shown card one-hot (52) + [shown_flag, kept_flag, rejected_flag].
    x = torch.zeros(MAX_PLAYERS * (52 + 3), dtype=torch.float32)
    for p in range(min(len(shown_history), n_players, MAX_PLAYERS)):
        info = shown_history[p]
        if info is None:
            continue
        shown_card, kept = info
        base = p * (52 + 3)
        x[base + card_to_id(shown_card)] = 1.0
        x[base + 52] = 1.0
        x[base + 53] = 1.0 if kept else 0.0
        x[base + 54] = 0.0 if kept else 1.0
    return x


def encode_announcement_context(
    announced_points: Sequence[Optional[int]],
    passed: Sequence[bool],
    n_players: int,
    announced_flags: Optional[Sequence[bool]] = None,
) -> torch.Tensor:
    # Per player: one-hot announced points [0..7] + [announced_flag, pass_flag].
    per_player = ANNOUNCE_POINT_BINS + 2
    x = torch.zeros(MAX_PLAYERS * per_player, dtype=torch.float32)
    for p in range(min(n_players, MAX_PLAYERS)):
        base = p * per_player
        points = announced_points[p] if p < len(announced_points) else None
        did_pass = bool(passed[p]) if p < len(passed) else False
        did_announce = (
            bool(announced_flags[p]) if announced_flags is not None and p < len(announced_flags) else points is not None
        )
        if points is not None:
            points_idx = max(0, min(int(points), ANNOUNCE_POINT_BINS - 1))
            x[base + points_idx] = 1.0
        if did_announce:
            x[base + ANNOUNCE_POINT_BINS] = 1.0
        if did_pass:
            x[base + ANNOUNCE_POINT_BINS + 1] = 1.0
    return x


def mask_announced_points_for_observer(
    announced_points: Sequence[Optional[int]],
    passed: Sequence[bool],
    n_players: int,
    self_player: int,
) -> Tuple[List[Optional[int]], List[bool]]:
    masked_points: List[Optional[int]] = [None for _ in range(n_players)]
    announced_flags: List[bool] = [False for _ in range(n_players)]
    for p in range(min(n_players, MAX_PLAYERS)):
        points = announced_points[p] if p < len(announced_points) else None
        did_pass = bool(passed[p]) if p < len(passed) else False
        did_announce = points is not None and not did_pass
        announced_flags[p] = did_announce
        if p == self_player:
            masked_points[p] = points
    return masked_points, announced_flags


def encode_tie_reveal_events(events: Sequence[TieRevealEvent]) -> torch.Tensor:
    per_event = MAX_PLAYERS + MAX_TIE_COMPONENTS + 2 + 1
    x = torch.zeros(MAX_TIE_REVEAL_EVENTS * per_event, dtype=torch.float32)
    for i, ev in enumerate(events[:MAX_TIE_REVEAL_EVENTS]):
        base = i * per_event
        if 0 <= ev.player < MAX_PLAYERS:
            x[base + ev.player] = 1.0
        comp_idx = max(0, min(int(ev.component_idx), MAX_TIE_COMPONENTS - 1))
        x[base + MAX_PLAYERS + comp_idx] = 1.0
        if ev.outcome == "pass":
            x[base + MAX_PLAYERS + MAX_TIE_COMPONENTS + 1] = 1.0
        elif ev.outcome == "reveal" and ev.revealed_value is not None:
            x[base + MAX_PLAYERS + MAX_TIE_COMPONENTS] = 1.0
            # Rank/suit tie fields are bounded; normalize to a stable range.
            x[base + MAX_PLAYERS + MAX_TIE_COMPONENTS + 2] = float(ev.revealed_value) / 14.0
        else:
            # "higher" leaves both flags at 0 and hides the exact value.
            pass
    return x


def build_context_features(
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
    showdown_starter: Optional[int] = None,
) -> torch.Tensor:
    x_scores = encode_score_context(
        scores,
        n_players,
        self_player,
        target_score,
        showdown_starter=showdown_starter,
    )
    x_last_round_scores = encode_last_round_scores(last_round_scores, n_players, target_score)
    x_table = encode_table_history(table_history, n_players)
    x_discards = encode_discard_history(discard_history, n_players)
    x_shown = encode_shown_history(shown_history, n_players)
    masked_announced_points, announced_flags = mask_announced_points_for_observer(
        announced_points,
        passed,
        n_players,
        self_player,
    )
    x_announce = encode_announcement_context(
        masked_announced_points,
        passed,
        n_players,
        announced_flags=announced_flags,
    )
    x_reveals = encode_tie_reveal_events(tie_reveal_events)
    return torch.cat([x_scores, x_last_round_scores, x_table, x_discards, x_shown, x_announce, x_reveals], dim=0)


def build_obs_draw(
    hand5: Sequence[Card],
    phase: int,
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
) -> torch.Tensor:
    x_hand = encode_hand_fixed5(hand5)
    x_phase = one_hot(3, phase)
    x_led = one_hot(5, 0)
    x_trick = one_hot(5, 0)
    x_context = build_context_features(
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
    )
    return torch.cat([x_hand, x_phase, x_led, x_trick, x_context], dim=0)


def build_obs_trick(
    hand: Sequence[Card],
    led_suit: Optional[Suit],
    trick_no: int,
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
) -> torch.Tensor:
    x_hand = encode_hand_padded(hand)
    x_phase = one_hot(3, 2)
    led_idx = 0 if led_suit is None else int(led_suit)
    x_led = one_hot(5, led_idx)
    x_trick = one_hot(5, trick_no)
    x_context = build_context_features(
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
    )
    return torch.cat([x_hand, x_phase, x_led, x_trick, x_context], dim=0)


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.in_dim = (
            260
            + 3
            + 5
            + 5
            + (MAX_PLAYERS * 3)
            + MAX_PLAYERS
            + (MAX_TRICKS * MAX_PLAYERS * 52)
            + (DRAW_PHASES * MAX_PLAYERS * 52)
            + (MAX_PLAYERS * (52 + 3))
            + (MAX_PLAYERS * (ANNOUNCE_POINT_BINS + 2))
            + (MAX_TIE_REVEAL_EVENTS * (MAX_PLAYERS + MAX_TIE_COMPONENTS + 2 + 1))
        )
        self.fc1 = nn.Linear(self.in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        # 64 draw actions: lower 5 bits = discard mask, bit 5 = reject shown card
        # when exactly one card is discarded.
        self.discard_head = nn.Linear(hidden, 64)
        self.trick_head = nn.Linear(hidden, 5)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        return self.discard_head(h), self.trick_head(h), self.value_head(h).squeeze(-1)


@dataclass
class StepRecord:
    obs: torch.Tensor
    action: int
    phase: int
    legal_mask: Optional[torch.Tensor]
    reward: float


def sample_categorical(logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> int:
    if mask is not None:
        very_neg = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
        logits = torch.where(mask > 0, logits, very_neg)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    return int(dist.sample().item())


def build_draw_action_mask(cards_left: int, phase: int) -> torch.Tensor:
    mask = torch.zeros(64, dtype=torch.float32)
    for a in range(64):
        discard_mask = a & 0b1_1111
        reject_single_draw = ((a >> 5) & 1) == 1
        n_discards = bin(discard_mask).count("1")
        if phase == 0 and reject_single_draw:
            continue
        if n_discards != 1 and reject_single_draw:
            continue
        needed_cards = n_discards + (1 if (n_discards == 1 and reject_single_draw) else 0)
        if needed_cards <= cards_left:
            mask[a] = 1.0
    return mask


def run_episode(
    model: PolicyNet,
    rng: random.Random,
    device: torch.device,
    n_players: int,
    target_score: int,
    gamma: float,
    winner_bonus: float,
) -> Tuple[List[List[StepRecord]], int, List[int], int]:
    scores = [0] * n_players
    trajs: List[List[StepRecord]] = [[] for _ in range(n_players)]
    circle_round: List[Optional[int]] = [None for _ in range(n_players)]
    announcement_order_start = 0
    rounds = 0

    while True:
        rounds += 1
        round_start_scores = scores[:]
        deck = Deck(rng)
        hands = [deck.draw(5) for _ in range(n_players)]
        table_history: List[List[Optional[Card]]] = [[None for _ in range(n_players)] for _ in range(MAX_TRICKS)]
        discard_history: List[List[List[Card]]] = [[[] for _ in range(n_players)] for _ in range(DRAW_PHASES)]
        shown_history: List[Optional[Tuple[Card, bool]]] = [None for _ in range(n_players)]
        announced_points: List[Optional[int]] = [None for _ in range(n_players)]
        passed: List[bool] = [False for _ in range(n_players)]
        tie_reveal_events: List[TieRevealEvent] = []
        showdown_starter: Optional[int] = None

        for off in range(n_players):
            p = (announcement_order_start + off) % n_players
            obs = build_obs_draw(
                hands[p],
                phase=0,
                scores=scores,
                last_round_scores=round_start_scores,
                n_players=n_players,
                self_player=p,
                target_score=target_score,
                showdown_starter=announcement_order_start,
                table_history=table_history,
                discard_history=discard_history,
                shown_history=shown_history,
                announced_points=announced_points,
                passed=passed,
                tie_reveal_events=tie_reveal_events,
            ).to(device)
            discard_logits, _, _ = model(obs)
            draw_mask = build_draw_action_mask(len(deck.cards), phase=0).to(device)
            a = sample_categorical(discard_logits, mask=draw_mask)
            discard_mask = a & 0b1_1111
            reject_single_draw = False
            draw_outcome = apply_draw(
                deck,
                hands[p],
                discard_mask,
                reject_single_draw=reject_single_draw,
                reveal_single_draw=False,
            )
            discard_history[0][p].extend(draw_outcome.discarded_cards)
            trajs[p].append(
                StepRecord(
                    obs=obs.detach().cpu(),
                    action=a,
                    phase=0,
                    legal_mask=draw_mask.detach().cpu(),
                    reward=0.0,
                )
            )

        phase1_winner, phase1_hv = best_hand_index(hands)
        if phase1_hv.category == Category.STRAIGHT_FLUSH:
            winner = phase1_winner
            trajs[winner][-1].reward += winner_bonus
            return trajs, winner, scores, rounds

        announce_result = resolve_first_scoring_announcements(hands, start_player=announcement_order_start)
        announced_points = announce_result.announced_points
        passed = announce_result.passed
        tie_reveal_events = announce_result.tie_reveal_events
        showdown_starter = announce_result.showdown_starter
        if announce_result.scoring_winner is not None and announce_result.winning_points > 0:
            w1 = announce_result.scoring_winner
            scores[w1] += announce_result.winning_points
            trajs[w1][-1].reward += announce_result.winning_points / target_score
            if scores[w1] >= target_score and circle_round[w1] is None:
                circle_round[w1] = rounds

        for off in range(n_players):
            p = (announcement_order_start + off) % n_players
            obs = build_obs_draw(
                hands[p],
                phase=1,
                scores=scores,
                last_round_scores=round_start_scores,
                n_players=n_players,
                self_player=p,
                target_score=target_score,
                showdown_starter=showdown_starter,
                table_history=table_history,
                discard_history=discard_history,
                shown_history=shown_history,
                announced_points=announced_points,
                passed=passed,
                tie_reveal_events=tie_reveal_events,
            ).to(device)
            discard_logits, _, _ = model(obs)
            draw_mask = build_draw_action_mask(len(deck.cards), phase=1).to(device)
            a = sample_categorical(discard_logits, mask=draw_mask)
            discard_mask = a & 0b1_1111
            reject_single_draw = ((a >> 5) & 1) == 1 and bin(discard_mask).count("1") == 1
            draw_outcome = apply_draw(
                deck,
                hands[p],
                discard_mask,
                reject_single_draw=reject_single_draw,
                reveal_single_draw=True,
            )
            discard_history[1][p].extend(draw_outcome.discarded_cards)
            if draw_outcome.shown_card is not None and draw_outcome.kept_shown is not None:
                shown_history[p] = (draw_outcome.shown_card, draw_outcome.kept_shown)
            trajs[p].append(
                StepRecord(
                    obs=obs.detach().cpu(),
                    action=a,
                    phase=1,
                    legal_mask=draw_mask.detach().cpu(),
                    reward=0.0,
                )
            )

        w2, hv2 = best_hand_index(hands)
        if hv2.category == Category.STRAIGHT_FLUSH:
            winner = w2
            trajs[winner][-1].reward += winner_bonus
            return trajs, winner, scores, rounds
        scores[w2] += hv2.points
        trajs[w2][-1].reward += hv2.points / target_score
        if scores[w2] >= target_score and circle_round[w2] is None:
            circle_round[w2] = rounds

        trick_hands = [h[:] for h in hands]
        leader = showdown_starter if showdown_starter is not None else announcement_order_start
        last_winner = leader

        for trick_no in range(5):
            led_suit: Optional[Suit] = None
            played: List[Tuple[int, Card]] = []

            for off in range(n_players):
                p = (leader + off) % n_players
                hand = trick_hands[p]
                legal = legal_indices(hand, led_suit)

                obs = build_obs_trick(
                    hand,
                    led_suit,
                    trick_no,
                    scores=scores,
                    last_round_scores=round_start_scores,
                    n_players=n_players,
                    self_player=p,
                    target_score=target_score,
                    table_history=table_history,
                    discard_history=discard_history,
                    shown_history=shown_history,
                    announced_points=announced_points,
                    passed=passed,
                    tie_reveal_events=tie_reveal_events,
                    showdown_starter=showdown_starter,
                ).to(device)
                _, trick_logits, _ = model(obs)

                mask = torch.zeros(5, dtype=torch.float32, device=device)
                for idx in legal:
                    mask[idx] = 1.0

                a = sample_categorical(trick_logits, mask=mask)
                if a not in legal:
                    a = legal[0]

                card = hand.pop(a)
                if led_suit is None:
                    led_suit = card.suit
                played.append((p, card))
                table_history[trick_no][p] = card

                trajs[p].append(
                    StepRecord(
                        obs=obs.detach().cpu(),
                        action=a,
                        phase=2,
                        legal_mask=mask.detach().cpu(),
                        reward=0.0,
                    )
                )

            assert led_suit is not None
            eligible = [(p, c) for (p, c) in played if c.suit == led_suit]
            winner, _ = max(eligible, key=lambda pc: (int(pc[1].rank), int(pc[1].suit)))
            leader = winner
            last_winner = winner

        scores[last_winner] += 5
        trajs[last_winner][-1].reward += 5 / target_score
        if scores[last_winner] >= target_score and circle_round[last_winner] is None:
            circle_round[last_winner] = rounds

        announcement_order_start = (announcement_order_start + 1) % n_players

        # A player can only win from showdown starting in rounds after
        # the round where they first reached the target ("circle").
        if circle_round[last_winner] is not None and circle_round[last_winner] < rounds:
            winner = last_winner
            trajs[winner][-1].reward += winner_bonus
            return trajs, winner, scores, rounds


def play_episode_with_policies(
    models: Sequence[PolicyNet],
    rng: random.Random,
    device: torch.device,
    target_score: int,
) -> Tuple[int, List[int], int]:
    n_players = len(models)
    scores = [0] * n_players
    circle_round: List[Optional[int]] = [None for _ in range(n_players)]
    announcement_order_start = 0
    rounds = 0

    while True:
        rounds += 1
        round_start_scores = scores[:]
        deck = Deck(rng)
        hands = [deck.draw(5) for _ in range(n_players)]
        table_history: List[List[Optional[Card]]] = [[None for _ in range(n_players)] for _ in range(MAX_TRICKS)]
        discard_history: List[List[List[Card]]] = [[[] for _ in range(n_players)] for _ in range(DRAW_PHASES)]
        shown_history: List[Optional[Tuple[Card, bool]]] = [None for _ in range(n_players)]
        announced_points: List[Optional[int]] = [None for _ in range(n_players)]
        passed: List[bool] = [False for _ in range(n_players)]
        tie_reveal_events: List[TieRevealEvent] = []
        showdown_starter: Optional[int] = None

        for off in range(n_players):
            p = (announcement_order_start + off) % n_players
            obs = build_obs_draw(
                hands[p],
                phase=0,
                scores=scores,
                last_round_scores=round_start_scores,
                n_players=n_players,
                self_player=p,
                target_score=target_score,
                showdown_starter=announcement_order_start,
                table_history=table_history,
                discard_history=discard_history,
                shown_history=shown_history,
                announced_points=announced_points,
                passed=passed,
                tie_reveal_events=tie_reveal_events,
            ).to(device)
            with torch.no_grad():
                discard_logits, _, _ = models[p](obs)
            draw_mask = build_draw_action_mask(len(deck.cards), phase=0).to(device)
            a = sample_categorical(discard_logits, mask=draw_mask)
            discard_mask = a & 0b1_1111
            draw_outcome = apply_draw(
                deck,
                hands[p],
                discard_mask,
                reject_single_draw=False,
                reveal_single_draw=False,
            )
            discard_history[0][p].extend(draw_outcome.discarded_cards)

        phase1_winner, phase1_hv = best_hand_index(hands)
        if phase1_hv.category == Category.STRAIGHT_FLUSH:
            return phase1_winner, scores, rounds

        announce_result = resolve_first_scoring_announcements(hands, start_player=announcement_order_start)
        announced_points = announce_result.announced_points
        passed = announce_result.passed
        tie_reveal_events = announce_result.tie_reveal_events
        showdown_starter = announce_result.showdown_starter
        if announce_result.scoring_winner is not None and announce_result.winning_points > 0:
            w1 = announce_result.scoring_winner
            scores[w1] += announce_result.winning_points
            if scores[w1] >= target_score and circle_round[w1] is None:
                circle_round[w1] = rounds

        for off in range(n_players):
            p = (announcement_order_start + off) % n_players
            obs = build_obs_draw(
                hands[p],
                phase=1,
                scores=scores,
                last_round_scores=round_start_scores,
                n_players=n_players,
                self_player=p,
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
                discard_logits, _, _ = models[p](obs)
            draw_mask = build_draw_action_mask(len(deck.cards), phase=1).to(device)
            a = sample_categorical(discard_logits, mask=draw_mask)
            discard_mask = a & 0b1_1111
            reject_single_draw = ((a >> 5) & 1) == 1 and bin(discard_mask).count("1") == 1
            draw_outcome = apply_draw(
                deck,
                hands[p],
                discard_mask,
                reject_single_draw=reject_single_draw,
                reveal_single_draw=True,
            )
            discard_history[1][p].extend(draw_outcome.discarded_cards)
            if draw_outcome.shown_card is not None and draw_outcome.kept_shown is not None:
                shown_history[p] = (draw_outcome.shown_card, draw_outcome.kept_shown)

        w2, hv2 = best_hand_index(hands)
        if hv2.category == Category.STRAIGHT_FLUSH:
            return w2, scores, rounds
        scores[w2] += hv2.points
        if scores[w2] >= target_score and circle_round[w2] is None:
            circle_round[w2] = rounds

        trick_hands = [h[:] for h in hands]
        leader = showdown_starter if showdown_starter is not None else announcement_order_start
        last_winner = leader

        for trick_no in range(5):
            led_suit: Optional[Suit] = None
            played: List[Tuple[int, Card]] = []

            for off in range(n_players):
                p = (leader + off) % n_players
                hand = trick_hands[p]
                legal = legal_indices(hand, led_suit)

                obs = build_obs_trick(
                    hand,
                    led_suit,
                    trick_no,
                    scores=scores,
                    last_round_scores=round_start_scores,
                    n_players=n_players,
                    self_player=p,
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
                    _, trick_logits, _ = models[p](obs)

                mask = torch.zeros(5, dtype=torch.float32, device=device)
                for idx in legal:
                    mask[idx] = 1.0
                a = sample_categorical(trick_logits, mask=mask)
                if a not in legal:
                    a = legal[0]

                card = hand.pop(a)
                if led_suit is None:
                    led_suit = card.suit
                played.append((p, card))
                table_history[trick_no][p] = card

            assert led_suit is not None
            eligible = [(p, c) for (p, c) in played if c.suit == led_suit]
            winner, _ = max(eligible, key=lambda pc: (int(pc[1].rank), int(pc[1].suit)))
            leader = winner
            last_winner = winner

        scores[last_winner] += 5
        if scores[last_winner] >= target_score and circle_round[last_winner] is None:
            circle_round[last_winner] = rounds

        announcement_order_start = (announcement_order_start + 1) % n_players
        if circle_round[last_winner] is not None and circle_round[last_winner] < rounds:
            return last_winner, scores, rounds


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)

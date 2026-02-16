from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from game_logic import Card, Category, Deck, Suit, apply_draw, best_hand_index, card_to_id, legal_indices


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


def build_obs_draw(hand5: Sequence[Card], phase: int) -> torch.Tensor:
    x_hand = encode_hand_fixed5(hand5)
    x_phase = one_hot(3, phase)
    x_led = one_hot(5, 0)
    x_trick = one_hot(5, 0)
    return torch.cat([x_hand, x_phase, x_led, x_trick], dim=0)


def build_obs_trick(hand: Sequence[Card], led_suit: Optional[Suit], trick_no: int) -> torch.Tensor:
    x_hand = encode_hand_padded(hand)
    x_phase = one_hot(3, 2)
    led_idx = 0 if led_suit is None else int(led_suit)
    x_led = one_hot(5, led_idx)
    x_trick = one_hot(5, trick_no)
    return torch.cat([x_hand, x_phase, x_led, x_trick], dim=0)


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.in_dim = 260 + 3 + 5 + 5
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
    starting_player = 0
    rounds = 0

    while True:
        rounds += 1
        deck = Deck(rng)
        hands = [deck.draw(5) for _ in range(n_players)]

        for p in range(n_players):
            obs = build_obs_draw(hands[p], phase=0).to(device)
            discard_logits, _, _ = model(obs)
            draw_mask = build_draw_action_mask(len(deck.cards), phase=0).to(device)
            a = sample_categorical(discard_logits, mask=draw_mask)
            discard_mask = a & 0b1_1111
            reject_single_draw = False
            apply_draw(deck, hands[p], discard_mask, reject_single_draw=reject_single_draw)
            trajs[p].append(
                StepRecord(
                    obs=obs.detach().cpu(),
                    action=a,
                    phase=0,
                    legal_mask=draw_mask.detach().cpu(),
                    reward=0.0,
                )
            )

        w1, hv1 = best_hand_index(hands)
        if hv1.category == Category.STRAIGHT_FLUSH:
            winner = w1
            trajs[winner][-1].reward += winner_bonus
            return trajs, winner, scores, rounds
        scores[w1] += hv1.points
        trajs[w1][-1].reward += hv1.points / target_score
        if scores[w1] >= target_score and circle_round[w1] is None:
            circle_round[w1] = rounds

        for p in range(n_players):
            obs = build_obs_draw(hands[p], phase=1).to(device)
            discard_logits, _, _ = model(obs)
            draw_mask = build_draw_action_mask(len(deck.cards), phase=1).to(device)
            a = sample_categorical(discard_logits, mask=draw_mask)
            discard_mask = a & 0b1_1111
            reject_single_draw = ((a >> 5) & 1) == 1 and bin(discard_mask).count("1") == 1
            apply_draw(deck, hands[p], discard_mask, reject_single_draw=reject_single_draw)
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
        leader = starting_player
        last_winner = leader

        for trick_no in range(5):
            led_suit: Optional[Suit] = None
            played: List[Tuple[int, Card]] = []

            for off in range(n_players):
                p = (leader + off) % n_players
                hand = trick_hands[p]
                legal = legal_indices(hand, led_suit)

                obs = build_obs_trick(hand, led_suit, trick_no).to(device)
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

        starting_player = (starting_player + 1) % n_players

        # A player can only win from showdown starting in rounds after
        # the round where they first reached the target ("circle").
        if circle_round[last_winner] is not None and circle_round[last_winner] < rounds:
            winner = last_winner
            trajs[winner][-1].reward += winner_bonus
            return trajs, winner, scores, rounds


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)

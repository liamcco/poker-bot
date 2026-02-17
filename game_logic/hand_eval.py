from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Sequence, Tuple

from .cards import Card


class Category(IntEnum):
    NOTHING = 0
    PAIR = 1
    TWOPAIR = 2
    TRIPS = 3
    STRAIGHT = 4
    FLUSH = 5
    FULLHOUSE = 6
    QUADS = 7
    STRAIGHT_FLUSH = 8


@dataclass(frozen=True)
class HandValue:
    category: Category
    key: Tuple[int, ...]

    @property
    def points(self) -> int:
        # Straight flush is handled as immediate game win, not round points.
        if self.category == Category.STRAIGHT_FLUSH:
            return 0
        return int(self.category)


def _detect_straight(ranks: Sequence[int]) -> Tuple[bool, int]:
    """Return (is_straight, high_rank), where wheel A-2-3-4-5 is 5-high."""
    unique_ranks_desc = sorted(set(ranks), reverse=True)
    if len(unique_ranks_desc) != 5:
        return (False, 0)
    if unique_ranks_desc[0] - unique_ranks_desc[4] == 4 and all(
        unique_ranks_desc[i] - 1 == unique_ranks_desc[i + 1] for i in range(4)
    ):
        return (True, unique_ranks_desc[0])
    if unique_ranks_desc == [14, 5, 4, 3, 2]:
        return (True, 5)
    return (False, 0)


def evaluate_hand(cards: Sequence[Card]) -> HandValue:
    """Evaluate a 5-card hand and return deterministic tie-break key data."""
    ranks = [int(card.rank) for card in cards]
    suits = [int(card.suit) for card in cards]
    ranks_desc = sorted(ranks, reverse=True)
    is_flush = len(set(suits)) == 1
    is_straight, straight_high = _detect_straight(ranks)

    rank_counts: Dict[int, int] = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1

    # Each item is (count, rank), sorted to prioritize bigger groups/ranks first.
    grouped_counts = sorted(((count, rank) for rank, count in rank_counts.items()), reverse=True)

    def sorted_suits_for_rank(rank: int) -> List[int]:
        return sorted((int(card.suit) for card in cards if int(card.rank) == rank), reverse=True)

    if is_flush and is_straight:
        straight_top_rank = 5 if straight_high == 5 else straight_high
        straight_top_suit = max(int(card.suit) for card in cards if int(card.rank) == straight_top_rank)
        key = (Category.STRAIGHT_FLUSH, straight_high, straight_top_suit)
        return HandValue(Category.STRAIGHT_FLUSH, tuple(int(value) for value in key))

    if grouped_counts[0][0] == 4:
        four_rank = grouped_counts[0][1]
        kicker_rank = max(rank for rank in ranks if rank != four_rank)
        key = (
            Category.QUADS,
            four_rank,
            kicker_rank,
            *sorted_suits_for_rank(four_rank),
            max(int(card.suit) for card in cards if int(card.rank) == kicker_rank),
        )
        return HandValue(Category.QUADS, tuple(int(value) for value in key))

    if grouped_counts[0][0] == 3 and grouped_counts[1][0] == 2:
        trips_rank = grouped_counts[0][1]
        pair_rank = grouped_counts[1][1]
        high_rank = max(trips_rank, pair_rank)
        low_rank = min(trips_rank, pair_rank)
        key = (Category.FULLHOUSE, high_rank, low_rank, *sorted_suits_for_rank(high_rank), *sorted_suits_for_rank(low_rank))
        return HandValue(Category.FULLHOUSE, tuple(int(value) for value in key))

    if is_flush:
        key = (Category.FLUSH, *ranks_desc, max(suits))
        return HandValue(Category.FLUSH, tuple(int(value) for value in key))

    if is_straight:
        straight_top_rank = 5 if straight_high == 5 else straight_high
        straight_top_suit = max(int(card.suit) for card in cards if int(card.rank) == straight_top_rank)
        key = (Category.STRAIGHT, straight_high, straight_top_suit)
        return HandValue(Category.STRAIGHT, tuple(int(value) for value in key))

    if grouped_counts[0][0] == 3:
        trips_rank = grouped_counts[0][1]
        kicker_ranks = sorted((rank for rank in ranks if rank != trips_rank), reverse=True)
        key = (
            Category.TRIPS,
            trips_rank,
            *kicker_ranks,
            *sorted_suits_for_rank(trips_rank),
            max(int(card.suit) for card in cards if int(card.rank) == kicker_ranks[0]),
            max(int(card.suit) for card in cards if int(card.rank) == kicker_ranks[1]),
        )
        return HandValue(Category.TRIPS, tuple(int(value) for value in key))

    if grouped_counts[0][0] == 2 and grouped_counts[1][0] == 2:
        pair_rank_a = grouped_counts[0][1]
        pair_rank_b = grouped_counts[1][1]
        high_pair_rank = max(pair_rank_a, pair_rank_b)
        low_pair_rank = min(pair_rank_a, pair_rank_b)
        kicker_rank = max(rank for rank in ranks if rank != high_pair_rank and rank != low_pair_rank)
        key = (
            Category.TWOPAIR,
            high_pair_rank,
            low_pair_rank,
            kicker_rank,
            *sorted_suits_for_rank(high_pair_rank),
            *sorted_suits_for_rank(low_pair_rank),
            max(int(card.suit) for card in cards if int(card.rank) == kicker_rank),
        )
        return HandValue(Category.TWOPAIR, tuple(int(value) for value in key))

    if grouped_counts[0][0] == 2:
        pair_rank = grouped_counts[0][1]
        kicker_ranks = sorted((rank for rank in ranks if rank != pair_rank), reverse=True)
        key = (
            Category.PAIR,
            pair_rank,
            *kicker_ranks,
            *sorted_suits_for_rank(pair_rank),
            max(int(card.suit) for card in cards if int(card.rank) == kicker_ranks[0]),
            max(int(card.suit) for card in cards if int(card.rank) == kicker_ranks[1]),
            max(int(card.suit) for card in cards if int(card.rank) == kicker_ranks[2]),
        )
        return HandValue(Category.PAIR, tuple(int(value) for value in key))

    highest_suit_per_rank = [max(int(card.suit) for card in cards if int(card.rank) == rank) for rank in ranks_desc]
    key = (Category.NOTHING, *ranks_desc, *highest_suit_per_rank)
    return HandValue(Category.NOTHING, tuple(int(value) for value in key))


def best_hand_index(hands: List[List[Card]]) -> Tuple[int, HandValue]:
    """Return (winner_index, winner_hand_value) across all players."""
    hand_values = [evaluate_hand(hand) for hand in hands]
    best_idx = max(range(len(hand_values)), key=lambda idx: (hand_values[idx].category, hand_values[idx].key))
    return best_idx, hand_values[best_idx]

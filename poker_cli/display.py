from __future__ import annotations

from typing import List

from game_logic import Card, Category, Suit

# Compact card rendering used by the TUI and logs.
RANK_TO_STR = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}

SUIT_TO_STR = {
    Suit.CLUBS: "♣",
    Suit.DIAMONDS: "♦",
    Suit.HEARTS: "♥",
    Suit.SPADES: "♠",
}


def bit_count(value: int) -> int:
    """Return the number of set bits in an integer mask."""
    return bin(value).count("1")


def card_str(card: Card) -> str:
    """Format a card as short rank+suit text (example: 'A♠')."""
    return f"{RANK_TO_STR[int(card.rank)]}{SUIT_TO_STR[card.suit]}"


def remove_by_discard_mask(hand: List[Card], discard_mask_5bits: int) -> List[Card]:
    """
    Remove cards from a 5-card hand using a bitmask and return removed cards.

    This mutates the `hand` list in-place.
    """
    discard_indices = [i for i in range(5) if (discard_mask_5bits >> i) & 1]
    discarded_cards = [hand[i] for i in discard_indices]
    for i in sorted(discard_indices, reverse=True):
        hand.pop(i)
    return discarded_cards


def category_name(cat: Category) -> str:
    """Return human-readable poker category labels for scoring logs."""
    names = {
        Category.NOTHING: "Pass",
        Category.PAIR: "Pair",
        Category.TWOPAIR: "Two Pair",
        Category.TRIPS: "Trips",
        Category.STRAIGHT: "Straight",
        Category.FLUSH: "Flush",
        Category.FULLHOUSE: "Full House",
        Category.QUADS: "Quads",
        Category.STRAIGHT_FLUSH: "Straight Flush",
    }
    return names[cat]


def announced_hand_name(points: int) -> str:
    """Map phase-1 announcement points to names shown in UI logs."""
    names = {
        1: "pair",
        2: "two pair",
        3: "trips",
        4: "straight",
        5: "flush",
        6: "full house",
        7: "quads",
    }
    return names.get(points, f"{points} point(s)")

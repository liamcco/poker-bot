from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import List


class Suit(IntEnum):
    CLUBS = 1
    DIAMONDS = 2
    HEARTS = 3
    SPADES = 4


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit


_SUIT_TO_INDEX = {
    Suit.CLUBS: 0,
    Suit.DIAMONDS: 1,
    Suit.HEARTS: 2,
    Suit.SPADES: 3,
}


def card_to_id(card: Card) -> int:
    """Return a stable [0, 51] card ID used by model encoders."""
    rank_idx = int(card.rank) - 2
    suit_idx = _SUIT_TO_INDEX[card.suit]
    return suit_idx * 13 + rank_idx


class Deck:
    """Mutable deck that supports drawing from the end in O(1)."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.cards: List[Card] = [Card(rank, suit) for suit in Suit for rank in Rank]
        self.rng.shuffle(self.cards)

    def draw(self, n: int) -> List[Card]:
        """Draw exactly n cards or raise if the deck is exhausted."""
        if n == 0:
            return []
        if len(self.cards) < n:
            raise RuntimeError("Deck out of cards")
        drawn = self.cards[-n:]
        del self.cards[-n:]
        return drawn

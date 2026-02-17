from __future__ import annotations

from typing import List, Optional

from .cards import Card, Suit


def legal_indices(hand: List[Card], led_suit: Optional[Suit]) -> List[int]:
    """Return legal play indices, enforcing follow-suit when possible."""
    if led_suit is None:
        return list(range(len(hand)))

    can_follow_suit = any(card.suit == led_suit for card in hand)
    if not can_follow_suit:
        return list(range(len(hand)))

    return [idx for idx, card in enumerate(hand) if card.suit == led_suit]

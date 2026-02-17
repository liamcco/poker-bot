from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .cards import Card, Deck


@dataclass(frozen=True)
class DrawOutcome:
    discarded_cards: List[Card]
    shown_card: Optional[Card]
    kept_shown: Optional[bool]


def apply_draw(
    deck: Deck,
    hand: List[Card],
    discard_mask_5bits: int,
    reject_single_draw: bool = False,
    reveal_single_draw: bool = False,
) -> DrawOutcome:
    """
    Apply a draw action in-place and return the public outcome details.

    Draw-2 special case:
    - if exactly one card is discarded and reveal_single_draw=True, the first
      replacement is shown and may be rejected for one hidden replacement.
    """
    discard_indices = [idx for idx in range(5) if (discard_mask_5bits >> idx) & 1]
    discarded_cards = [hand[idx] for idx in discard_indices]

    for idx in sorted(discard_indices, reverse=True):
        hand.pop(idx)

    discard_count = len(discard_indices)
    shown_card: Optional[Card] = None
    kept_shown: Optional[bool] = None

    if discard_count == 1 and reveal_single_draw:
        first_replacement = deck.draw(1)[0]
        shown_card = first_replacement

        if reject_single_draw:
            if len(deck.cards) > 0:
                accepted_replacement = deck.draw(1)[0]
                kept_shown = False
            else:
                # If the deck is exhausted, the shown card must be kept.
                accepted_replacement = first_replacement
                kept_shown = True
        else:
            accepted_replacement = first_replacement
            kept_shown = True

        hand.append(accepted_replacement)
    else:
        hand.extend(deck.draw(discard_count))

    if len(hand) != 5:
        raise RuntimeError("Hand must remain 5 cards after draw")

    return DrawOutcome(discarded_cards=discarded_cards, shown_card=shown_card, kept_shown=kept_shown)

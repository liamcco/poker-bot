from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple


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


def card_to_id(c: Card) -> int:
    rank_idx = int(c.rank) - 2
    suit_idx = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}[c.suit]
    return suit_idx * 13 + rank_idx


class Deck:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.cards: List[Card] = [Card(rank, suit) for suit in Suit for rank in Rank]
        self.rng.shuffle(self.cards)

    def draw(self, n: int) -> List[Card]:
        if n == 0:
            return []
        if len(self.cards) < n:
            raise RuntimeError("Deck out of cards")
        out = self.cards[-n:]
        del self.cards[-n:]
        return out


class Category(IntEnum):
    HIGH = 0
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
        if self.category == Category.STRAIGHT_FLUSH:
            return 0
        return int(self.category)


@dataclass(frozen=True)
class DrawOutcome:
    discarded_cards: List[Card]
    shown_card: Optional[Card]
    kept_shown: Optional[bool]


@dataclass(frozen=True)
class AnnouncementResult:
    announced_points: List[Optional[int]]
    passed: List[bool]
    first_announcer: Optional[int]
    scoring_winner: Optional[int]
    winning_points: int
    showdown_starter: int
    tie_reveal_events: List["TieRevealEvent"]


@dataclass(frozen=True)
class TieRevealEvent:
    component_idx: int
    player: int
    revealed_value: Optional[int]


def _is_straight(ranks: List[int]) -> Tuple[bool, int]:
    uniq = sorted(set(ranks), reverse=True)
    if len(uniq) != 5:
        return (False, 0)
    if uniq[0] - uniq[4] == 4 and all(uniq[i] - 1 == uniq[i + 1] for i in range(4)):
        return (True, uniq[0])
    if uniq == [14, 5, 4, 3, 2]:
        return (True, 5)
    return (False, 0)


def evaluate_hand(cards: Sequence[Card]) -> HandValue:
    ranks = [int(c.rank) for c in cards]
    suits = [int(c.suit) for c in cards]
    ranks_desc = sorted(ranks, reverse=True)
    is_flush = len(set(suits)) == 1
    is_straight, straight_high = _is_straight(ranks)

    counts: Dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    groups = sorted(((cnt, r) for r, cnt in counts.items()), reverse=True)

    def rank_suits(rr: int) -> List[int]:
        return sorted([int(c.suit) for c in cards if int(c.rank) == rr], reverse=True)

    if is_flush and is_straight:
        high_rank = 5 if straight_high == 5 else straight_high
        high_suit = max(int(c.suit) for c in cards if int(c.rank) == high_rank)
        key = (Category.STRAIGHT_FLUSH, straight_high, high_suit)
        return HandValue(Category.STRAIGHT_FLUSH, tuple(int(x) for x in key))

    if groups[0][0] == 4:
        quad = groups[0][1]
        kicker = max(r for r in ranks if r != quad)
        key = (
            Category.QUADS,
            quad,
            kicker,
            *rank_suits(quad),
            max(int(c.suit) for c in cards if int(c.rank) == kicker),
        )
        return HandValue(Category.QUADS, tuple(int(x) for x in key))

    if groups[0][0] == 3 and groups[1][0] == 2:
        trips = groups[0][1]
        pair = groups[1][1]
        hi = max(trips, pair)
        lo = min(trips, pair)
        key = (Category.FULLHOUSE, hi, lo, *rank_suits(hi), *rank_suits(lo))
        return HandValue(Category.FULLHOUSE, tuple(int(x) for x in key))

    if is_flush:
        key = (Category.FLUSH, *ranks_desc, max(suits))
        return HandValue(Category.FLUSH, tuple(int(x) for x in key))

    if is_straight:
        high_rank = 5 if straight_high == 5 else straight_high
        high_suit = max(int(c.suit) for c in cards if int(c.rank) == high_rank)
        key = (Category.STRAIGHT, straight_high, high_suit)
        return HandValue(Category.STRAIGHT, tuple(int(x) for x in key))

    if groups[0][0] == 3:
        trips = groups[0][1]
        kickers = sorted((r for r in ranks if r != trips), reverse=True)
        key = (
            Category.TRIPS,
            trips,
            *kickers,
            *rank_suits(trips),
            max(int(c.suit) for c in cards if int(c.rank) == kickers[0]),
            max(int(c.suit) for c in cards if int(c.rank) == kickers[1]),
        )
        return HandValue(Category.TRIPS, tuple(int(x) for x in key))

    if groups[0][0] == 2 and groups[1][0] == 2:
        p1 = groups[0][1]
        p2 = groups[1][1]
        hi = max(p1, p2)
        lo = min(p1, p2)
        kicker = max(r for r in ranks if r != hi and r != lo)
        key = (
            Category.TWOPAIR,
            hi,
            lo,
            kicker,
            *rank_suits(hi),
            *rank_suits(lo),
            max(int(c.suit) for c in cards if int(c.rank) == kicker),
        )
        return HandValue(Category.TWOPAIR, tuple(int(x) for x in key))

    if groups[0][0] == 2:
        pair = groups[0][1]
        kickers = sorted((r for r in ranks if r != pair), reverse=True)
        key = (
            Category.PAIR,
            pair,
            *kickers,
            *rank_suits(pair),
            max(int(c.suit) for c in cards if int(c.rank) == kickers[0]),
            max(int(c.suit) for c in cards if int(c.rank) == kickers[1]),
            max(int(c.suit) for c in cards if int(c.rank) == kickers[2]),
        )
        return HandValue(Category.PAIR, tuple(int(x) for x in key))

    suit_by_rank = [max(int(c.suit) for c in cards if int(c.rank) == r) for r in ranks_desc]
    key = (Category.HIGH, *ranks_desc, *suit_by_rank)
    return HandValue(Category.HIGH, tuple(int(x) for x in key))


def best_hand_index(hands: List[List[Card]]) -> Tuple[int, HandValue]:
    vals = [evaluate_hand(h) for h in hands]
    i = max(range(len(vals)), key=lambda k: (vals[k].category, vals[k].key))
    return i, vals[i]


def resolve_first_scoring_announcements(
    hands: Sequence[Sequence[Card]],
    start_player: int,
) -> AnnouncementResult:
    n_players = len(hands)
    announced_points: List[Optional[int]] = [None for _ in range(n_players)]
    passed = [False for _ in range(n_players)]

    best_points = 0
    first_announcer: Optional[int] = None
    scoring_winner: Optional[int] = None
    contenders: List[int] = []
    tie_reveal_events: List[TieRevealEvent] = []

    for off in range(n_players):
        p = (start_player + off) % n_players
        hv = evaluate_hand(hands[p])
        points = hv.points
        if points > best_points:
            announced_points[p] = points
            best_points = points
            scoring_winner = p
            contenders = [p]
            if first_announcer is None:
                first_announcer = p
        elif points == best_points and points > 0:
            announced_points[p] = points
            contenders.append(p)
        else:
            passed[p] = True

    showdown_starter = start_player
    if len(contenders) > 1:
        ordered_contenders = sorted(
            contenders,
            key=lambda p: (p - showdown_starter) % n_players,
        )
        tie_keys: Dict[int, Tuple[int, ...]] = {p: evaluate_hand(hands[p]).key[1:] for p in contenders}
        active = set(contenders)
        max_components = max((len(tie_keys[p]) for p in contenders), default=0)

        for comp_idx in range(max_components):
            current_best: Optional[int] = None
            comp_values: Dict[int, int] = {}

            for p in ordered_contenders:
                if p not in active:
                    continue
                v = tie_keys[p][comp_idx] if comp_idx < len(tie_keys[p]) else -1
                comp_values[p] = v
                if current_best is None or v > current_best:
                    tie_reveal_events.append(TieRevealEvent(component_idx=comp_idx, player=p, revealed_value=v))
                    current_best = v
                else:
                    tie_reveal_events.append(TieRevealEvent(component_idx=comp_idx, player=p, revealed_value=None))

            if current_best is None:
                continue
            active = {p for p in active if comp_values.get(p, -1) == current_best}
            if len(active) <= 1:
                break

        if active:
            scoring_winner = sorted(active, key=lambda p: (p - showdown_starter) % n_players)[0]

    return AnnouncementResult(
        announced_points=announced_points,
        passed=passed,
        first_announcer=first_announcer,
        scoring_winner=scoring_winner,
        winning_points=best_points,
        showdown_starter=showdown_starter,
        tie_reveal_events=tie_reveal_events,
    )


def apply_draw(
    deck: Deck,
    hand: List[Card],
    discard_mask_5bits: int,
    reject_single_draw: bool = False,
    reveal_single_draw: bool = False,
) -> DrawOutcome:
    discard_indices = [i for i in range(5) if (discard_mask_5bits >> i) & 1]
    discarded_cards = [hand[i] for i in discard_indices]
    for i in sorted(discard_indices, reverse=True):
        hand.pop(i)

    n_discards = len(discard_indices)
    shown_card: Optional[Card] = None
    kept_shown: Optional[bool] = None
    if n_discards == 1 and reveal_single_draw:
        first_card = deck.draw(1)[0]
        shown_card = first_card
        if reject_single_draw:
            if len(deck.cards) > 0:
                replacement = deck.draw(1)[0]
                kept_shown = False
            else:
                replacement = first_card
                kept_shown = True
        else:
            replacement = first_card
            kept_shown = True
        hand.append(replacement)
    else:
        hand.extend(deck.draw(n_discards))

    if len(hand) != 5:
        raise RuntimeError("Hand must remain 5 cards after draw")
    return DrawOutcome(discarded_cards=discarded_cards, shown_card=shown_card, kept_shown=kept_shown)


def legal_indices(hand: List[Card], led_suit: Optional[Suit]) -> List[int]:
    if led_suit is None:
        return list(range(len(hand)))
    can_follow = any(c.suit == led_suit for c in hand)
    if not can_follow:
        return list(range(len(hand)))
    return [i for i, c in enumerate(hand) if c.suit == led_suit]

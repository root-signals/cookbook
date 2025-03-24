from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"


class Rank(Enum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


@dataclass
class Card:
    rank: Rank
    suit: Suit

    def __str__(self):
        return f"{self.rank.value}{self.suit.value}"


def check_hand(cards: list[Card]) -> int:
    ranks = [card.rank for card in cards]
    rank_values = {
        Rank.TWO: 2,
        Rank.THREE: 3,
        Rank.FOUR: 4,
        Rank.FIVE: 5,
        Rank.SIX: 6,
        Rank.SEVEN: 7,
        Rank.EIGHT: 8,
        Rank.NINE: 9,
        Rank.TEN: 10,
        Rank.JACK: 11,
        Rank.QUEEN: 12,
        Rank.KING: 13,
        Rank.ACE: 14,
    }

    rank_counts: dict[Rank, int] = {}
    for rank in ranks:
        if rank in rank_counts:
            rank_counts[rank] += 1
        else:
            rank_counts[rank] = 1

    pairs = [rank for rank, count in rank_counts.items() if count == 2]
    three_of_a_kind = [rank for rank, count in rank_counts.items() if count == 3]
    four_of_a_kind = [rank for rank, count in rank_counts.items() if count == 4]

    if pairs and three_of_a_kind:
        return 700 + rank_values[three_of_a_kind[0]]  # Full house
    elif four_of_a_kind:
        return 800 + rank_values[four_of_a_kind[0]]  # Four of a kind
    elif three_of_a_kind:
        return 400 + rank_values[three_of_a_kind[0]]  # Three of a kind
    elif len(pairs) == 2:
        return 300 + max(rank_values[p] for p in pairs)  # Two pair
    elif pairs:
        return 200 + rank_values[pairs[0]]  # One pair
    else:
        return max(rank_values[r] for r in ranks)  # High card

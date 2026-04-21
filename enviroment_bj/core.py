from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import IntEnum
import random
from typing import Any, Sequence


CARD_RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
TEN_VALUE_RANKS = {"10", "J", "Q", "K"}
BET_ACTION_ORDER = ("bet_1x", "bet_2x", "bet_3x", "bet_4x")
PLAYING_ACTION_ORDER = ("stand", "hit", "double", "split", "surrender", "insurance")
ACTION_ORDER = BET_ACTION_ORDER + PLAYING_ACTION_ORDER
BET_ACTION_MULTIPLIERS = {action_name: index + 1 for index, action_name in enumerate(BET_ACTION_ORDER)}
SUPPORTED_BET_MULTIPLIERS = tuple(BET_ACTION_MULTIPLIERS[action_name] for action_name in BET_ACTION_ORDER)
VALID_RANKS = set(CARD_RANKS)


class Action(IntEnum):
    BET_1X = 0
    BET_2X = 1
    BET_3X = 2
    BET_4X = 3
    STAND = 4
    HIT = 5
    DOUBLE = 6
    SPLIT = 7
    SURRENDER = 8
    INSURANCE = 9


ACTION_ALIASES = {
    "bet_1x": "bet_1x",
    "bet1": "bet_1x",
    "1x": "bet_1x",
    "apostar_1x": "bet_1x",
    "apostar1": "bet_1x",
    "bet_2x": "bet_2x",
    "bet2": "bet_2x",
    "2x": "bet_2x",
    "apostar_2x": "bet_2x",
    "apostar2": "bet_2x",
    "bet_3x": "bet_3x",
    "bet3": "bet_3x",
    "3x": "bet_3x",
    "apostar_3x": "bet_3x",
    "apostar3": "bet_3x",
    "bet_4x": "bet_4x",
    "bet4": "bet_4x",
    "4x": "bet_4x",
    "apostar_4x": "bet_4x",
    "apostar4": "bet_4x",
    "stand": "stand",
    "plant": "stand",
    "plantarse": "stand",
    "stay": "stand",
    "hit": "hit",
    "pedir": "hit",
    "double": "double",
    "doblar": "double",
    "split": "split",
    "dividir": "split",
    "surrender": "surrender",
    "rendirse": "surrender",
    "insurance": "insurance",
    "seguro": "insurance",
}


def rank_value(rank: str) -> int:
    if rank == "A":
        return 11
    if rank in TEN_VALUE_RANKS:
        return 10
    return int(rank)


def split_value(rank: str) -> int:
    if rank in TEN_VALUE_RANKS:
        return 10
    if rank == "A":
        return 11
    return int(rank)


def hand_value(cards: Sequence[str]) -> tuple[int, bool]:
    total = 0
    aces = 0

    for card in cards:
        if card == "A":
            aces += 1
            total += 1
        elif card in TEN_VALUE_RANKS:
            total += 10
        else:
            total += int(card)

    is_soft = aces > 0 and total + 10 <= 21
    if is_soft:
        total += 10
    return total, is_soft


def is_natural_blackjack(cards: Sequence[str], from_split: bool = False) -> bool:
    if from_split or len(cards) != 2:
        return False
    return hand_value(cards)[0] == 21


def validate_shoe_cards(
    cards: Sequence[str],
    *,
    n_decks: int,
    total_cards: int,
    strict: bool = False,
) -> None:
    invalid_cards = sorted({card for card in cards if card not in VALID_RANKS})
    if invalid_cards:
        raise ValueError(f"Invalid card ranks in shoe: {', '.join(invalid_cards)}")

    if total_cards < len(cards):
        raise ValueError("total_cards must be greater than or equal to the number of loaded cards")

    max_cards = 52 * n_decks
    if len(cards) > max_cards:
        raise ValueError("The loaded shoe contains more cards than the configured number of decks")

    if strict:
        if total_cards > max_cards:
            raise ValueError("Strict shoe validation requires total_cards to fit in the configured decks")

        counts = Counter(cards)
        max_rank_count = 4 * n_decks
        impossible_ranks = [rank for rank, count in sorted(counts.items()) if count > max_rank_count]
        if impossible_ranks:
            details = ", ".join(f"{rank}={counts[rank]}" for rank in impossible_ranks)
            raise ValueError(f"Strict shoe validation found impossible card counts: {details}")


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def coerce_action_name(action: Any) -> str:
    if isinstance(action, Action):
        return ACTION_ORDER[int(action)]

    if isinstance(action, int):
        try:
            return ACTION_ORDER[action]
        except IndexError as exc:
            raise ValueError(f"Unknown action index: {action}") from exc

    if isinstance(action, str):
        normalized = ACTION_ALIASES.get(action.strip().lower())
        if normalized is None:
            raise ValueError(f"Unknown action name: {action}")
        return normalized

    if isinstance(action, dict):
        if "action" in action:
            return coerce_action_name(action["action"])

        truthy_actions: list[str] = []
        for key, value in action.items():
            alias = ACTION_ALIASES.get(str(key).strip().lower())
            if alias is None:
                continue
            if _is_truthy(value):
                truthy_actions.append(alias)

        if len(truthy_actions) != 1:
            raise ValueError(
                "Action payload must contain exactly one truthy action flag or an 'action' key"
            )
        return truthy_actions[0]

    raise TypeError(f"Unsupported action payload type: {type(action)!r}")


@dataclass(slots=True)
class HandState:
    cards: list[str]
    bet: float
    doubled: bool = False
    from_split: bool = False
    split_depth: int = 0
    split_aces: bool = False
    closed: bool = False
    surrendered: bool = False
    action_count: int = 0
    close_reason: str | None = None
    settlement: str | None = None
    reward: float = 0.0

    def total(self) -> int:
        return hand_value(self.cards)[0]

    def is_soft(self) -> bool:
        return hand_value(self.cards)[1]

    def is_blackjack(self) -> bool:
        return is_natural_blackjack(self.cards, from_split=self.from_split)

    def is_bust(self) -> bool:
        return self.total() > 21


@dataclass(slots=True)
class Shoe:
    n_decks: int
    penetration: float
    rng: random.Random = field(default_factory=random.Random)
    use_cut_card: bool = False
    standard_total_cards: int = field(init=False)
    total_cards: int = field(init=False)
    cards: deque[str] = field(init=False)
    cut_card_reached: bool = field(init=False)

    def __post_init__(self) -> None:
        self.standard_total_cards = 52 * self.n_decks
        self.total_cards = self.standard_total_cards
        self.cards = deque()
        self.cut_card_reached = False
        self.shuffle()

    @property
    def remaining_cards(self) -> int:
        return len(self.cards)

    def shuffle(self) -> None:
        cards = [rank for _ in range(self.n_decks * 4) for rank in CARD_RANKS]
        self.rng.shuffle(cards)
        self.cards = deque(cards)
        self.total_cards = self.standard_total_cards
        self.cut_card_reached = False

    def draw(self) -> str:
        if not self.cards:
            raise RuntimeError("The shoe is empty. Reshuffle before drawing again.")
        card = self.cards.popleft()
        if self.use_cut_card and not self.cut_card_reached and self.remaining_cards <= self._minimum_remaining_cards():
            self.cut_card_reached = True
        return card

    def should_reshuffle(self) -> bool:
        if self.use_cut_card:
            return self.cut_card_reached
        minimum_remaining = int(self.total_cards * (1 - self.penetration))
        return self.remaining_cards <= minimum_remaining

    def _minimum_remaining_cards(self) -> int:
        return int(self.total_cards * (1 - self.penetration))

    def composition(self) -> dict[str, int]:
        counts = Counter(self.cards)
        return {rank: counts.get(rank, 0) for rank in CARD_RANKS}

    def force_order(self, cards: Sequence[str], total_cards: int | None = None) -> None:
        normalized_total = len(cards) if total_cards is None else total_cards
        validate_shoe_cards(cards, n_decks=self.n_decks, total_cards=normalized_total)
        self.cards = deque(cards)
        self.total_cards = normalized_total
        self.cut_card_reached = self.use_cut_card and self.remaining_cards <= self._minimum_remaining_cards()

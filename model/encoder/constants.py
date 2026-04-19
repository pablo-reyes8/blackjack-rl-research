from __future__ import annotations

from enviroment_bj.core import ACTION_ORDER, CARD_RANKS


CARD_TO_INDEX = {rank: index for index, rank in enumerate(CARD_RANKS)}
DOUBLE_ALLOWED_ON_VALUES = ("any_two_cards", "hard_9_10_11", "hard_10_11")
SPLIT_RULE_VALUES = ("same_rank", "same_value")
REWARD_MODE_VALUES = ("round_end",)
PROGRESS_BUCKET_VALUES = ("early", "mid", "late")
HAND_SETTLEMENT_VALUES = ("loss", "push", "win", "blackjack", "surrender")
PUBLIC_ACTION_TOKENS = (
    "pad",
    "unk",
    "table:deal_round",
    "dealer:offer_insurance",
    "player:hit",
    "player:stand",
    "player:double",
    "player:split",
    "player:surrender",
    "player:insurance",
    "dealer:reveal_hole",
    "dealer:hit",
    "dealer:stand",
    "table:settle_round",
    "table:reshuffle",
)
PUBLIC_ACTION_TO_INDEX = {token: index for index, token in enumerate(PUBLIC_ACTION_TOKENS)}

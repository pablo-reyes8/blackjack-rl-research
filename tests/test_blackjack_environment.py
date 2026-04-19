from __future__ import annotations

import unittest

from enviroment_bj import (
    BlackjackConfig,
    BlackjackEnvironment,
    BlackjackJSONWrapper,
    BlackjackTextGame,
    ObservationConfig,
    StartStateConfig,
)


class BlackjackEnvironmentTests(unittest.TestCase):
    def make_observation_config(self, profile: str = "minimal_basic_strategy", **overrides: object) -> ObservationConfig:
        observation = ObservationConfig.for_profile(profile)
        for key, value in overrides.items():
            setattr(observation, key, value)
        observation.__post_init__()
        return observation

    def make_env(
        self,
        *,
        observation_profile: str = "minimal_basic_strategy",
        observation_overrides: dict[str, object] | None = None,
        start_state: StartStateConfig | None = None,
        **config_overrides: object,
    ) -> BlackjackEnvironment:
        observation = self.make_observation_config(observation_profile, **(observation_overrides or {}))
        config_kwargs = {
            "n_decks": 1,
            "shoe_penetration": 1.0,
            "observation": observation,
        }
        config_kwargs.update(config_overrides)
        config = BlackjackConfig(**config_kwargs)
        return BlackjackEnvironment(config=config, seed=11, start_state=start_state)

    def assert_clean_agent_observation(self, observation: dict[str, object]) -> None:
        self.assertNotIn("legal_actions", observation)
        self.assertNotIn("can_hit", observation)
        self.assertNotIn("can_double", observation)
        self.assertNotIn("can_split", observation)
        self.assertNotIn("can_surrender", observation)
        self.assertNotIn("pair_for_split", observation)

    def test_default_realistic_profile_exposes_visible_rules_and_temporal_history(self) -> None:
        env = self.make_env(observation_profile="table_realistic_default")
        env.load_shoe(["10", "6", "7", "10", "5", "2"], total_cards=6)

        response = env.reset()
        observation = response["observation"]

        self.assert_clean_agent_observation(observation)
        self.assertEqual(observation["profile"], "table_realistic_default")
        self.assertEqual(observation["mode"], "table_raw")
        self.assertEqual(observation["current_hand_cards"], ["10", "7"])
        self.assertEqual(observation["current_bet"], 1.0)
        self.assertEqual(observation["hand_context"]["current_hand_index"], 0)
        self.assertEqual(observation["temporal_context"]["rounds_since_shuffle"], 1)
        self.assertEqual(observation["temporal_context"]["player_hands_seen_since_shuffle"], 1)
        self.assertEqual(observation["observed_cards_history"]["10"], 1)
        self.assertEqual(observation["observed_cards_history"]["6"], 1)
        self.assertEqual(observation["discard_summary"]["observed_cards_count"], 3)
        self.assertIn("dealer_hits_soft_17", response["table_rules"])
        self.assertNotIn("n_decks", response["table_rules"])
        self.assertNotIn("shoe_penetration", response["table_rules"])

    def test_fully_observable_profile_exposes_hidden_rules_and_exact_shoe(self) -> None:
        env = self.make_env(observation_profile="fully_observable_sim")
        env.load_shoe(["10", "6", "7", "10", "5", "2"], total_cards=6)

        response = env.reset()
        observation = response["observation"]

        self.assertIn("n_decks", response["table_rules"])
        self.assertIn("shoe_penetration", response["table_rules"])
        self.assertIn("dealer_peeks_for_blackjack", response["table_rules"])
        self.assertEqual(observation["exact_shoe_composition"]["5"], 1)
        self.assertEqual(observation["exact_shoe_composition"]["2"], 1)
        self.assertEqual(observation["exact_shoe_composition"]["10"], 0)

    def test_recent_actions_and_hand_counts_persist_between_rounds(self) -> None:
        env = self.make_env(
            observation_profile="table_realistic_default",
            observation_overrides={
                "obs_include_recent_actions": True,
                "obs_include_last_hand_outcome": True,
                "obs_recent_actions_window": 8,
            },
        )
        env.load_shoe(
            ["10", "6", "7", "10", "10", "9", "5", "2", "10", "K", "8"],
            total_cards=11,
        )

        env.reset()
        env.step("stand")
        second_round = env.reset()
        temporal = second_round["observation"]["temporal_context"]
        tokens = [event["token"] for event in temporal["recent_actions"]]

        self.assertEqual(temporal["rounds_since_shuffle"], 2)
        self.assertEqual(temporal["player_hands_seen_since_shuffle"], 2)
        self.assertEqual(temporal["dealer_hands_seen_since_shuffle"], 2)
        self.assertEqual(temporal["last_round_outcome"]["reward"], 1.0)
        self.assertIn("player:stand", tokens)
        self.assertIn("table:settle_round", tokens)
        self.assertEqual(tokens[-1], "table:deal_round")

    def test_observed_cards_history_resets_after_shuffle_when_enabled(self) -> None:
        env = self.make_env(
            observation_profile="table_realistic_default",
            shoe_penetration=0.5,
        )
        env.load_shoe(["10", "6", "7", "10", "10"], total_cards=8)

        env.reset()
        env.step("stand")
        next_round = env.reset()

        self.assertTrue(next_round["info"]["public_state"]["shoe"]["reshuffled_on_reset"])
        self.assertEqual(next_round["observation"]["discard_summary"]["observed_cards_count"], 3)

    def test_unknown_progress_start_burns_hidden_rounds_without_leaking_progress(self) -> None:
        env = self.make_env(
            observation_profile="table_realistic_unknown_progress",
            start_state=StartStateConfig(
                mode="unknown_progress",
                min_burned_rounds=3,
                max_burned_rounds=3,
                hide_reshuffle_progress_from_observation=True,
            ),
        )

        response = env.reset()
        observation = response["observation"]
        temporal = observation["temporal_context"]
        debug_state = env.get_debug_state()

        self.assertEqual(observation["profile"], "table_realistic_unknown_progress")
        self.assertNotIn("estimated_shoe_progress", temporal)
        self.assertNotIn("rounds_since_shuffle", temporal)
        self.assertEqual(temporal["rounds_played_total"], 1)
        self.assertEqual(temporal["shuffle_count"], 0)
        self.assertEqual(observation["discard_summary"]["observed_cards_count"], 3)
        self.assertEqual(sum(observation["observed_cards_history"].values()), 3)
        self.assertEqual(response["info"]["public_state"]["history"]["observed_cards_count"], 3)
        self.assertEqual(debug_state["history"]["hidden_burned_rounds"], 3)
        self.assertTrue(env.shoe.remaining_cards < env.shoe.total_cards - 4)

    def test_player_blackjack_auto_resolves_on_reset(self) -> None:
        env = self.make_env()
        env.load_shoe(["A", "6", "K", "10"], total_cards=4)

        response = env.reset()
        public_state = response["info"]["public_state"]

        self.assertTrue(response["done"])
        self.assertEqual(response["reward"], 1.5)
        self.assertEqual(response["action_mask"], [0, 0, 0, 0, 0, 0])
        self.assertEqual(public_state["player_hands"][0]["settlement"], "blackjack")
        self.assertEqual(public_state["dealer"]["cards"], ["6", "10"])
        self.assertEqual(public_state["dealer"]["has_blackjack"], False)

    def test_dealer_blackjack_auto_resolves_on_reset_with_peek(self) -> None:
        env = self.make_env()
        env.load_shoe(["9", "K", "7", "A"], total_cards=4)

        response = env.reset()
        public_state = response["info"]["public_state"]

        self.assertTrue(response["done"])
        self.assertEqual(response["reward"], -1.0)
        self.assertEqual(public_state["player_hands"][0]["settlement"], "loss")
        self.assertEqual(public_state["dealer"]["cards"], ["K", "A"])
        self.assertTrue(public_state["dealer"]["has_blackjack"])

    def test_blackjack_push_auto_resolves_on_reset(self) -> None:
        env = self.make_env()
        env.load_shoe(["A", "K", "10", "A"], total_cards=4)

        response = env.reset()
        public_state = response["info"]["public_state"]

        self.assertTrue(response["done"])
        self.assertEqual(response["reward"], 0.0)
        self.assertEqual(public_state["player_hands"][0]["settlement"], "push")
        self.assertEqual(public_state["dealer"]["cards"], ["K", "A"])

    def test_insurance_offsets_a_loss_when_dealer_has_blackjack(self) -> None:
        env = self.make_env()
        env.load_shoe(["9", "A", "7", "K"], total_cards=4)

        start = env.reset()
        self.assertIn("insurance", start["legal_actions"])
        self.assertFalse(start["done"])

        response = env.step({"insurance": 1})
        public_state = response["info"]["public_state"]

        self.assertTrue(response["done"])
        self.assertEqual(response["reward"], 0.0)
        self.assertEqual(public_state["insurance"]["bet"], 0.5)
        self.assertEqual(public_state["insurance"]["reward"], 1.0)
        self.assertEqual(response["info"]["insurance_reward"], 1.0)
        self.assertEqual(public_state["dealer"]["cards"], ["A", "K"])

    def test_insurance_is_lost_when_dealer_does_not_have_blackjack(self) -> None:
        env = self.make_env()
        env.load_shoe(["9", "A", "7", "9", "10"], total_cards=5)

        start = env.reset()
        self.assertIn("insurance", start["legal_actions"])

        insurance_response = env.step({"insurance": 1})
        self.assertFalse(insurance_response["done"])
        self.assertEqual(insurance_response["reward"], 0.0)
        self.assertEqual(insurance_response["observation"]["insurance_context"]["insurance_offer_active"], False)

        final_response = env.step("stand")
        public_state = final_response["info"]["public_state"]

        self.assertTrue(final_response["done"])
        self.assertEqual(final_response["reward"], -1.5)
        self.assertEqual(final_response["info"]["insurance_reward"], -0.5)
        self.assertEqual(public_state["dealer"]["cards"], ["A", "9"])
        self.assertEqual(public_state["player_hands"][0]["settlement"], "loss")

    def test_split_eights_and_double_after_split_work_in_sequence(self) -> None:
        env = self.make_env()
        env.load_shoe(["8", "6", "8", "10", "3", "K", "2", "10"], total_cards=8)

        start = env.reset()
        self.assertEqual(start["action_mask_by_name"]["split"], True)

        response = env.step("split")
        public_state = response["info"]["public_state"]

        self.assertFalse(response["done"])
        self.assertEqual(len(public_state["player_hands"]), 2)
        self.assertEqual(public_state["current_hand_index"], 0)
        self.assertEqual(public_state["current_hand"]["cards"], ["8", "3"])
        self.assertEqual(response["action_mask_by_name"]["double"], True)

        response = env.step("double")
        public_state = response["info"]["public_state"]
        self.assertFalse(response["done"])
        self.assertEqual(public_state["current_hand_index"], 1)
        self.assertEqual(public_state["current_hand"]["cards"], ["8", "K"])

        response = env.step("stand")
        public_state = response["info"]["public_state"]
        self.assertTrue(response["done"])
        self.assertEqual(response["reward"], 3.0)
        self.assertEqual(response["info"]["hand_settlements"], ["win", "win"])
        self.assertEqual(public_state["dealer"]["cards"], ["6", "10", "10"])

    def test_split_aces_are_locked_when_hits_are_disabled(self) -> None:
        env = self.make_env(resplit_aces_allowed=False)
        env.load_shoe(["A", "6", "A", "10", "9", "K", "10"], total_cards=7)

        start = env.reset()
        self.assertTrue(start["action_mask_by_name"]["split"])

        response = env.step("split")
        public_state = response["info"]["public_state"]

        self.assertTrue(response["done"])
        self.assertEqual(response["reward"], 2.0)
        self.assertEqual(
            [hand["close_reason"] for hand in public_state["player_hands"]],
            ["split_aces_locked", "split_aces_locked"],
        )
        self.assertEqual(public_state["dealer"]["cards"], ["6", "10", "10"])

    def test_resplit_aces_creates_a_third_hand(self) -> None:
        env = self.make_env()
        env.load_shoe(["A", "6", "A", "10", "A", "9", "K", "10", "10"], total_cards=9)

        start = env.reset()
        self.assertTrue(start["action_mask_by_name"]["split"])

        first_split = env.step("split")
        first_public = first_split["info"]["public_state"]
        self.assertFalse(first_split["done"])
        self.assertEqual(first_public["current_hand"]["cards"], ["A", "A"])
        self.assertTrue(first_split["action_mask_by_name"]["split"])

        second_split = env.step("split")
        public_state = second_split["info"]["public_state"]

        self.assertTrue(second_split["done"])
        self.assertEqual(second_split["reward"], 3.0)
        self.assertEqual(len(public_state["player_hands"]), 3)
        self.assertEqual(second_split["info"]["hand_settlements"], ["win", "win", "win"])

    def test_surrender_loses_half_bet(self) -> None:
        env = self.make_env()
        env.load_shoe(["10", "9", "6", "7"], total_cards=4)

        env.reset()
        response = env.step("surrender")
        public_state = response["info"]["public_state"]

        self.assertTrue(response["done"])
        self.assertEqual(response["reward"], -0.5)
        self.assertEqual(public_state["player_hands"][0]["settlement"], "surrender")
        self.assertEqual(public_state["dealer"]["cards"], ["9", "7"])

    def test_dealer_h17_and_s17_produce_different_outcomes(self) -> None:
        s17_env = self.make_env(dealer_hits_soft_17=False)
        s17_env.load_shoe(["10", "A", "7", "6", "5", "10"], total_cards=6)
        s17_env.reset()
        s17_response = s17_env.step("stand")

        self.assertTrue(s17_response["done"])
        self.assertEqual(s17_response["reward"], 0.0)
        self.assertEqual(s17_response["info"]["public_state"]["dealer"]["cards"], ["A", "6"])

        h17_env = self.make_env(dealer_hits_soft_17=True)
        h17_env.load_shoe(["10", "A", "7", "6", "5", "10"], total_cards=6)
        h17_env.reset()
        h17_response = h17_env.step("stand")

        self.assertTrue(h17_response["done"])
        self.assertEqual(h17_response["reward"], 1.0)
        self.assertEqual(h17_response["info"]["public_state"]["dealer"]["cards"], ["A", "6", "5", "10"])

    def test_multiple_rounds_keep_the_same_shoe_until_reset_is_needed(self) -> None:
        env = self.make_env()
        env.load_shoe(
            ["10", "6", "7", "10", "10", "9", "5", "2", "10", "K", "8"],
            total_cards=11,
        )

        first_round = env.reset()
        self.assertEqual(first_round["observation"]["current_hand_total"], 17)

        first_result = env.step("stand")
        self.assertTrue(first_result["done"])
        self.assertEqual(first_result["reward"], 1.0)
        self.assertEqual(first_result["info"]["public_state"]["player_hands"][0]["settlement"], "win")
        self.assertEqual(first_result["info"]["public_state"]["dealer"]["cards"], ["6", "10", "10"])
        self.assertEqual(env.shoe.remaining_cards, 6)

        second_round = env.reset()
        self.assertEqual(second_round["info"]["public_state"]["current_hand"]["cards"], ["9", "2"])
        self.assertEqual(second_round["observation"]["current_hand_total"], 11)

        second_result = env.step({"action": "double"})
        self.assertTrue(second_result["done"])
        self.assertEqual(second_result["reward"], 2.0)
        self.assertEqual(second_result["info"]["public_state"]["player_hands"][0]["bet"], 2.0)
        self.assertEqual(second_result["info"]["public_state"]["player_hands"][0]["settlement"], "win")
        self.assertEqual(env.shoe.remaining_cards, 0)

    def test_reset_reshuffles_only_after_the_round_crosses_penetration(self) -> None:
        env = self.make_env(shoe_penetration=0.5)
        env.load_shoe(["10", "6", "7", "10", "10"], total_cards=8)

        env.reset()
        response = env.step("stand")
        self.assertTrue(response["done"])
        self.assertEqual(response["reward"], 1.0)
        self.assertTrue(response["info"]["public_state"]["shoe"]["reshuffle_pending"])

        next_round = env.reset()
        self.assertTrue(next_round["info"]["public_state"]["shoe"]["reshuffled_on_reset"])
        self.assertEqual(next_round["info"]["public_state"]["shoe"]["remaining_cards"], 48)

    def test_transition_log_records_public_actions_and_drawn_cards(self) -> None:
        env = self.make_env(
            observation_profile="table_realistic_default",
            observation_overrides={"obs_include_recent_actions": True},
        )
        env.load_shoe(["10", "6", "6", "10", "K"], total_cards=5)

        env.reset()
        response = env.step("hit")
        last_transition = response["info"]["last_transition"]
        public_tokens = [event["token"] for event in last_transition["public_actions_added"]]

        self.assertTrue(response["done"])
        self.assertEqual(response["reward"], -1.0)
        self.assertEqual(last_transition["action"], "hit")
        self.assertEqual(
            last_transition["drawn_cards"],
            [{"recipient": "player", "card": "K", "visible": True, "hand_index": 0}],
        )
        self.assertEqual(last_transition["action_mask_before"], [1, 1, 1, 0, 1, 0])
        self.assertIn("player:hit", public_tokens)
        self.assertIn("dealer:reveal_hole", public_tokens)
        self.assertIn("table:settle_round", public_tokens)
        self.assertEqual(response["info"]["transition_log_length"], 2)
        self.assertEqual(len(env.get_transition_log()), 2)

    def test_load_shoe_rejects_invalid_cards(self) -> None:
        env = self.make_env()
        with self.assertRaises(ValueError):
            env.load_shoe(["1", "B"], total_cards=2)

    def test_strict_shoe_validation_rejects_impossible_rank_counts(self) -> None:
        env = self.make_env(strict_shoe_validation=True)
        with self.assertRaises(ValueError):
            env.load_shoe(["A", "A", "A", "A", "A"], total_cards=5)

    def test_wrapper_accepts_json_like_action_payloads(self) -> None:
        wrapper = BlackjackJSONWrapper(
            config=BlackjackConfig(
                n_decks=1,
                shoe_penetration=1.0,
                observation=ObservationConfig.for_profile("minimal_basic_strategy"),
            ),
            seed=11,
        )
        wrapper.environment.load_shoe(["10", "9", "6", "7"], total_cards=4)

        start = wrapper.reset()
        self.assertIn("surrender", start["legal_actions"])

        response = wrapper.step_from_json('{"surrender": 1, "stand": 0}')
        self.assertTrue(response["done"])
        self.assertEqual(response["reward"], -0.5)
        self.assertEqual(response["info"]["public_state"]["player_hands"][0]["settlement"], "surrender")

    def test_text_game_shows_a_compact_human_readable_round(self) -> None:
        game = BlackjackTextGame(
            config=BlackjackConfig(
                n_decks=1,
                shoe_penetration=1.0,
                observation=ObservationConfig.for_profile("minimal_basic_strategy"),
            ),
            seed=11,
        )
        game.wrapper.environment.load_shoe(
            ["10", "6", "7", "10", "10", "9", "5", "2", "10", "K", "8"],
            total_cards=11,
        )

        start = game.new_round()
        self.assertIn("Round 1", start)
        self.assertIn("Dealer: 6 ? (6)", start)
        self.assertIn("You[0]: 10 7 (17)", start)
        self.assertIn("Actions: stand, hit, double, surrender", start)

        end = game.stand()
        self.assertIn("Dealer: 6 10 10 (26)", end)
        self.assertIn("win", end)
        self.assertIn("Round result: +1.00", end)
        self.assertIn("Use game.new_round() to play again.", end)

    def test_text_game_requires_a_new_round_after_termination(self) -> None:
        game = BlackjackTextGame(
            config=BlackjackConfig(
                n_decks=1,
                shoe_penetration=1.0,
                observation=ObservationConfig.for_profile("minimal_basic_strategy"),
            ),
            seed=11,
        )
        game.wrapper.environment.load_shoe(["10", "6", "7", "10", "10"], total_cards=5)

        game.new_round()
        game.stand()

        with self.assertRaises(RuntimeError):
            game.hit()


if __name__ == "__main__":
    unittest.main()

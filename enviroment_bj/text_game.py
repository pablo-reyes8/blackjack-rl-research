from __future__ import annotations

from typing import Any

from .config import BlackjackConfig, StartStateConfig
from .wrapper import BlackjackJSONWrapper


class RenderedText(str):
    def __repr__(self) -> str:
        return str(self)

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(str(self))


class BlackjackTextGame:
    def __init__(
        self,
        config: BlackjackConfig | None = None,
        seed: int | None = None,
        start_state: StartStateConfig | None = None,
    ) -> None:
        self.wrapper = BlackjackJSONWrapper(config=config, seed=seed, start_state=start_state)
        self.state: dict[str, Any] | None = None

    def new_round(self) -> str:
        self.state = self.wrapper.reset()
        return self.render()

    def play(self, action: Any) -> str:
        if self.state is None:
            raise RuntimeError("No active round. Call new_round() first.")
        if self.state["done"]:
            raise RuntimeError("The round is over. Call new_round() to start again.")

        self.state = self.wrapper.step(action)
        return self.render()

    def status(self) -> str:
        return self.render()

    def hit(self) -> str:
        return self.play("hit")

    def stand(self) -> str:
        return self.play("stand")

    def double(self) -> str:
        return self.play("double")

    def split(self) -> str:
        return self.play("split")

    def surrender(self) -> str:
        return self.play("surrender")

    def insurance(self) -> str:
        return self.play("insurance")

    def render(self) -> RenderedText:
        if self.state is None:
            return RenderedText("No active round. Use game.new_round().")

        public_state = self.state["info"]["public_state"]
        lines = [
            f"Round {public_state['round_index']}",
            self._format_dealer(public_state["dealer"]),
            *self._format_player_hands(public_state),
        ]

        if self.state["done"]:
            lines.append(f"Round result: {self._format_reward(self.state['reward'])}")
            insurance_reward = public_state["insurance"]["reward"]
            if insurance_reward:
                lines.append(f"Insurance: {self._format_reward(insurance_reward)}")
            lines.append("Use game.new_round() to play again.")
        else:
            legal_actions = ", ".join(self.state["legal_actions"])
            lines.append(f"Actions: {legal_actions}")

        return RenderedText("\n".join(lines))

    def _format_dealer(self, dealer: dict[str, Any]) -> str:
        if dealer["hole_card_hidden"]:
            visible_cards = " ".join(dealer["cards"])
            return f"Dealer: {visible_cards} ? ({dealer['visible_total']})"

        cards = " ".join(dealer["cards"])
        return f"Dealer: {cards} ({dealer['total']})"

    def _format_player_hands(self, observation: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        active_index = observation["current_hand_index"]

        for hand in observation["player_hands"]:
            prefix = "->" if hand["index"] == active_index else "  "
            cards = " ".join(hand["cards"])
            label = f"{prefix} You[{hand['index']}]: {cards} ({hand['total']})"

            extras: list[str] = []
            if hand["is_soft"]:
                extras.append("soft")
            if hand["doubled"]:
                extras.append("doubled")
            if hand["from_split"]:
                extras.append("from_split")
            if hand["surrendered"]:
                extras.append("surrendered")
            if hand["settlement"] is not None:
                extras.append(hand["settlement"])
            if hand["reward"]:
                extras.append(self._format_reward(hand["reward"]))

            if extras:
                label = f"{label} | {', '.join(extras)}"

            lines.append(label)

        return lines

    def _format_reward(self, reward: float) -> str:
        return f"{reward:+.2f}"

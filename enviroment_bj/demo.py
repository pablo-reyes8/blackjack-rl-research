from __future__ import annotations

import json

from .wrapper import BlackjackJSONWrapper


def choose_simple_action(response: dict) -> str:
    legal_actions = response["legal_actions"]
    for preferred in ("insurance", "split", "double", "stand", "hit", "surrender"):
        if preferred in legal_actions:
            return preferred
    raise RuntimeError("No legal action is available")


def main() -> None:
    wrapper = BlackjackJSONWrapper(seed=7)
    response = wrapper.reset()
    print(json.dumps(response, indent=2))

    while not response["done"]:
        action = choose_simple_action(response)
        response = wrapper.step({"action": action})
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from contextlib import redirect_stdout
import io
import math
import random
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

import torch

from enviroment_bj.core import ACTION_ORDER
from inference.final_wrapper import (
    _resolve_betting_auxiliary_args,
    _resolve_count_auxiliary_args,
    _resolve_model_and_encoder_args_for_checkpoint,
    load_checkpoint_weights_for_eval,
)
from scripts.blackjack_rl_cli.common import load_checkpoint_payload, resolve_device
from training.final_wrapper import run_blackjack_stage


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PRESETS: dict[str, dict[str, str]] = {
    "05A": {
        "label": "05A count aux representation",
        "checkpoint_path": "outputs/models/KEEP_05A_count_aux_representation_acc0815.pt",
        "stage_name": "app_05a_count_aux_representation",
    },
    "04D": {
        "label": "04D betting weighted CE",
        "checkpoint_path": "outputs/models/KEEP_04D_betting_weighted_ce_best.pt",
        "stage_name": "app_04d_betting_weighted_ce",
    },
}

ACTION_LABELS = {
    "bet_1x": "Bet 1x",
    "bet_2x": "Bet 2x",
    "bet_3x": "Bet 3x",
    "bet_4x": "Bet 4x",
    "stand": "Stand",
    "hit": "Hit",
    "double": "Double",
    "split": "Split",
    "surrender": "Surrender",
    "insurance": "Insurance",
}

COUNT_BUCKET_LABELS = ("low", "medium", "high", "very_high")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


class BlackjackAppSession:
    def __init__(self, *, model_key: str = "05A", seed: int | None = None, device: str = "cpu") -> None:
        self._lock = threading.RLock()
        self.device_name = str(resolve_device(device))
        self.model_key = model_key if model_key in MODEL_PRESETS else "05A"
        self.seed = seed if seed is not None else random.randint(1, 2_000_000_000)
        self.runtime_root = Path(tempfile.gettempdir()) / "blackjack_rl_app_runtime"
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.completed_rounds = 0
        self.total_reward = 0.0
        self.outcome_counts = {"win": 0, "loss": 0, "push": 0, "blackjack": 0, "bust": 0, "surrender": 0}
        self.last_result: dict[str, Any] | None = None
        self.last_error: str | None = None
        self._load_runtime()

    @property
    def preset(self) -> dict[str, str]:
        return MODEL_PRESETS[self.model_key]

    def _load_runtime(self) -> None:
        checkpoint_path = (ROOT_DIR / self.preset["checkpoint_path"]).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
        stage_name = self.preset["stage_name"]
        betting_auxiliary_args = _resolve_betting_auxiliary_args(
            payload,
            axu_loss_bet=None,
            betting_auxiliary_threshold_2x=None,
            betting_auxiliary_threshold_3x=None,
            betting_auxiliary_threshold_4x=None,
            betting_auxiliary_min_observed_cards=None,
            betting_auxiliary_class_weights=None,
        )
        count_auxiliary_args = _resolve_count_auxiliary_args(payload)
        common_env_args = {
            "output_root": self.runtime_root,
            "stage_name": stage_name,
            "device": self.device_name,
            "start_mode": "unknown_progress",
            "min_burned_rounds": 10,
            "max_burned_rounds": 60,
            "bet_multipliers": (1, 2, 3, 4),
            "penetrations": (0.75,),
            "base_seed": int(self.seed),
            "n_decks": 8,
            "shoe_penetration": 0.75,
            "dealer_hits_soft_17": False,
            "blackjack_payout": 1.5,
            "double_allowed_on": "any_two_cards",
            "double_after_split_allowed": True,
            "split_rule": "same_value",
            "surrender_allowed": False,
            "insurance_allowed": False,
            "six_card_charlie_enabled": False,
            "blackjack_overrides": None,
            "observation_overrides": None,
            "start_state_overrides": None,
        }
        model_and_encoder_args = _resolve_model_and_encoder_args_for_checkpoint(
            payload,
            architecture=None,
            feedforward_hidden_dims=None,
            use_layer_norm=None,
            use_phase_adapters=None,
            use_module_gating=None,
            betting_auxiliary_args=betting_auxiliary_args,
            **common_env_args,
        )

        with redirect_stdout(io.StringIO()):
            setup = run_blackjack_stage(
                run_training=False,
                enable_prints=False,
                print_run_summary=False,
                eval_rounds=1,
                eval_max_decisions=1,
                total_epochs=1,
                env_steps_per_epoch=1,
                checkpoint_directory=self.runtime_root / stage_name,
                **common_env_args,
                **model_and_encoder_args,
                **betting_auxiliary_args,
                **count_auxiliary_args,
            )

        self.model = load_checkpoint_weights_for_eval(setup["model"], checkpoint_path, device=self.device_name)
        self.pipeline_config = setup["pipeline_config"]
        self.env = setup["envs"][0]
        self.response: dict[str, Any] | None = None
        self.hidden_state = None
        self.model_metadata = {
            "key": self.model_key,
            "label": self.preset["label"],
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT_DIR)),
            "architecture": getattr(self.model.config, "architecture", None),
            "state_dim": int(getattr(self.model, "state_dim", 0)),
            "device": self.device_name,
            "seed": self.seed,
            "encoder_profile": getattr(getattr(self.model.config, "encoder", None), "profile", None),
        }

    def _record_completed_round(self, response: Mapping[str, Any]) -> None:
        info = response.get("info") or {}
        public_state = info.get("public_state") or {}
        reward = float(info.get("round_reward", 0.0))
        settlements = [str(item) for item in info.get("hand_settlements", []) if item is not None]
        close_reasons = [str(item) for item in info.get("hand_close_reasons", []) if item is not None]

        if any(item == "blackjack" for item in settlements):
            result = "blackjack"
        elif any(item == "win" for item in settlements):
            result = "win"
        elif all(item == "push" for item in settlements) and settlements:
            result = "push"
        elif any(item == "surrender" for item in close_reasons):
            result = "surrender"
        elif any(item == "loss" for item in settlements):
            result = "loss"
        else:
            result = "push" if reward == 0 else ("win" if reward > 0 else "loss")

        if any(reason == "bust" for reason in close_reasons):
            self.outcome_counts["bust"] += 1
        if result in self.outcome_counts:
            self.outcome_counts[result] += 1
        self.completed_rounds += 1
        self.total_reward += reward
        self.last_result = {
            "result": result,
            "reward": reward,
            "settlements": settlements,
            "close_reasons": close_reasons,
            "dealer": public_state.get("dealer", {}),
            "player_hands": public_state.get("player_hands", []),
        }

    def _ensure_response(self) -> dict[str, Any]:
        if self.response is None:
            self.response = self.env.reset()
        return self.response

    def _model_forward(self, response: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
        with torch.no_grad():
            if getattr(self.model.config, "architecture", "feedforward") == "feedforward":
                output = self.model(response)
                return output, None
            output = self.model.forward_step(response, hidden_state=self.hidden_state)
            return output, output.get("hidden_state")

    def _recommendation(self, response: Mapping[str, Any], *, commit_hidden: bool = False) -> dict[str, Any]:
        if bool(response.get("done")):
            return {"suggested_action": None, "actions": [], "count_bucket": None}

        output, next_hidden = self._model_forward(response)
        if commit_hidden and next_hidden is not None:
            self.hidden_state = next_hidden

        q_values = output["q_values"].squeeze(0).detach().cpu()
        masked_q_values = output["masked_q_values"].squeeze(0).detach().cpu()
        action_mask = output["action_mask"].squeeze(0).detach().cpu().to(torch.bool)
        legal_indices = [idx for idx, allowed in enumerate(action_mask.tolist()) if allowed]
        best_index = int(masked_q_values.argmax(dim=-1).item()) if legal_indices else None

        actions = []
        best_legal_q = None
        if best_index is not None:
            best_legal_q = _as_float(q_values[best_index].item())
        for index, action_name in enumerate(ACTION_ORDER):
            legal = bool(action_mask[index].item())
            q_value = _as_float(q_values[index].item())
            actions.append(
                {
                    "name": action_name,
                    "label": ACTION_LABELS[action_name],
                    "legal": legal,
                    "q": q_value,
                    "masked_q": _as_float(masked_q_values[index].item()) if legal else None,
                    "is_best": best_index == index,
                    "delta_from_best": (
                        None if not legal or best_legal_q is None or q_value is None else q_value - best_legal_q
                    ),
                }
            )

        count_bucket = None
        logits = output.get("count_bucket_logits")
        if logits is not None:
            probabilities = torch.softmax(logits.squeeze(0).detach().cpu(), dim=-1).tolist()
            best_bucket_index = int(torch.tensor(probabilities).argmax().item())
            count_bucket = {
                "labels": list(COUNT_BUCKET_LABELS),
                "probabilities": [float(item) for item in probabilities],
                "best_label": COUNT_BUCKET_LABELS[best_bucket_index],
            }

        return {
            "suggested_action": ACTION_ORDER[best_index] if best_index is not None else None,
            "suggested_label": ACTION_LABELS[ACTION_ORDER[best_index]] if best_index is not None else None,
            "actions": actions,
            "count_bucket": count_bucket,
        }

    def _state_payload(self) -> dict[str, Any]:
        response = self._ensure_response()
        public_state = (response.get("info") or {}).get("public_state") or {}
        return _jsonable(
            {
                "model": self.model_metadata,
                "models": [
                    {"key": key, "label": preset["label"], "checkpoint_path": preset["checkpoint_path"]}
                    for key, preset in MODEL_PRESETS.items()
                ],
                "session": {
                    "completed_rounds": self.completed_rounds,
                    "total_reward": self.total_reward,
                    "average_reward": self.total_reward / self.completed_rounds if self.completed_rounds else 0.0,
                    "ev_per_100_hands": (
                        100.0 * self.total_reward / self.completed_rounds if self.completed_rounds else 0.0
                    ),
                    "outcomes": dict(self.outcome_counts),
                    "last_result": self.last_result,
                },
                "response": {
                    "reward": response.get("reward", 0.0),
                    "done": bool(response.get("done")),
                    "legal_actions": response.get("legal_actions", []),
                    "action_mask_by_name": response.get("action_mask_by_name", {}),
                    "info": response.get("info", {}),
                    "table_rules": response.get("table_rules", {}),
                    "observation": response.get("observation", {}),
                },
                "public_state": public_state,
                "recommendation": self._recommendation(response),
            }
        )

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state_payload()

    def step(self, action: str) -> dict[str, Any]:
        with self._lock:
            response = self._ensure_response()
            if bool(response.get("done")):
                raise ValueError("Round is over. Start a new hand before taking another action.")
            self._recommendation(response, commit_hidden=True)
            self.response = self.env.step(action)
            if bool(self.response.get("done")):
                self._record_completed_round(self.response)
            return self._state_payload()

    def play_suggestion(self) -> dict[str, Any]:
        with self._lock:
            response = self._ensure_response()
            recommendation = self._recommendation(response)
            action = recommendation.get("suggested_action")
            if not action:
                raise ValueError("No suggested action is available.")
            return self.step(str(action))

    def autoplay(self, *, max_steps: int = 20) -> dict[str, Any]:
        with self._lock:
            steps: list[str] = []
            for _ in range(max(1, int(max_steps))):
                response = self._ensure_response()
                if bool(response.get("done")):
                    break
                recommendation = self._recommendation(response)
                action = recommendation.get("suggested_action")
                if not action:
                    break
                self._recommendation(response, commit_hidden=True)
                self.response = self.env.step(str(action))
                steps.append(str(action))
                if bool(self.response.get("done")):
                    self._record_completed_round(self.response)
                    break
            payload = self._state_payload()
            payload["autoplay"] = {"steps": steps}
            return payload

    def new_round(self) -> dict[str, Any]:
        with self._lock:
            self.response = self.env.reset()
            if getattr(self.model.config, "architecture", "feedforward") != "feedforward":
                self.hidden_state = self.model.init_hidden(batch_size=1)
            return self._state_payload()

    def new_table(self, *, model_key: str | None = None, seed: int | None = None) -> dict[str, Any]:
        with self._lock:
            if model_key is not None:
                self.model_key = model_key if model_key in MODEL_PRESETS else "05A"
            self.seed = seed if seed is not None else random.randint(1, 2_000_000_000)
            self.completed_rounds = 0
            self.total_reward = 0.0
            self.outcome_counts = {"win": 0, "loss": 0, "push": 0, "blackjack": 0, "bust": 0, "surrender": 0}
            self.last_result = None
            self._load_runtime()
            self.response = self.env.reset()
            return self._state_payload()

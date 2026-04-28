from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import random
from time import perf_counter
from typing import Any

import torch

from enviroment_bj.core import ACTION_ORDER
from model.agents import DuelingRecurrentDoubleDQN, FeedForwardDoubleDQN, RecurrentDoubleDQN

from .checkpoints import CheckpointManager
from .betting_auxiliary import compute_observed_hi_lo_proxy_from_response
from .config import TrainingPipelineConfig
from .env_factory import clone_environments, normalize_envs
from .epsilon import DualEpsilonScheduler
from .evaluation import evaluate_policy
from .logging import TrainingLogger
from .metrics import BehaviorMetricsTracker, ScalarMetricAccumulator
from .policy import action_name_from_index, infer_decision_phase, resolve_epsilon_value, select_epsilon_greedy_action
from .replay_buffer import FeedForwardReplayBuffer, RecurrentReplayBuffer
from .step import build_optimizer, build_scheduler, hard_update_target, maybe_update_target, train_gradient_step
from .transfer_learning import encode_teacher_state, load_teacher_model


@dataclass(slots=True)
class EnvironmentRunnerState:
    env: Any
    response: dict[str, Any] | None = None
    encoded_response: dict[str, Any] | None = None
    hidden_state: Any = None
    pending_sequence: list[dict[str, Any]] = field(default_factory=list)
    pending_n_step: list[dict[str, Any]] = field(default_factory=list)


class BlackjackRLTrainer:
    def __init__(
        self,
        envs: Any,
        online_network: Any,
        *,
        pipeline_config: TrainingPipelineConfig | None = None,
        target_network: Any | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        self.pipeline_config = pipeline_config or TrainingPipelineConfig()
        self.envs = normalize_envs(envs)
        self.online_network = online_network
        self.device = self._resolve_device(self.pipeline_config.trainer.device)
        self.online_network.to(self.device)
        self.target_network = target_network or deepcopy(online_network)
        self.target_network.to(self.device)
        hard_update_target(self.online_network, self.target_network)
        self.optimizer = optimizer or build_optimizer(self.online_network, self.pipeline_config.optimization)
        self.scheduler = scheduler or build_scheduler(self.optimizer, self.pipeline_config.optimization)
        self.epsilon_scheduler = DualEpsilonScheduler(self.pipeline_config.epsilon)
        self.rng = random.Random(self.pipeline_config.trainer.seed)
        self.logger = TrainingLogger(self.pipeline_config.prints)
        self.checkpoints = CheckpointManager(self.pipeline_config.checkpoints)
        self.teacher_model = self._build_teacher_model()
        self.is_recurrent = self.online_network.config.architecture != "feedforward"
        self.replay_buffer = (
            RecurrentReplayBuffer(self.pipeline_config.replay_buffer, rng=self.rng)
            if self.is_recurrent
            else FeedForwardReplayBuffer(self.pipeline_config.replay_buffer, rng=self.rng)
        )
        self.env_states = [EnvironmentRunnerState(env=env) for env in self.envs]
        self.epoch_index = 0
        self.env_step_count = 0
        self.update_count = 0
        self.best_eval_metrics: dict[str, Any] | None = None
        self.training_history: list[dict[str, Any]] = []
        self._set_env_runtime_mode(enable_transition_recording=False)
        torch.manual_seed(self.pipeline_config.trainer.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.pipeline_config.trainer.seed)

    def _build_teacher_model(self) -> Any | None:
        transfer_config = self.pipeline_config.transfer
        if not transfer_config.enabled or not transfer_config.distillation.enabled:
            return None
        if transfer_config.teacher_checkpoint_path is None:
            raise ValueError("transfer.teacher_checkpoint_path is required when distillation is enabled")
        return load_teacher_model(transfer_config.teacher_checkpoint_path, device=self.device)

    def _set_env_runtime_mode(self, *, enable_transition_recording: bool) -> None:
        for env in self.envs:
            if hasattr(env, "set_runtime_options"):
                env.set_runtime_options(
                    enable_transition_recording=enable_transition_recording,
                    compact_response_mode=not enable_transition_recording,
                )

    def _encode_response_for_storage(self, response: dict[str, Any], *, env: Any | None = None) -> dict[str, Any]:
        encoded = self.online_network.encoder.encode_state_only(response)
        output = {
            "state_vector": encoded["state_vector"].detach().cpu(),
            "action_mask": encoded["action_mask"].detach().cpu(),
        }
        if self.teacher_model is not None:
            teacher_state = encode_teacher_state(self.teacher_model, response)
            output["teacher_state_vector"] = teacher_state["state_vector"]
            output["teacher_action_mask"] = teacher_state["action_mask"]
        if self.pipeline_config.betting_auxiliary.enabled:
            n_decks = getattr(getattr(env, "config", None), "n_decks", 8)
            auxiliary = compute_observed_hi_lo_proxy_from_response(response, n_decks=int(n_decks))
            output["betting_auxiliary"] = {
                "true_count_proxy": torch.tensor(auxiliary["true_count_proxy"], dtype=torch.float32),
                "observed_cards": torch.tensor(auxiliary["observed_cards"], dtype=torch.long),
            }
        return output

    def _resolve_device(self, device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(device_name)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch_index": self.epoch_index,
            "env_step_count": self.env_step_count,
            "update_count": self.update_count,
            "epsilon_scheduler": self.epsilon_scheduler.state_dict(),
            "best_eval_metrics": deepcopy(self.best_eval_metrics),
            "device": str(self.device),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch_index = int(state.get("epoch_index", 0))
        self.env_step_count = int(state.get("env_step_count", 0))
        self.update_count = int(state.get("update_count", 0))
        self.epsilon_scheduler.load_state_dict(state.get("epsilon_scheduler", {}))
        self.best_eval_metrics = deepcopy(state.get("best_eval_metrics"))
        best_metric_name = self.pipeline_config.checkpoints.best_metric_name
        if self.best_eval_metrics is not None and best_metric_name in self.best_eval_metrics:
            self.checkpoints.best_metric_value = float(self.best_eval_metrics[best_metric_name])

    def _n_step_enabled(self) -> bool:
        return self.pipeline_config.n_step.enabled and self.pipeline_config.n_step.n_steps > 1

    def _build_n_step_transition(self, transitions: list[dict[str, Any]]) -> dict[str, Any]:
        reward = 0.0
        steps_used = 0
        gamma = self.pipeline_config.trainer.loss.gamma
        max_n_steps = self.pipeline_config.n_step.n_steps
        last_transition = transitions[0]

        for step_index, transition in enumerate(transitions[:max_n_steps]):
            reward += (gamma ** step_index) * float(transition["reward"])
            steps_used += 1
            last_transition = transition
            if transition["done"]:
                break

        return {
            **transitions[0],
            "reward": reward,
            "done": bool(last_transition["done"]),
            "next_state": last_transition["next_state"],
            "next_action_mask": last_transition["next_action_mask"],
            "n_steps": steps_used,
        }

    def _store_processed_transition(self, env_state: EnvironmentRunnerState, transition: dict[str, Any]) -> None:
        if self.is_recurrent:
            env_state.pending_sequence.append(transition)
            should_finalize = len(env_state.pending_sequence) >= self.pipeline_config.replay_buffer.sequence_length
            if transition["done"] and (
                self.pipeline_config.trainer.sequence_end_on_done
                or self.pipeline_config.trainer.reset_hidden_on_round_end
            ):
                should_finalize = True
            if should_finalize:
                self._commit_pending_sequence(env_state)
            return

        self.replay_buffer.add(transition)

    def _store_transition(self, env_state: EnvironmentRunnerState, transition: dict[str, Any]) -> None:
        if not self._n_step_enabled():
            self._store_processed_transition(env_state, {**transition, "n_steps": 1})
            return

        env_state.pending_n_step.append(transition)
        if transition["done"]:
            while env_state.pending_n_step:
                processed = self._build_n_step_transition(env_state.pending_n_step)
                self._store_processed_transition(env_state, processed)
                env_state.pending_n_step.pop(0)
            return

        if len(env_state.pending_n_step) >= self.pipeline_config.n_step.n_steps:
            processed = self._build_n_step_transition(env_state.pending_n_step)
            self._store_processed_transition(env_state, processed)
            env_state.pending_n_step.pop(0)

    def _buffer_ready(self) -> bool:
        return len(self.replay_buffer) >= self.pipeline_config.replay_buffer.warmup_size and self.replay_buffer.can_sample()

    def warmup(self) -> dict[str, Any]:
        tracker = BehaviorMetricsTracker()
        while len(self.replay_buffer) < self.pipeline_config.replay_buffer.warmup_size:
            self.collect_experience(num_steps=1, tracker=tracker, epsilon=1.0)
            self.logger.log_warmup(buffer_size=len(self.replay_buffer), target_size=self.pipeline_config.replay_buffer.warmup_size)
        self.logger.log_warmup(
            buffer_size=len(self.replay_buffer),
            target_size=self.pipeline_config.replay_buffer.warmup_size,
            force=True,
        )
        return tracker.summary()

    def _estimated_updates_this_epoch(self) -> int:
        updates = 0
        env_steps_at_epoch_start = self.env_step_count
        nominal_env_steps_this_epoch = self.pipeline_config.trainer.env_steps_per_epoch * len(self.env_states)
        for global_env_step in range(env_steps_at_epoch_start + 1, env_steps_at_epoch_start + nominal_env_steps_this_epoch + 1):
            if global_env_step % self.pipeline_config.trainer.train_frequency == 0:
                updates += self.pipeline_config.trainer.updates_per_train_step
        if self.pipeline_config.trainer.max_updates_per_epoch is not None:
            updates = min(updates, self.pipeline_config.trainer.max_updates_per_epoch)
        return max(updates, 1)

    def build_run_summary(self) -> dict[str, Any]:
        reference_env = self.envs[0]
        model_config = self.online_network.config
        parameter_count = sum(parameter.numel() for parameter in self.online_network.parameters())
        return {
            "architecture": model_config.architecture,
            "recurrent_type": getattr(model_config, "recurrent_type", "none")
            if model_config.architecture != "feedforward"
            else "none",
            "encoder_profile": model_config.encoder.profile,
            "observation_profile": reference_env.config.observation.profile,
            "start_state_mode": reference_env.start_state.mode,
            "device": str(self.device),
            "total_epochs": self.pipeline_config.trainer.total_epochs,
            "num_envs": len(self.envs),
            "nominal_env_steps_per_epoch": self.pipeline_config.trainer.env_steps_per_epoch * len(self.envs),
            "estimated_updates_per_epoch": self._estimated_updates_this_epoch(),
            "parameter_count": parameter_count,
            "optimizer": self.pipeline_config.optimization.optimizer,
            "learning_rate": self.pipeline_config.optimization.learning_rate,
            "loss_type": self.pipeline_config.trainer.loss.loss_type,
            "gamma": self.pipeline_config.trainer.loss.gamma,
            "gradient_clipping": self.pipeline_config.optimization.gradient_clipping,
            "max_grad_norm": self.pipeline_config.optimization.max_grad_norm,
            "warmup_size": self.pipeline_config.replay_buffer.warmup_size,
            "buffer_capacity": self.pipeline_config.replay_buffer.capacity,
            "batch_size": self.pipeline_config.replay_buffer.batch_size,
            "sequence_length": self.pipeline_config.replay_buffer.sequence_length,
            "min_sequence_length": self.pipeline_config.replay_buffer.min_sequence_length,
            "epsilon_betting_start": self.pipeline_config.epsilon.betting.start,
            "epsilon_betting_end": self.pipeline_config.epsilon.betting.end,
            "epsilon_betting_decay_steps": self.pipeline_config.epsilon.betting.decay_steps,
            "epsilon_playing_start": self.pipeline_config.epsilon.playing.start,
            "epsilon_playing_end": self.pipeline_config.epsilon.playing.end,
            "epsilon_playing_decay_steps": self.pipeline_config.epsilon.playing.decay_steps,
            "epsilon_start": self.pipeline_config.epsilon.playing.start,
            "epsilon_end": self.pipeline_config.epsilon.playing.end,
            "epsilon_decay_steps": self.pipeline_config.epsilon.playing.decay_steps,
            "target_update_mode": self.pipeline_config.target_update.mode,
            "target_hard_interval": self.pipeline_config.target_update.hard_update_interval,
            "target_soft_tau": self.pipeline_config.target_update.soft_tau,
            "eval_rounds": self.pipeline_config.evaluation.num_rounds,
            "eval_max_decisions": self.pipeline_config.evaluation.max_decisions,
            "checkpoint_dir": str(self.pipeline_config.checkpoints.directory_path),
            "n_step_enabled": self.pipeline_config.n_step.enabled,
            "n_step_size": self.pipeline_config.n_step.n_steps,
            "phase_loss_weights_enabled": self.pipeline_config.trainer.loss.phase_weights.enabled,
            "betting_loss_weight": self.pipeline_config.trainer.loss.phase_weights.betting_weight,
            "playing_loss_weight": self.pipeline_config.trainer.loss.phase_weights.playing_weight,
            "use_module_gating": bool(getattr(model_config, "use_module_gating", False)),
            "use_phase_adapters": bool(getattr(model_config, "use_phase_adapters", False)),
            "transfer_enabled": self.pipeline_config.transfer.enabled,
            "warm_start_checkpoint_path": self.pipeline_config.transfer.warm_start_checkpoint_path,
            "teacher_checkpoint_path": self.pipeline_config.transfer.teacher_checkpoint_path,
            "distillation_enabled": self.pipeline_config.transfer.distillation.enabled,
            "distillation_mode": self.pipeline_config.transfer.distillation.mode,
            "distillation_weight": self.pipeline_config.transfer.distillation.weight,
            "distillation_final_weight": self.pipeline_config.transfer.distillation.final_weight,
            "betting_auxiliary_enabled": self.pipeline_config.betting_auxiliary.enabled,
            "betting_auxiliary_mode": self.pipeline_config.betting_auxiliary.mode,
            "betting_auxiliary_weight": self.pipeline_config.betting_auxiliary.weight,
            "betting_auxiliary_final_weight": self.pipeline_config.betting_auxiliary.final_weight,
            "betting_auxiliary_min_observed_cards": self.pipeline_config.betting_auxiliary.min_observed_cards,
            "n_decks": reference_env.config.n_decks,
            "shoe_penetration": reference_env.config.shoe_penetration,
            "dealer_hits_soft_17": reference_env.config.dealer_hits_soft_17,
            "blackjack_payout": reference_env.config.blackjack_payout,
            "double_allowed_on": reference_env.config.double_allowed_on,
            "split_rule": reference_env.config.split_rule,
            "double_after_split_allowed": reference_env.config.double_after_split_allowed,
        }

    def _count_train_triggers_between(self, start_env_steps: int, end_env_steps: int) -> int:
        frequency = self.pipeline_config.trainer.train_frequency
        return max(0, (end_env_steps // frequency) - (start_env_steps // frequency))

    def _stack_recurrent_hidden_state(self, hidden_states: list[Any]) -> Any:
        first_state = hidden_states[0]
        if isinstance(first_state, tuple):
            return (
                torch.cat([state[0] for state in hidden_states], dim=1),
                torch.cat([state[1] for state in hidden_states], dim=1),
            )
        return torch.cat(hidden_states, dim=1)

    def _split_recurrent_hidden_state(self, hidden_state: Any, batch_size: int) -> list[Any]:
        if isinstance(hidden_state, tuple):
            h, c = hidden_state
            return [(h[:, index : index + 1].contiguous(), c[:, index : index + 1].contiguous()) for index in range(batch_size)]
        return [hidden_state[:, index : index + 1].contiguous() for index in range(batch_size)]

    def _batched_policy_inference(self, env_states: list[EnvironmentRunnerState]) -> tuple[torch.Tensor, torch.Tensor, list[Any] | None]:
        encoded_states = [env_state.encoded_response for env_state in env_states]
        with torch.no_grad():
            if self.is_recurrent:
                hidden_batch = self._stack_recurrent_hidden_state([env_state.hidden_state for env_state in env_states])
                batch = {
                    "state_vector": torch.stack([state["state_vector"] for state in encoded_states], dim=0).unsqueeze(1),
                    "action_mask": torch.stack([state["action_mask"] for state in encoded_states], dim=0).unsqueeze(1),
                    "padding_mask": torch.ones((len(encoded_states), 1), dtype=torch.bool),
                    "module_tensors": {},
                    "metadata": {"batch_size": len(encoded_states), "sequence_lengths": [1] * len(encoded_states)},
                }
                policy_output = self.online_network(batch, hidden_state=hidden_batch)
                next_hidden_states = self._split_recurrent_hidden_state(policy_output["hidden_state"], len(env_states))
                return (
                    policy_output["masked_q_values"].squeeze(1),
                    policy_output["action_mask"].squeeze(1),
                    next_hidden_states,
                )

            batch = {
                "state_vector": torch.stack([state["state_vector"] for state in encoded_states], dim=0),
                "action_mask": torch.stack([state["action_mask"] for state in encoded_states], dim=0),
                "module_tensors": {},
                "metadata": {"batch_size": len(encoded_states)},
            }
            policy_output = self.online_network(batch)
            return policy_output["masked_q_values"], policy_output["action_mask"], None

    def _temporary_eval_mode(self) -> tuple[bool, bool]:
        online_was_training = self.online_network.training
        if online_was_training:
            self.online_network.eval()
        return online_was_training, False

    def _restore_mode(self, online_was_training: bool, target_was_training: bool) -> None:
        if online_was_training:
            self.online_network.train()
        if target_was_training:
            self.target_network.train()

    def collect_experience(
        self,
        *,
        num_steps: int,
        tracker: BehaviorMetricsTracker,
        epsilon: float | None = None,
    ) -> None:
        online_was_training, target_was_training = self._temporary_eval_mode()
        try:
            for _ in range(num_steps):
                for env_state in self.env_states:
                    self._ensure_active_response(env_state, tracker)

                masked_q_values_batch, action_mask_batch, next_hidden_states = self._batched_policy_inference(self.env_states)
                if next_hidden_states is not None:
                    for env_state, hidden_state in zip(self.env_states, next_hidden_states):
                        env_state.hidden_state = hidden_state

                for env_index, env_state in enumerate(self.env_states):
                    masked_q_values = masked_q_values_batch[env_index]
                    action_mask = action_mask_batch[env_index]
                    decision_phase = infer_decision_phase(env_state.response)
                    epsilon_value = epsilon if epsilon is not None else self.epsilon_scheduler.value(decision_phase)

                    action_index, was_random = select_epsilon_greedy_action(
                        masked_q_values=masked_q_values,
                        action_mask=action_mask,
                        epsilon=epsilon_value,
                        rng=self.rng,
                    )
                    action_name = action_name_from_index(action_index, ACTION_ORDER)
                    try:
                        next_response = env_state.env.step(action_name)
                    except RuntimeError as exc:
                        if "shoe is empty" not in str(exc).lower() and "round is over" not in str(exc).lower():
                            raise
                        env_state.response = None
                        env_state.pending_n_step = []
                        continue

                    table_key = f"{env_state.env.start_state.mode}|{env_state.env.config.observation.profile}"
                    tracker.record_decision(
                        env_state.response,
                        action_name,
                        was_random=was_random,
                        table_key=table_key,
                        env_key=str(env_index),
                    )
                    tracker.record_round_result(next_response, env_key=str(env_index))
                    self.env_step_count += 1
                    self.epsilon_scheduler.step(decision_phase)
                    encoded_next_response = self._encode_response_for_storage(next_response, env=env_state.env)

                    transition = {
                        "state": env_state.encoded_response,
                        "next_state": encoded_next_response,
                        "action": action_index,
                        "reward": float(next_response["reward"]),
                        "done": bool(next_response["done"]),
                        "action_mask": env_state.encoded_response["action_mask"],
                        "next_action_mask": encoded_next_response["action_mask"],
                    }

                    self._store_transition(env_state, transition)

                    if next_response["done"]:
                        env_state.response = None
                        env_state.encoded_response = None
                        if self.is_recurrent and self.pipeline_config.trainer.reset_hidden_on_round_end:
                            env_state.hidden_state = self.online_network.init_hidden(batch_size=1, device=self.device)
                    else:
                        env_state.response = next_response
                        env_state.encoded_response = encoded_next_response
        finally:
            self._restore_mode(online_was_training, target_was_training)

    def _ensure_active_response(self, env_state: EnvironmentRunnerState, tracker: BehaviorMetricsTracker) -> None:
        while env_state.response is None or env_state.response["done"]:
            if env_state.response is not None and env_state.response["done"]:
                if self.is_recurrent and self.pipeline_config.trainer.reset_hidden_on_round_end:
                    env_state.hidden_state = self.online_network.init_hidden(batch_size=1, device=self.device)

            if env_state.response is None and env_state.pending_n_step:
                env_state.pending_n_step = []

            env_state.response = env_state.env.reset()
            env_state.encoded_response = self._encode_response_for_storage(env_state.response, env=env_state.env)
            if self.is_recurrent and env_state.hidden_state is None:
                env_state.hidden_state = self.online_network.init_hidden(batch_size=1, device=self.device)

    def _commit_pending_sequence(self, env_state: EnvironmentRunnerState) -> None:
        if not env_state.pending_sequence:
            return
        sequence = {
            key: [step[key] for step in env_state.pending_sequence]
            for key in ("state", "next_state", "action", "reward", "done", "action_mask", "next_action_mask", "n_steps")
        }
        self.replay_buffer.add(sequence)
        env_state.pending_sequence = []

    def _flush_pending_sequences(self) -> None:
        if not self.is_recurrent or not self.pipeline_config.trainer.flush_partial_sequences_at_epoch_end:
            return
        for env_state in self.env_states:
            self._commit_pending_sequence(env_state)

    def train_step(self) -> dict[str, Any]:
        batch = self.replay_buffer.sample()
        result = train_gradient_step(
            online_network=self.online_network,
            target_network=self.target_network,
            optimizer=self.optimizer,
            batch=batch,
            loss_config=self.pipeline_config.trainer.loss,
            optimization_config=self.pipeline_config.optimization,
            scheduler=self.scheduler,
            teacher_model=self.teacher_model,
            distillation_config=self.pipeline_config.transfer.distillation,
            betting_auxiliary_config=self.pipeline_config.betting_auxiliary,
            update_count=self.update_count,
        )
        self.update_count += 1
        target_synced = maybe_update_target(
            self.online_network,
            self.target_network,
            self.update_count,
            self.pipeline_config.target_update,
        )
        metrics = dict(result["metrics"])
        metrics.update(
            {
                "buffer_size": float(len(self.replay_buffer)),
                **self.epsilon_scheduler.current_values(),
                "epsilon": self.epsilon_scheduler.current_values()["epsilon_playing"],
                "target_synced": float(target_synced),
            }
        )
        result["metrics"] = metrics
        return result

    def evaluate(self) -> dict[str, Any]:
        eval_envs = clone_environments(self.envs, seed_offset=self.pipeline_config.trainer.seed + 100_000)
        for env in eval_envs:
            if hasattr(env, "set_runtime_options"):
                env.set_runtime_options(enable_transition_recording=False, compact_response_mode=True)
        return evaluate_policy(
            envs=eval_envs,
            model=self.online_network,
            epsilon=self.pipeline_config.epsilon,
            num_rounds=self.pipeline_config.evaluation.num_rounds,
            max_decisions=self.pipeline_config.evaluation.max_decisions,
            rng=random.Random(self.pipeline_config.trainer.seed + self.epoch_index),
            reset_hidden_on_round_end=self.pipeline_config.trainer.reset_hidden_on_round_end,
            betting_auxiliary_config=self.pipeline_config.betting_auxiliary,
        )

    def train_one_epoch(self) -> dict[str, Any]:
        self.epoch_index += 1
        epoch_start_time = perf_counter()
        self.logger.log_epoch_start(
            epoch=self.epoch_index,
            total_epochs=self.pipeline_config.trainer.total_epochs,
        )
        if not self._buffer_ready():
            self.warmup()

        behavior_tracker = BehaviorMetricsTracker()
        optimization_tracker = ScalarMetricAccumulator()
        updates_this_epoch = 0
        total_updates_this_epoch = self._estimated_updates_this_epoch()

        for step_in_epoch in range(1, self.pipeline_config.trainer.env_steps_per_epoch + 1):
            env_steps_before_collect = self.env_step_count
            self.collect_experience(num_steps=1, tracker=behavior_tracker)
            if self.logger.should_log_collection(step_in_epoch):
                self.logger.log_collection(
                    env_step_in_epoch=step_in_epoch,
                    total_env_steps_in_epoch=self.pipeline_config.trainer.env_steps_per_epoch,
                    buffer_size=len(self.replay_buffer),
                    warmup_target=self.pipeline_config.replay_buffer.warmup_size,
                    metrics=behavior_tracker.summary(),
                )

            train_triggers = self._count_train_triggers_between(env_steps_before_collect, self.env_step_count)
            if self._buffer_ready() and train_triggers > 0:
                for _ in range(train_triggers * self.pipeline_config.trainer.updates_per_train_step):
                    if (
                        self.pipeline_config.trainer.max_updates_per_epoch is not None
                        and updates_this_epoch >= self.pipeline_config.trainer.max_updates_per_epoch
                    ):
                        break
                    train_result = self.train_step()
                    updates_this_epoch += 1
                    optimization_tracker.update(train_result["metrics"])
                    self.logger.log_update(
                        update_in_epoch=updates_this_epoch,
                        total_updates_in_epoch=total_updates_this_epoch,
                        metrics=train_result["metrics"],
                    )

                    if (
                        self.pipeline_config.checkpoints.save_periodic
                        and self.update_count % self.pipeline_config.checkpoints.periodic_interval_updates == 0
                    ):
                        path = self.checkpoints.save_periodic(self, metrics=train_result["metrics"])
                        if path is not None:
                            self.logger.log_checkpoint(kind="periodic", path=str(path))

        self._flush_pending_sequences()

        train_metrics = behavior_tracker.summary()
        optimization_metrics = optimization_tracker.summary()
        epoch_summary = {
            **train_metrics,
            **optimization_metrics,
            "epoch": float(self.epoch_index),
            "env_steps": float(self.env_step_count),
            "updates": float(self.update_count),
            "updates_this_epoch": float(updates_this_epoch),
            "buffer_size": float(len(self.replay_buffer)),
            **self.epsilon_scheduler.current_values(),
            "epsilon": self.epsilon_scheduler.current_values()["epsilon_playing"],
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

        eval_metrics: dict[str, Any] | None = None
        if self.pipeline_config.evaluation.enabled and self.epoch_index % self.pipeline_config.evaluation.every_n_epochs == 0:
            eval_metrics = self.evaluate()
            self.logger.log_evaluation(metrics=eval_metrics)
            self.logger.log_train_val_comparison(train_metrics=epoch_summary, eval_metrics=eval_metrics)
            best_path = self.checkpoints.save_best(self, metrics=eval_metrics)
            if best_path is not None:
                self.best_eval_metrics = dict(eval_metrics)
                best_metric_name = self.pipeline_config.checkpoints.best_metric_name
                best_metric_value = float(eval_metrics[best_metric_name]) if best_metric_name in eval_metrics else None
                self.logger.log_checkpoint(
                    kind="best_eval",
                    path=str(best_path),
                    metric_name=best_metric_name,
                    metric_value=best_metric_value,
                )

        latest_path = self.checkpoints.save_latest(self, metrics=eval_metrics or epoch_summary)
        if latest_path is not None:
            self.logger.log_checkpoint(kind="latest", path=str(latest_path))

        final_summary = {
            **epoch_summary,
            "eval": eval_metrics,
        }
        self.training_history.append(final_summary)
        self.logger.log_epoch_summary(summary=epoch_summary)
        self.logger.log_epoch_time(epoch_time_sec=perf_counter() - epoch_start_time)
        return final_summary

    def train(self) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        self.logger.log_run_summary(summary=self.build_run_summary())
        for _ in range(self.pipeline_config.trainer.total_epochs):
            history.append(self.train_one_epoch())
        return {
            "history": history,
            "best_eval_metrics": deepcopy(self.best_eval_metrics),
            "state": self.state_dict(),
            "checkpoint_dir": str(self.pipeline_config.checkpoints.directory_path),
        }


def _ensure_supported_model(model: Any) -> None:
    if not isinstance(model, (FeedForwardDoubleDQN, RecurrentDoubleDQN, DuelingRecurrentDoubleDQN)):
        raise TypeError("Unsupported model type for BlackjackRLTrainer")


def build_trainer(
    envs: Any,
    model: Any,
    *,
    pipeline_config: TrainingPipelineConfig | None = None,
    target_network: Any | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> BlackjackRLTrainer:
    _ensure_supported_model(model)
    return BlackjackRLTrainer(
        envs,
        model,
        pipeline_config=pipeline_config,
        target_network=target_network,
        optimizer=optimizer,
        scheduler=scheduler,
    )

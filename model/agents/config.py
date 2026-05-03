from __future__ import annotations

from dataclasses import dataclass, field

from model.encoder import EncoderConfig


@dataclass(slots=True)
class AgentNetworkConfig:
    architecture: str = "feedforward"
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig.for_profile("table_realistic_default"))
    activation: str = "relu"
    use_layer_norm: bool = False
    dropout: float = 0.0
    feedforward_hidden_dims: tuple[int, ...] = (256, 256)
    projection_dim: int = 256
    recurrent_hidden_dim: int = 256
    recurrent_num_layers: int = 1
    recurrent_type: str = "gru"
    head_hidden_dim: int = 128
    value_hidden_dim: int = 128
    advantage_hidden_dim: int = 128
    use_phase_adapters: bool = True
    use_module_gating: bool = True
    use_count_auxiliary_head: bool = False
    count_auxiliary_hidden_dim: int = 128
    count_auxiliary_num_buckets: int = 4
    mask_bet_features_for_playing: bool = False
    play_feature_mask_module_names: tuple[str, ...] = ("bet", "betting_context")

    def __post_init__(self) -> None:
        if self.architecture not in {"feedforward", "recurrent", "dueling_recurrent"}:
            raise ValueError("architecture must be 'feedforward', 'recurrent', or 'dueling_recurrent'")
        if not isinstance(self.encoder, EncoderConfig):
            raise TypeError("encoder must be an EncoderConfig instance")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.recurrent_type not in {"gru", "lstm"}:
            raise ValueError("recurrent_type must be 'gru' or 'lstm'")
        if self.recurrent_num_layers <= 0:
            raise ValueError("recurrent_num_layers must be positive")
        if not self.feedforward_hidden_dims:
            raise ValueError("feedforward_hidden_dims must contain at least one hidden layer")
        if self.projection_dim <= 0 or self.recurrent_hidden_dim <= 0:
            raise ValueError("projection_dim and recurrent_hidden_dim must be positive")
        if self.head_hidden_dim <= 0 or self.value_hidden_dim <= 0 or self.advantage_hidden_dim <= 0:
            raise ValueError("head_hidden_dim, value_hidden_dim, and advantage_hidden_dim must be positive")
        if self.count_auxiliary_hidden_dim <= 0:
            raise ValueError("count_auxiliary_hidden_dim must be positive")
        if self.count_auxiliary_num_buckets <= 0:
            raise ValueError("count_auxiliary_num_buckets must be positive")
        if not isinstance(self.mask_bet_features_for_playing, bool):
            raise TypeError("mask_bet_features_for_playing must be bool")
        if not isinstance(self.play_feature_mask_module_names, tuple):
            self.play_feature_mask_module_names = tuple(self.play_feature_mask_module_names)
        if any(not isinstance(name, str) for name in self.play_feature_mask_module_names):
            raise TypeError("play_feature_mask_module_names must contain strings")
        if any(dim <= 0 for dim in self.feedforward_hidden_dims):
            raise ValueError("feedforward_hidden_dims must contain positive sizes")

    @classmethod
    def for_architecture(
        cls,
        architecture: str,
        *,
        encoder_profile: str = "table_realistic_default",
        **overrides: object,
    ) -> AgentNetworkConfig:
        encoder_config = EncoderConfig.for_profile(encoder_profile)

        if architecture == "feedforward":
            config = cls(
                architecture=architecture,
                encoder=encoder_config,
                activation="relu",
                use_layer_norm=False,
                dropout=0.0,
                feedforward_hidden_dims=(256, 256),
            )
        elif architecture == "recurrent":
            config = cls(
                architecture=architecture,
                encoder=encoder_config,
                activation="relu",
                use_layer_norm=True,
                dropout=0.0,
                projection_dim=256,
                recurrent_hidden_dim=256,
                recurrent_num_layers=1,
                recurrent_type="gru",
                head_hidden_dim=128,
            )
        elif architecture == "dueling_recurrent":
            config = cls(
                architecture=architecture,
                encoder=encoder_config,
                activation="relu",
                use_layer_norm=True,
                dropout=0.0,
                projection_dim=256,
                recurrent_hidden_dim=256,
                recurrent_num_layers=1,
                recurrent_type="gru",
                value_hidden_dim=128,
                advantage_hidden_dim=128,
            )
        else:
            raise ValueError("Unsupported architecture")

        for key, value in overrides.items():
            if key == "encoder_profile":
                continue
            setattr(config, key, value)
        config.__post_init__()
        return config

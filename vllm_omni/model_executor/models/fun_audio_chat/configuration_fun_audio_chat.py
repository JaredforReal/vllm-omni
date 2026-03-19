# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration class for Fun-Audio-Chat model.

This provides a proper PretrainedConfig subclass that correctly parses
the nested configuration (text_config, audio_config) from the model's config.json.
"""

from transformers import Qwen3Config
from transformers.configuration_utils import PretrainedConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class FunAudioChatAudioEncoderConfig(PretrainedConfig):
    """Audio configuration for Fun-Audio-Chat."""

    model_type = "funaudiochat_audio_encoder"

    def __init__(
        self,
        num_mel_bins: int = 128,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        encoder_layers: int = 32,
        d_model: int = 1280,
        dropout: float = 0,
        attention_dropout: float = 0,
        activation_function: str = "gelu",
        activation_dropout: float = 0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        max_source_positions: int = 1500,
        n_window: int = 100,
        output_dim: int = 4096,
        bos_token_id: int = 6561,
        codebook_size: int = 6565,
        continuous_features_mode: str = "replace",
        crq_transformer_config: dict | None = None,
        eos_token_id: int = 6562,
        group_size: int = 5,
        enable_audio_invert_tower: bool = True,
        pad_token_id: int = 6563,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim

        # Additional audio encoder parameters
        self.bos_token_id = bos_token_id
        self.codebook_size = codebook_size
        self.continuous_features_mode = continuous_features_mode
        self.crq_transformer_config = crq_transformer_config
        self.eos_token_id = eos_token_id
        self.group_size = group_size
        self.enable_audio_invert_tower = enable_audio_invert_tower
        self.pad_token_id = pad_token_id


class FunAudioChatConfig(PretrainedConfig):
    """Configuration class for Fun-Audio-Chat model.

    This handles the nested text_config and audio_config properly,
    converting them from dicts to PretrainedConfig objects.
    """

    model_type = "funaudiochat"
    attribute_map = {
        "audio_token_id": "audio_token_index",
    }
    sub_configs = {
        "text_config": Qwen3Config,
        "audio_config": FunAudioChatAudioEncoderConfig,
    }

    def __init__(
        self,
        text_config: dict | Qwen3Config | None = None,
        audio_config: dict | FunAudioChatAudioEncoderConfig | None = None,
        audio_token_index: int = 151626,
        ignore_index: int = -100,
        hidden_size: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_token_index = audio_token_index
        self.ignore_index = ignore_index

        if isinstance(text_config, dict):
            text_config = Qwen3Config(**text_config)
        if isinstance(audio_config, dict):
            audio_config = FunAudioChatAudioEncoderConfig(**audio_config)

        self.text_config = text_config
        self.audio_config = audio_config

    def get_text_config(self, **kwargs) -> Qwen3Config:
        """Return the text config for compatibility with vLLM."""
        return self.text_config

    @property
    def vocab_size(self) -> int:
        """Return vocab size from text config."""
        return getattr(self.text_config, "vocab_size", 151936)

    @property
    def hidden_size(self) -> int:
        """Return hidden size from text config."""
        return getattr(self.text_config, "hidden_size", 4096)

    @property
    def num_hidden_layers(self) -> int:
        """Return number of hidden layers from text config."""
        return getattr(self.text_config, "num_hidden_layers", 36)

    @property
    def num_attention_heads(self) -> int:
        """Return number of attention heads from text config."""
        return getattr(self.text_config, "num_attention_heads", 32)

    @property
    def num_key_value_heads(self) -> int:
        """Return number of key-value heads from text config."""
        return getattr(self.text_config, "num_key_value_heads", 8)

    @property
    def max_position_embeddings(self) -> int:
        """Return max position embeddings from text config."""
        return getattr(self.text_config, "max_position_embeddings", 262144)


__all__ = ["FunAudioChatConfig", "FunAudioChatAudioEncoderConfig"]

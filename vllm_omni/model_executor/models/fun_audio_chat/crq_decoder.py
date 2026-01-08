# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CRQ Transformer Decoder for Fun-Audio-Chat speech synthesis.

The CRQ (Codec Residual Quantization) Decoder is the second stage in the
Fun-Audio-Chat S2S pipeline. It takes LLM hidden states and generates
discrete speech tokens at 25Hz that can be converted to audio by CosyVoice.

Architecture:
- Pre-matching: Linear projection to expand hidden states by group_size (5x)
- CRQ Transformer: Qwen3-based transformer for autoregressive token generation
- LM Head: Projects to codebook vocabulary for token prediction

Reference: https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B
Official implementation: Fun-Audio-Chat/funaudiochat/modeling_funaudiochat.py
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class FunAudioChatCRQDecoder(nn.Module, SupportsPP):
    """
    CRQ Transformer Decoder for speech token generation.

    This implements the autoregressive decoder that converts LLM hidden states
    into discrete speech tokens. The decoder uses a Qwen3-based transformer
    with interleaved token generation across group_size (5) positions.

    Following official implementation in modeling_funaudiochat.py:
    - FunAudioChatDecoder class
    - crq_generate_forward method for inference

    Input: LLM hidden states [batch, seq_len, hidden_size]
    Output: Speech tokens [batch, seq_len * group_size] with values in [0, codebook_size)
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.audio_config = self.config.audio_config

        # Get decoder config from audio_config
        self.group_size = getattr(self.audio_config, "group_size", 5)
        self.hidden_size = getattr(self.audio_config, "output_dim", 3584)
        self.codebook_size = getattr(self.audio_config, "codebook_size", 6565)
        self.bos_token_id = getattr(self.audio_config, "bos_token_id", 6561)
        self.eos_token_id = getattr(self.audio_config, "eos_token_id", 6562)
        self.pad_token_id = getattr(self.audio_config, "pad_token_id", 6564)

        logger.info(
            f"Initializing CRQ Decoder: group_size={self.group_size}, "
            f"hidden_size={self.hidden_size}, codebook_size={self.codebook_size}"
        )

        # Pre-matching: Expand hidden states by group_size
        # Official: self.pre_matching = nn.Linear(self.hidden_size, self.hidden_size * self.group_size, bias=True)
        self.pre_matching = nn.Linear(self.hidden_size, self.hidden_size * self.group_size, bias=True)

        # CRQ Transformer: Qwen3-based decoder
        crq_config_dict = getattr(self.audio_config, "crq_transformer_config", None)
        if crq_config_dict is None:
            # Default CRQ config based on Fun-Audio-Chat
            crq_config_dict = {
                "model_type": "qwen3",
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 3072,
                "vocab_size": self.codebook_size,
                "max_position_embeddings": 32768,
                "head_dim": 64,
                "rope_theta": 1000000.0,
            }

        self.crq_config = AutoConfig.for_model(**crq_config_dict)
        self.crq_transformer = AutoModel.from_config(self.crq_config)

        # Remove embedding layer (we use our own embeddings from lm_head.weight)
        if hasattr(self.crq_transformer, "embed_tokens"):
            del self.crq_transformer.embed_tokens

        crq_hidden_size = self.crq_config.hidden_size

        # Input/Output matching layers
        # Official: self.input_matching = nn.Linear(self.hidden_size, crq_transformer_config.hidden_size, bias=False)
        self.input_matching = nn.Linear(self.hidden_size, crq_hidden_size, bias=False)
        # Official: self.output_matching = nn.Linear(crq_transformer_config.hidden_size, self.hidden_size, bias=False)
        self.output_matching = nn.Linear(crq_hidden_size, self.hidden_size, bias=False)

        # LM head for codebook prediction
        # Official: self.lm_head = nn.Linear(config.output_dim, config.codebook_size, bias=False)
        # Note: lm_head weights are tied to audio_tower.embed_tokens in official impl
        self.lm_head = nn.Linear(self.hidden_size, self.codebook_size, bias=False)

        # For empty intermediate tensors
        self.make_empty_intermediate_tensors = lambda: None

        # Cache for generated speech tokens to avoid regenerating on subsequent vLLM forward calls
        # vLLM calls forward repeatedly for token-by-token generation, but CRQ generates all at once
        self._cached_speech_tokens: torch.Tensor | None = None
        self._cached_hidden_states: torch.Tensor | None = None
        self._generation_complete: bool = False

    def reset_generation_cache(self):
        """Reset generation cache for new request."""
        self._cached_speech_tokens = None
        self._cached_hidden_states = None
        self._generation_complete = False

    def get_embeddings(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Get embeddings for audio tokens from LM head weights.

        Official implementation: self.lm_head.weight.data[audio_tokens]
        Since lm_head is tied to embed_tokens, this effectively uses the embedding table.
        """
        return self.lm_head.weight.data[audio_tokens]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """
        Forward pass for CRQ decoder.

        Args:
            input_ids: Not used directly (placeholder for vLLM interface)
            positions: Position IDs
            additional_information: Contains:
                - thinker_hidden_states: [batch, seq_len, hidden_size] from Stage 0
                - text_embeds: [batch, seq_len, hidden_size] from Stage 0

        Returns:
            OmniOutput with speech_tokens in multimodal_outputs
        """
        # Get device from model weights for dummy tensors
        device = self.lm_head.weight.device
        dtype = self.lm_head.weight.dtype

        # Determine number of tokens from positions tensor for dummy run
        # vLLM expects hidden_states shape [num_tokens, hidden_size]
        num_tokens = positions.numel() if positions is not None else 1

        # vLLM-omni passes runtime_additional_information as a list of dicts (one per request)
        # We need to extract the first request's info
        if additional_information is None:
            runtime_info = kwargs.get("runtime_additional_information", [])
            logger.debug(
                f"CRQ Decoder: kwargs keys = {list(kwargs.keys())}, "
                f"runtime_info type = {type(runtime_info)}, len = {len(runtime_info) if runtime_info else 0}"
            )
            if runtime_info and isinstance(runtime_info, list) and len(runtime_info) > 0:
                additional_information = runtime_info[0]  # Single request batch for stage pipeline
                info_keys = list(additional_information.keys()) if additional_information else None
                logger.debug(f"CRQ Decoder: extracted additional_information keys = {info_keys}")

        if additional_information is None or not additional_information:
            logger.warning(f"CRQ Decoder: No additional_information provided, kwargs keys = {list(kwargs.keys())}")
            # Check if we have cached results from a previous generation
            # This happens when vLLM calls forward again after generation is complete
            if self._generation_complete and self._cached_speech_tokens is not None:
                logger.debug("CRQ Decoder: Returning cached result (no additional_information, generation complete)")
                return OmniOutput(
                    text_hidden_states=self._cached_hidden_states,
                    multimodal_outputs={"speech_tokens": self._cached_speech_tokens},
                )
            # Return dummy tensors with correct shape for vLLM's _dummy_run
            return OmniOutput(
                text_hidden_states=torch.zeros(num_tokens, self.hidden_size, device=device, dtype=dtype),
                multimodal_outputs={"speech_tokens": None},
            )

        # Extract hidden states from Stage 0
        # Official: speech_inputs_embeds = last_hidden_state + text_embeds.detach()
        thinker_hidden_states = additional_information.get("thinker_hidden_states")
        text_embeds = additional_information.get("text_embeds")

        if thinker_hidden_states is None:
            logger.warning("CRQ Decoder: No thinker_hidden_states in additional_information")
            # Check if we have cached results from a previous generation
            if self._generation_complete and self._cached_speech_tokens is not None:
                logger.debug("CRQ Decoder: Returning cached result (no hidden_states, generation complete)")
                return OmniOutput(
                    text_hidden_states=self._cached_hidden_states,
                    multimodal_outputs={"speech_tokens": self._cached_speech_tokens},
                )
            return OmniOutput(
                text_hidden_states=torch.zeros(num_tokens, self.hidden_size, device=device, dtype=dtype),
                multimodal_outputs={"speech_tokens": None},
            )

        # New generation request - clear any cached state from previous request
        self._generation_complete = False
        self._cached_speech_tokens = None
        self._cached_hidden_states = None

        # Move tensors from CPU to GPU (they come from additional_information_cpu)
        # and ensure correct dtype
        model_device = self.lm_head.weight.device
        model_dtype = self.lm_head.weight.dtype
        thinker_hidden_states = thinker_hidden_states.to(device=model_device, dtype=model_dtype)
        if text_embeds is not None:
            text_embeds = text_embeds.to(device=model_device, dtype=model_dtype)

        device = model_device
        dtype = model_dtype

        # Ensure 3D shape [batch, seq_len, hidden_size]
        if thinker_hidden_states.dim() == 2:
            thinker_hidden_states = thinker_hidden_states.unsqueeze(0)

        # Combine hidden states with text embeddings (following official impl)
        # Official: speech_inputs_embeds = speech_inputs_embeds + text_embeds.detach()
        if text_embeds is not None:
            if text_embeds.dim() == 2:
                text_embeds = text_embeds.unsqueeze(0)
            # Ensure same shape
            if text_embeds.shape[1] != thinker_hidden_states.shape[1]:
                min_len = min(text_embeds.shape[1], thinker_hidden_states.shape[1])
                text_embeds = text_embeds[:, :min_len, :]
                thinker_hidden_states = thinker_hidden_states[:, :min_len, :]
            speech_inputs_embeds = thinker_hidden_states + text_embeds.detach()
        else:
            speech_inputs_embeds = thinker_hidden_states

        # Generate speech tokens autoregressively using official crq_generate_forward logic
        speech_tokens = self._crq_generate_forward(speech_inputs_embeds)

        # Cache results for subsequent forward calls
        # vLLM may call forward multiple times, we return cached results after first generation
        # We return EOS-biased hidden states so vLLM samples EOS and terminates
        eos_hidden_states = self._get_eos_hidden_states(device, dtype)
        self._cached_speech_tokens = speech_tokens
        self._cached_hidden_states = eos_hidden_states
        self._generation_complete = True

        logger.info(f"CRQ Decoder: Generation complete, cached {speech_tokens.shape[1]} speech tokens")

        return OmniOutput(
            text_hidden_states=eos_hidden_states,
            multimodal_outputs={"speech_tokens": speech_tokens},
        )

    def _get_eos_hidden_states(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get hidden states that will produce EOS-biased logits when passed through lm_head.

        This is used to signal to vLLM that generation is complete.
        We use the EOS embedding from lm_head.weight as the hidden state.
        When compute_logits calls lm_head(hidden_states), the dot product with EOS embedding
        will be maximized, causing EOS to be sampled.
        """
        # Get EOS embedding from lm_head weight
        eos_embedding = self.lm_head.weight.data[self.eos_token_id]  # [hidden_size]
        # Normalize to unit vector for stable dot product
        eos_embedding = eos_embedding / (eos_embedding.norm() + 1e-6)
        # Return as [1, hidden_size] for single token
        return eos_embedding.unsqueeze(0).to(device=device, dtype=dtype)

    def _crq_generate_forward(
        self,
        inputs_embeds: torch.Tensor,
        temperature: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate speech tokens autoregressively using CRQ decoder.

        This follows the official crq_generate_forward implementation in
        modeling_funaudiochat.py (FunAudioChatDecoder class).

        Note: Unlike the official implementation which runs incrementally per text token,
        this version generates all speech tokens at once from the complete hidden states.
        This is necessary for vLLM's staged pipeline architecture.

        Args:
            inputs_embeds: Combined hidden states [batch, seq_len, hidden_size]
                          This is (last_hidden_state + text_embeds.detach())
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Speech tokens [batch, seq_len * group_size]
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # Step 1: Pre-matching - expand to group_size
        # Shape: [bs, slen, hidden_size] -> [bs, slen, hidden_size * group_size]
        expanded = self.pre_matching(inputs_embeds)

        # Reshape to [bs, slen * group_size, hidden_size]
        hidden_states = expanded.reshape(batch_size, seq_len * self.group_size, -1)
        total_positions = seq_len * self.group_size

        # Initialize audio embeddings with BOS token
        bos_embedding = self.get_embeddings(torch.tensor([self.bos_token_id], device=device))
        audio_embeds = bos_embedding.unsqueeze(0).repeat(batch_size, 1, 1).to(dtype=dtype)

        all_tokens: list[torch.Tensor] = []
        past_key_values = None

        # Generate tokens position by position
        # Official implementation generates group_size tokens per "step" (per text token)
        # Here we process all positions sequentially to match the expected behavior
        for pos in range(total_positions):
            # Determine input based on position within group
            group_idx = pos % self.group_size

            if group_idx == 0:
                # First position in group: use context from start to current
                # This matches: hidden_states[:, : slen * group_size - (group_size - 0 - 1)]
                # For pos=0: we use hidden_states[:, :1]
                input_embeds = hidden_states[:, : pos + 1] + audio_embeds
            else:
                # Subsequent positions: only current position
                input_embeds = hidden_states[:, pos : pos + 1] + audio_embeds.unsqueeze(1)

            # Project to CRQ transformer dimension
            input_embeds = self.input_matching(input_embeds)

            # Forward through CRQ transformer with KV cache
            outputs = self.crq_transformer(
                inputs_embeds=input_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

            # Get last hidden state and compute logits
            last_hidden = self.output_matching(outputs.last_hidden_state[:, -1:, :])
            logits = self.lm_head(last_hidden).squeeze(1)  # [batch, vocab]

            # Sample next token
            if do_sample and temperature > 0:
                logits = logits / temperature
                probs = torch.nn.functional.softmax(logits.float(), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            all_tokens.append(next_token)

            # Check for EOS
            if (next_token == self.eos_token_id).any():
                logger.info(f"CRQ Decoder: EOS detected at position {pos}")
                break

            # Update audio embeddings for next position
            audio_embeds = self.get_embeddings(next_token).to(dtype=dtype)

        # Clean up KV cache to free memory
        del past_key_values
        del hidden_states
        del expanded

        # Concatenate all tokens
        if all_tokens:
            speech_tokens = torch.stack(all_tokens, dim=1)  # [batch, num_tokens]
        else:
            speech_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=device)

        logger.info(
            f"CRQ Decoder: Generated {speech_tokens.shape[1]} speech tokens "
            f"from {seq_len} input steps (max {total_positions})"
        )
        return speech_tokens

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        prefix: str = "",
    ) -> set[str]:
        """Load CRQ decoder weights from audio_invert_tower prefix.

        Weight mapping from official model:
        - audio_invert_tower.pre_matching -> pre_matching
        - audio_invert_tower.crq_transformer -> crq_transformer
        - audio_invert_tower.input_matching -> input_matching
        - audio_invert_tower.output_matching -> output_matching
        - audio_invert_tower.lm_head -> lm_head

        Note: In official impl, lm_head.weight is tied to audio_tower.embed_tokens.weight
        We handle this by also accepting audio_tower.embed_tokens weights.

        Args:
            weights: Iterable of (name, tensor) tuples from checkpoint
            prefix: The prefix used in parent model's state_dict (e.g., "crq_decoder.")
                   This is used to return correct weight names for vLLM's loader.
        """
        loaded_weights: set[str] = set()

        # Collect weights
        weights_dict: dict[str, torch.Tensor] = {}
        embed_tokens_weight: torch.Tensor | None = None

        for name, weight in weights:
            # Handle audio_invert_tower prefix
            if name.startswith("audio_invert_tower."):
                param_name = name.replace("audio_invert_tower.", "")
                weights_dict[param_name] = weight
            # Also capture embed_tokens for lm_head weight tying
            elif name == "audio_tower.embed_tokens.weight":
                embed_tokens_weight = weight

        # Load into current model
        state_dict = self.state_dict()

        for name, weight in weights_dict.items():
            if name in state_dict:
                if state_dict[name].shape == weight.shape:
                    state_dict[name].copy_(weight)
                    # Return the weight name as it appears in the parent model's state_dict
                    loaded_weights.add(f"{prefix}{name}")
                else:
                    logger.warning(f"Shape mismatch for {name}: expected {state_dict[name].shape}, got {weight.shape}")
            else:
                logger.debug(f"Skipping weight: {name}")

        # Handle lm_head weight tying with embed_tokens
        # If lm_head.weight wasn't loaded but embed_tokens is available, use it
        if "lm_head.weight" not in weights_dict and embed_tokens_weight is not None:
            if state_dict["lm_head.weight"].shape == embed_tokens_weight.shape:
                state_dict["lm_head.weight"].copy_(embed_tokens_weight)
                loaded_weights.add(f"{prefix}lm_head.weight")
                logger.info("Tied lm_head.weight to audio_tower.embed_tokens.weight")

        logger.info(f"Loaded {len(loaded_weights)} CRQ decoder weights")
        return loaded_weights


__all__ = ["FunAudioChatCRQDecoder"]

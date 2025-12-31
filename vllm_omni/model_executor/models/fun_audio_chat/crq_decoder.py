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
        if additional_information is None:
            logger.warning("CRQ Decoder: No additional_information provided")
            return OmniOutput(
                text_hidden_states=torch.zeros(1, self.hidden_size),
                multimodal_outputs={"speech_tokens": None},
            )

        # Extract hidden states from Stage 0
        # Official: speech_inputs_embeds = last_hidden_state + text_embeds.detach()
        thinker_hidden_states = additional_information.get("thinker_hidden_states")
        text_embeds = additional_information.get("text_embeds")

        if thinker_hidden_states is None:
            logger.warning("CRQ Decoder: No thinker_hidden_states provided")
            return OmniOutput(
                text_hidden_states=torch.zeros(1, self.hidden_size),
                multimodal_outputs={"speech_tokens": None},
            )

        device = thinker_hidden_states.device
        dtype = thinker_hidden_states.dtype

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
            speech_inputs_embeds = thinker_hidden_states + text_embeds.detach().to(device, dtype)
        else:
            speech_inputs_embeds = thinker_hidden_states

        # Generate speech tokens autoregressively using official crq_generate_forward logic
        speech_tokens = self._crq_generate_forward(speech_inputs_embeds)

        return OmniOutput(
            text_hidden_states=thinker_hidden_states.reshape(-1, thinker_hidden_states.shape[-1]),
            multimodal_outputs={"speech_tokens": speech_tokens},
        )

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
        # Official: inputs_embeds = self.pre_matching(inputs_embeds)
        # Shape: [bs, slen, hidden_size] -> [bs, slen, hidden_size * group_size]
        expanded = self.pre_matching(inputs_embeds)

        # Official: hidden_states = inputs_embeds.reshape(bs, slen * self.group_size, -1)
        # Shape: [bs, slen * group_size, hidden_size]
        hidden_states = expanded.reshape(batch_size, seq_len * self.group_size, -1)

        # Initialize audio embeddings with BOS token
        # Official:
        # self.crq_audio_embeds = (
        #     self.get_embeddings(self.config.bos_token_id)[None, None, :]
        #     .repeat(bs, 1, 1)
        #     .to(dtype=hidden_states.dtype, device=hidden_states.device)
        # )
        audio_embeds = (
            self.get_embeddings(torch.tensor([self.bos_token_id], device=device))
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .to(dtype=dtype)
        )

        all_tokens = []
        past_key_values = None

        # Accumulated generated tokens for potential logits processing
        generated_tokens: list[torch.Tensor] = []

        # Generate step by step
        for step in range(seq_len):
            step_tokens = []

            for i in range(self.group_size):
                # Official logic from crq_generate_forward:
                # if i == 0:
                #     input_embeds = (
                #         hidden_states[:, : slen * self.group_size - (self.group_size - i - 1)] + self.crq_audio_embeds
                #     )
                # else:
                #     input_embeds = (
                #         hidden_states[:, slen * self.group_size - (self.group_size - i)] + self.crq_audio_embeds
                #     ).unsqueeze(1)

                if i == 0:
                    # First position in group: use context from start to current position
                    # end_idx = total_len - (group_size - i - 1) = total_len - group_size + 1
                    # For step=0, i=0: end_idx = total_len - 4 (assuming group_size=5)
                    # But we're generating step by step, so we use:
                    # For step=s: end_idx = (s + 1) * group_size - (group_size - 1) = s * group_size + 1
                    end_idx = step * self.group_size + 1
                    input_embeds = hidden_states[:, :end_idx] + audio_embeds
                else:
                    # Subsequent positions: only current position
                    # pos_idx = total_len - (group_size - i) for the original full-sequence logic
                    # For step-by-step: pos_idx = step * group_size + i
                    pos_idx = step * self.group_size + i
                    input_embeds = hidden_states[:, pos_idx : pos_idx + 1] + audio_embeds.unsqueeze(1)

                # Project to CRQ transformer dimension
                # Official: input_embeds = self.input_matching(input_embeds)
                input_embeds = self.input_matching(input_embeds)

                # Forward through CRQ transformer with KV cache
                # Official:
                # outputs = self.crq_transformer(
                #     inputs_embeds=input_embeds,
                #     past_key_values=self.crq_past_key_values,
                #     use_cache=True,
                #     return_dict=True,
                # )
                # self.crq_past_key_values = outputs.past_key_values
                outputs = self.crq_transformer(
                    inputs_embeds=input_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values

                # Project back and compute logits
                # Official:
                # lhidden_states = outputs.last_hidden_state
                # lhidden_states = self.output_matching(lhidden_states)
                # logits = self.lm_head(lhidden_states)
                lhidden_states = self.output_matching(outputs.last_hidden_state)
                logits = self.lm_head(lhidden_states)

                # Sampling step
                # Official: crq_audio_tokens, logits = self.sampling_step(logits)
                next_token_logits = logits[:, -1, :].clone().float()

                # Apply temperature
                if do_sample and temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                step_tokens.append(next_tokens)
                generated_tokens.append(next_tokens)

                # Update audio embeddings for next position
                # Official: self.crq_audio_embeds = self.get_embeddings(crq_audio_tokens)
                audio_embeds = self.get_embeddings(next_tokens).to(dtype=dtype)

            # Concatenate tokens for this step
            # Official: self.crq_generate_tokens = torch.cat(self.crq_generate_tokens, dim=1)
            step_tokens_tensor = torch.stack(step_tokens, dim=1)  # [batch, group_size]
            all_tokens.append(step_tokens_tensor)

            # Check for EOS in any position
            if (step_tokens_tensor == self.eos_token_id).any():
                logger.info(f"CRQ Decoder: EOS detected at step {step}")
                break

        # Concatenate all tokens
        if all_tokens:
            speech_tokens = torch.cat(all_tokens, dim=1)  # [batch, num_steps * group_size]
        else:
            speech_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=device)

        logger.info(f"CRQ Decoder: Generated {speech_tokens.shape[1]} speech tokens from {seq_len} input steps")
        return speech_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load CRQ decoder weights from audio_invert_tower prefix.

        Weight mapping from official model:
        - audio_invert_tower.pre_matching -> pre_matching
        - audio_invert_tower.crq_transformer -> crq_transformer
        - audio_invert_tower.input_matching -> input_matching
        - audio_invert_tower.output_matching -> output_matching
        - audio_invert_tower.lm_head -> lm_head

        Note: In official impl, lm_head.weight is tied to audio_tower.embed_tokens.weight
        We handle this by also accepting audio_tower.embed_tokens weights.
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
                    loaded_weights.add(f"audio_invert_tower.{name}")
                else:
                    logger.warning(f"Shape mismatch for {name}: expected {state_dict[name].shape}, got {weight.shape}")
            else:
                logger.debug(f"Skipping weight: {name}")

        # Handle lm_head weight tying with embed_tokens
        # If lm_head.weight wasn't loaded but embed_tokens is available, use it
        if "lm_head.weight" not in weights_dict and embed_tokens_weight is not None:
            if state_dict["lm_head.weight"].shape == embed_tokens_weight.shape:
                state_dict["lm_head.weight"].copy_(embed_tokens_weight)
                loaded_weights.add("audio_tower.embed_tokens.weight (tied to lm_head)")
                logger.info("Tied lm_head.weight to audio_tower.embed_tokens.weight")

        logger.info(f"Loaded {len(loaded_weights)} CRQ decoder weights")
        return loaded_weights


__all__ = ["FunAudioChatCRQDecoder"]

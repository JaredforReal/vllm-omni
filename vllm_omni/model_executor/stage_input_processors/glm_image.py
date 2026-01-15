# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for GLM-Image: AR → Diffusion transition."""

from math import sqrt
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def _upsample_token_ids(token_ids: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
    """Upsample token IDs by 2x using nearest neighbor interpolation.

    GLM-Image AR model generates tokens at 32x downsampling, but DiT expects
    16x downsampling, so we need to upsample by 2x.

    Args:
        token_ids: Prior token IDs of shape [num_tokens]
        token_h: Height in token space (at 32x downsampling)
        token_w: Width in token space (at 32x downsampling)

    Returns:
        Upsampled token IDs of shape [num_tokens * 4]
    """
    token_ids = token_ids.view(1, 1, token_h, token_w)
    token_ids = torch.nn.functional.interpolate(token_ids.float(), scale_factor=2, mode="nearest").to(dtype=torch.long)
    token_ids = token_ids.view(-1)
    return token_ids


def _parse_generated_tokens(
    token_ids: list[int],
    height: int,
    width: int,
    factor: int = 32,
) -> tuple[torch.Tensor, int, int]:
    """Parse AR-generated tokens to extract prior_token_ids.

    The AR model generates tokens in a specific format:
    - For text-to-image: small_image_tokens + large_image_tokens + EOS
    - For image-to-image: large_image_tokens + EOS

    We need to extract the large_image_tokens and upsample them.

    Args:
        token_ids: Generated token IDs from AR model
        height: Target image height
        width: Target image width
        factor: Downsampling factor (default 32 for AR output)

    Returns:
        Tuple of (upsampled_prior_token_ids, pixel_height, pixel_width)
    """
    # Calculate token dimensions
    token_h = height // factor
    token_w = width // factor
    large_image_tokens = token_h * token_w

    # Calculate small image dimensions (used in text-to-image)
    ratio = token_h / token_w
    prev_token_h = int(sqrt(ratio) * (factor // 2))
    prev_token_w = int(sqrt(1 / ratio) * (factor // 2))
    small_image_tokens = prev_token_h * prev_token_w

    # Determine if this is text-to-image (has small + large) or image-to-image (large only)
    total_expected_t2i = small_image_tokens + large_image_tokens + 1  # +1 for EOS
    total_expected_i2i = large_image_tokens + 1

    token_tensor = torch.tensor(token_ids, dtype=torch.long)

    if len(token_ids) >= total_expected_t2i:
        # Text-to-image: extract large image tokens after small image tokens
        large_start = small_image_tokens
        large_end = large_start + large_image_tokens
        prior_token_ids_d32 = token_tensor[large_start:large_end]
    elif len(token_ids) >= total_expected_i2i:
        # Image-to-image: large image tokens are at the beginning
        prior_token_ids_d32 = token_tensor[:large_image_tokens]
    else:
        # Fallback: use whatever tokens we have
        logger.warning(
            f"Unexpected token count: {len(token_ids)}, expected at least {total_expected_i2i}. Using available tokens."
        )
        prior_token_ids_d32 = token_tensor[:large_image_tokens]

    # Upsample from 32x to 16x
    prior_token_ids = _upsample_token_ids(prior_token_ids_d32, token_h, token_w)

    return prior_token_ids, height, width


def ar2diffusion(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """
    Process AR stage outputs to create Diffusion stage inputs.

    This function bridges the AR model (which generates prior_token_ids) and
    the Diffusion pipeline (which uses them for conditioned denoising).

    Workflow:
    1. Extract generated token_ids from AR stage output
    2. Parse and upsample prior_token_ids (32x → 16x)
    3. Package into diffusion request format with original prompt info

    Args:
        stage_list: List of stage objects containing outputs
        engine_input_source: Source stage IDs (typically [0] for AR stage)
        prompt: Original prompt data (contains height, width, prompt text, images)
        requires_multimodal_data: Whether to pass multimodal data (condition images)

    Returns:
        List of dicts containing diffusion request parameters
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    ar_outputs = stage_list[source_stage_id].engine_outputs
    diffusion_inputs = []

    # Normalize prompt to list
    if not isinstance(prompt, list):
        prompt = [prompt] if prompt is not None else [{}]

    for i, ar_output in enumerate(ar_outputs):
        output = ar_output.outputs[0]
        generated_token_ids = output.token_ids

        # Get original prompt info
        original_prompt = prompt[i] if i < len(prompt) else {}
        if isinstance(original_prompt, (OmniTokensPrompt, TextPrompt)):
            original_prompt = dict(original_prompt) if hasattr(original_prompt, "__iter__") else {}

        # Extract dimensions from original prompt or use defaults
        height = original_prompt.get("height", 1024)
        width = original_prompt.get("width", 1024)
        text_prompt = original_prompt.get("prompt", "")

        # Parse and upsample prior tokens
        prior_token_ids, pixel_h, pixel_w = _parse_generated_tokens(generated_token_ids, height, width)

        # Build diffusion input
        # The diffusion stage expects these in OmniDiffusionRequest format
        diffusion_input = {
            "prompt": text_prompt,
            "height": pixel_h,
            "width": pixel_w,
            "extra": {
                "prior_token_ids": prior_token_ids,
                # Pass condition image info for image-to-image mode
                "prior_token_image_ids": output.multimodal_output.get("prior_token_image_ids")
                if hasattr(output, "multimodal_output") and output.multimodal_output
                else None,
            },
        }

        # Include multimodal data (condition images) if required
        if requires_multimodal_data:
            mm_data = original_prompt.get("multi_modal_data")
            if mm_data:
                diffusion_input["pil_image"] = mm_data.get("image")

        # Copy other relevant parameters from original prompt
        for key in ["seed", "num_inference_steps", "guidance_scale", "negative_prompt"]:
            if key in original_prompt:
                diffusion_input[key] = original_prompt[key]

        diffusion_inputs.append(diffusion_input)

    logger.debug(f"ar2diffusion: processed {len(ar_outputs)} AR outputs → {len(diffusion_inputs)} diffusion inputs")

    return diffusion_inputs

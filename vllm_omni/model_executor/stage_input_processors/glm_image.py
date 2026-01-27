# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for GLM-Image: AR → Diffusion transition."""

import time
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
    """Parse AR-generated tokens to extract prior_token_ids."""
    # Calculate token dimensions for target image
    token_h = height // factor
    token_w = width // factor
    large_image_tokens = token_h * token_w

    # Calculate small preview image dimensions (used in text-to-image)
    small_token_h = token_h // 2
    small_token_w = token_w // 2
    small_image_tokens = small_token_h * small_token_w

    token_tensor = torch.tensor(token_ids, dtype=torch.long)

    # Remove EOS token (16385) from the end if present
    eos_token_id = 16385
    if len(token_ids) > 0 and token_ids[-1] == eos_token_id:
        token_tensor = token_tensor[:-1]
        logger.debug(f"[_parse_generated_tokens] Removed EOS token, new length={len(token_tensor)}")

    actual_tokens = len(token_tensor)

    logger.debug(
        f"[_parse_generated_tokens] height={height}, width={width}, "
        f"token_h={token_h}, token_w={token_w}, "
        f"large_image_tokens={large_image_tokens}, small_image_tokens={small_image_tokens}, "
        f"actual_tokens={actual_tokens}"
    )

    if actual_tokens >= small_image_tokens + large_image_tokens:
        # Text-to-image: extract large image tokens after small image tokens
        large_start = small_image_tokens
        large_end = large_start + large_image_tokens
        prior_token_ids_d32 = token_tensor[large_start:large_end]
        actual_h, actual_w = token_h, token_w
        logger.info(
            f"[_parse_generated_tokens] t2i mode: extracting tokens [{large_start}:{large_end}], "
            f"grid={actual_h}x{actual_w}"
        )
    elif actual_tokens >= large_image_tokens:
        # Image-to-image: large image tokens are at the beginning
        prior_token_ids_d32 = token_tensor[:large_image_tokens]
        actual_h, actual_w = token_h, token_w
        logger.info(
            f"[_parse_generated_tokens] i2i mode: extracting tokens [0:{large_image_tokens}], "
            f"grid={actual_h}x{actual_w}"
        )
    else:
        # Insufficient tokens - try to infer the actual grid size
        import math

        for scale in [1, 2, 4]:
            test_h = token_h // scale
            test_w = token_w // scale
            test_small_h = test_h // 2
            test_small_w = test_w // 2
            test_large = test_h * test_w
            test_small = test_small_h * test_small_w

            if actual_tokens >= test_small + test_large:
                prior_token_ids_d32 = token_tensor[test_small : test_small + test_large]
                actual_h, actual_w = test_h, test_w
                height = test_h * factor
                width = test_w * factor
                logger.warning(f"Adjusted grid to {test_h}x{test_w}, output will be {height}x{width}")
                break
            elif actual_tokens >= test_large:
                prior_token_ids_d32 = token_tensor[:test_large]
                actual_h, actual_w = test_h, test_w
                height = test_h * factor
                width = test_w * factor
                logger.warning(f"Adjusted grid to {test_h}x{test_w}, output will be {height}x{width}")
                break
        else:
            sqrt_tokens = int(math.sqrt(actual_tokens))
            actual_h = actual_w = sqrt_tokens
            usable_tokens = sqrt_tokens * sqrt_tokens
            prior_token_ids_d32 = token_tensor[:usable_tokens]
            height = sqrt_tokens * factor
            width = sqrt_tokens * factor
            logger.error(f"Grid pattern mismatch. Using {sqrt_tokens}x{sqrt_tokens}, output: {height}x{width}")

    # Upsample from 32x to 16x
    prior_token_ids = _upsample_token_ids(prior_token_ids_d32, actual_h, actual_w)

    return prior_token_ids, height, width


def ar2diffusion(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Process AR stage outputs to create Diffusion stage inputs."""
    t_start = time.perf_counter()

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
        if isinstance(original_prompt, dict):
            pass
        elif hasattr(original_prompt, "_asdict"):
            original_prompt = original_prompt._asdict()
        elif hasattr(original_prompt, "__dict__"):
            original_prompt = vars(original_prompt)
        else:
            original_prompt = {}

        height = original_prompt.get("height", 1024)
        width = original_prompt.get("width", 1024)
        text_prompt = original_prompt.get("prompt", "")

        # Parse and upsample prior tokens
        t_parse_start = time.perf_counter()
        prior_token_ids, pixel_h, pixel_w = _parse_generated_tokens(generated_token_ids, height, width)
        t_parse_end = time.perf_counter()

        # Get prior_token_image_ids from AR model output (for i2i mode)
        # This contains VQ-VAE tokens from input image, used for KV cache conditioning
        # NOTE: multimodal_output is attached to ar_output (RequestOutput), NOT output (CompletionOutput)
        prior_token_image_ids = None

        # Debug: log available attributes
        logger.debug(
            f"[ar2diffusion] Request {i}: "
            f"ar_output type={type(ar_output).__name__}, "
            f"has multimodal_output={hasattr(ar_output, 'multimodal_output')}"
        )

        # Check ar_output (RequestOutput) for multimodal_output - this is the correct location
        if hasattr(ar_output, "multimodal_output") and ar_output.multimodal_output:
            mm_output = ar_output.multimodal_output
            logger.debug(
                f"[ar2diffusion] Request {i}: multimodal_output keys={list(mm_output.keys()) if isinstance(mm_output, dict) else type(mm_output)}"
            )
            if isinstance(mm_output, dict):
                raw_prior_image_ids = mm_output.get("prior_token_image_ids")
                if raw_prior_image_ids is not None:
                    # Wrap in list if it's a single tensor (expected by diffusion pipeline)
                    if isinstance(raw_prior_image_ids, torch.Tensor):
                        prior_token_image_ids = [raw_prior_image_ids]
                        logger.info(
                            f"[ar2diffusion] Request {i}: got prior_token_image_ids tensor, shape={raw_prior_image_ids.shape}"
                        )
                    elif isinstance(raw_prior_image_ids, list):
                        prior_token_image_ids = raw_prior_image_ids
                        shapes = [t.shape if isinstance(t, torch.Tensor) else type(t) for t in raw_prior_image_ids]
                        logger.info(f"[ar2diffusion] Request {i}: got prior_token_image_ids list, shapes={shapes}")
        else:
            # Fallback: also check output (CompletionOutput) in case of different vLLM versions
            if hasattr(output, "multimodal_output") and output.multimodal_output:
                mm_output = output.multimodal_output
                logger.debug(f"[ar2diffusion] Request {i}: found multimodal_output on CompletionOutput (fallback)")
                if isinstance(mm_output, dict):
                    raw_prior_image_ids = mm_output.get("prior_token_image_ids")
                    if raw_prior_image_ids is not None:
                        if isinstance(raw_prior_image_ids, torch.Tensor):
                            prior_token_image_ids = [raw_prior_image_ids]
                        elif isinstance(raw_prior_image_ids, list):
                            prior_token_image_ids = raw_prior_image_ids

        diffusion_input = {
            "prompt": text_prompt,
            "height": pixel_h,
            "width": pixel_w,
            "extra": {
                "prior_token_ids": prior_token_ids,
                "prior_token_image_ids": prior_token_image_ids,
            },
        }

        # Log whether this is t2i or i2i mode
        mode = "i2i" if prior_token_image_ids is not None else "t2i"
        logger.info(
            f"[ar2diffusion] Request {i}: mode={mode}, "
            f"prior_token_image_ids={'present' if prior_token_image_ids else 'None'}"
        )

        if requires_multimodal_data:
            mm_data = original_prompt.get("multi_modal_data")
            if mm_data:
                diffusion_input["pil_image"] = mm_data.get("image")

        for key in ["seed", "num_inference_steps", "guidance_scale", "negative_prompt"]:
            if key in original_prompt:
                diffusion_input[key] = original_prompt[key]

        diffusion_inputs.append(diffusion_input)
        logger.info(
            f"[Profile] ar2diffusion request {i}: parse_tokens={t_parse_end - t_parse_start:.4f}s, "
            f"num_ar_tokens={len(generated_token_ids)}, prior_shape={prior_token_ids.shape}"
        )

    t_end = time.perf_counter()
    logger.info(
        f"[Profile] ar2diffusion total: {t_end - t_start:.4f}s, "
        f"processed {len(ar_outputs)} AR outputs → {len(diffusion_inputs)} diffusion inputs"
    )

    return diffusion_inputs

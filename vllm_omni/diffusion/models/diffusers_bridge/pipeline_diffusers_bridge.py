# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diffusers Bridge Pipeline -- zero-code integration of any diffusers model.

Wraps an arbitrary ``diffusers.DiffusionPipeline`` behind the vLLM-Omni
``nn.Module`` contract so that it can be served through the OpenAI-compatible
API without writing a dedicated pipeline class.

Usage (CLI)::

    python -m vllm_omni.entrypoints.openai.api_server \\
        --model stabilityai/stable-diffusion-3-medium-diffusers \\
        --model-class-name DiffusersBridge \\
        --dtype bfloat16

Limitations (Layer 0):
    - No CFG parallelism, feature caching, or torch.compile
    - No TP / SP / PP distributed parallelism
    - Text encoder outputs are NOT cached across requests
    - Only single-GPU inference is supported
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

import torch
from PIL import Image
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = logging.getLogger(__name__)


def get_diffusers_bridge_post_process_func(od_config: OmniDiffusionConfig):
    """Return a no-op post-process for the bridge.

    The bridge already converts outputs to PIL inside ``forward()``.
    """
    return lambda x: x


class DiffusersBridgePipeline(nn.Module):
    """Wrap any ``diffusers.DiffusionPipeline`` for vLLM-Omni serving.

    The pipeline is loaded via ``DiffusionPipeline.from_pretrained()`` in
    ``__init__`` and stored as a plain attribute (bypassing ``nn.Module``
    registration) so that the vLLM weight-loading machinery does not attempt
    to reload the weights.

    Contract:
        ``forward(req) -> DiffusionOutput``   (request-mode, full denoise loop)
        ``load_weights(weights) -> set[str]``  (no-op, weights already loaded)
    """

    # No external weight sources -- everything is loaded by from_pretrained().
    weights_sources: tuple = ()

    # Class-level capability flags (checked by registry / profiler).
    support_image_input: bool = False
    support_audio_input: bool = False
    support_audio_output: bool = False
    color_format: str = "RGB"

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        model_path = od_config.model

        # ------------------------------------------------------------------
        # Load the full diffusers pipeline
        # ------------------------------------------------------------------
        from diffusers import DiffusionPipeline  # noqa: C901

        logger.info("DiffusersBridge: loading %s ...", model_path)
        dtype = getattr(od_config, "dtype", torch.bfloat16) or torch.bfloat16

        local_files_only = os.path.isdir(model_path)

        # Determine the concrete pipeline class from model_index.json
        # so that we use the correct from_pretrained variant.
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )

        # Move everything to target device
        pipe = pipe.to(self.device)

        # Store as a *plain* attribute to bypass nn.Module.__setattr__
        # which would otherwise register it as a submodule and cause the
        # weight-loading path to try to load duplicate weights.
        object.__setattr__(self, "_pipe", pipe)

        # Expose scheduler for the engine
        self.scheduler = pipe.scheduler

        # Auto-detect capabilities
        self._detect_capabilities()

        logger.info(
            "DiffusersBridge: loaded %s (class=%s)",
            model_path,
            pipe.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Capability detection
    # ------------------------------------------------------------------

    def _detect_capabilities(self) -> None:
        """Auto-detect model capabilities from the pipeline components."""
        pipe = self._pipe
        # Check if the pipeline accepts image input
        import inspect

        call_sig = inspect.signature(pipe.__call__)
        image_params = {"image", "init_image", "latents"}
        if image_params & set(call_sig.parameters):
            DiffusersBridgePipeline.support_image_input = True

    # ------------------------------------------------------------------
    # Core interface: forward
    # ------------------------------------------------------------------

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Execute the full denoising pipeline for *req*.

        Maps ``OmniDiffusionRequest`` fields to the underlying diffusers
        pipeline's ``__call__`` kwargs.
        """
        pipe = self._pipe
        params = req.sampling_params

        # -- Extract prompt(s) and negative_prompt(s) ----------------------
        prompts: list[str] = []
        negative_prompts: list[str] = []
        images: list[Any] = []

        for p in req.prompts:
            if isinstance(p, str):
                prompts.append(p)
                negative_prompts.append("")
            elif isinstance(p, dict):
                prompts.append(p.get("prompt") or p.get("text") or "")
                negative_prompts.append(p.get("negative_prompt") or p.get("negative_prompt_embeds") or "")
                img = p.get("image") or p.get("init_image")
                if img is not None:
                    images.append(img)
            elif isinstance(p, list):
                # token ids -- not supported in bridge mode
                raise ValueError("DiffusersBridge does not support token-id prompts. Pass a text prompt instead.")
            else:
                prompts.append(str(p))
                negative_prompts.append("")

        prompt = prompts[0] if len(prompts) == 1 else prompts
        negative_prompt = (
            negative_prompts[0]
            if len(negative_prompts) == 1 and negative_prompts[0]
            else (negative_prompts if any(negative_prompts) else None)
        )

        # -- Build kwargs for the diffusers pipeline ----------------------
        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "generator": params.generator,
            "num_images_per_prompt": (params.num_outputs_per_prompt if params.num_outputs_per_prompt > 0 else 1),
        }

        # Height / width
        if params.height is not None:
            call_kwargs["height"] = params.height
        if params.width is not None:
            call_kwargs["width"] = params.width

        # Inference steps
        if params.num_inference_steps is not None:
            call_kwargs["num_inference_steps"] = params.num_inference_steps

        # Guidance scale -- only pass when explicitly provided
        if params.guidance_scale_provided and params.guidance_scale > 0:
            call_kwargs["guidance_scale"] = params.guidance_scale

        # Negative prompt
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt

        # Image input (img2img, instruct-pix2pix, etc.)
        if images:
            call_kwargs["image"] = images[0] if len(images) == 1 else images
            # Some pipelines also need `strength` for img2img
            if params.strength is not None:
                call_kwargs["strength"] = params.strength

        # Number of frames (video models)
        if params.num_frames > 1:
            call_kwargs["num_frames"] = params.num_frames

        # FPS (video models)
        if params.fps is not None:
            call_kwargs["fps"] = params.fps

        # Seed (fallback when generator is not set)
        if call_kwargs["generator"] is None and params.seed is not None:
            call_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(params.seed)

        # Extra args pass-through
        if params.extra_args:
            call_kwargs.update(params.extra_args)

        # -- Call the pipeline -------------------------------------------
        logger.debug(
            "DiffusersBridge: calling %s(%s)",
            pipe.__class__.__name__,
            ", ".join(f"{k}={v!r}" for k, v in call_kwargs.items() if k != "prompt"),
        )

        with torch.inference_mode():
            result = pipe(**call_kwargs)

        # -- Normalise output to DiffusionOutput -------------------------
        output = self._normalize_output(result)

        return DiffusionOutput(output=output)

    # ------------------------------------------------------------------
    # Output normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_output(result: Any) -> list[Image.Image] | torch.Tensor:
        """Convert the diffusers pipeline output to a list of PIL images or tensor."""
        # Common patterns:
        #   - StableDiffusionPipelineOutput: .images (list[PIL])
        #   - ImagePipelineOutput: .images (list[PIL])
        #   - FlaxImagePipelineOutput: .images
        #   - np.ndarray / torch.Tensor (some pipelines)
        if hasattr(result, "images"):
            return result.images
        if isinstance(result, (list, tuple)):
            return list(result)
        if isinstance(result, torch.Tensor):
            return result
        if isinstance(result, Image.Image):
            return [result]
        # Fallback: try to iterate
        try:
            return list(result)
        except TypeError:
            return result

    # ------------------------------------------------------------------
    # Weight loading (no-op)
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """All weights are already loaded via ``from_pretrained``.

        Return the set of all parameter names so that the loader's strict
        check considers them "loaded".
        """
        return {name for name, _ in self.named_parameters()}

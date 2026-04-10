# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DiffusersBridgePipeline."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.diffusers_bridge.pipeline_diffusers_bridge import (
    DiffusersBridgePipeline,
)
from vllm_omni.diffusion.registry import DiffusionModelRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class TestBridgeRegistry:
    """Test that DiffusersBridge is properly registered."""

    def test_registered_in_registry(self):
        """DiffusersBridge should be resolvable from the model registry."""
        # _try_load_model_cls returns the class or None
        cls = DiffusionModelRegistry._try_load_model_cls("DiffusersBridge")
        assert cls is not None
        assert cls is DiffusersBridgePipeline

    def test_registry_has_post_process_func(self):
        """The post-process func mapping should include the bridge."""
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        assert "DiffusersBridgePipeline" in _DIFFUSION_POST_PROCESS_FUNCS

    def test_no_cache_acceleration(self):
        """The bridge should be in the no-cache-acceleration set."""
        from vllm_omni.diffusion.registry import _NO_CACHE_ACCELERATION

        assert "DiffusersBridgePipeline" in _NO_CACHE_ACCELERATION


class TestBridgeForward:
    """Test the forward() request-mapping logic with a mock pipeline."""

    @pytest.fixture
    def bridge(self):
        """Create a bridge with a mock diffusers pipeline."""
        od_config = MagicMock()
        od_config.model = "test-model"
        od_config.dtype = torch.float32

        # Create a minimal bridge without calling from_pretrained
        with patch.object(DiffusersBridgePipeline, "__init__", lambda self, **kw: None):
            pipe = DiffusersBridgePipeline.__new__(DiffusersBridgePipeline)
            pipe.od_config = od_config
            pipe.device = torch.device("cpu")

            # Mock the diffusers pipeline
            mock_pipe = MagicMock()
            img = Image.new("RGB", (64, 64))
            mock_pipe.return_value = MagicMock(images=[img])
            mock_pipe.__class__.__name__ = "MockPipeline"
            object.__setattr__(pipe, "_pipe", mock_pipe)
            pipe.scheduler = MagicMock()

            return pipe

    def _make_request(self, prompt="hello", **param_overrides):
        """Helper to create an OmniDiffusionRequest."""
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        params = OmniDiffusionSamplingParams(
            num_inference_steps=20,
            guidance_scale=7.5,
            guidance_scale_provided=True,
            height=512,
            width=512,
            seed=42,
            **param_overrides,
        )
        return OmniDiffusionRequest(
            prompts=[{"prompt": prompt, "negative_prompt": "bad"}],
            sampling_params=params,
            request_ids=["test-req-0"],
            request_id="test-req-0",
        )

    def test_forward_text_to_image(self, bridge):
        """Basic text-to-image request mapping."""
        req = self._make_request("a cat")

        output = bridge.forward(req)
        assert isinstance(output, DiffusionOutput)
        assert output.output is not None

        # Verify the underlying pipeline was called
        bridge._pipe.assert_called_once()
        call_kwargs = bridge._pipe.call_args[1]
        assert call_kwargs["prompt"] == "a cat"
        assert call_kwargs["guidance_scale"] == 7.5
        assert call_kwargs["num_inference_steps"] == 20
        assert call_kwargs["height"] == 512
        assert call_kwargs["width"] == 512

    def test_forward_guidance_scale_not_provided(self, bridge):
        """When guidance_scale_provided is False, guidance_scale should not be passed."""
        req = self._make_request()
        req.sampling_params.guidance_scale_provided = False
        req.sampling_params.guidance_scale = 0.0

        bridge.forward(req)
        call_kwargs = bridge._pipe.call_args[1]
        assert "guidance_scale" not in call_kwargs

    def test_forward_negative_prompt(self, bridge):
        """Negative prompt should be passed through."""
        req = self._make_request("a cat")

        bridge.forward(req)
        call_kwargs = bridge._pipe.call_args[1]
        assert call_kwargs.get("negative_prompt") == "bad"

    def test_forward_seed_generator(self, bridge):
        """When no generator but seed is set, a new generator should be created."""
        req = self._make_request()
        req.sampling_params.generator = None
        req.sampling_params.seed = 12345

        bridge.forward(req)
        call_kwargs = bridge._pipe.call_args[1]
        assert call_kwargs["generator"] is not None

    def test_forward_string_prompt(self, bridge):
        """Plain string prompts should work."""
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        params = OmniDiffusionSamplingParams(num_inference_steps=10, seed=1)
        req = OmniDiffusionRequest(
            prompts=["a dog"],
            sampling_params=params,
            request_ids=["req-1"],
        )

        bridge.forward(req)
        call_kwargs = bridge._pipe.call_args[1]
        assert call_kwargs["prompt"] == "a dog"

    def test_forward_extra_args_passthrough(self, bridge):
        """Extra args should be forwarded to the diffusers pipeline."""
        req = self._make_request()
        req.sampling_params.extra_args = {"cross_attention_kwargs": {"scale": 0.5}}

        bridge.forward(req)
        call_kwargs = bridge._pipe.call_args[1]
        assert call_kwargs["cross_attention_kwargs"] == {"scale": 0.5}


class TestBridgeOutputNormalization:
    """Test the output normalization logic."""

    def test_pil_images_output(self):
        """Pipeline output with .images attribute."""
        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (64, 64))]
        output = DiffusersBridgePipeline._normalize_output(mock_result)
        assert len(output) == 1
        assert isinstance(output[0], Image.Image)

    def test_list_output(self):
        """Pipeline returning a list."""
        output = DiffusersBridgePipeline._normalize_output([1, 2, 3])
        assert output == [1, 2, 3]

    def test_tensor_output(self):
        """Pipeline returning a tensor."""
        t = torch.randn(1, 3, 64, 64)
        output = DiffusersBridgePipeline._normalize_output(t)
        assert output is t


class TestBridgeLoadWeights:
    """Test that load_weights is a proper no-op."""

    def test_load_weights_returns_param_names(self):
        """load_weights should return all named parameters."""
        pipe = DiffusersBridgePipeline.__new__(DiffusersBridgePipeline)
        pipe._modules = {}  # nn.Module internal dict
        pipe._parameters = {}

        # Add a fake parameter
        pipe.register_parameter("fake_weight", torch.nn.Parameter(torch.zeros(1)))

        result = pipe.load_weights([])
        assert "fake_weight" in result

    def test_load_weights_empty_model(self):
        """load_weights on a model with no parameters should return empty set."""
        pipe = DiffusersBridgePipeline.__new__(DiffusersBridgePipeline)
        pipe._modules = {}
        pipe._parameters = {}

        result = pipe.load_weights([])
        assert result == set()

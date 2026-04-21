# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark online serving for GLM-Image (T2I and I2I modes).

Sends requests to the /v1/chat/completions endpoint and reports end-to-end
latency, throughput, and per-stage durations (when the server is started with
--enable-diffusion-pipeline-profiler and/or --enable-ar-profiler).

Usage:
    # Text-to-image (T2I)
    python benchmarks/glm_image/benchmark_glm_image.py \
        --mode t2i --num-prompts 10

    # Image-to-image (I2I)
    python benchmarks/glm_image/benchmark_glm_image.py \
        --mode i2i --num-prompts 10

    # Custom dataset
    python benchmarks/glm_image/benchmark_glm_image.py \
        --mode i2i --dataset custom \
        --dataset-path prompts.json --num-prompts 5
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import numpy as np
from PIL import Image
from tqdm.asyncio import tqdm

# Import backends from the diffusion benchmark (add parent dirs to path)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diffusion"))
from backends import RequestFuncInput, RequestFuncOutput, async_request_chat_completions

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass
class GLMImageRequest:
    prompt: str
    image_path: str | None = None  # Only for I2I mode


class RandomDataset:
    """Generate synthetic prompts (and optional random images for I2I)."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.num_prompts = args.num_prompts
        self._random_image_paths: list[str] | None = None
        if args.mode == "i2i":
            self._random_image_paths = self._generate_random_images()

    def _generate_random_images(self) -> list[str]:
        paths: list[str] = []
        for i in range(self.args.num_input_images):
            img = Image.new("RGB", (512, 512), (128 + i * 30 % 128, 64, 192))
            path = os.path.join(tempfile.gettempdir(), f"glm_image_bench_input_{i}.png")
            img.save(path)
            paths.append(path)
        return paths

    def __len__(self) -> int:
        return self.num_prompts

    def __getitem__(self, idx: int) -> GLMImageRequest:
        image_path = None
        if self._random_image_paths is not None:
            image_path = self._random_image_paths[idx % len(self._random_image_paths)]
        return GLMImageRequest(
            prompt=f"A beautiful scene with vivid colors and intricate details, prompt {idx}",
            image_path=image_path,
        )

    def get_requests(self) -> list[GLMImageRequest]:
        return [self[i] for i in range(len(self))]


class CustomDataset:
    """Load prompts and optional image paths from a JSON file.

    Expected format:
    [
        {"prompt": "A cat sitting on a windowsill"},
        {"prompt": "Make it look like winter", "image_path": "/path/to/img.png"}
    ]
    """

    def __init__(self, args: argparse.Namespace):
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for custom dataset")
        with open(args.dataset_path) as f:
            raw = json.load(f)
        self.items: list[GLMImageRequest] = []
        for item in raw:
            self.items.append(
                GLMImageRequest(
                    prompt=item.get("prompt", ""),
                    image_path=item.get("image_path"),
                )
            )
        if args.num_prompts and len(self.items) > args.num_prompts:
            self.items = self.items[: args.num_prompts]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> GLMImageRequest:
        return self.items[idx]

    def get_requests(self) -> list[GLMImageRequest]:
        return list(self.items)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


async def iter_requests(
    requests_list: list[RequestFuncInput],
    request_rate: float,
) -> Any:
    """Yield requests; Poisson inter-arrival when request_rate is finite."""
    import random as _random

    for i, req in enumerate(requests_list):
        if request_rate != float("inf") and i > 0:
            await asyncio.sleep(_random.expovariate(request_rate))
        yield req


def calculate_metrics(
    outputs: list[RequestFuncOutput],
    total_duration: float,
) -> dict[str, Any]:
    success = [o for o in outputs if o.success]
    errors = [o for o in outputs if not o.success]
    latencies = [o.latency for o in success]
    peak_mems = [o.peak_memory_mb for o in success if o.peak_memory_mb > 0]

    stage_duration_lists: dict[str, list[float]] = {}
    for o in success:
        for stage, dur in (o.stage_durations or {}).items():
            stage_duration_lists.setdefault(stage, []).append(dur)

    metrics: dict[str, Any] = {
        "duration": total_duration,
        "completed_requests": len(success),
        "failed_requests": len(errors),
        "throughput_qps": len(success) / total_duration if total_duration > 0 else 0,
        "latency_mean": float(np.mean(latencies)) if latencies else 0,
        "latency_median": float(np.median(latencies)) if latencies else 0,
        "latency_p99": float(np.percentile(latencies, 99)) if latencies else 0,
        "latency_p95": float(np.percentile(latencies, 95)) if latencies else 0,
        "peak_memory_mb_max": max(peak_mems) if peak_mems else 0,
        "stage_durations_mean": {s: float(np.mean(v)) for s, v in stage_duration_lists.items()},
        "stage_durations_p50": {s: float(np.percentile(v, 50)) for s, v in stage_duration_lists.items()},
    }
    return metrics


async def benchmark(args: argparse.Namespace) -> None:
    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"

    # Load dataset
    if args.dataset == "random":
        dataset = RandomDataset(args)
    elif args.dataset == "custom":
        dataset = CustomDataset(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    glm_requests = dataset.get_requests()
    print(f"Prepared {len(glm_requests)} requests (mode={args.mode})")

    # Convert to RequestFuncInput
    requests_list: list[RequestFuncInput] = []
    for req in glm_requests:
        image_paths = [req.image_path] if req.image_path else None
        requests_list.append(
            RequestFuncInput(
                prompt=req.prompt,
                api_url=api_url,
                model=args.model,
                width=args.width,
                height=args.height,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                image_paths=image_paths,
            )
        )

    # Concurrency semaphore
    semaphore = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None

    async def limited_request(req: RequestFuncInput, session: aiohttp.ClientSession, pbar: Any):
        if semaphore:
            async with semaphore:
                return await async_request_chat_completions(req, session, pbar)
        return await async_request_chat_completions(req, session, pbar)

    # Warmup
    async with aiohttp.ClientSession() as session:
        if args.warmup_requests and requests_list:
            print(f"Running {args.warmup_requests} warmup request(s)...")
            for i in range(args.warmup_requests):
                warm_req = requests_list[i % len(requests_list)]
                await limited_request(warm_req, session, None)

        # Main benchmark
        pbar = tqdm(total=len(requests_list), disable=args.disable_tqdm)
        start_time = time.perf_counter()
        tasks = []
        async for req in iter_requests(requests_list, args.request_rate):
            tasks.append(asyncio.create_task(limited_request(req, session, pbar)))
        outputs = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_time
        pbar.close()

    # Calculate and print metrics
    metrics = calculate_metrics(outputs, total_duration)
    metrics["mode"] = args.mode
    metrics["model"] = args.model

    print(f"\n{' GLM-Image Benchmark Result ':=^60}")
    print(f"{'Mode:':<40} {args.mode}")
    print(f"{'Model:':<40} {args.model}")
    print(f"{'Dataset:':<40} {args.dataset}")
    print("-" * 50)
    print(f"{'Benchmark duration (s):':<40} {metrics['duration']:.2f}")
    print(f"{'Request rate:':<40} {args.request_rate}")
    print(f"{'Max concurrency:':<40} {args.max_concurrency}")
    print(f"{'Successful requests:':<40} {metrics['completed_requests']}/{len(requests_list)}")
    print("-" * 50)
    print(f"{'Throughput (req/s):':<40} {metrics['throughput_qps']:.2f}")
    print(f"{'Latency Mean (s):':<40} {metrics['latency_mean']:.4f}")
    print(f"{'Latency Median (s):':<40} {metrics['latency_median']:.4f}")
    print(f"{'Latency P95 (s):':<40} {metrics['latency_p95']:.4f}")
    print(f"{'Latency P99 (s):':<40} {metrics['latency_p99']:.4f}")

    if metrics["peak_memory_mb_max"] > 0:
        print("-" * 50)
        print(f"{'Peak Memory Max (MB):':<40} {metrics['peak_memory_mb_max']:.2f}")

    if metrics["stage_durations_mean"]:
        print("-" * 50)
        print("Stage Durations Mean (s):")
        for stage, val in sorted(metrics["stage_durations_mean"].items()):
            print(f"  {stage + ':':<38} {val:.4f}")

    print("=" * 60)

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GLM-Image T2I/I2I serving.")
    parser.add_argument("--mode", type=str, default="t2i", choices=["t2i", "i2i"], help="Generation mode.")
    parser.add_argument("--dataset", type=str, default="random", choices=["random", "custom"], help="Dataset type.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to custom dataset JSON.")
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of requests.")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Max concurrent requests.")
    parser.add_argument("--request-rate", type=float, default=float("inf"), help="Requests per second.")
    parser.add_argument("--warmup-requests", type=int, default=1, help="Number of warmup requests.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Diffusion denoising steps.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--model", type=str, default="default", help="Model name.")
    parser.add_argument("--host", type=str, default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=8091, help="Server port.")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSON file for metrics.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable progress bar.")
    parser.add_argument(
        "--num-input-images",
        type=int,
        default=1,
        help="Number of synthetic input images for I2I mode (random dataset).",
    )
    args = parser.parse_args()
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()

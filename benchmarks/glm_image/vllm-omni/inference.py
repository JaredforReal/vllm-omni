# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Batch inference script for GLM-Image with prompts from CSV file.

This script reads prompts from a CSV file and processes them sequentially,
measuring the total time and average time per image.

Usage:
    python batch_inference.py \
        --model-path /path/to/glm-image \
        --csv-path dedup_prompts.csv \
        --output-dir ./outputs \
        --num-prompts 100
"""

import argparse
import csv
import os
import time
from pathlib import Path

from PIL import Image

from vllm_omni.entrypoints.omni import Omni

# Default stage config path (relative to vllm_omni package)
DEFAULT_CONFIG_PATH = "vllm_omni/model_executor/stage_configs/glm_image.yaml"

SEED = 42

# GLM-Image special tokens
GLM_IMAGE_EOS_TOKEN_ID = 16385
GLM_IMAGE_VISION_VOCAB_SIZE = 16512


def compute_max_tokens(height: int, width: int, factor: int = 32) -> int:
    """Compute max_new_tokens for GLM-Image AR generation."""
    token_h = height // factor
    token_w = width // factor
    large_tokens = token_h * token_w
    small_h = token_h // 2
    small_w = token_w // 2
    small_tokens = small_h * small_w
    return small_tokens + large_tokens + 1


def load_prompts_from_csv(csv_path: str, num_prompts: int = 100) -> list[dict]:
    """
    Load prompts from CSV file.

    Args:
        csv_path: Path to CSV file
        num_prompts: Maximum number of prompts to load

    Returns:
        List of dicts containing log_id and code (prompt)
    """
    prompts = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_prompts:
                break
            prompts.append(
                {
                    "log_id": row.get("log_id", f"prompt_{i}"),
                    "code": row.get("code", ""),
                }
            )
    return prompts


def save_image(image: Image.Image, output_path: str) -> None:
    """Save an image to file path."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    image.save(output_path)


def build_prompt_dict(
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    seed: int = SEED,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.5,
) -> dict:
    """Build prompt dict for text-to-image generation."""
    return {
        "prompt": prompt,
        "height": height,
        "width": width,
        "mm_processor_kwargs": {
            "target_h": height,
            "target_w": width,
        },
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }


def main(args: argparse.Namespace) -> None:
    """Main entry point for batch inference."""
    print("=" * 60)
    print("GLM-Image Batch Inference from CSV")
    print("=" * 60)

    # Validate arguments
    if not args.model_path:
        raise ValueError("--model-path is required")
    if not args.csv_path:
        raise ValueError("--csv-path is required")
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    # Determine config path
    config_path = args.config_path
    if config_path is None:
        if os.path.exists(DEFAULT_CONFIG_PATH):
            config_path = DEFAULT_CONFIG_PATH
        else:
            script_dir = Path(__file__).parent.parent.parent.parent
            config_path = script_dir / "vllm_omni/model_executor/stage_configs/glm_image.yaml"
            if not config_path.exists():
                raise FileNotFoundError("Stage config not found. Please specify --config-path.")
            config_path = str(config_path)

    print(f"Model path: {args.model_path}")
    print(f"Config path: {config_path}")
    print(f"CSV path: {args.csv_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of prompts: {args.num_prompts}")
    print(f"Image size: {args.height}x{args.width}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts from CSV
    print("\nLoading prompts from CSV...")
    prompts_data = load_prompts_from_csv(args.csv_path, args.num_prompts)
    actual_num_prompts = len(prompts_data)
    print(f"Loaded {actual_num_prompts} prompts")

    if actual_num_prompts == 0:
        print("No prompts found in CSV. Exiting.")
        return

    # Initialize Omni
    print("\nInitializing Omni with multistage pipeline...")
    init_start_time = time.time()

    omni = Omni(
        model=args.model_path,
        stage_configs_path=config_path,
        log_stats=args.enable_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    init_time = time.time() - init_start_time
    print(f"Initialization completed in {init_time:.2f}s")

    # Prepare sampling parameters
    from vllm import SamplingParams

    calculated_max_tokens = compute_max_tokens(args.height, args.width)

    ar_sampling_params = SamplingParams(
        temperature=0.9,
        top_p=0.75,
        top_k=GLM_IMAGE_VISION_VOCAB_SIZE,
        max_tokens=calculated_max_tokens,
        stop_token_ids=[GLM_IMAGE_EOS_TOKEN_ID],
        seed=args.seed,
        detokenize=False,
    )

    diffusion_sampling_params = {
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
    }

    sampling_params_list = [ar_sampling_params, diffusion_sampling_params]

    # Build all prompt dicts
    print("\nBuilding prompt dicts...")
    all_prompts = []
    prompt_id_to_log_id = {}  # Map request index to log_id
    skipped_count = 0

    for idx, prompt_data in enumerate(prompts_data):
        log_id = prompt_data["log_id"]
        prompt_text = prompt_data["code"]

        if not prompt_text.strip():
            print(f"Skipping empty prompt at index {idx} (log_id: {log_id})")
            skipped_count += 1
            continue

        prompt_dict = build_prompt_dict(
            prompt=prompt_text,
            height=args.height,
            width=args.width,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        prompt_id_to_log_id[len(all_prompts)] = log_id
        all_prompts.append(prompt_dict)

    valid_num_prompts = len(all_prompts)
    print(f"Valid prompts: {valid_num_prompts}, Skipped: {skipped_count}")

    if valid_num_prompts == 0:
        print("No valid prompts found. Exiting.")
        omni.close()
        return

    # Process all prompts in one batch (Omni handles them sequentially internally)
    print(f"\nProcessing {valid_num_prompts} prompts...")
    print("-" * 60)

    total_gen_time = 0.0
    successful_count = 0
    failed_count = 0
    individual_times = []
    request_start_times = {}

    # Record start time for all requests
    batch_start_time = time.time()

    try:
        output_count = 0
        for stage_outputs in omni.generate(all_prompts, sampling_params_list, py_generator=True):
            if stage_outputs.final_output_type == "image":
                for output in stage_outputs.request_output:
                    request_id = output.request_id
                    # Extract the index from request_id (format: "idx_uuid")
                    try:
                        req_idx = int(request_id.split("_")[0])
                        log_id = prompt_id_to_log_id.get(req_idx, request_id)
                    except (ValueError, IndexError):
                        log_id = request_id

                    images = output.images if hasattr(output, "images") else []
                    if not images and hasattr(output, "multimodal_output"):
                        images = output.multimodal_output.get("images", [])

                    for img_idx, img in enumerate(images):
                        if isinstance(img, Image.Image):
                            output_path = os.path.join(args.output_dir, f"{log_id}_{img_idx}.png")
                            save_image(img, output_path)

                    output_count += 1
                    successful_count += 1
                    current_time = time.time()
                    elapsed = current_time - batch_start_time
                    print(f"[{output_count}/{valid_num_prompts}] Generated - {log_id} (elapsed: {elapsed:.2f}s)")

    except Exception as e:
        print(f"Error during generation: {e}")
        failed_count = valid_num_prompts - successful_count

    total_gen_time = time.time() - batch_start_time

    # Print statistics
    print("\n" + "=" * 60)
    print("BATCH INFERENCE STATISTICS")
    print("=" * 60)
    print(f"Total prompts:     {valid_num_prompts}")
    print(f"Successful:        {successful_count}")
    print(f"Failed:            {failed_count}")
    print(f"Skipped (empty):   {skipped_count}")
    print("-" * 60)
    print(f"Initialization time:   {init_time:.2f}s")
    print(f"Total generation time: {total_gen_time:.2f}s")

    if successful_count > 0:
        avg_time = total_gen_time / successful_count
        print(f"Average time per image: {avg_time:.2f}s")
        print(f"\nThroughput: {successful_count / total_gen_time:.2f} images/second")

    print(f"\nOutput directory: {args.output_dir}")
    print("=" * 60)
    print("=" * 60)

    # Cleanup
    omni.close()
    print("\nDone!")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GLM-Image Batch Inference from CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to GLM-Image model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CSV file containing prompts (must have 'code' column)",
    )

    # Optional arguments
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to stage config YAML file (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./batch_outputs",
        help="Output directory for generated images (default: ./batch_outputs)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to process from CSV (default: 100)",
    )

    # Generation parameters
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height (default: 1024)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width (default: 1024)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of diffusion denoising steps (default: 50)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale (default: 1.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED})",
    )

    # Runtime options
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        default=False,
        help="Enable statistics logging",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=600,
        help="Timeout for stage initialization in seconds (default: 600)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

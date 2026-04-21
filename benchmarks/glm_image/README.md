# GLM-Image Benchmarks

Benchmark GLM-Image T2I (text-to-image) and I2I (image-to-image) serving performance via the `/v1/chat/completions` endpoint.

## Setup

Start the GLM-Image server with profiling flags for stage-level timing:

```bash
vllm serve <model_path> --omni --port 8091 \
    --enable-diffusion-pipeline-profiler \
    --enable-ar-profiler
```

## Usage

### Text-to-Image (T2I)

```bash
python benchmarks/glm_image/benchmark_glm_image.py \
    --mode t2i --num-prompts 10
```

### Image-to-Image (I2I) with random images

```bash
python benchmarks/glm_image/benchmark_glm_image.py \
    --mode i2i --num-prompts 10
```

### Custom dataset

Create a JSON file (`prompts.json`):

```json
[
    {"prompt": "A cat sitting on a windowsill"},
    {"prompt": "Make it look like winter", "image_path": "/path/to/image.png"}
]
```

```bash
python benchmarks/glm_image/benchmark_glm_image.py \
    --mode i2i --dataset custom \
    --dataset-path prompts.json --num-prompts 5
```

### Full options

```bash
python benchmarks/glm_image/benchmark_glm_image.py \
    --mode t2i \
    --num-prompts 50 \
    --max-concurrency 4 \
    --request-rate 2.0 \
    --width 1024 --height 1024 \
    --num-inference-steps 50 \
    --warmup-requests 2 \
    --output-file results.json
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `t2i` | `t2i` or `i2i` |
| `--dataset` | `random` | `random` (synthetic) or `custom` (JSON file) |
| `--dataset-path` | - | JSON file path (required for `custom`) |
| `--num-prompts` | `10` | Number of benchmark requests |
| `--max-concurrency` | `1` | Max concurrent requests |
| `--request-rate` | `inf` | Requests per second (Poisson arrival) |
| `--warmup-requests` | `1` | Warmup requests before measurement |
| `--width` | `1024` | Output image width |
| `--height` | `1024` | Output image height |
| `--num-inference-steps` | `50` | Diffusion denoising steps |
| `--seed` | - | Random seed |
| `--model` | `default` | Model name |
| `--host` | `localhost` | Server host |
| `--port` | `8091` | Server port |
| `--output-file` | - | JSON output file for metrics |

## Stage Durations

When the server is started with `--enable-diffusion-pipeline-profiler` and `--enable-ar-profiler`, the benchmark reports per-stage timing:

- `ar_stage_0`: AR stage generation time
- `vae.encode`, `vae.decode`, `diffuse`, `text_encoder.forward`: Diffusion sub-stage times

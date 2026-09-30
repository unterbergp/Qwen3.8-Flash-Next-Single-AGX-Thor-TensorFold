<h1 align="center">Qwen3.8 Flash Next on one Jetson AGX Thor (TensorFold)</h1>

> **This fork runs on NVIDIA Jetson AGX Thor.** It is MiaAI Lab's DGX Spark TensorFold recipe plus the few changes
> Thor needs. This section documents the port; everything after the divider is the upstream Spark README, and its
> numbers are Spark numbers. For the Spark version use
> [MiaAI Lab's repository](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold).

- [Quick start on Thor](#quick-start-on-thor)
- [What TensorFold is, compared with the vLLM Thor fork](#what-tensorfold-is-compared-with-the-vllm-thor-fork)
- [Why it runs on Thor: what was checked](#why-it-runs-on-thor-what-was-checked)
- [What changed for Thor](#what-changed-for-thor)
- [Measured on Thor](#measured-on-thor)
- [Reproducing the measurements](#reproducing-the-measurements)
- [Thor notes and troubleshooting](#thor-notes-and-troubleshooting)

## Quick start on Thor

```bash
git clone https://github.com/unterbergp/Qwen3.8-Flash-Next-Single-AGX-Thor-TensorFold.git
cd Qwen3.8-Flash-Next-Single-AGX-Thor-TensorFold
sudo nvpmodel -m 0 && sudo jetson_clocks   # MAXN and fixed clocks, as measured below (optional)
./start.sh                                   # first run: pulls the image (~11 GB), downloads ~106 GiB, then serves
./stop.sh                                    # stop it and free the memory
```

The API is `http://<thor-address>:8888/v1`, model `Qwen3.8-Flash-Next`. Every setting, command and check in the
Spark README below applies unchanged (`./start.sh restart`, `PARALLEL`, `CONTEXT`, `VISION=0`, `tools/*.py`, ...).

Tested 2026-09-30 on AGX Thor (sm_110, 20 SMs, 122 GiB shared RAM, WD SN5000S NVMe), Jetson Linux R38.2.2, driver
580.00 / CUDA 13.0, Docker with the nvidia runtime, MAXN power mode with the GPU at 1,575 MHz. Image:
`ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold:v0.3.6.3-c1f5d72f8d16` (the Spark image,
unchanged). Defaults: 5 streams x 262,144 tokens, int8 KV, n-gram tables on SSD, image and video input on.

## What TensorFold is, compared with the vLLM Thor fork

The earlier Thor port ([unterbergp/Qwen3.8-Flash-Next-AGX-Thor](https://github.com/unterbergp/Qwen3.8-Flash-Next-AGX-Thor), `start-thor.sh`)
serves the NVFP4 checkpoint with vLLM. It needed Thor workarounds: Marlin MoE instead of CUTLASS, persistent top-k
instead of the cooperative kernel, V2 runner fixes for CUDA graphs and MTP, and one sequence at a time.

TensorFold is a different engine, not a vLLM option. None of those workarounds apply to it.

| | vLLM Thor fork | This fork (TensorFold v0.3.6.3) |
| --- | --- | --- |
| Engine | vLLM (`vllm/vllm-openai:qwen38-flash-next`) | TensorFold on `nvcr.io/nvidia/pytorch:26.07-py3` |
| Checkpoint | NVFP4 | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` (MLX 4-bit, group 32, MTP head) |
| Kernels | vLLM's prebuilt kernels (CUTLASS / Marlin, FlashInfer) | its own CUDA extensions and Triton kernels, compiled at first start for the GPU present |
| Drafting | MTP, 3 tokens | MTP, up to 6 tokens with a 60% confidence cut-off, plus prompt-lookup drafts |
| n-gram (PLE) tables | memory-mapped | read from NVMe per lookup (`--ple-on-ssd`), leaving 29.8 GiB to the KV cache |
| Streams x context | 1 x 262k, BF16 KV | 5 x 262k, int8 KV (1.31 M-token pool) |
| Images / video | — | yes (patch 0008) |

MiaAI's patches (`patches/`, described in [What the patches change](#what-the-patches-change)) add read-ahead and a
native C++ reader for the SSD tables, a register-spill fix in the sparse-attention block select, configurable
prompt chunks, prompt-copy drafts, vision, and stats. They change speed, not output. None of them is specific to GB10.

## Why it runs on Thor: what was checked

The review looked for anything that ties TensorFold to GB10 (sm_121), then tested each risk on the Thor:

1. **Kernel architecture.** `tensorfold/cuda/build.py` builds every extension with
   `-gencode=arch=compute_XY,code=sm_XY` taken from `torch.cuda.get_device_capability()`. The minimum is sm_90.
   Thor is sm_110, so it gets native code; nothing is hard-coded to sm_121.
2. **Instructions used.** The inline PTX is sm_89/sm_90-level: FP8 `mma.sync.m16n8k32.e4m3`, bf16 `mma.sync`,
   thread-block clusters (at most 8, the portable size), `cp.async`. It uses no `tcgen05`, block-scaled FP4 `mma`
   or other sm_12x-only instructions. The rest of Flash Next is Triton.
3. **The container on Thor's driver.** The image's CUDA 13.3 runs on driver 580.00 (CUDA 13.0) in *minor version
   compatibility* mode. torch 2.13 (nv26.07) sees `NVIDIA Thor, (11, 0), 20 SMs, is_integrated=1`, and bf16 matmul
   works. Because the kernels are native sm_110 code, no driver JIT is needed.
4. **Building the extensions.** All six extensions Flash Next uses were compiled and loaded inside the image on
   Thor: `tensorfold_qmm_v3` (31 s), `tensorfold_gdn_v2` (26 s), `tensorfold_experts_v6` (52 s),
   `tensorfold_qwen4_exp_gdn` (20 s), `tensorfold_qwen4_exp_gdn_io` (20 s) and the patched SSD reader (17 s).
   A Triton kernel also compiled and ran. They are cached in `~/.cache/tensorfold-qwen38`.
5. **Memory model.** `capacity.unified()` uses `is_integrated`, which is true on Thor. The budget is therefore
   `MemAvailable - RAM/10`: 105.6-105.8 GiB on an idle Thor, against 102.5 GiB for the default setting.
   That is a little more room than a Spark's ~103-104 GiB.
6. **Tuning constants.** The patches' Triton tile sizes (`SELECT_TILE`, `num_warps`) are per-SM choices. The
   native SSD reader's 64 threads are sized to the NVMe queue, not the CPU. Both carry over to Thor's 14 CPU cores
   and 20 SMs unchanged.

## What changed for Thor

Only the scripts changed; the image and patches are the Spark's.

| File | Change | Why |
| --- | --- | --- |
| `scripts/config.sh` | new `GPU_ARGS` (default `--runtime nvidia --gpus all`), used by `start.sh` | JetPack 7 rejects bare `--gpus all`: *"invoking the NVIDIA Container Runtime Hook directly (e.g. specifying the docker --gpus flag) is not supported. Please use the NVIDIA Container Runtime (e.g. specify the --runtime=nvidia flag)"*. A Spark accepts this too. |
| `start.sh` | loading heartbeat = drop in `MemAvailable` since launch | Thor's `nvidia-smi` reports memory as "Not Supported", so the per-process query always showed 0 |
| `start.sh`, `scripts/prepare.sh`, `scripts/config.sh` | Thor in messages and comments (sm_110 kernel compile, Thor budget) | accuracy |
| `tools/thor-decode.py` | new: the vLLM Thor fork's decode workload plus a concurrency sweep | head-to-head measurement |
| `README.md` | this section | documentation |

## Measured on Thor

These results come from one start with the defaults. The server was idle between runs.

**Decode, head to head with the vLLM Thor fork.** Both used the same workload: the prompts from its
`bench/thor-decode.py`, temperature 0, thinking off, 400 tokens, and the median of 3. Figures are end-to-end,
including prefill and HTTP.

| Workload | TensorFold (this fork) | vLLM Thor fork (2026-09-26) | Change |
| --- | ---: | ---: | ---: |
| Prose | **46.0 tok/s** (219 of 378 drafts accepted) | 36.3-36.7 tok/s | **+26%** |
| Code | **67.3 tok/s** (303 of 416 drafts accepted) | 59.0-59.5 tok/s | **+14%** |

**Concurrent requests.** Prose, sampled with seeds, thinking off, 400 tokens. The vLLM fork serves one stream.

| Streams | Aggregate | Per stream | Spark (upstream, 4-5 streams) |
| ---: | ---: | ---: | ---: |
| 1 | 41.9 tok/s | 41.9 tok/s | 62.4 |
| 2 | 61.2 tok/s | 30.6 tok/s | 90.5 |
| 5 | **78.3 tok/s** | 15.9 tok/s | 119.3 |

**Prefill** (`tools/bench.py thor`, `tools/needle.py`, with the default 2,048-row chunks):

| Prompt tokens | 850 | 3,217 | 12,644 | 50,325 | 194,893 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Thor | 929 tok/s | 1,065 tok/s | 1,119 tok/s | 945 tok/s | 867 tok/s (225 s) |
| Spark (upstream) | | ~2,400 tok/s | ~2,500 tok/s | ~2,400 tok/s | ~2,000 tok/s (~97-102 s) |

`tools/bench.py` also reported decode at 46.8 tok/s (code, greedy) and 49.3 tok/s (chat, sampled), each the median
of 5.

**Checks.** All passed:

- `tools/needle.py` returned the passphrase from a 194,893-token prompt (CORRECT).
- `tools/toolcheck.py` returned the array parameter as a JSON array.
- `tools/visioncheck.py` named both the red circle and the blue square.
- The smoke test ran at start.
- The lowest `MemAvailable` during the 195k-token prompt was 12 GiB (the Spark's lowest was 8.3 GiB).
- The weights loaded in 126 s. The startup estimate was 102.50 GiB within a budget of 105.58 GiB.

**Reading the numbers.** Single-stream decode is limited by memory bandwidth. Thor and GB10 both have 273 GB/s
LPDDR5X, so Thor reaches ~74% of the Spark's single-stream speed. The gap comes from the lower GPU clock and from
compute in the MTP verify pass.

Prefill and many-stream decode are limited by compute. Thor has 20 SMs at 1.575 GHz against GB10's 48. That gives
~45% of the Spark's prefill and ~65% of its 5-stream aggregate.

**Summary.** The Spark's gains carry over fully for single-request chat and code: this fork is faster than the
vLLM fork and serves 5 full-context streams where that fork serves 1. Long prompts take about 2.3x longer than on
a Spark. vLLM Thor prefill has not been measured, so there is no head-to-head prefill comparison.

## Reproducing the measurements

```bash
./start.sh                       # server on :8888, defaults
python3 tools/thor-decode.py     # decode head to head + 1/2/5 streams
python3 tools/bench.py thor      # prefill 0.85k-50k + short decode
python3 tools/needle.py          # 195k-token needle (~4 min on Thor)
python3 tools/toolcheck.py; python3 tools/visioncheck.py
```

For the vLLM side, stop this server and run `./start-thor.sh` and `bench/thor-decode.py` in the vLLM Thor fork.

## Thor notes and troubleshooting

- **One large model at a time.** This server holds ~110 GiB of the shared memory, and the vLLM Thor fork also uses
  port 8888. Run `./stop.sh` before starting vLLM or a llama.cpp server, and stop those before `./start.sh`.
  `start.sh` warns when less than 115 GiB is available.
- **The "CUDA Minor Version Compatibility mode ENABLED" warning** (`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH ... cuInit()=803`)
  at every start is expected, because the container's CUDA 13.3 is newer than the Thor driver's 13.0. It is
  harmless; it would matter only if a kernel needed driver-side PTX JIT for features newer than 13.0.
- **`--gpus` errors** ("use the NVIDIA Container Runtime"): keep `GPU_ARGS` at its default. The nvidia runtime must
  be listed in `docker info | grep -i runtime`.
- **First start after an image change** compiles the CUDA extensions (about 3 minutes on Thor). Later starts reuse
  `~/.cache/tensorfold-qwen38`.
- **Clocks.** The numbers above were measured in MAXN with the GPU at its 1,575 MHz maximum. Check the clock with
  `cat /sys/class/devfreq/gpu-gpc-0/cur_freq`. Lower power modes reduce throughput, prefill most of all.
- **Faster prefill** (text only): `VISION=0 ./start.sh restart` uses 4,096-row chunks. That is 2-5% faster on a
  Spark; it has not been measured on Thor.

---

<h1 align="center">Qwen3.8 Flash Next on one DGX Spark (TensorFold)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Serve **Qwen3.8 Flash Next** from a single NVIDIA DGX Spark (GB10, 128 GB) through an OpenAI-compatible API, with
**5 concurrent requests at the full 262,144-token context** and **image and video input**. It runs
[TensorFold](https://github.com/ashhart/TensorFold) v0.3.6.3 in NVIDIA's PyTorch container, plus a small set of
patches that make prompt processing about **1.7x faster** without changing a single output token, and that give the
model its own vision tower on CUDA.

- Checkpoint: [`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP)
  (MLX 4-bit, group size 32, with the MTP draft head)
- API model id: `Qwen3.8-Flash-Next`
- KV pool: **1,310,720 tokens** (5 streams x 262,144, int8 KV cache, ~23.4 GiB), 25% more than 4 streams
- Images and videos in chat messages (`image_url` / `video_url` parts), see [Images and video](#images-and-video)
- One command: `./start.sh` sets everything up on the first run and starts the server; `./stop.sh` stops it

## Performance

One DGX Spark, int8 KV cache, n-gram tables read from SSD and MTP drafting, measured through the OpenAI API. The 5
concurrent requests row is from the current default (5 streams x 262,144 tokens); the other rows and the prefill table
were measured with 4 streams x 262,144 tokens.

**Decode, prose**

| Concurrent requests | Aggregate | Per request | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 62.4 tok/s | 62.4 tok/s | 152 ms |
| 2 | 90.5 tok/s | 46.3 tok/s | 257 ms |
| 4 | 106.7 tok/s | 28.9 tok/s | 436 ms |
| 5 | 119.3 tok/s | 27.0 tok/s | 528 ms |

**Prefill**

| Prompt | Tokens | Prefill speed | Time to first token |
| ---: | ---: | ---: | ---: |
| 8k | 8,229 | 2,503 tok/s | 3.29 s |
| 16k | 16,425 | 2,520 tok/s | 6.52 s |
| 32k | 32,806 | 2,499 tok/s | 13.13 s |
| 64k | 65,575 | 2,414 tok/s | 27.17 s |
| 128k | 131,110 | 2,200 tok/s | 59.60 s |

Against unpatched TensorFold v0.3.6.2 with the same settings, prefill went from ~1,350-1,490 tok/s to ~2,340-2,480
tok/s (3k-50k-token prompts), a ~195k-token prompt from ~208 s to ~97 s, and single-request decode rose ~4%.
Every reply stayed byte-identical. The prefill table used 4,096-row prompt chunks, which `VISION=0` keeps; with image
input on (the default), chunks are 2,048 rows to make room for the vision tower: prompts of 12k-150k tokens took 4-5%
longer in our runs (e.g. 149k tokens in 74.0 s instead of 70.5 s), and a ~195k-token prompt ~102 s. Decode is
unchanged.

## Requirements

- A DGX Spark (or another GB10 system with 128 GB unified memory) with nothing else large on the GPU: the default
  setting needs ~115 GiB free when the server starts (see [KV pool and memory](#kv-pool-and-memory)).
- Docker with the NVIDIA container runtime, and your user in the `docker` group.
- ~160 GB free disk on a fresh machine: ~125 GB for the checkpoint download under `~/.cache/huggingface`
  (~114 GB) and ~35 GB for the image under Docker's root (~24 GB); `scripts/prepare.sh` checks both.
- Optional: the `hf` CLI on the host (faster, resumable download) and a Hugging Face token in
  `~/.cache/huggingface/token` or `HF_TOKEN`.

## Quick start

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git
cd Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
./start.sh
```

That is all. The first run sets everything up (see below): it pulls the prebuilt image (~11 GB) and downloads the
~106 GiB checkpoint, then compiles the CUDA kernels for the GB10 (a few minutes, once). Later starts take ~2.5 minutes to load the
weights. `start.sh` shows each step, the server's log and the loading progress, runs a smoke test, prints
`Qwen3.8-Flash-Next is now LIVE! on port 8888` with the endpoint, and returns you to the shell.

```bash
curl -s http://<spark-address>:8888/v1/models

curl -s http://<spark-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 1000
}'
```

Any OpenAI client works with `base_url = "http://<spark-address>:8888/v1"` and the model `Qwen3.8-Flash-Next`.
Streaming, tool calls (typed parameters, e.g. arrays come back as JSON arrays), reasoning content, images and
videos are supported. The model thinks before it answers (`reasoning_content`), so give replies enough `max_tokens`.

```bash
./start.sh restart                            # restart it, e.g. after changing a setting
./stop.sh                                     # stop the server and free the GPU memory
docker logs -f qwen38-flash-next-tf           # server log
curl -s http://<spark-address>:8888/health    # busy flag and live token totals
```

## Images and video

The model's own vision tower (27 layers, 0.84 GiB, from the same checkpoint) turns images and video frames into
tokens, placed with Qwen's 3-D rotary positions, the way the reference implementation does. Send them as OpenAI-style
content parts in a user message:

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://<spark-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}},
    {"type": "text", "text": "What is in this picture?"}]}],
  "max_tokens": 2000
}'
```

A video is a `video_url` part (`{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}`); with
OpenAI's Python client, pass the same dicts in `messages`.

| | Images | Videos |
| --- | --- | --- |
| Formats | JPEG, PNG, WebP | MP4, WebM, MOV, MKV (anything FFmpeg decodes) |
| Per request | up to 50 (all of a chat's turns count), 10 MB each, 64 MB in all | up to 2, 64 MB each, 96 MB in all, up to an hour of footage |
| Tokens | up to 16,384 for all images, at most 4,096 an image (50 images: ~320 each; `"detail": "low"`: 256 an image) | 2 frames a second (at most 256 frames, spread over the whole video), each pair of frames one timestamped block; up to 16,384 tokens a request (`TENSORFOLD_VIDEO_TOKENS`) |

By default only data URLs are accepted; `VISION_URLS=1` also lets the server fetch public `https://` URLs. Image
and video prompts are not kept for prefix reuse, so each turn of a chat with images processes them again. A request
body can be up to 96 MiB (base64 makes data URLs a third larger than the files). Text requests are unaffected:
their replies stay byte-identical with vision on. `VISION=0 ./start.sh restart` serves text only.

## Other languages

**Recommended only for replies mostly in Chinese or Japanese; leave it off otherwise.**

MTP drafts may only propose tokens from a list, and TensorFold's list (79,591 tokens) is English and code: it holds
50 Chinese characters and 433 Cyrillic tokens. Replies in other languages still come out right (every token is
checked against the full vocabulary), but fewer drafts are accepted, so they decode slower. A second image adds a
language's tokens to that list (patch `patches/languages/0010`). It is opt-in: put the language in a `.env` file next
to `start.sh` and restart.

```bash
echo 'DRAFT_LANGUAGE=zh' >> .env     # or ja; several: zh,ja
./start.sh restart                   # switches to the language image (pulled or built the first time)
```

Remove the line (or leave it empty) and `./start.sh restart` to go back to the default image. The output is
byte-identical with either image; only speed changes. Measured on one Spark (one stream, recipe sampling, seed 1234,
one boot per arm, 2026-09-29):

| Replies in | Default image | Language image | Change |
| --- | --- | --- | --- |
| Chinese, thinking off / on | 35.5 / 38.4 tok/s | 45.9 / 50.6 tok/s | **+29% / +32%** |
| Japanese, thinking off / on | 39.5 / 46.0 tok/s | 46.9 / 49.2 tok/s | **+19% / +7%** |

The larger list makes every draft step a little slower, which is why it does not pay off for English or code.
`DRAFT_LANGUAGE` also accepts `ru`, `de`, `fr` and `pt`, but those have not been measured to help, so they are not
recommended. The language token lists come from the vLLM recipe's language draft vocabularies; see
[`CREDITS.md`](CREDITS.md).

## What `start.sh` and `scripts/prepare.sh` do

**`./start.sh`** works in five steps, each shown as it runs:

1. **Setup:** runs `scripts/prepare.sh` whenever the setup is not ready: on the first run, after the patches change,
   or with another model or image. It compares what `prepare.sh` last left ready with the current settings, so later
   starts skip it instantly.
2. **Checks:** the arguments (with TensorFold's own parser, in a throwaway container), the previous server, the
   port and the free memory.
3. **Launch:** `tensorfold serve` with the settings from `scripts/config.sh`.
4. **Loading:** the server's log as it comes, and every 15 s the elapsed time and how much of the startup estimate is
   on the GPU. If the server stops, the last log lines and the reason are shown.
5. **Smoke test:** one chat completion, then the LIVE message and the endpoint.

If the server is already running, `./start.sh` says so and leaves it alone; `./start.sh restart` stops it and
starts it again. It stops the server only after the setup and the argument check pass, so a typo leaves the running
server alone and the server is down only while it restarts. Stopping cuts off requests still running (`stop.sh` warns
when there are any). Extra arguments go to `tensorfold serve` after the defaults, so they win
(`./start.sh restart --context 131072`); `./start.sh --help` lists the options. `FOREGROUND=1 ./start.sh` stays
attached to the server's log and exits with its exit code (for a systemd unit).

**`scripts/prepare.sh`** does the one-time setup, and is safe to re-run (each step skips work already done):

1. Preflight: Docker, the NVIDIA runtime, disk space.
2. The image `tensorfold-qwen38:v0.3.6.3`: TensorFold v0.3.6.3 with every `patches/*.patch` applied, plus
   `transformers` (the vision tower) and PyAV (video decoding), on NVIDIA's PyTorch container
   (`nvcr.io/nvidia/pytorch:26.07-py3`). With `DRAFT_LANGUAGE` set it is `tensorfold-qwen38:v0.3.6.3-languages`
   instead, which also applies `patches/languages/*.patch`. It first tries the matching prebuilt image from GitHub
   Container Registry (`ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold:v0.3.6.3-<patches hash>`,
   ~11 GB; `:latest` is the default image, `:languages` the language image); if that tag is not there (e.g. after
   you change `patches/`), or with `PULL=0`, it builds the image locally instead (a few minutes).
3. Downloads the checkpoint into `~/.cache/huggingface` (resumable).
4. Verifies the checkpoint with `tensorfold info`.

Run it yourself to download ahead of time or to rebuild the image from scratch:

```bash
scripts/prepare.sh             # set up without starting the server
scripts/prepare.sh --rebuild   # rebuild the image from scratch
PREPARE=1 ./start.sh restart   # force prepare.sh, then restart; PREPARE=0 skips the check
```

After changing `patches/`, `scripts/publish-image.sh` pushes the new image to GitHub Container Registry
(`latest` and `v0.3.6.3-<patches hash>`), and `DRAFT_LANGUAGE=zh scripts/publish-image.sh` the language image
(`languages` and its own `v0.3.6.3-<patches hash>`).

## KV pool and memory

TensorFold gives every stream its own cache for a full window, so the KV pool is streams x window:

| | Default |
| --- | ---: |
| Streams (`PARALLEL`) | 5 |
| Window per stream (`CONTEXT`, the model's native maximum) | 262,144 tokens |
| **KV pool** | **1,310,720 tokens** (4 streams: 1,048,576) |
| KV precision (`KV_DTYPE`) | int8 (an fp16 scale per 32 values) |
| Memory a stream, allocated (server log) | 4,799 MiB: the KV cache, the sparse-attention index and the stream's own buffers |
| **Memory for the pool, allocated** | **~23.4 GiB** (5 x 4,799 MiB) |

The server reports these at every start: `5 streams of 262144 prompt/reply tokens (4799 MiB a stream)` and
`startup estimate 102.50 GiB within 103.26 GiB` (the budget varies a little from start to start).

Where the memory goes at the default setting (TensorFold's startup estimate):

| | GiB |
| --- | ---: |
| Model weights (the 29.8 GiB of n-gram tables stay on the SSD with `PLE_ON_SSD=1`) | 75.2 |
| Vision tower (`VISION=1`) | 0.84 |
| Stream caches (5 x 4.49, the context-sized part) | 22.5 |
| Fixed buffers (DeltaNet states, decode windows, 2,048-row prompt-chunk scratch, 8 saved prompt states) | 4.0 |
| **Startup estimate** | **102.5** |

The vision tower's scratch (~0.8 GiB at most, measured on a 4,096-token image and a 256-frame video) is taken only
while an image or video encodes and is handed back right after; startup reserves none for it
(`TENSORFOLD_VISION_WORKSPACE_MIB`). With `VISION=0` the tower is not loaded and the chunk scratch is 4,096 rows
(estimate 102.6 GiB).

TensorFold's budget is the free memory at start (`MemAvailable`) minus a host reserve of a tenth of RAM (12.2 GiB),
so ~103-104 GiB on an otherwise idle Spark. The reserve covers what the estimate leaves out (CUDA context, workspaces,
the Python process) and the host itself: on the Spark's unified memory, running out tends to freeze the machine
rather than fail an allocation. At the default setting (vision on) the host kept at least 8.3 GiB free through a
195k-token prompt with image, video and text requests running alongside (9.7 GiB with `VISION=0` under a 195k-token
prompt and 5 concurrent long requests).

Other settings that fit the same budget (TensorFold's own estimate):

| Setting | KV pool | Estimate | Note |
| --- | ---: | ---: | --- |
| `PARALLEL=4` (int8) | 1,048,576 | 97.7 GiB | more headroom |
| `PARALLEL=5` (int8, default) | 1,310,720 | 102.5 GiB | |
| `PARALLEL=6 CONTEXT=220000` (int8) | 1,320,000 | ~103 GiB | shorter windows, one more stream |
| `PARALLEL=6 KV_DTYPE=int4` | 1,572,864 | 97.6 GiB | int4 changes outputs slightly; quality not measured here |
| `PARALLEL=8 KV_DTYPE=int4 CONTEXT=250000` | 2,000,000 | ~103 GiB | tight |
| `PARALLEL=3 KV_DTYPE=bf16` | 786,432 | 102.0 GiB | full-precision KV |

A setting that does not fit is refused at startup, before any weights load, with a message naming a window that
fits.

## Configuration

Every setting lives in [`scripts/config.sh`](scripts/config.sh) and can be overridden from the environment
(`PARALLEL=4 ./start.sh`), in a `.env` file next to `start.sh` (`KEY=value` lines, e.g. `PARALLEL=4`; the environment
wins over it), or with `tensorfold serve` flags (`./start.sh --context 131072`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `PARALLEL` | `5` | requests decoded together (streams) |
| `CONTEXT` | `262144` | prompt + reply window per stream |
| `KV_DTYPE` | `int8` | `bf16`, `int8` or `int4` KV cache |
| `PLE_ON_SSD` | `1` | read the 29.8 GiB n-gram tables from SSD instead of RAM, leaving that memory to the KV cache |
| `VISION` | `1` | image and video input (`--vision`); `0` serves text only |
| `VISION_URLS` | `0` | `1` also accepts public `https://` image and video URLs (default: data URLs only) |
| `DRAFT_LANGUAGE` | empty | `zh` or `ja`: serve the language image, for replies mostly in that language ([Other languages](#other-languages)) |
| `MTP_DRAFTS` / `MTP_CONFIDENCE` | `6` / `0.60` | at most 6 MTP drafts a round; a chain stops before a draft under 60% |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | default sampling (Qwen's thinking-mode values); a request's own values win |
| `THINKING` | `1` | open a think block by default; `0` answers directly unless a request asks to think |
| `SERVED_NAME` | `Qwen3.8-Flash-Next` | the model id in `/v1/models` and in replies |
| `PORT` / `HOST` | `8888` / `0.0.0.0` | where the API listens |
| `TENSORFOLD_PREFILL_ROWS` | `2048` (`4096` with `VISION=0`) | rows per prompt chunk (patch 0006); 4,096 is 2-5% faster from 3k tokens and takes 0.94 GiB more |
| `TENSORFOLD_MTP_COPY` | `1` | prompt-lookup drafts for text that repeats the prompt (patch 0007; needs `PARALLEL` >= 2); `0` turns them off |
| `TENSORFOLD_MAX_IMAGES` / `TENSORFOLD_IMAGE_TOKENS` | `50` / `16384` | images a request may carry and the tokens they share, each at most 4,096 (patch 0009) |
| `TENSORFOLD_VIDEO_TOKENS` | `16384` | a request's video token budget |
| `TENSORFOLD_VISION_WORKSPACE_MIB` | `0` | what startup reserves for the vision tower's scratch |
| `PREPARE` | `auto` | `start.sh` runs `scripts/prepare.sh` when needed; `1` always, `0` never |
| `PULL` | `1` | `prepare.sh` tries the prebuilt image first; `0` always builds locally |
| `STOP_TIMEOUT` | `30` | seconds `stop.sh` gives the server to shut down before removing it |

Any `TENSORFOLD_*` variable in the environment is passed into the container (`TENSORFOLD_NO_UPDATE_CHECK=1`, the
default, stops TensorFold asking GitHub for a newer release at each start). Less common settings are described in
`scripts/config.sh`: `MODEL_ID`, `TF_VERSION`, `TF_REPO`, `BASE_IMAGE` (the patches are made for TensorFold v0.3.6.3;
after changing any of these run `scripts/prepare.sh --rebuild`), `IMAGE`, `CONTAINER_NAME`, `GHCR_IMAGE`, `HF_CACHE`
(default `$HF_HOME` or `~/.cache/huggingface`), `KERNEL_CACHE`, `MIN_FREE_GB`, `IMAGE_FREE_GB`. `start.sh` also takes
`FOREGROUND=1`, `WAIT_TIMEOUT` (seconds, default 1800) and `HF_HUB_OFFLINE=0` (let the server reach Hugging Face; by
default it serves from the local cache only).

### Thinking and sampling

By default the model thinks before it answers, with Qwen's recommended thinking-mode sampling: temperature 1.0,
top_p 0.95, top_k 20. TensorFold has no min_p, presence penalty or repetition penalty, which is the same as
min_p 0.0, presence_penalty 0.0 and repetition_penalty 1.0; requests that send those fields are served as if they
had not. Per request:

- `temperature`, `top_p`, `top_k` and `seed` override the defaults (`temperature: 0` decodes greedily).
- `"chat_template_kwargs": {"enable_thinking": false}` answers without thinking, and
  `"chat_template_kwargs": {"reasoning_effort": "low"}` (or `"xhigh"`) sets Qwen's reasoning effort; without it the
  template's default (medium) applies. A top-level OpenAI-style `reasoning_effort` field is ignored.
- The reasoning comes back in `reasoning_content`, the answer in `content`.

## What the patches change

`scripts/prepare.sh` bakes every `patches/*.patch` into the image (unified diffs against TensorFold's site-packages,
applied with `patch -p0`), and `start.sh` rebuilds or re-pulls the image by itself when the patches change.

| Patch | Change | Effect |
| --- | --- | --- |
| `0001-cuda-live-token-counters` | `/health` reports live token totals | monitoring ([upstream #79](https://github.com/ashhart/TensorFold/pull/79)) |
| `0002-flash-next-ssd-read-ahead` | a prompt chunk's n-gram rows are read from SSD while the GPU processes the previous chunk | multi-chunk prefill +50% |
| `0003-flash-next-ssd-native-reader` | those reads run on a C++ thread pool outside the Python GIL | short prompts' time to first token -35%, 3k-12k prefill +10-40% on top, decode +4% |
| `0004-flash-next-qsa-tiled-select` | the sparse-attention block selection no longer spills registers past 128k tokens | 149k-token prompts 25% faster; decode at 149k context +19% |
| `0005-cuda-stream-draft-stats` | `drafted` / `accepted` counts in concurrent requests' stats | observability |
| `0006-flash-next-prefill-rows` | configurable prompt chunk size (port of [#40](https://github.com/ashhart/TensorFold/pull/40)) | +2-5% at 4,096 rows |
| `0007-flash-next-copy-drafts` | drafts copied from earlier text when the reply repeats the prompt | +6% on quoting and editing replies |
| `0008-flash-next-vision` | image and video input for Flash Next on CUDA: the Qwen3.5 vision tower, interleaved 3-D rotary positions in the attention and sparse-attention kernels, video frames in timestamped blocks | `--vision` (TensorFold's own `--vision` covers only the dense 27B) |
| `0009-flash-next-many-images` | up to 50 images a request sharing 16,384 tokens (4,096 at most an image), encoded by the vision tower in bounded runs; request bodies up to 96 MiB | many-image chats; one image is encoded exactly as before |
| `0011-cuda-request-log` | one timestamped (UTC) log line as each request starts (prompt tokens, max tokens, images, live streams, queue, host MemAvailable) and as it finishes or fails | after a crash or GPU hang, `docker logs` shows which requests were running; no speed change |
| `languages/0010-flash-next-draft-languages` | only in the opt-in language image (`DRAFT_LANGUAGE`): Chinese and Japanese (also Russian, German, French, Portuguese) tokens added to the list MTP drafts from | Chinese +29-32%, Japanese +7-19% decode ([Other languages](#other-languages)) |

Typed tool-call parameters (this recipe's former patch 0001, [#75](https://github.com/ashhart/TensorFold/pull/75))
are part of TensorFold v0.3.6.3.

**Outputs are unchanged.** Every speed patch changes speed only: drafts are verified against the model's own keyed
samples, and the prefill changes read the same bytes and select the same attention blocks. This was checked by
comparing reply hashes (sampled and greedy, prompts up to 149k tokens) against unpatched TensorFold, with vision on
and off, and with a ~195k-token needle-in-a-haystack test. Text rows take exactly the rotary path they always did;
on image and video prompts, drafted replies equal the serial reference too. Any request can also be sent with
`"draft": false` to get TensorFold's serial, one-token-at-a-time reference.

## Checks

The scripts in `tools/` talk to the running server (`API_URL`, default `http://127.0.0.1:8888`; or just `PORT`),
from this machine or another one (`API_URL=http://<spark-address>:8888 tools/bench.py`):

| Script | What it does |
| --- | --- |
| `tools/bench.py [label]` | prefill at ~0.85k / 3.2k / 12.6k / 50k tokens (fresh random prompts) and a short decode check |
| `tools/needle.py` | hides a passphrase in a ~195k-token prompt and checks the model returns it |
| `tools/toolcheck.py` | makes a tool call with an array parameter and checks it comes back as a JSON array |
| `tools/visioncheck.py` | sends a drawn image (a red circle and a blue square) and checks the model names both |

## Repository layout

```
start.sh      set up (first run) and start the server
stop.sh       stop it
scripts/      prepare.sh (image + checkpoint), config.sh (all settings), publish-image.sh (push the image to GHCR),
              banner.sh (start.sh's banner)
patches/      patches baked into the image; patches/languages/ only into the language image (DRAFT_LANGUAGE)
tools/        benchmark and checks
.github/      issue and pull request templates, GitHub Sponsors
CREDITS.md    who and what this builds on
```

## License

MIT, see [`LICENSE`](LICENSE), which also carries TensorFold's MIT notice for the patches. The model weights, downloaded from Hugging Face and not
part of this repository, are under the Qwen Community License 1.0.

**Third-party software in the image.** The prebuilt image (and the one `scripts/prepare.sh` builds) is based on
NVIDIA's PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3`, redistributed as a value-added runtime image. The NVIDIA
software in it is governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/),
which the container prints at every start (it shows in `start.sh`'s output); by pulling or running the image you
accept them. The image also contains Hugging Face `transformers` (Apache 2.0) and PyAV (BSD) with its FFmpeg
libraries (LGPL). The MIT license above covers this repository's scripts and patches only.

## Credits

Built on [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart ([ashhart](https://github.com/ashhart)), [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
by Qwen, and [Vontra's MLX 4-bit checkpoint](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP), with a
prompt-chunk change by MovieMaker93 ([TensorFold #40](https://github.com/ashhart/TensorFold/pull/40)). The full list,
including the runtime stack and licenses, is in [`CREDITS.md`](CREDITS.md).

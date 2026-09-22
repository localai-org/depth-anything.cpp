# depth-anything.cpp

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team**, the folks behind LocalAI, the open-source AI engine that runs any model (LLMs, vision, voice, image, video) on any hardware, no GPU required.

[![Model on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/mudler/depth-anything.cpp-gguf)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![LocalAI](https://img.shields.io/badge/LocalAI-Run_Locally-orange)](https://github.com/mudler/LocalAI)

A from-scratch C++17/[ggml](https://github.com/ggml-org/ggml) port of [Depth Anything 3](https://github.com/bytedance-seed/depth-anything-3) (ByteDance) for **dependency-free monocular metric depth + camera pose** inference. One self-contained GGUF file, no Python, no PyTorch, no CUDA toolkit at inference, just a small native library and CLI, and now **faster than PyTorch on CPU**, bit-exact against the original.

![depth-anything.cpp vs PyTorch on CPU: same depth, ggml finishes first](benchmarks/media/depth_race.gif)

> The same photo, depth computed side by side on CPU: identical output, depth-anything.cpp gets there first ([full clip](benchmarks/media/depth_race.mp4)).

Given an image it recovers a dense **metric depth** map, per-pixel **confidence**, the camera **extrinsics (3x4)** and **intrinsics (3x3)**, an optional **sky** mask, a back-projected **3D point cloud**, and exports to **glb / COLMAP / PLY**. Everything is verified numerically equal to the reference DA3 forward (correlation 1.0), component by component.

---

## Features

- **Monocular metric depth + camera pose** from a single image, plus multi-view depth+pose.
- **Full DA3 family.** small (ViT-S), base (ViT-B), large (ViT-L), giant (ViT-g), metric-large, mono-large, and the nested giant+large metric model - all driven by metadata baked into the GGUF.
- **The whole output surface:** depth, confidence, sky mask, extrinsics/intrinsics, ray-based pose, 3D Gaussians / point cloud, and `glb` / `COLMAP` / `PLY` export.
- **Self-contained GGUF.** Every dimension, hyperparameter and preprocessing constant lives inside the file. The loader reads them; nothing is hardcoded, no external config or vocab is shipped.
- **Quantization** to f16 / q8_0 / q6_k / q5_k / q4_k - q4_k is **99 MB** (0.25x the f32) and near-lossless.
- **CPU-first, GPU-ready.** Tuned CPU path (tinyBLAS, Winograd, flash-attention) plus CUDA / Metal / Vulkan ggml backends.
- **Flat C API** (`include/da_capi.h`) - embed from C, C++, Go, or Rust. Powers the [LocalAI](#use-it-from-localai) backend.
- **SpexAI sensor-calibrated depth.** Run DA2 or single-file DA3 on a main-camera image, fit it to projected sensor ranges, and emit calibrated uint16 millimetres.
- **Parity-first.** Every component is gated against PyTorch-dumped reference tensors; the end-to-end depth matches the real `net()` at correlation 1.0.

---

## Supported models

Convert any of the official [Depth-Anything-3](https://huggingface.co/depth-anything) checkpoints to GGUF. All run through the same metadata-driven engine:

| Model | Backbone | Output | Notes |
|-------|----------|--------|-------|
| `DA3-SMALL` | ViT-S | depth + conf + pose | smallest / fastest |
| `DA3-BASE` | ViT-B | depth + conf + pose | the default anchor |
| `DA3-LARGE` | ViT-L | depth + conf + pose | higher quality |
| `DA3-GIANT` | ViT-g | depth + conf + pose + 3D Gaussians | reconstruction |
| `DA3MONO-LARGE` | ViT-L | depth + **sky** | monocular DPT head |
| `DA3METRIC-LARGE` | ViT-L | metric depth + sky | metric branch |
| `DA3NESTED-GIANT-LARGE` | ViT-g + ViT-L | aligned **metric** depth + pose | two-branch alignment |

### Depth Anything V2

The same engine also runs [Depth Anything **V2**](https://huggingface.co/depth-anything) checkpoints (single-image depth only — no confidence, pose or sky). Relative models emit inverse depth (ReLU); metric models emit depth in metres (Sigmoid × `max_depth`).

| Model | Backbone | Output | Notes |
|-------|----------|--------|-------|
| `Depth-Anything-V2-Small` | ViT-S | relative depth | smallest / fastest |
| `Depth-Anything-V2-Base` | ViT-B | relative depth | |
| `Depth-Anything-V2-Large` | ViT-L | relative depth | higher quality |
| `Depth-Anything-V2-Metric-Hypersim-{Small,Base,Large}` | ViT-S/B/L | metric depth (m), indoor | `max_depth=20` |
| `Depth-Anything-V2-Metric-VKITTI-{Small,Base,Large}` | ViT-S/B/L | metric depth (m), outdoor | `max_depth=80` |

> DA2 is depth only (no pose/confidence). The ViT-g (Giant) DA2 checkpoint is not shipped — its `Depth-Anything-V2-Giant` HF repo is gated/unreleased.

---

## Performance

depth-anything.cpp is now **faster than PyTorch on CPU**: **1.20x at f32** and **1.31x at q8_0** on the production @504 path, while also running the same model in **half the memory**, **loading ~6.7x faster**, shipping as a **99 MB** quantized file, and needing **no Python / PyTorch / CUDA** at inference, all while staying **bit-exact** (correlation 1.0 vs the reference).

CPU, AMD Ryzen 9 9950X3D (16-core / 32-thread x86), `threads=16`, 504x336, sustained (`repeat=25`), PyTorch f32 baseline (full methodology + the CPU-optimization history in [`benchmarks/BENCHMARK.md`](benchmarks/BENCHMARK.md)):

| engine | quant | model MB | load ms | infer ms | peak RAM MB | vs PyTorch |
|--------|-------|---------:|--------:|---------:|------------:|-----------:|
| PyTorch | f32 | 516 | 749 | 416.9 | 1328 | 1.00x |
| **C++/ggml** | f32 | 393 | **112** | **346.4** | **614** | **1.20x** |
| **C++/ggml** | q8_0 | 142 | **40** | **319.4** | **363** | **1.31x** |
| **C++/ggml** | q4_k | **99** | **25** | 395.2 | **320** | 1.05x |

What flipped it: two positional embeddings (the DPT head's UV embedding ~90 ms, the backbone's bicubic pos-embed ~10 ms) were recomputed every forward with single-threaded scalar sin/cos and bicubic loops, even though they depend only on the input geometry and are identical every call. Caching them removed ~95 ms of per-forward host overhead (PyTorch builds the same embeddings with vectorized ops). These are x86 + oneDNN numbers; see [`benchmarks/BENCHMARK.md`](benchmarks/BENCHMARK.md). Camera pose adds a few ms on top of depth at f32. q8_0 is near-lossless; q4_k is the smallest. Bit-exact parity holds at every quant down to f16.

On **GPU** (NVIDIA GB10, via `-DDA_GGML_CUDA=ON`) the ggml CUDA path with flash attention **ties PyTorch's tuned cuDNN** (47.3 vs 47.3 ms/forward, ~47 ms across every quant) and loads 1.75-2.9x faster, so it wins the cold start (~548 vs ~926 ms). Details in [`benchmarks/BENCHMARK.md`](benchmarks/BENCHMARK.md#gpu-nvidia-gb10-grace-blackwell).

![GPU inference speed](benchmarks/media/gpu_speed.png) ![GPU load time](benchmarks/media/gpu_load.png)

![inference speed](benchmarks/media/infer_speed.png) ![peak memory](benchmarks/media/memory.png)

### SpexAI depth-upscale capacity

`depth-upscale` is the SpexAI-specific capture workflow: it combines a DA2 or single-file
DA3 prediction from an RGB TIFF with a paired sensor TIFF. The sensor image stores each uint16
depth sample in its first two interleaved uint8 channels (high byte, then low byte); the command
resizes and smooths the model map, polynomial-fits it to valid sensor pixels, and writes an
uncompressed single-channel uint16 TIFF in calibrated sensor units. The library API also accepts
a sparse range image already projected into the main camera by PointClouds, preserving its
millimetre scale rather than stretching each result to 0..65535.

Measured as one cold task on the supplied SpexAI captures (3072×2048 RGB → 1280×720 depth)
with `depth-anything2-large-q4_k.gguf` (289.1 MiB), four CPU threads:

| Device        | Wall time | Peak host RSS | Peak GPU memory |   Capacity guidance   |
|---------------|----------:|--------------:|----------------:|-----------------------|
| CPU           | 31.14 s   | 833.1 MiB     |              —  | 1 GiB RAM per worker  |
| NVIDIA M2000M | **9.73 s**| **661.9 MiB** | 2,473 MiB task allocation; 2,806 MiB global peak (333 MiB baseline) | 1 GiB RAM + 3 GiB VRAM per worker; one worker on a 4 GiB card |
| old SciPy CPU | 4.79 s    | 883.6 MiB     |                 |                       |

The CUDA run (`-DDA_GGML_CUDA=ON`, `DA_DEVICE=CUDA0`) offloaded 403 weights and used
flash attention. GPU memory was sampled every 100 ms; its ≈2.8 GiB global plateau persisted
for several seconds. The CUDA output was validated as a 1280×720, single-channel uint16 TIFF.

### See it run

Real photos through the actual CLI, input next to the colorized depth (turbo):

![real-photo depth maps from depth-anything.cpp](benchmarks/media/depth_demo.png)

More plots: [load time](benchmarks/media/load_time.png), [model size](benchmarks/media/model_size.png), [quantization tradeoff](benchmarks/media/quant_tradeoff.png).

---

## Build

```sh
git clone --recursive https://github.com/mudler/depth-anything.cpp
cd depth-anything.cpp
cmake -B build -DDA_BUILD_CLI=ON
cmake --build build -j
# -> build/examples/cli/da3-cli
```

The default build has no libtiff dependency. To build the TIFF-based `depth-upscale` workflow, install libtiff's development package (for example, `libtiff-dev` on Debian/Ubuntu) and configure with `-DDA_ENABLE_TIFF=ON`.

### CMake options

| Option | Default | Effect |
|--------|---------|--------|
| `DA_BUILD_CLI` | ON | build the `da3-cli` tool |
| `DA_BUILD_TESTS` | OFF | build the ctest parity suite |
| `DA_ENABLE_TIFF` | OFF | enable libtiff-backed TIFF I/O and the `depth-upscale` CLI workflow |
| `DA_SHARED` | OFF | build `libdepthanything.so` (static ggml, PIC) for embedding |
| `DA_GGML_LLAMAFILE` | ON | tinyBLAS AVX-512/AVX2 matmul kernels (faster CPU) |
| `DA_GGML_CUDA` | OFF | CUDA backend (`-DCMAKE_CUDA_ARCHITECTURES=native` auto) |
| `DA_GGML_METAL` | OFF | Apple Metal backend |
| `DA_GGML_VULKAN` | OFF | Vulkan backend |

GPU example: `cmake -B build -DDA_GGML_CUDA=ON && cmake --build build -j`. See [`docs/GPU.md`](docs/GPU.md).

---

## Python environment setup (conversion only)

Conversion and parity checks need a Python env; **inference does not**.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py --repo depth-anything/DA3-BASE --out models/DA3-BASE
```

## Converting a model

```sh
# depth + pose models (small/base/large/giant)
python scripts/convert_da3_to_gguf.py --model models/DA3-BASE --output models/depth-anything-base-f32.gguf

# monocular (depth + sky) - DA3MONO-LARGE
python scripts/convert_mono_to_gguf.py --model models/DA3MONO-LARGE --output models/depth-anything-mono-large-f32.gguf

# nested two-branch metric model
python scripts/convert_nested_to_gguf.py --model models/DA3NESTED-GIANT-LARGE --output-prefix models/depth-anything-nested

# Depth Anything V2 — relative (encoder vits/vitb/vitl)
python scripts/convert_da2_to_gguf.py --encoder vitl --ckpt models/depth_anything_v2_vitl.pth \
    --output models/depth-anything2-large-f32.gguf --name Depth-Anything-V2-Large

# Legacy depth-upscale model: yuvraj108c's static ViT-B ONNX export.
# This writes the depth-upscale default and needs only onnx, numpy, and gguf.
python scripts/convert_da2_onnx_to_gguf.py \
    --onnx ../depth-upscale/models/depth_anything_v2/1/model.onnx \
    --output models/depth-anything2-base-f32.gguf

# Depth Anything V2 — metric (add --max-depth: 20 for Hypersim/indoor, 80 for VKITTI/outdoor)
python scripts/convert_da2_to_gguf.py --encoder vits --ckpt models/depth_anything_v2_metric_hypersim_vits.pth \
    --output models/depth-anything2-metric-hypersim-small-f32.gguf \
    --name Depth-Anything-V2-Metric-Hypersim-Small --max-depth 20
python scripts/convert_da2_to_gguf.py --encoder vitl --ckpt models/depth_anything_v2_metric_vkitti_vitl.pth \
    --output models/depth-anything2-metric-vkitti-large-f32.gguf \
    --name Depth-Anything-V2-Metric-VKITTI-Large --max-depth 80
```

## Quantization

```sh
build/examples/cli/da3-cli quantize models/depth-anything-base-f32.gguf models/depth-anything-base-q4_k.gguf q4_k
# types: f16 | q8_0 | q6_k | q5_k | q4_k
```

---

## Running inference

```sh
CLI=build/examples/cli/da3-cli
M=models/depth-anything-base-f32.gguf

# Depth -> PFM (lossless float) + a colorizable grayscale PNG
$CLI depth --model $M --input photo.jpg --pfm depth.pfm --png depth.png

# Depth + camera pose (extrinsics 3x4 / intrinsics 3x3) as JSON
$CLI depth --model $M --input photo.jpg --pose pose.json

# Ray-based pose (solved from the auxiliary ray field; needs a --with-aux GGUF)
$CLI depth --model models/depth-anything-base-aux-f32.gguf --input photo.jpg --ray-pose --pose pose.json

# Monocular model: depth + sky mask
$CLI depth --model models/depth-anything-mono-large-f32.gguf --input photo.jpg --pfm depth.pfm --sky sky.pfm

# Nested metric-scale depth (two GGUFs)
$CLI depth --model nested-anyview.gguf --metric-model nested-metric.gguf --input photo.jpg --pfm metric.pfm

# Sensor-calibrated DA2/DA3 depth TIFF (build with -DDA_ENABLE_TIFF=ON).
# depth-upscale defaults to the converted
# yuvraj108c Depth Anything V2 ViT-B model at models/depth-anything2-base-f32.gguf.
# The sensor input stores each 16-bit depth sample in its first two 8-bit channels
# (high byte, low byte); pass --model to override the default.
$CLI depth-upscale --input rgb_capture.tiff --sensor-depth sensor_depth.tiff --tiff calibrated_depth.tiff

# Multi-view depth + pose
$CLI depth --model $M --input a.jpg --input b.jpg --out-prefix scene

# 3D export from a single image
$CLI depth --model $M --input photo.jpg --glb scene.glb --colmap colmap_out/
$CLI reconstruct --model models/depth-anything-giant-f32.gguf --input photo.jpg --ply cloud.ply

# Model metadata (arch, dims, head config, quant)
$CLI info --model $M
```

The `.glb` (glTF 2.0) and COLMAP `cameras/images/points3D` writers are dependency-free (no trimesh / pycolmap) and parity-checked against the reference exporters. See [`docs/EXPORT.md`](docs/EXPORT.md).

---

## Use it from LocalAI

depth-anything.cpp ships as a native [LocalAI](https://github.com/mudler/LocalAI) backend (Go gRPC + purego, the `.so` static-links ggml - no external `libggml`). It exposes a typed **`Depth`** gRPC RPC and a **`POST /v1/depth`** REST endpoint returning the *full* output - per-pixel depth, confidence, sky, extrinsics/intrinsics, the 3D point cloud, and glb/COLMAP exports - not just a normalized PNG.

```sh
# once the backend + models are in the gallery:
local-ai run depth-anything-3-base
```

```sh
# REST: full depth field + pose + points in one typed response
curl http://localhost:8080/v1/depth -H 'Content-Type: application/json' -d '{
  "model": "depth-anything-3-base",
  "src": "photo.jpg",
  "include_depth": true, "include_pose": true, "include_points": true
}'
```

Gallery entries cover base (q4_k/q8_0/f16/f32), small, large, giant, and mono-large. The LocalAI backend lives in the [LocalAI repo](https://github.com/mudler/LocalAI) under `backend/go/depth-anything-cpp/`.

---

## C API

A flat C ABI (`include/da_capi.h`, `abi_version` 12) over `libdepthanything.so`:

```c
da_ctx* ctx = da_capi_load("model.gguf", /*threads*/ 8);
// nested metric model (two branches): da_capi_load_nested(anyview, metric, threads)
int h, w, is_metric;
float *depth, *conf, *sky, ext[12], intr[9];
da_capi_depth_dense(ctx, "photo.jpg", &h, &w, &depth, &conf, &sky, ext, intr, &is_metric);
// ... use depth[h*w], conf, pose ...
da_capi_free_floats(depth);

int n; float *xyz; unsigned char *rgb;
da_capi_points(ctx, "photo.jpg", /*conf_thresh*/ 1.0f, &n, &xyz, &rgb);   // 3D cloud
da_capi_export_glb(ctx, "photo.jpg", "scene.glb");

// PCL projects sensor points into the main-camera grid first. The result and
// output are caller-owned uint16 millimetre buffers of size range_h*range_w.
da_capi_depth_upscale(ctx, "main.tiff", projected_range_mm, range_h, range_w,
                      /*degree*/ 2, /*gaussian_sigma*/ 1.0f, dense_range_mm);
da_capi_free(ctx);
```

Opaque handles, C types only, `da_capi_last_error` for diagnostics. Build it with `-DDA_SHARED=ON`; add `-DDA_ENABLE_TIFF=ON` when `image_path` can be a TIFF.

---

## Parity & tests

The port is verified **numerically equal to the reference**, per component, not just end-to-end. The ctest suite (`-DDA_BUILD_TESTS=ON`, 37 tests) gates the preprocessing, backbone, attention, DPT head, depth, pose, ray head, ray→pose solver, exporters, and the C API against PyTorch-dumped tensors. End-to-end depth correlates 1.0 with the real DA3 forward; the full verification and parity numbers are in [`docs/VERIFICATION.md`](docs/VERIFICATION.md).

```sh
cmake -B build -DDA_BUILD_TESTS=ON && cmake --build build -j && ctest --test-dir build
```

---

## Why depth-anything.cpp

Depth Anything 3 is a great model, but running it for inference drags in a heavy Python/PyTorch/CUDA stack. This is a clean C++17/ggml port focused purely on inference:

- **No Python at inference.** A single `libdepthanything.so` behind a flat C API, easy to embed from C, C++, Go, or Rust.
- **Faster than PyTorch on CPU** (1.20x f32, 1.31x q8_0), in **half the memory**, with **~6.7x faster load** and **bit-exact** output.
- **Small and portable.** Self-contained GGUF with f16 / q8_0 / K-quants, on CPU and any ggml GPU backend.
- **The whole model.** Depth, confidence, pose, sky, ray-pose, 3D point cloud, and glb/COLMAP/PLY export, all parity-checked, across the full DA3 model family.

---

## Citation

If you use depth-anything.cpp, please cite this repository and the original model:

```bibtex
@software{depth_anything_cpp,
  title  = {depth-anything.cpp: a C++/ggml inference engine for Depth Anything 3},
  author = {Di Giacinto, Ettore and Palethorpe, Richard},
  url    = {https://github.com/mudler/depth-anything.cpp},
  year   = {2026}
}
```

Depth Anything 3 is by ByteDance Seed ([bytedance-seed/depth-anything-3](https://github.com/bytedance-seed/depth-anything-3)).

## Author

Ettore Di Giacinto ([@mudler](https://github.com/mudler)).

## License

depth-anything.cpp is released under the [MIT License](LICENSE). Model weights
retain the license of their original Depth Anything checkpoint; converting them
to GGUF does not change that license. For the models supported by the demo:

| Model | Weight license | Commercial use |
| --- | --- | --- |
| DA3 Small | Apache-2.0 | Permitted under the license terms |
| DA3 Base (default) | Apache-2.0 | Permitted under the license terms |
| DA3 Large | CC BY-NC 4.0 | Non-commercial only |
| DA3 Giant | CC BY-NC 4.0 | Non-commercial only |
| DA3 Nested | CC BY-NC 4.0 | Non-commercial only |

Always check the
[official model table](https://github.com/ByteDance-Seed/Depth-Anything-3#%EF%B8%8F-model-cards)
and the individual Hugging Face model card before distributing or commercially
deploying a checkpoint, since upstream terms may change or differ for other
model variants.

---

Built by the [LocalAI](https://github.com/mudler/LocalAI) team. If you want to run depth (and LLMs, vision, voice, image, and video models) locally on any hardware with an OpenAI-compatible API, [give LocalAI a star](https://github.com/mudler/LocalAI).

# ScalePainter

Training-free image inpainting with a frozen Infinity next-scale visual autoregressive model.

This v0.1 source package is being validated. CPU controller/input checks and fixed-case frozen metric checks have passed; full-model GPU generation parity is pending. It has not yet been published as a validated release.

The implementation separates visible-context construction, token commitment, residual reclamation, mask alignment, model loading and evaluation. The paper v8.22 configuration is documented for reproduction in the [paper-to-code mapping](docs/PAPER_V8_22.md). Future experiments can add separate configurations and adapters.

## Installation

Inference uses Linux, Python 3.10 and a CUDA GPU with enough memory for Infinity 8B, the VAE and the text encoder. The recorded inference environment uses PyTorch 2.5.1 with CUDA 12.1 and FlashAttention 2.8.3. See [environment details](docs/ENVIRONMENTS.md); a complete fresh-environment installation is still being checked.

```bash
python -m pip install -r requirements/inference.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install --no-deps -e .
```

Model files are supplied separately; follow [WEIGHTS.md](docs/WEIGHTS.md) for pinned downloads and verification. Pass paths to the Infinity 8B safetensors directory, the patchified 14-bit VAE checkpoint and a local FLAN-T5-XL directory. Checkpoints and datasets are not included in the source package.

## Single image

Use an RGB image and a mask of the same native dimensions. White mask pixels are unknown; black pixels are visible. An empty mask or a mask covering the entire image is rejected.

```bash
scalepainter infer \
  --image example.png --mask mask.png --prompt "a small wooden chair" \
  --model-path weights/Infinity/infinity_8b_weights \
  --vae-path weights/Infinity/infinity_vae_d56_f8_14_patchify.pth \
  --text-encoder-path weights/flan-t5-xl \
  --seed 0 --output outputs/example
```

Generic input uses aspect-preserving letterboxing on a 1024-square canvas. Hidden RGB is removed before the first image resize, and masks use conservative footprint coverage. The raw result is restored to the original image dimensions. Paper geometry is a separate protocol and does not use this generic transform.

Output directories must be new. Each successful case produces `raw.png`, `audit.json` and `result.json`. The raw image is the generated result; it is not composited with the original image. `--save-arrays` also saves control arrays and the continuous final latent for regression analysis.

## Batch and ablations

Create a JSONL manifest. File paths are relative to the manifest unless absolute. Case IDs must be unique and contain letters, numbers, periods, underscores or hyphens.

```json
{"id":"sample_001","image":"images/001.png","mask":"masks/001.png","prompt":"a wooden chair","seed":0,"dataset":"generic"}
```

```bash
scalepainter batch --manifest cases.jsonl --output outputs/check --dry-run
scalepainter batch --manifest cases.jsonl --output outputs/ablations \
  --variants A0 A1 A2 A3 A4 A_all A_wo1 A_wo2 \
  --model-path weights/Infinity/infinity_8b_weights \
  --vae-path weights/Infinity/infinity_vae_d56_f8_14_patchify.pth \
  --text-encoder-path weights/flan-t5-xl
```

Aliases `A_wo3=A4` and `A_wo4=A3` are deduplicated. Failures remain in `summary.json`; the command exits nonzero when any case fails. Black outputs are not silently discarded. A dry run validates inputs and writes a plan without loading a model.

Paper mode requires prepared 1024-square placement masks, original native images, explicit dataset, frozen seed and mask stratum. See [DATA.md](docs/DATA.md) and [EVALUATION.md](docs/EVALUATION.md). Generic runs must not be described as reproducing the paper protocol. The paper's main quality table reports IR, PickScore, HPSv2, CLIP-S, BG LPIPS and BG SSIM; BG PSNR is used in component ablations. Show-o 1.3B is a paper baseline, but its inference code is not included in this package.

## Validation and extension

```bash
python -m unittest discover -s tests -v
```

These CPU tests exercise input invariance, geometry, batch failure accounting, aliases and evaluation population joins. They do not validate full-model numerical equivalence. Read [EXTENDING.md](docs/EXTENDING.md) before adding datasets, baselines or experimental settings.

The core vendor code derives from FoundationVision/Infinity. Its MIT notice is retained in [third_party/Infinity-LICENSE](third_party/Infinity-LICENSE). Model weights and datasets have their own upstream terms.

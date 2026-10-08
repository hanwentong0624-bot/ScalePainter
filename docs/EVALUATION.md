# Evaluation protocol

The independent `evaluation/` directory runs in the recorded Python 3.9 environment. Do not install the inference package into that environment or combine its dependencies with the inference requirements.

```bash
python scripts/prepare_evaluation.py \
  --cases data/prepared_editbench/cases.jsonl \
  --run outputs/paper_editbench --output outputs/editbench_evaluation.jsonl
python -m pip install -r requirements/evaluation.txt
python -m evaluation.evaluate --manifest outputs/editbench_evaluation.jsonl \
  --assets configs/metric_assets.example.json --output outputs/metrics --device cuda:0 --aesthetic
```

The manifest utility preserves every planned case/variant, including missing or failed generations, and checks generation prompts and seeds against the prepared dataset. It rejects dry runs. Use analogous paths for BrushBench.

One real case passed end-to-end CPU loading and computation of all seven evaluator metrics. All eight historical metrics passed fixed-case checks on 12 cases: background PSNR/SSIM agree within 1e-12; GPU ImageReward, CLIP, HPSv2.1, LPIPS and aesthetic values are exact; PickScore differs by at most 0.0000343323, below the original 0.001 tolerance. This GPU check placed one metric model at a time on the device, with HPS weight caching disabled; see the environment notes. A command being available does not mean the full 840-case experiment has been rerun.

Each manifest row supplies `id`, `dataset`, `variant`, `raw`, `source`, `mask`, `global_prompt` and `local_prompt`. Paths are relative to the manifest. The source is the protocol's prepared, complete reference image, **not** the visible-only conditioning proxy. The placement mask and reference must use the same coordinates. The raw result must be saved before any presentation composite.

For EditBench, use the original `prompt_full` as `global_prompt` and `prompt_mask_rich` as `local_prompt`. For BrushBench, use the original caption for both. Do not use the mask-size-expanded generation prompt for evaluation.

The asset JSON supplies local paths under these keys:

| Key | Asset |
| --- | --- |
| `clip` | CLIP checkpoint/tokenizer used by the frozen CLIPScore evaluator |
| `image_reward` | Directory containing `ImageReward.pt` and `med_config.json` |
| `bert_tokenizer` | Local BERT-base-uncased tokenizer for ImageReward |
| `hps` | HPSv2.1 checkpoint |
| `pickscore` | Frozen PickScore model directory |
| `pick_processor` | Processor directory paired with that PickScore checkpoint |
| `aesthetic_clip` | Optional aesthetic CLIP checkpoint |
| `aesthetic_head` | Optional aesthetic prediction head |

The manifests in `configs/metric_weights/` pin upstream revisions, file sizes and SHA-256 hashes. All 31 files across the six Hugging Face assets match the frozen local assets and their pinned upstream metadata. The optional aesthetic head is pinned to its original Git repository commit. Model files are downloaded separately and are governed by their upstream licenses.

From the source root, prepare the assets before starting evaluation:

```bash
for asset in clip image_reward bert_tokenizer hps pickscore pick_processor aesthetic; do
  python scripts/download_weights.py \
    --manifest "configs/metric_weights/${asset}.json" \
    --destination "weights/metrics/${asset}"
done
export TORCH_HOME="$(pwd)/weights/torch_cache"
python scripts/download_weights.py --manifest configs/metric_weights/squeezenet.json \
  --destination "$TORCH_HOME"
```

Keep `TORCH_HOME` set when running the evaluator. The SqueezeNet features use the [torchvision 0.20.1 weight URL](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/squeezenet.py); LPIPS calibration weights come with the pinned torchmetrics package. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` after preparation when running offline. Add `--verify-only` to any download command to check existing files without downloading replacements.

`configs/metric_assets.example.json` resolves paths relative to its own directory and matches these commands. The example enables `--aesthetic` to reproduce the historical eight-metric evaluator; omit the aesthetic download and flag when reporting the paper-aligned subset. For the v8.22 paper, the main quality table contains six metrics: IR, PickScore, HPSv2, CLIP-S, BG LPIPS and BG SSIM. BG PSNR is reported for component ablations, while Aesthetic is historical auxiliary output. Do not substitute a different checkpoint and call the result paper reproduction.

| Reported metric | Frozen definition |
| --- | --- |
| `IR_x10` | ImageReward with the original global prompt, multiplied by 10 |
| `PickScore_x100` | Generated-versus-source pairwise softmax probability, multiplied by 100 |
| `HPSv2_x100` | HPSv2.1 with the original global prompt, multiplied by 100 |
| `CLIP_S_x100` | torchmetrics CLIPScore on the tight mask crop and original local prompt; its output is already scaled |
| `BG_LPIPS_x1000` | SqueezeNet LPIPS between the source and a background composite, multiplied by 1000 |
| `BG_SSIM_x10` | Background mean of the SSIM map, multiplied by 10 |
| `BG_PSNR` | PSNR from background RGB mean squared error; used for component ablations |

Core metrics resize image inputs to 512 square with LANCZOS, and masks with nearest-neighbor interpolation. Preference scoring loads the saved raw/reference images and uses the frozen processor's own image preprocessing. Its default batch size is 2, matching G40. The LPIPS background composite replaces the unknown region with the source solely for measuring background error; it is not a generated output or ranking image for the other metrics.

The paper's quality means are independent full-population results over EditBench (240 cases) and BrushBench (600 cases), with no post-hoc filtering. Runtime is a separate protocol over 48 cases from each benchmark: all methods in the runtime table use one A100 GPU and the same warmed-model end-to-end boundary, including preprocessing, generation, one final decode and inverse restoration, excluding model loading, dataset I/O, file writing and metric computation. The CLI's `generation_seconds` measures only the autoregressive stage and must not replace the paper's end-to-end measurement. See the [paper-to-code mapping](PAPER_V8_22.md) for boundaries and result provenance. Show-o 1.3B is included as a paper baseline; this package does not include its inference implementation or claim to reproduce it from the package alone.

`--aesthetic` adds the historical aesthetic metric. Black outputs remain in the population. Missing or failed cases remain explicit failures, and incomplete summaries must not be presented as complete-population paper results. Summary means retain infinite values and skip NaN like the frozen evaluator, with finite/nonfinite counts included.

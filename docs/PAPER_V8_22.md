# Paper v8.22 and source release v0.1

The manuscript version is **v8.22**; the independently versioned source package is **v0.1**. This candidate retains the frozen G40/G41 numerical implementation. Full-model output regression is still pending. The paper's completed research results are not evidence that this extracted package has passed that regression.

## Paper-to-code mapping

Paths below are relative to the source root.

| Paper component or protocol | Source/configuration | Reproduction boundary |
| --- | --- | --- |
| Visible-context construction (BGCI) | `scalepainter/paper_input.py`, `proxy.py`, `geometry.py` | Remove hidden RGB before the first RGB resize or encoder operation. Paper and generic geometry are separate. |
| Coverage-calibrated commitment (DCTC) | `scalepainter/commit.py`, `coverage_carrier.py`, `config.py` | Preserve sampled discrete endpoints and continuous residual updates; defaults use coverage commitment. |
| Descendant reclamation (SLC) | `scalepainter/commit.py`, `terminal.py`, `no_reclamation.py` | Area threshold 0.22; coverage threshold 0.10; terminal correction through Transformer grid side 16. Keep each route's initial-scale convention. |
| Conditional phase alignment (MATC) | `scalepainter/phase.py`, `phase_geometry.py`, `phase_stability.py`, `translation.py` | Mask-only decision; common proxy/mask shift; inverse translation restores the saved visible strip. |
| Prompt canonicalization | `scalepainter/prompt.py`, `config.py`, `pipeline.py` | Generation only; evaluation keeps the original prompt. Frozen manifest strata and per-case seeds are retained. |
| Frozen sampling | `scalepainter/config.py`, `backbone.py`, `pipeline.py` | Infinity 8B; 13 stages; CFG 4.0, temperature 0.5, top-k 900, top-p 0.97. |
| Eight unique ablations | `scalepainter/config.py`; `scalepainter batch --variants ...` | `A0 A1 A2 A3 A4 A_all A_wo1 A_wo2`; aliases `A_wo3=A4`, `A_wo4=A3` do not create independent evidence. |
| Frozen benchmark population | `scalepainter/data/`, `configs/paper_fingerprints.json` | EditBench 240 and BrushBench 600; see [DATA.md](DATA.md). |
| Main quality table | `evaluation/`, `scripts/prepare_evaluation.py` | Six displayed metrics; the evaluator additionally retains BG PSNR for ablations, and optional historical Aesthetic. See [EVALUATION.md](EVALUATION.md). |
| Runtime table | Separate matched-runtime evidence | Shared 96 cases (48+48), one device, warmed-model end-to-end boundary. CLI stage timings do not reproduce this table. |

The v8.22 Method source is byte-identical to the previously audited v8.19 Method (SHA-256 `30c0f7c47ccefadc015e99cee2151470ca26ede89d7e6e674e60532bfbf58e70`). This does not substitute for checking source and dependency identity or final sampled outputs.

## Geometry and timing details

The manuscript describes aspect-preserving cover resize and center crop. The frozen paper entry in this release specifically accepts native benchmark images and an already prepared 1024-square placement mask. It conservatively projects the mask to native pixel footprints, fills unknown native pixels from visible neighbors, resizes this safe condition with LANCZOS, and builds the final-grid nearest-visible proxy. It requires exact integer mask-to-native scale factors. Do not replace this path with an arbitrary cover/crop transform on an unprepared photograph and call it the same protocol. The manuscript's general description does not specify all these frozen preprocessing steps; the release preserves the audited implementation.

For ordinary images, use generic mode: native image and mask share dimensions, hidden RGB is isolated first, then aspect-preserving letterboxing is applied. Output is restored to native dimensions. Generic mode is not the benchmark reproduction transform.

`audit.json`'s `generation_seconds` measures the synchronized autoregressive call. It excludes proxy construction, text/VAE encoding and final decode/restoration. The paper's separate warmed-model timing includes preprocessing, generation, one final decode and inverse restoration, while excluding model loading, dataset I/O, file writing and metric computation. Neither CLI stage timing nor full-population historical recorded seconds should be substituted for that measurement.

## Show-o and result provenance

Show-o 1.3B is already a v8.22 baseline. The recorded implementation follows the official 512-pixel inpainting path (`inference_t2i.py` with `configs/showo_demo_512x512.yaml`): 16 denoising steps, guidance 1.75, temperature 1.0, float32, 1024 image tokens, and the raw full-token decode before compositing. Its conditioning uses the audited safe input. It is separate from ScalePainter's 1024-pixel Infinity trajectory.

The paper's Show-o quality overlay was checked against completed common-evaluator means for 240 EditBench and 600 BrushBench cases, with zero invalid/all-black outputs. All six displayed numbers match their source CSV exactly. The matched runtime overlay uses the passed 96-case block `block_0002`, mean 7.689956156730962 s/image; it is separate from ordinary generation logs.

The internal evidence inventory pins the quality source CSV to SHA-256 `640498e3fdda714a5c8185002a85dfc99ec22de4cf92de2ace60d27d6e0eba7c` and runtime summary to `f633244b3ca871af43872d9885edfbe65ddc803763cf77322d0868996f9710bb`. These identify research evidence, not newly rerun release results. This package contains the ScalePainter inference/evaluation path, not a bundled Show-o implementation or the internal evidence archive. Future baseline adapters can follow [EXTENDING.md](EXTENDING.md).

## Validation scope

The release explicitly normalizes upstream temporal scales to 1, as the frozen image runner does; a CPU pipeline-to-VAE integration test covers all 13 scales. The loader converts the same transformer blocks to bfloat16 before CUDA transfer to avoid a transient full-float32 loading peak. Final intended parameter dtypes are unchanged. On 2026-10-07, independent verification of saved outputs passed all 12 fixed real cases across eight variants (96 pairs): final RGB pixels, 13-scale sampled bits, control arrays, prompt, phase and seed matched the frozen references exactly. This covers both benchmarks and SLC routes and active/inactive MATC. The verification used ordinary GPU residency; earlier memory-constrained attempts remain diagnostic history and are not paper timing measurements.

Controller, data-preparation, metric and installation checks remain applicable only where their source/configuration/dependencies are unchanged. The 96-pair output regression is a separate full-model check. The full quality populations have not been rerun for this release; unspecified future baselines are not prerequisites for this engineering gate.

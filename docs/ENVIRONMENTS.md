# Environments

The release work uses separate inference and evaluation environments. Their versions must remain separate; do not combine them into one claimed environment.

## Inference environment

The currently reverified inference environment reports:

| Component | Version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| transformers | 4.57.6 |
| flash-attn | 2.8.3 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| Pillow | 12.2.0 |
| timm | 0.6.13 |
| PyYAML | 6.0.3 |
| einops | 0.8.2 |
| safetensors | 0.8.0 |
| tqdm | 4.67.3 |
| pytz | 2026.2 |
| imageio | 2.37.3 |
| sentencepiece | 0.2.1 |
| protobuf | 3.20.3 |

## Evaluation environment

The currently reverified evaluation environment reports:

| Component | Version |
|---|---|
| Python | 3.9.25 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| transformers | 4.46.3 |
| numpy | 1.26.4 |
| Pillow | 9.5.0 |
| scipy | 1.13.1 |
| lpips | 0.1.4 |
| image-reward | 1.5 |
| hpsv2 | 1.2.0 |

The evaluation requirements file also pins the following packages. These are dependency pins for the evaluation environment; they are not an assertion that each package has been independently reinstalled in a clean environment:

| Component | Evaluation pin |
|---|---|
| torchmetrics | 1.6.1 |
| scikit-image | 0.24.0 |
| pandas | 2.3.3 |
| opencv-python | 4.9.0.80 |
| einops | 0.8.2 |
| timm | 0.6.13 |
| safetensors | 0.7.0 |

The two environment contracts must not be combined. In particular, the inference environment's `safetensors` 0.8.0 and the evaluation requirement's 0.7.0 are recorded separately. Packages not listed above, including any optional evaluator or model service dependency, should be treated as version-unknown until separately verified.

## Validation status

The following bounded checks have completed:

- A fresh Windows CPU utility virtual environment installed the wheel successfully; isolated imports and a 12-case dry run passed.
- An independent target installation passed all inference, backbone, and VAE imports. It used existing CUDA dependencies; this is not evidence of a fresh complete CUDA dependency installation.
- Background PSNR and background SSIM for 12 frozen outputs match the frozen metrics within `1e-12` in the evaluation environment.
- One real case loaded all core metric models and computed all seven metrics on CPU using the independent evaluator and existing local assets. This verifies the entry point, not CPU/GPU numerical equivalence.
- All six model-based metrics, including the optional aesthetic score, passed GPU comparison on 12 frozen cases. ImageReward, CLIP, HPSv2.1, background LPIPS and aesthetic scores were exactly equal; the largest PickScore difference was 0.0000343323, below the frozen 0.001 tolerance. Models were placed on the GPU individually. HPS autocast weight caching was disabled for this bounded validation; its float16 arithmetic and outputs were unchanged. Peak reserved memory stayed below 4.2 GB.

Full-model generated outputs and a fresh complete CUDA dependency installation remain unverified. Loading all evaluation models together, as the normal evaluator does, requires more memory than the individual-model validation peak.

## Reproducibility boundary

Inference and evaluation are different environment contracts. Run generation with the inference environment and metric computation with the evaluation environment when following the recorded evidence protocol. Record the environment label with each output and metric artifact.

The current release acceptance state is **not complete**. These checks establish bounded utility, controller and fixed-case metric validation. They do not certify fresh full CUDA installation, full-model generation parity, a new complete-population experiment, or numerical equivalence across different environments.

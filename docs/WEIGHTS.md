# Model weights

Model files are downloaded from the original providers and are not redistributed in this source release. The manifests pin repositories, immutable revisions, filenames, sizes and SHA-256 hashes.

| Component | Original repository | Pinned revision |
| --- | --- | --- |
| Infinity 8B and 14-bit patchified VAE | [FoundationVision/Infinity](https://huggingface.co/FoundationVision/Infinity) | `e7108923249fb3c836ebfbc3f4b4d318d6131293` |
| FLAN-T5-XL | [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl) | `7d6315df2c2fb742f0f5b556879d730926ca9001` |

After installing the inference dependencies:

```bash
python scripts/download_weights.py --manifest configs/infinity_weights.json --destination weights/Infinity
python scripts/download_weights.py --manifest configs/flan_t5_weights.json --destination weights/flan-t5-xl
```

These downloads total approximately 46 GB. Additional space is needed for package caches and inference outputs. `--verify-only` checks existing files without downloading. A conflicting existing file is left unchanged and reported as an error.

The model loader uses the safetensors shards, the VAE file and local FLAN-T5-XL assets directly. It does not slim, delete or rewrite checkpoints. All nine FLAN files were matched to the recorded server files and the pinned upstream revision; the Infinity manifest preserves the verified research checkpoint hashes.

FLAN-T5-XL is published under Apache 2.0 according to its [model card](https://huggingface.co/google/flan-t5-xl). Refer to each provider's terms when obtaining or redistributing model assets; the source package's MIT license applies to its code.

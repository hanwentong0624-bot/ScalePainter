# Benchmark preparation

Obtain EditBench and BrushBench from their original distributors. Images, checkpoints and complete benchmark annotations are not redistributed here. The release includes preprocessing code and a fingerprint inventory of the frozen case population.

EditBench expects `annotations_natural.csv`, `annotations_generated.csv`, the `references_*` image folders and the `masks_*` RGB masked-input folders. Those masked inputs are not binary masks. The adapter reconstructs the known benchmark hole using the frozen exact-black component selection and reference-difference rule, then writes a 1024-square binary placement mask.

```bash
python -m scalepainter.data.prepare editbench \
  --root data/EditBench --output data/prepared_editbench \
  --fingerprints configs/paper_fingerprints.json
```

BrushBench expects its original image mapping and C-order `inpainting_mask` run-length encodings. If using an already decoded mask collection, supply `--mask-root` with its `mapping_file.json` and `masks/` directory; each mask entry must explicitly record `mask_order: "C"`. The adapter does not guess orientation or transpose an unverified mask.

```bash
python -m scalepainter.data.prepare brushbench \
  --root data/BrushBench --output data/prepared_brushbench \
  --fingerprints configs/paper_fingerprints.json
```

Each preparation writes `cases.jsonl` and a separate prepared reference/mask directory. It keeps the original native image as the inference input path. The complete reference image is used only for the prescribed evaluation, while generation removes hidden RGB before its first resampling operation.

The frozen population contains 240 EditBench cases with seeds 9000–9239 and 600 BrushBench cases with seeds 10000–10599. Fingerprints check case IDs, original image bytes, binary mask pixels, prompts, mask strata and seeds. Image PNG byte hashes can vary across encoder versions even when every decoded pixel agrees, so mask validation uses decoded pixel hashes.

Run each prepared dataset separately:

```bash
scalepainter batch --manifest data/prepared_editbench/cases.jsonl \
  --geometry paper --variants A0 A1 A2 A3 A4 A_all A_wo1 A_wo2 \
  --model-path weights/Infinity/infinity_8b_weights \
  --vae-path weights/Infinity/infinity_vae_d56_f8_14_patchify.pth \
  --text-encoder-path weights/flan-t5-xl --output outputs/paper_editbench
```

Use the analogous command with the BrushBench manifest and a new output directory. Eight unique configurations on the full population create 6720 generations; this is a full reproduction command, not a claim that these runs were performed during release validation. For a quick check, use `--dry-run` or explicitly select a small subset with `--max-cases` and report it as a subset.

Preparation passed the frozen fingerprint checks for all 240 EditBench and 600 BrushBench cases in the recorded server environment. BrushBench validation used the verified C-order mask collection. Do not disable a fingerprint failure to label mismatched data as paper reproduction.

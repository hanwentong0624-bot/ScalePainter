"""Evaluate explicit raw/reference pairs in the separate metric environment."""
import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .protocol import _calculate_core_metrics

METRICS = ['IR_x10', 'PickScore_x100', 'HPSv2_x100', 'CLIP_S_x100', 'BG_LPIPS_x1000', 'BG_SSIM_x10', 'BG_PSNR']


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='JSONL rows: id, dataset, variant, raw, source, mask, global_prompt, local_prompt')
    parser.add_argument('--assets', required=True, help='JSON paths: clip, image_reward, bert_tokenizer, hps, pickscore, pick_processor')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=2, help='Frozen G40 preference batch size is 2')
    parser.add_argument('--aesthetic', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive')
    manifest = Path(args.manifest).resolve()
    records = [json.loads(line) for line in manifest.read_text('utf-8').splitlines() if line.strip()]
    if not records:
        raise ValueError('Evaluation manifest has no cases')
    keys = set()
    for row in records:
        key = tuple(row[k] for k in ['dataset', 'variant', 'id'])
        if key in keys:
            raise ValueError(f'Duplicate evaluation row {key}')
        keys.add(key)
        for field in ['global_prompt', 'local_prompt']:
            if not str(row[field]).strip():
                raise ValueError(f'{key}: empty {field}')
        for field in ['raw', 'source', 'mask']:
            value = Path(row[field])
            row[field+'_path'] = str((manifest.parent/value).resolve() if not value.is_absolute() else value)
    assets_path = Path(args.assets).resolve()
    assets = json.loads(assets_path.read_text('utf-8'))
    assets = {key:str((assets_path.parent/Path(value)).resolve()) for key,value in assets.items()}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'protocol.json', {'metric_size':512, 'pickscore':'generated-vs-source paired probability',
        'clip_s':'tight-mask crop with original local prompt', 'raw_only':True,
        'expected_rows':len(records), 'batch_size':args.batch_size, 'device':args.device,
        'validation_status':'12-case frozen metric check passed; full-population rerun not implied'})
    from .calculator import CoreCalculator
    calculator = CoreCalculator(assets, args.device)
    results = []
    for row in records:
        try:
            if row.get('generation_status', 'success') != 'success':
                raise ValueError('Generation is failed or missing; retained in expected population')
            result = _calculate_core_metrics(calculator, row, 512)
            result['status'] = 'core_complete'
        except Exception as error:
            result = dict(row, status='failed', error=f'{type(error).__name__}: {error}')
        results.append(result)
        save_json(output/'partial.json', results)
    del calculator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from .preference import PickScoreScorer, AestheticScorer
    pick = PickScoreScorer(Path(assets['pickscore']), Path(assets['pick_processor']), args.device)
    aesthetic = AestheticScorer(Path(assets['aesthetic_clip']), Path(assets['aesthetic_head']), args.device) if args.aesthetic else None
    eligible = [r for r in results if r['status']=='core_complete']
    for start in range(0, len(eligible), args.batch_size):
        group = eligible[start:start+args.batch_size]
        generated, references = [], []
        try:
            generated = [Image.open(r['raw_path']).convert('RGB') for r in group]
            references = [Image.open(r['source_path']).convert('RGB') for r in group]
            values = pick.score_pairs(generated, references, [r['global_prompt'] for r in group])
            aesthetics = aesthetic.score(generated) if aesthetic else [None]*len(group)
            for row, value, score in zip(group, values, aesthetics):
                row.update(PickScore_x100=float(value), status='success')
                if score is not None:
                    row['AestheticScore'] = float(score)
        except Exception as error:
            for row in group:
                row.update(status='failed', error=f'{type(error).__name__}: {error}')
        finally:
            for image in generated + references:
                image.close()
    fields = list(dict.fromkeys(k for row in results for k in row))
    with (output/'per_case.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    summaries = []
    for dataset, variant in sorted({(r['dataset'],r['variant']) for r in records}):
        group = [r for r in results if (r['dataset'],r['variant'])==(dataset,variant)]
        successful = [r for r in group if r['status']=='success']
        summary = {'dataset':dataset, 'variant':variant, 'expected':len(group), 'successful':len(successful),
                   'failed':len(group)-len(successful), 'complete_population':len(group)==len(successful),
                   'raw_all_black':sum(bool(r.get('raw_all_black')) for r in group)}
        for metric in METRICS + (['AestheticScore'] if args.aesthetic else []):
            values = [float(r[metric]) for r in successful if metric in r]
            finite = [v for v in values if np.isfinite(v)]
            # Match the frozen pandas mean: skip NaN, retain infinities.
            valid = [v for v in values if not np.isnan(v)]
            mean = float(np.mean(valid)) if valid else None
            if mean is not None and not np.isfinite(mean):
                mean = 'Infinity' if np.isposinf(mean) else '-Infinity' if np.isneginf(mean) else None
            summary[metric] = {'mean':mean,
                               'finite_count':len(finite), 'nonfinite_count':len(values)-len(finite)}
        summaries.append(summary)
    save_json(output/'summary.json', summaries)
    return 1 if any(r['status']!='success' for r in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())

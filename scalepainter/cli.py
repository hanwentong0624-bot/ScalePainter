"""Single-image and manifest-based generation, with explicit failure accounting."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

from . import __version__
from .config import ALIASES, VARIANTS, SamplingConfig
from .geometry import prepare_input


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load_pair(image_path, mask_path):
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert('RGB')).copy()
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert('L')) > 127
    return rgb, mask


def save_rgb(path, array):
    # Frozen research save_rgb uses round-to-nearest before uint8 conversion.
    rgb = np.clip(np.rint(np.asarray(array) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def read_manifest(path):
    path = Path(path).resolve()
    records, seen = [], set()
    for number, line in enumerate(path.read_text('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f'Line {number}: expected a JSON object')
        for field in ['id', 'image', 'mask', 'prompt']:
            if field not in row:
                raise ValueError(f'Line {number}: missing {field}')
        identifier = str(row['id'])
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', identifier) or identifier in seen:
            raise ValueError(f'Line {number}: unsafe or duplicate case id {identifier!r}')
        seen.add(identifier)
        for field in ['image', 'mask']:
            item = Path(row[field])
            row[field] = str((path.parent / item).resolve() if not item.is_absolute() else item)
        records.append(row)
    if not records:
        raise ValueError('Manifest has no cases')
    return records


def parser():
    root = argparse.ArgumentParser(description='ScalePainter v0.1 image inpainting')
    root.add_argument('--version', action='version', version=__version__)
    commands = root.add_subparsers(dest='command', required=True)
    single = commands.add_parser('infer', help='Generate one image')
    single.add_argument('--image', required=True)
    single.add_argument('--mask', required=True)
    single.add_argument('--prompt', required=True)
    single.add_argument('--dataset', choices=['generic', 'EditBench', 'BrushBench'], default='generic')
    single.add_argument('--stratum', choices=['small', 'medium', 'large'])
    batch = commands.add_parser('batch', help='Generate JSONL manifest cases')
    batch.add_argument('--manifest', required=True)
    batch.add_argument('--max-cases', type=int)
    for command in [single, batch]:
        command.add_argument('--output', required=True, help='New output directory; existing directories are never overwritten')
        command.add_argument('--variants', nargs='+', choices=list(VARIANTS)+list(ALIASES), default=['A_all'])
        command.add_argument('--seed', type=int, default=0)
        command.add_argument('--geometry', choices=['generic', 'paper'], default='generic')
        command.add_argument('--cfg', type=float, default=4.0)
        command.add_argument('--tau', type=float, default=0.5)
        command.add_argument('--top-k', type=int, default=900)
        command.add_argument('--top-p', type=float, default=0.97)
        command.add_argument('--model-path')
        command.add_argument('--vae-path')
        command.add_argument('--text-encoder-path')
        command.add_argument('--device', default='cuda:0')
        command.add_argument('--save-arrays', action='store_true', help='Save per-scale controls and final continuous latent for regression checks')
        command.add_argument('--dry-run', action='store_true', help='Validate inputs and save a plan without loading the model')
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if not all(math.isfinite(v) for v in [args.cfg, args.tau, args.top_p]) or args.tau <= 0 or not 0 <= args.top_p <= 1 or args.top_k < 0:
        raise ValueError('Expected tau > 0, top_k >= 0, 0 <= top_p <= 1')
    if args.geometry == 'paper' and (args.cfg, args.tau, args.top_k, args.top_p) != (4.0, .5, 900, .97):
        raise ValueError('Paper mode freezes sampling; use generic mode for exploratory settings')
    variants = list(dict.fromkeys(ALIASES.get(v, v) for v in args.variants))
    if args.command == 'batch':
        records = read_manifest(args.manifest)
        if args.max_cases is not None:
            if args.max_cases < 1:
                raise ValueError('--max-cases must be positive')
            records = records[:args.max_cases]
    else:
        records = [{'id': 'image', 'image': args.image, 'mask': args.mask, 'prompt': args.prompt,
                    'dataset': args.dataset, 'stratum': args.stratum, 'seed': args.seed}]
    # Validate every selected case before loading an expensive model.
    for row in records:
        if not str(row['prompt']).strip():
            raise ValueError(f"{row['id']}: empty prompt")
        if args.geometry == 'paper' and (row.get('dataset') not in {'EditBench', 'BrushBench'} or row.get('stratum') not in {'small', 'medium', 'large'} or 'seed' not in row):
            raise ValueError('Paper cases require explicit dataset, stratum and frozen seed in a manifest')
        seed = row.get('seed', args.seed)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError(f"{row['id']}: seed must be an integer in [0, 2**32)")
        rgb, mask = load_pair(row['image'], row['mask'])
        prepare_input(rgb, mask, mode=args.geometry)
        row['image_sha256'], row['mask_sha256'] = sha256(row['image']), sha256(row['mask'])
        row['seed'] = seed
    sampling = SamplingConfig(args.cfg, args.tau, args.top_k, args.top_p)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {'version': __version__, 'geometry': args.geometry, 'sampling': asdict(sampling),
            'variants': variants, 'requested_variants': args.variants, 'cases': records,
            'expected_generations': len(records)*len(variants), 'dry_run': args.dry_run}
    write_json(output/'plan.json', plan)
    if args.dry_run:
        print(f'Validated {len(records)} cases, {len(variants)} unique variants. No inference was run.')
        return 0
    for field in ['model_path', 'vae_path', 'text_encoder_path']:
        if not getattr(args, field):
            raise ValueError(f'--{field.replace("_", "-")} is required for inference')
    from .pipeline import ScalePainter
    engine = ScalePainter.from_pretrained(model_path=args.model_path, vae_path=args.vae_path,
                                         text_encoder_path=args.text_encoder_path, device=args.device)
    results = []
    for row in records:
        rgb, mask = load_pair(row['image'], row['mask'])
        for variant in variants:
            target = output/row['id']/variant
            target.mkdir(parents=True, exist_ok=False)
            result = {'id': row['id'], 'variant': variant, 'seed': row['seed']}
            try:
                raw, audit, arrays = engine(rgb, mask, row['prompt'], seed=row['seed'], variant=variant,
                    geometry_mode=args.geometry, dataset=row.get('dataset', 'generic'), stratum=row.get('stratum'),
                    sampling=sampling, return_arrays=args.save_arrays)
                save_rgb(target/'raw.png', raw)
                write_json(target/'audit.json', audit)
                if arrays is not None:
                    np.savez_compressed(target/'controls.npz', **{k:v.detach().cpu().numpy() for k,v in arrays.items()})
                result.update(status='success', raw_sha256=sha256(target/'raw.png'), raw=str((target/'raw.png').relative_to(output)))
            except Exception as error:
                result.update(status='failed', error_type=type(error).__name__, error=str(error))
                print(f"{row['id']}/{variant}: {type(error).__name__}: {error}", file=sys.stderr)
            results.append(result)
            write_json(target/'result.json', result)
            write_json(output/'summary.json', {'expected': plan['expected_generations'], 'completed': len(results),
                'successful': sum(r['status']=='success' for r in results), 'failed': sum(r['status']=='failed' for r in results), 'results': results})
    return 1 if any(r['status']=='failed' for r in results) else 0

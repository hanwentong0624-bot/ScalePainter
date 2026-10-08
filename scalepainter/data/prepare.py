"""Prepare original images and frozen placement masks without copying datasets."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .editbench_masks import extract_editbench_hole_mask, resize_binary_mask
from .brushbench_masks import decode_rle_c_order
from ..config import mask_stratum


def relative(path, root):
    return os.path.relpath(Path(path).resolve(), root).replace(os.sep, '/')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_row(output, dataset, index, original, mask, prompt, global_prompt, local_prompt):
    if not mask.any() or mask.all():
        raise ValueError(f'{index}: invalid empty or full placement mask')
    target = output/'prepared'/str(index)
    target.mkdir(parents=True, exist_ok=False)
    with Image.open(original) as image:
        reference = image.convert('RGB').resize((1024, 1024), Image.Resampling.LANCZOS)
    reference.save(target/'reference.png')
    model_mask = Image.fromarray(np.asarray(mask, dtype=np.uint8)*255).resize((1024, 1024), Image.Resampling.NEAREST)
    model_mask.save(target/'mask.png')
    binary = np.asarray(model_mask)>127
    ratio = float(np.asarray(mask, dtype=bool).mean())
    return {'id':str(index), 'dataset':dataset, 'seed':index,
            'image':relative(original, output), 'source':relative(target/'reference.png',output),
            'mask':relative(target/'mask.png',output), 'prompt':prompt,
            'global_prompt':global_prompt, 'local_prompt':local_prompt,
            'stratum':mask_stratum(ratio,dataset), 'mask_ratio':ratio,
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'global_prompt_sha256':hashlib.sha256(global_prompt.encode()).hexdigest(),
            'local_prompt_sha256':hashlib.sha256(local_prompt.encode()).hexdigest(),
            'native_sha256':sha256(original), 'mask_pixels_sha256':hashlib.sha256(binary.astype(np.uint8).tobytes()).hexdigest()}


def editbench(root, output):
    rows, index = [], 9000
    for split in ['natural', 'generated']:
        with (root/f'annotations_{split}.csv').open(encoding='utf-8-sig', newline='') as stream:
            annotations = [r for r in csv.DictReader(stream) if str(r.get('aos','')).strip()]
        for row in annotations:
            original = root/f'references_{split}'/(row['aos']+'.png')
            masked = root/f'masks_{split}'/(row['aos']+'.png')
            with Image.open(original) as image, Image.open(masked) as masked_image:
                mask, audit = extract_editbench_hole_mask(masked_image.convert('RGB'), image.convert('RGB'), difference_threshold=8.0)
            rows.append(prepare_row(output, 'EditBench', index, original, mask, row['prompt_mask-rich'].strip(),
                                    row['prompt_full'].strip(), row['prompt_mask-rich'].strip()))
            (output/'prepared'/str(index)/'mask_extraction.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
            index += 1
    return rows


def brushbench(root, output, mask_root=None):
    mapping = json.loads((root/'mapping_file.json').read_text('utf-8'))
    masks = json.loads((mask_root/'mapping_file.json').read_text('utf-8')) if mask_root else mapping
    rows = []
    for offset, key in enumerate(sorted(masks)):
        mask_item = masks[key]
        original_id = str(mask_item.get('original_id', Path(key).stem))
        item = mapping[original_id]
        image_path = root/item['image']
        with Image.open(image_path) as image:
            shape = (image.height, image.width)
        if mask_root:
            if mask_item.get('mask_order') != 'C':
                raise ValueError(f'{key}: expected explicit C-order mask provenance')
            with Image.open(mask_root/'masks'/key) as image:
                mask = np.asarray(image.convert('L')) > 127
        else:
            if 'inpainting_mask' not in mask_item:
                raise ValueError('Mapping lacks original inpainting_mask RLE; supply --mask-root with verified C-order masks')
            mask = decode_rle_c_order(mask_item['inpainting_mask'], shape).astype(bool)
        if mask.shape != shape:
            raise ValueError(f'{key}: native image and mask dimensions differ')
        prompt = str(mask_item.get('caption') or item.get('caption') or '').strip()
        if not prompt:
            raise ValueError(f'{key}: missing caption')
        rows.append(prepare_row(output, 'BrushBench', 10000+offset, image_path, mask, prompt, prompt, prompt))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['editbench','brushbench'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mask-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-count', type=int)
    parser.add_argument('--fingerprints', type=Path, help='Optional frozen case/hash inventory to enforce exact population and mask pixels')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = editbench(args.root.resolve(),output) if args.dataset=='editbench' else brushbench(args.root.resolve(),output,args.mask_root.resolve() if args.mask_root else None)
    expected = args.expected_count if args.expected_count is not None else (240 if args.dataset=='editbench' else 600)
    if len(rows) != expected:
        raise ValueError(f'Expected {expected} cases, prepared {len(rows)}')
    if args.fingerprints:
        entries = json.loads(args.fingerprints.read_text('utf-8'))['cases']
        frozen = {str(r['id']):r for r in entries if r['dataset']==rows[0]['dataset']}
        if set(frozen) != {r['id'] for r in rows}:
            raise ValueError('Case population differs from the frozen protocol')
        for row in rows:
            for field in ['native_sha256','mask_pixels_sha256','prompt_sha256','global_prompt_sha256','local_prompt_sha256','stratum','seed']:
                if row[field] != frozen[row['id']][field]:
                    raise ValueError(f"{row['id']}: frozen {field} mismatch")
    (output/'cases.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n',encoding='utf-8')
    print(f'Prepared {len(rows)} cases in {output}')


if __name__ == '__main__':
    main()

"""Join a prepared benchmark manifest with an explicit generation plan.

Keep missing/failed generations in the population so evaluation cannot silently
report a survivors-only result. Run this utility from either Python environment.
"""
import argparse
import json
import os
from pathlib import Path


def build_rows(cases_path, run_dir, output_path):
    cases_path, run_dir, output_path = map(lambda p:Path(p).resolve(), [cases_path, run_dir, output_path])
    cases = {}
    for line in cases_path.read_text('utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row['id'])
        if key in cases:
            raise ValueError(f'Duplicate prepared case {key}')
        cases[key] = row
    plan = json.loads((run_dir/'plan.json').read_text('utf-8'))
    if plan['dry_run']:
        raise ValueError('A dry run has no generated images to evaluate')
    rows = []
    for planned in plan['cases']:
        case = cases[str(planned['id'])]
        for field in ['dataset', 'seed', 'prompt']:
            if case[field] != planned[field]:
                raise ValueError(f"{case['id']}: generation {field} does not match prepared case")
        for variant in plan['variants']:
            row = {key:case[key] for key in ['id','dataset','global_prompt','local_prompt']}
            row['variant'] = variant
            for key in ['source','mask']:
                path = (cases_path.parent/case[key]).resolve()
                row[key] = os.path.relpath(path, output_path.parent).replace(os.sep, '/')
            raw = run_dir/str(case['id'])/variant/'raw.png'
            row['raw'] = os.path.relpath(raw, output_path.parent).replace(os.sep, '/')
            result_path = raw.parent/'result.json'
            result = json.loads(result_path.read_text('utf-8')) if result_path.exists() else {}
            row['generation_status'] = result.get('status', 'missing')
            rows.append(row)
    if len(rows) != plan['expected_generations']:
        raise ValueError('Generation plan population is inconsistent')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows(args.cases, args.run, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(json.dumps(row) for row in rows)+'\n')
    print(f'Prepared {len(rows)} evaluation rows, including failed or missing generations.')


if __name__ == '__main__':
    main()

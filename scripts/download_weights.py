"""Download or verify only the files listed in a pinned model manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request


def matches(path, record):
    if path.stat().st_size != record['size']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest() == record['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text('utf-8'))
    root = args.destination.resolve()
    for record in manifest['files']:
        target = (root/record['filename']).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Manifest filename escapes the destination')
        if not record.get('sha256'):
            raise ValueError(f"Missing frozen SHA-256 for {record['filename']}")
        if target.exists():
            if not matches(target, record):
                raise ValueError(f'Existing file does not match the manifest; left unchanged: {target}')
        elif args.verify_only:
            raise FileNotFoundError(target)
        else:
            if 'url' in record:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    try:
                        with urllib.request.urlopen(record['url'], timeout=60) as response:
                            shutil.copyfileobj(response, stream)
                    except BaseException:
                        stream.close()
                        temporary.unlink()
                        raise
                try:
                    if not matches(temporary, record):
                        raise ValueError(f'Download does not match the frozen manifest: {target}')
                    if target.exists():
                        raise FileExistsError(target)
                    temporary.rename(target)
                finally:
                    temporary.unlink(missing_ok=True)
            else:
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id=manifest['repository'], revision=manifest['revision'],
                                filename=record['filename'], local_dir=str(root))
            if not matches(target, record):
                raise ValueError(f'Download does not match the frozen manifest: {target}')
        print(f"Verified {record['filename']}")


if __name__ == '__main__':
    main()

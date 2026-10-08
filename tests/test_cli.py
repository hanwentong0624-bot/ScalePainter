"""Batch orchestration checks without loading a model or GPU."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scalepainter.cli import main, read_manifest


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.fromarray(np.full((32, 40, 3), 128, dtype=np.uint8)).save(self.root/'image.png')
        mask = np.zeros((32, 40), dtype=np.uint8)
        mask[8:24, 10:30] = 255
        Image.fromarray(mask).save(self.root/'mask.png')

    def manifest(self, rows):
        path = self.root/'cases.jsonl'
        path.write_text('\n'.join(json.dumps(dict(image='image.png', mask='mask.png', **row)) for row in rows))
        return path

    def test_duplicate_ids_rejected(self):
        manifest = self.manifest([dict(id='case', prompt='object')]*2)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            read_manifest(manifest)

    def test_invalid_seed_rejected_before_output_creation(self):
        manifest = self.manifest([dict(id='case', prompt='object', seed=-1)])
        output = self.root/'output'
        with self.assertRaisesRegex(ValueError, 'seed must'):
            main(['batch', '--manifest', str(manifest), '--output', str(output), '--dry-run'])
        self.assertFalse(output.exists())

    def test_failures_retained_and_aliases_deduplicated(self):
        manifest = self.manifest([dict(id='bad', prompt='fail'), dict(id='good', prompt='object')])
        calls = []
        class Engine:
            @classmethod
            def from_pretrained(cls, **kwargs):
                return cls()

            def __call__(self, rgb, mask, prompt, **kwargs):
                calls.append((prompt, kwargs['variant']))
                if prompt == 'fail':
                    raise RuntimeError('intentional case failure')
                # Black images must remain successful outputs, not be filtered.
                return np.zeros_like(rgb, dtype=np.float32), {}, None
        output = self.root/'output'
        with patch.dict('sys.modules', {'scalepainter.pipeline':SimpleNamespace(ScalePainter=Engine)}):
            code = main(['batch', '--manifest', str(manifest), '--output', str(output),
                         '--variants', 'A4', 'A_wo3', '--model-path', 'model',
                         '--vae-path', 'vae', '--text-encoder-path', 'text'])
        self.assertEqual(code, 1)
        self.assertEqual(calls, [('fail', 'A4'), ('object', 'A4')])
        summary = json.loads((output/'summary.json').read_text())
        self.assertEqual((summary['expected'], summary['successful'], summary['failed']), (2, 1, 1))
        self.assertTrue((output/'good/A4/raw.png').exists())
        self.assertFalse((output/'bad/A4/raw.png').exists())


if __name__ == '__main__':
    unittest.main()

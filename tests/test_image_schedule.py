"""CPU pipeline integration check; requires the inference PyTorch dependency."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Install inference dependencies for the pipeline test')
class ImageScheduleTests(unittest.TestCase):
    def test_pipeline_uses_image_time_and_patchified_vae_grids(self):
        import torch
        from scalepainter.pipeline import ScalePainter

        class EncodeReached(Exception):
            pass

        class RecordingVAE:
            def parameters(self):
                yield torch.zeros(1)

            def encode(self, image, *, scale_schedule):
                self.schedule = scale_schedule
                self.image_shape = tuple(image.shape)
                raise EncodeReached()

        vae = RecordingVAE()
        engine = ScalePainter(None, vae, None, None)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        with patch('scalepainter.pipeline.prepare_input', return_value=(rgb, mask, None, {})):
            with self.assertRaises(EncodeReached):
                engine(rgb, mask, 'a chair', variant='A0')
        self.assertEqual(vae.image_shape, (1, 3, 4, 4))
        edges = [2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128]
        self.assertEqual(vae.schedule, [(1, edge, edge) for edge in edges])


if __name__ == '__main__':
    unittest.main()

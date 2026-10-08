"""CPU invariance and geometry checks; no model or CUDA required."""
import unittest

import numpy as np

from scalepainter.geometry import prepare_input, resize_mask_conservative, restore_native
from scalepainter.paper_input import native_unknown_mask
from scalepainter.proxy import build_nearest_proxy


class GeometryTests(unittest.TestCase):
    def test_hidden_rgb_never_changes_condition(self):
        rng = np.random.default_rng(74)
        for h, w in [(31, 19), (513, 777), (640, 2048), (1, 40), (1111, 13)]:
            with self.subTest(shape=(h, w)):
                rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
                mask = np.zeros((h, w), bool)
                mask[:, w//3:2*w//3] = True
                baseline, placement, geo, _ = prepare_input(rgb, mask)
                for fill in [0, 255]:
                    changed = rgb.copy()
                    changed[mask] = fill
                    safe, new_mask, new_geo, _ = prepare_input(changed, mask)
                    np.testing.assert_array_equal(safe, baseline)
                    np.testing.assert_array_equal(new_mask, placement)
                    self.assertEqual(new_geo, geo)
                proxy = build_nearest_proxy(baseline, placement).image
                self.assertEqual(restore_native(proxy, geo).shape, rgb.shape)

    def test_paper_hidden_footprint_invariance(self):
        rng = np.random.default_rng(144)
        mask = np.zeros((1024, 1024), bool)
        mask[173:513, 281:781] = True
        native = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        hidden = native_unknown_mask(mask, native.shape[:2])
        baseline = prepare_input(native, mask, mode='paper')[0]
        changed = native.copy()
        changed[hidden] = rng.integers(0, 256, (int(hidden.sum()), 3), dtype=np.uint8)
        np.testing.assert_array_equal(prepare_input(changed, mask, mode='paper')[0], baseline)

    def test_thin_mask_survives_downsampling(self):
        mask = np.zeros((2048, 2048), bool)
        mask[101, 333] = True
        resized = resize_mask_conservative(mask, 17, 19)
        self.assertTrue(resized.any())
        self.assertTrue(resized[101*17//2048, 333*19//2048])

    def test_empty_and_full_masks_fail(self):
        rgb = np.zeros((32, 40, 3), np.uint8)
        for fill in [False, True]:
            with self.assertRaises(ValueError):
                prepare_input(rgb, np.full((32, 40), fill))

    def test_paper_rejects_generic_geometry(self):
        rgb = np.zeros((513, 777, 3), np.uint8)
        mask = np.zeros((1024, 1024), bool)
        mask[100:200, 100:200] = True
        with self.assertRaises(ValueError):
            prepare_input(rgb, mask, mode='paper')


if __name__ == '__main__':
    unittest.main()

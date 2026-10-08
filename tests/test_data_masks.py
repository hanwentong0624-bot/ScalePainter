import unittest

import numpy as np

from scalepainter.data.brushbench_masks import decode_rle_c_order
from scalepainter.data.editbench_masks import extract_editbench_hole_mask


class BenchmarkMaskTests(unittest.TestCase):
    def test_brushbench_rle_is_row_major(self):
        actual = decode_rle_c_order([2, 2, 6, 1], (2, 3))
        np.testing.assert_array_equal(actual, [[0, 1, 1], [0, 0, 1]])

    def test_editbench_ignores_naturally_black_component(self):
        reference = np.full((64, 64, 3), 180, dtype=np.uint8)
        reference[2:9, 2:9] = 0
        masked = reference.copy()
        masked[20:40, 24:48] = 0
        actual, audit = extract_editbench_hole_mask(masked, reference, difference_threshold=8.0)
        expected = np.zeros((64, 64), bool)
        expected[20:40, 24:48] = True
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()

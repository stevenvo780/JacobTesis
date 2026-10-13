"""Regression tests for stable categorical land-use colors."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.image import AxesImage

from src.viz import plot_land_use


class LandUseColorsTest(unittest.TestCase):
    def assert_land_use_plot(self, values):
        land_use = np.array(values, dtype=np.int8)
        original = land_use.copy()
        figures_before = set(plt.get_fignums())
        images = []
        real_imshow = plt.imshow

        def capture_image(*args, **kwargs):
            image = real_imshow(*args, **kwargs)
            images.append(image)
            return image

        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "land_use.png"
                with (
                    matplotlib.rc_context({"savefig.bbox": None, "text.usetex": False}),
                    patch.object(plt, "imshow", wraps=capture_image),
                ):
                    plot_land_use(land_use, str(output))

                self.assertEqual(len(images), 1)
                image = images[0]
                self.assertIsInstance(image, AxesImage)
                np.testing.assert_array_equal(land_use, original)
                np.testing.assert_array_equal(image.get_array(), original)
                self.assertEqual(land_use.dtype, original.dtype)
                self.assertFalse(plt.fignum_exists(image.figure.number))
                self.assertEqual(set(plt.get_fignums()), figures_before)
                self.assertGreater(output.stat().st_size, 0)
                self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(plt.imread(output).shape[:2], (700, 700))
                np.testing.assert_allclose(image.figure.get_size_inches(), [5, 5])
                self.assertEqual(image.get_interpolation(), "nearest")

                colorbar = image.colorbar
                np.testing.assert_array_equal(colorbar.get_ticks(), [0, 1, 2, 3])
                self.assertEqual(
                    [label.get_text() for label in colorbar.ax.get_yticklabels()],
                    ["rural", "urbano", "vegetacion", "agua"],
                )
                for code, color in enumerate(
                    ["#c2b280", "#444444", "#3c8d0d", "#2a6fbb"]
                ):
                    with self.subTest(code=code):
                        np.testing.assert_allclose(
                            image.cmap(image.norm(code)),
                            to_rgba(color),
                            rtol=0,
                            atol=1e-12,
                        )
        finally:
            for number in set(plt.get_fignums()) - figures_before:
                plt.close(number)

    def test_without_water_keeps_vegetation_green(self):
        self.assert_land_use_plot([[0, 1, 2], [2, 1, 0]])

    def test_all_land_use_codes_keep_their_colors(self):
        self.assert_land_use_plot([[0, 1], [2, 3]])

    def test_singleton_land_use_codes_keep_their_colors(self):
        for code in (0, 1, 2, 3):
            with self.subTest(singleton=code):
                self.assert_land_use_plot([[code]])

    def test_without_rural_keeps_the_remaining_colors(self):
        self.assert_land_use_plot([[1, 2, 3], [3, 2, 1]])


if __name__ == "__main__":
    unittest.main()

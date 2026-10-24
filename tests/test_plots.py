"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the units and scaling of the PEC stick plot.
"""

import contextlib
import io
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from colradpy import colradpy


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"


class PecPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_pec_sticks_units_and_scaling(self):
        # The label gave PECs in cm^-3 and a factor 1E<n> while the sticks were multiplied
        # by 10^|n|, so the label read as PEC x 10^n instead of PEC / 10^n.
        m = colradpy(str(INPUT / "cpb03_ls#be0.dat"), metas=np.array([0]),
                     temp_grid=np.array([10.]), electron_den=np.array([1e13]))
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            m.solve_cr()
            m.plot_pec_sticks(temp=[0], dens=[0], meta=[0])
        ax = plt.gca()
        pecs = m.data["processed"]["pecs"][:, 0, 0, 0]
        scaling = int(np.floor(np.log10(pecs.max())))
        self.assertEqual(ax.get_ylabel(), "PEC (10$^{" + str(scaling) + "}$ ph cm$^{3}$ s$^{-1}$)")
        # lines between degenerate levels have an infinite wavelength and no stick
        finite = np.isfinite(m.data["processed"]["wave_air"])
        segments = ax.collections[0].get_segments()
        heights = np.array([s[1, 1] for s, f in zip(segments, finite) if f])
        np.testing.assert_allclose(heights * 10.**scaling, pecs[finite], rtol=1e-12)
        self.assertTrue(1 <= heights.max() < 10)

    def be0_model(self):
        m = colradpy(str(INPUT / "cpb03_ls#be0.dat"), metas=np.array([0]),
                     temp_grid=np.array([10.]), electron_den=np.array([1e13]))
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            m.solve_cr()
        return m

    def test_wavelength_range(self):
        # The axis was fixed to 0-1300 nm, which hid 9 of the 31 Be I lines of at least 1e-3
        # of the strongest (up to 3561 nm).
        m = self.be0_model()
        wave = m.data["processed"]["wave_air"]
        pecs = m.data["processed"]["pecs"][:, 0, 0, 0]
        shown = np.isfinite(wave) & (pecs >= 1e-3 * pecs.max())
        with contextlib.redirect_stdout(io.StringIO()):
            m.plot_pec_sticks(temp=[0], dens=[0], meta=[0])
        low, high = plt.gca().get_xlim()
        self.assertLessEqual(low, wave[shown].min())
        self.assertGreaterEqual(high, wave[shown].max())
        self.assertGreater(wave[shown].max(), 3500)
        with contextlib.redirect_stdout(io.StringIO()):
            m.plot_pec_sticks(temp=[0], dens=[0], meta=[0], wave_range=(200, 500))
        np.testing.assert_allclose(plt.gca().get_xlim(), (200, 500))

    def test_zero_pecs(self):
        # log10 of a zero largest PEC failed
        m = self.be0_model()
        m.data["processed"]["pecs"][:, 0, 0, 0] = 0
        with contextlib.redirect_stdout(io.StringIO()):
            m.plot_pec_sticks(temp=[0], dens=[0], meta=[0])
        self.assertEqual(plt.gca().get_ylabel(), "PEC (10$^{0}$ ph cm$^{3}$ s$^{-1}$)")

    def test_default_style_and_returned_figures(self):
        # plot_pec_sticks set the axes line width and font weight with rc(), so every later
        # figure of the session was drawn with thick axes and semibold text. The plot methods
        # now use the session's style, leave it unchanged and return their figures.
        m = self.be0_model()
        before = dict(plt.rcParams)
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            sticks = m.plot_pec_sticks(temp=[0], dens=[0], meta=[0])
            ratio_t = m.plot_pec_ratio_temp(0, 1, [0], [0])
            ratio_n = m.plot_pec_ratio_dens(0, 1, [0], [0])
        self.assertEqual(dict(plt.rcParams), before)
        for figures in (sticks, ratio_t, ratio_n):
            self.assertEqual(len(figures), 1)
            ax = figures[0].axes[0]
            self.assertEqual(ax.spines["left"].get_linewidth(), plt.rcParams["axes.linewidth"])
            self.assertEqual(ax.title.get_fontweight(), plt.rcParams["axes.titleweight"])
        self.assertEqual(ratio_n[0].axes[0].get_xscale(), "log")


if __name__ == "__main__":
    unittest.main()

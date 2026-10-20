"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks reading adf11 files (ADAS appxa-11) and ionization balances built from
them, with small unresolved files written here for a two-stage element.
"""

import contextlib
import io
import os
import tempfile
import unittest

import numpy as np

from colradpy.ionization_balance_class import ionization_balance
from colradpy.read_adf11 import adf11_class, read_adf11


LOG_DENS = np.array([10.0, 11.0, 12.0])        # log10 cm-3
LOG_TEMP = np.array([0.0, 0.5, 1.0, 1.5])      # log10 eV
# log10 coefficients for stages Z1 = 1 and 2, (temperatures, densities)
VALUES = {
    "scd": {1: -8.0 + 0.4 * LOG_TEMP[:, None] - 0.01 * LOG_DENS[None, :],
            2: -10.0 + 0.6 * LOG_TEMP[:, None] - 0.02 * LOG_DENS[None, :]},
    "acd": {1: -12.0 - 0.5 * LOG_TEMP[:, None] + 0.03 * LOG_DENS[None, :],
            2: -11.5 - 0.4 * LOG_TEMP[:, None] + 0.01 * LOG_DENS[None, :]},
    "ccd": {1: -9.0 + 0.1 * LOG_TEMP[:, None] + 0.0 * LOG_DENS[None, :],
            2: -8.5 + 0.2 * LOG_TEMP[:, None] + 0.0 * LOG_DENS[None, :]},
    "prb": {1: -26.0 + 0.2 * LOG_TEMP[:, None] + 0.0 * LOG_DENS[None, :],
            2: -25.5 + 0.3 * LOG_TEMP[:, None] + 0.0 * LOG_DENS[None, :]},
}


def write_adf11(directory, cls, comment_start="C" + "-" * 79):
    """An unresolved adf11 file of class cls for nuclear charge 2."""
    def rows(values):
        return ["".join(f"{v:10.5f}" for v in values[k:k + 8]) for k in range(0, len(values), 8)]

    lines = [f"{2:5d}{len(LOG_DENS):5d}{len(LOG_TEMP):5d}{1:5d}{2:5d}     /TEST              /GCR PROJECT",
             "-" * 80] + rows(LOG_DENS) + rows(LOG_TEMP)
    for z1, block in VALUES[cls].items():
        lines.append("-" * 20 + f"/ Z1= {z1:<3d}/ DATE= 18/03/97")
        for row in block:
            lines += rows(row)
    lines += [comment_start, "C  written for the ColRadPy tests", "C" + "-" * 79]
    path = os.path.join(directory, f"{cls}96_te.dat")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def balance(paths, **kwargs):
    # The grids are file grid points, where interpolation gives the tabulated values.
    with contextlib.redirect_stdout(io.StringIO()):
        result = ionization_balance(paths, [], 10**LOG_TEMP[1:3], 10**LOG_DENS[:2],
                                    adf11_files=True, **kwargs)
        result.populate_ion_matrix()
    return result


class ReadAdf11Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_class_from_file_name(self):
        self.assertEqual(adf11_class("/data/scd_and_acd/acd96_c.dat"), ("acd", False))
        self.assertEqual(adf11_class(r"C:\adf11\scd96r_c.dat"), ("scd", True))
        # prb and prc contain an r but are unresolved unless the r follows the year.
        self.assertEqual(adf11_class("prb96_c.dat"), ("prb", False))
        self.assertEqual(adf11_class("prc96r_c.dat"), ("prc", True))

    def test_values(self):
        for cls in VALUES:
            with self.subTest(cls=cls):
                adf11 = read_adf11(write_adf11(self.tmp.name, cls))["input_file"]
                self.assertFalse(adf11["resolved"])
                np.testing.assert_allclose(adf11["dens_grid"], LOG_DENS)
                np.testing.assert_allclose(adf11["temp_grid"], LOG_TEMP)
                for z1, block in VALUES[cls].items():
                    np.testing.assert_allclose(adf11[str(z1 - 1)][0, 0], block, atol=5e-6)

    def test_comments_starting_with_dashes(self):
        # Some files start their comments with a line of dashes instead of C-----.
        path = write_adf11(self.tmp.name, "acd", comment_start="-" * 80)
        adf11 = read_adf11(path)["input_file"]
        np.testing.assert_allclose(adf11["1"][0, 0], VALUES["acd"][2], atol=5e-6)


class Adf11BalanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Directory names containing other class names must not change the class.
        self.directory = os.path.join(self.tmp.name, "scd_acd_ccd")
        os.mkdir(self.directory)
        self.paths = {cls: write_adf11(self.directory, cls) for cls in VALUES}

    def expected(self, cls, z1):
        return 10**VALUES[cls][z1][1:3, :2]

    def test_coefficients_and_matrix(self):
        for order in (("scd", "acd"), ("acd", "scd")):
            with self.subTest(order=order):
                result = balance([self.paths[c] for c in order])
                gcrs = result.data["cr_data"]["gcrs"]
                for stage, z1 in (("0", 1), ("1", 2)):
                    for cls in ("scd", "acd"):
                        np.testing.assert_allclose(gcrs[stage][cls][0, 0], self.expected(cls, z1),
                                                   rtol=1e-4)
                    np.testing.assert_array_equal(gcrs[stage]["qcd"], 0)
                    np.testing.assert_array_equal(gcrs[stage]["xcd"], 0)
                matrix = result.data["ion_matrix"]
                np.testing.assert_allclose(matrix[1, 0], self.expected("scd", 1), rtol=1e-4)
                np.testing.assert_allclose(matrix[0, 1], self.expected("acd", 1), rtol=1e-4)
                np.testing.assert_allclose(matrix.sum(axis=0), 0, atol=1e-20)

    def test_charge_exchange(self):
        hdens = 10**LOG_DENS[:2] / 10
        result = balance([self.paths[c] for c in ("ccd", "acd", "scd")], use_cx=True,
                         htemp_grid=10**LOG_TEMP[1:3], hdens_grid=hdens)
        self.assertTrue(result.data["user"]["use_cx"].all())
        matrix = result.data["ion_matrix"]
        expected = self.expected("acd", 1) + self.expected("ccd", 1) * hdens / 10**LOG_DENS[:2]
        np.testing.assert_allclose(matrix[0, 1], expected, rtol=1e-4)
        np.testing.assert_allclose(matrix.sum(axis=0), 0, atol=1e-20)

    def test_missing_files(self):
        with self.assertRaises(ValueError):
            balance([self.paths["scd"]])
        with self.assertRaises(ValueError):
            balance([self.paths["scd"], self.paths["acd"]], use_cx=True)

    def test_solutions(self):
        result = balance([self.paths["acd"], self.paths["scd"]])
        with np.errstate(all="ignore"):
            result.solve_time_independent()
            result.solve_no_source(td_t=np.array([1e-4, 1e-2]))
            result.solve_source(ne_tau=np.array([1e8, 1e10]))
        steady = result.data["processed"]["pops_ss"]
        np.testing.assert_allclose(steady.sum(axis=0), 1.0)
        self.assertEqual(result.data["processed"]["pops_td"].shape, (3, 2, 2, 2))
        self.assertEqual(result.data["processed"]["pops_td_source"].shape, (3, 2, 2, 2))


if __name__ == "__main__":
    unittest.main()

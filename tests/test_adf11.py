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


#metastable resolved files (ADAS writes F10.5 without the leading zero, e.g. -.69877) for nuclear
#charge 2 with 2, 2 and 1 metastables in the stages
METAS = (2, 2, 1)
LOG_TEMP_R = np.array([-0.69877, 0.00020, 0.5, 1.30103])


def fortran(v):
    text = f"{v:.5f}"
    if text.startswith("-0."):
        text = "-" + text[2:]
    elif text.startswith("0."):
        text = text[1:]
    return text.rjust(10)


def resolved_value(cls, z1, i, j, t, d):
    """A distinct log10 coefficient for every block, temperature and density."""
    return -10. - z1 - 0.1 * i - 0.01 * j + 0.3 * LOG_TEMP_R[t] - 0.02 * LOG_DENS[d] - 0.5 * ("sqax".index(cls[0]))


def write_resolved(directory, cls):
    def rows(values):
        return ["".join(fortran(v) for v in values[k:k + 8]) for k in range(0, len(values), 8)]
    lines = [f"{2:5d}{len(LOG_DENS):5d}{len(LOG_TEMP_R):5d}{1:5d}{2:5d}     /TEST              /GCR PROJECT",
             "-" * 80, "".join(f"{m:5d}" for m in METAS), "-" * 80] + rows(LOG_DENS) + rows(LOG_TEMP_R)
    for z1 in (1, 2):
        if cls in ("scd", "acd"):
            pairs = [(("IPRT", p), ("IGRD", g)) for g in range(1, METAS[z1 - 1] + 1) for p in range(1, METAS[z1] + 1)]
        elif cls == "qcd":
            pairs = [(("IGRD", i), ("JGRD", j)) for i in range(1, METAS[z1 - 1] + 1) for j in range(1, METAS[z1 - 1] + 1) if i != j]
        else:
            pairs = [(("IPRT", i), ("JPRT", j)) for i in range(1, METAS[z1] + 1) for j in range(1, METAS[z1] + 1) if i != j]
        for (a, i), (b, j) in pairs:
            lines.append("-" * 21 + f"/ {a}= {i:<3d}/ {b}= {j:<3d}/--------/ Z1= {z1:<3d}/ DATE= 04/11/99")
            for t in range(len(LOG_TEMP_R)):
                lines += rows([resolved_value(cls, z1, i, j, t, d) for d in range(len(LOG_DENS))])
    lines += ["C" + "-" * 79, "C  written for the ColRadPy tests", "C" + "-" * 79]
    path = os.path.join(directory, f"{cls}96r_te.dat")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


class ResolvedAdf11Tests(unittest.TestCase):
    """The resolved files of OpenADAS (e.g. scd96r_c.dat) write numbers below one without the
    leading zero, which the reader skipped, so it read the first block as grid values and failed."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.paths = {cls: write_resolved(tmp.name, cls) for cls in ("scd", "acd", "qcd", "xcd")}

    def test_values(self):
        for cls, path in self.paths.items():
            with self.subTest(cls=cls):
                data = read_adf11(path)["input_file"]
                self.assertTrue(data["resolved"])
                np.testing.assert_array_equal(data["metas"], METAS)
                np.testing.assert_allclose(data["temp_grid"], LOG_TEMP_R, atol=1e-12)
                for z1 in (1, 2):
                    block = data[str(z1 - 1)]
                    for idx in np.ndindex(block.shape[:2]):
                        i, j = idx[0] + 1, idx[1] + 1
                        if cls in ("scd", "acd"):  # [ground metastable, parent]
                            expected = [[resolved_value(cls, z1, j, i, t, d) for d in range(len(LOG_DENS))]
                                        for t in range(len(LOG_TEMP_R))]
                        elif i == j:  # not in the file: a zero coefficient
                            self.assertTrue(np.isneginf(block[idx]).all())
                            continue
                        else:  # [from, to]
                            expected = [[resolved_value(cls, z1, i, j, t, d) for d in range(len(LOG_DENS))]
                                        for t in range(len(LOG_TEMP_R))]
                        np.testing.assert_allclose(block[idx], expected, atol=1e-12)

    def test_balance(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = ionization_balance(list(self.paths.values()), [], 10**LOG_TEMP_R[1:3], 10**LOG_DENS[:2],
                                        adf11_files=True)
            result.populate_ion_matrix()
            result.solve_time_independent()
        self.assertEqual(result.data["ion_matrix"].shape[0], sum(METAS))
        np.testing.assert_allclose(result.data["processed"]["pops_ss"].sum(axis=0), 1, rtol=1e-12)
        # The QCD and XCD diagonals, which the files do not give, were read as log10 = 0, a rate
        # of 1 cm3/s that only cancelled to rounding in the matrix, so its columns did not
        # conserve particles (3.8e-4 for the OpenADAS carbon files).
        gcrs = result.data["cr_data"]["gcrs"]
        for stage in gcrs.values():
            for c in ("qcd", "xcd"):
                np.testing.assert_array_equal(np.einsum("ii...->i...", stage[c]), 0)
        matrix = result.data["ion_matrix"]
        np.testing.assert_allclose(matrix.sum(axis=0), 0, atol=1e-14 * np.abs(matrix).max())


def write_blocks(directory, name, blocks, metas=METAS, number=fortran):
    """An adf11 file for nuclear charge 2 with the blocks (header, values (temperature, density))."""
    def rows(values):
        return ["".join(number(v) for v in values[k:k + 8]) for k in range(0, len(values), 8)]
    lines = [f"{2:5d}{len(LOG_DENS):5d}{len(LOG_TEMP_R):5d}{1:5d}{2:5d}     /TEST              /GCR PROJECT",
             "-" * 80]
    if metas:
        lines += ["".join(f"{m:5d}" for m in metas), "-" * 80]
    lines += rows(LOG_DENS) + rows(LOG_TEMP_R)
    for header, values in blocks:
        lines.append(header)
        for row in values:
            lines += rows(row)
    lines += ["C" + "-" * 79, "C  written for the ColRadPy tests", "C" + "-" * 79]
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


class Adf11BlockLabelTests(unittest.TestCase):
    """Blocks are placed by their labels. The reader took the first two numbers of a block line as
    IPRT and IGRD, but the radiated power files of ADAS give IGRD= g / IPRT= 0 (plt96r) or
    IPRT= p / IGRD= 0 (prb96r, prc96r): resolved plt blocks went to the column -1 (the last), and
    resolved prb and prc files failed with IndexError where a stage has more parents than
    metastables."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.count = 0

    def block(self, labels, z1):
        """A block line and distinct values, as written."""
        self.count += 1
        header = "-" * 21 + "".join(f"/ {k}= {v:<3d}" for k, v in labels) + f"/--------/ Z1= {z1:<3d}/ DATE= 04/11/99"
        values = np.array([[float(fortran(-26. - 0.25 * self.count + 0.3 * t - 0.02 * d)) for d in LOG_DENS]
                           for t in LOG_TEMP_R])
        return header, values

    def test_line_power_of_each_metastable(self):
        blocks = {(z1, g): self.block((("IGRD", g), ("IPRT", 0)), z1) for z1 in (1, 2) for g in range(1, METAS[z1 - 1] + 1)}
        data = read_adf11(write_blocks(self.dir, "plt96r_te.dat", blocks.values()))["input_file"]
        for (z1, g), (header, values) in blocks.items():
            self.assertEqual(data[str(z1 - 1)].shape[:2], (METAS[z1 - 1], 1))
            np.testing.assert_array_equal(data[str(z1 - 1)][g - 1, 0], values)

    def test_recombination_power_of_each_parent(self):
        metas = (1, 2, 1)   # a stage with more parents than metastables
        blocks = {(z1, p): self.block((("IPRT", p), ("IGRD", 0)), z1) for z1 in (1, 2) for p in range(1, metas[z1] + 1)}
        data = read_adf11(write_blocks(self.dir, "prb96r_te.dat", blocks.values(), metas=metas))["input_file"]
        for (z1, p), (header, values) in blocks.items():
            self.assertEqual(data[str(z1 - 1)].shape[:2], (metas[z1], 1))
            np.testing.assert_array_equal(data[str(z1 - 1)][p - 1, 0], values)

    def test_recombination_power_of_each_pair(self):
        # prb93r gives IPRT= p / IGRD= g, like acd
        blocks = {(z1, g, p): self.block((("IPRT", p), ("IGRD", g)), z1) for z1 in (1, 2)
                  for g in range(1, METAS[z1 - 1] + 1) for p in range(1, METAS[z1] + 1)}
        data = read_adf11(write_blocks(self.dir, "prb93r_te.dat", blocks.values()))["input_file"]
        for (z1, g, p), (header, values) in blocks.items():
            np.testing.assert_array_equal(data[str(z1 - 1)][g - 1, p - 1], values)

    def test_specific_line_power(self):
        # pls blocks give the wavelength of the line, '/ IGRD= 1  / IPRT= 0  / 1638.59/'; they
        # were stored as keys of a dict instead of an array
        blocks = []
        for z1, wave in ((1, 1638.59), (2, 303.78)):
            header, values = self.block((), z1)
            blocks.append((header.replace("/--------/", f"/ IGRD= 1  / IPRT= 0  / {wave:7.2f}/"), values))
        data = read_adf11(write_blocks(self.dir, "pls96_te.dat", blocks, metas=None))["input_file"]
        for z1, (header, values) in enumerate(blocks, 1):
            np.testing.assert_array_equal(data[str(z1 - 1)][0, 0], values)
        self.assertEqual([b["wavelength"] for b in data["blocks"]], [1638.59, 303.78])

    def test_classes_given_as_blocks(self):
        # ecd files (effective ionisation potentials in eV, not log10) have a block for each
        # charge from Z1 = 0 and failed with KeyError, as did ycd and zcd
        blocks = [self.block((("IGRD", 1), ("IPRT", 1)), z1) for z1 in (0, 1, 2)]
        path = write_blocks(self.dir, "ecd96_te.dat", blocks, metas=None, number=lambda v: f"{-v:10.3E}")
        data = read_adf11(path)["input_file"]
        self.assertEqual(data["class"], "ecd")
        self.assertEqual([b["z1"] for b in data["blocks"]], [0, 1, 2])
        for block, (header, values) in zip(data["blocks"], blocks):
            np.testing.assert_allclose(block["data"], -values, rtol=1e-3)

    def test_unresolved_labels(self):
        # Files without metastables have one block for each stage, whatever their labels:
        # plt12_h gives IPRT= 1 / IGRD= 0
        blocks = [self.block((("IPRT", 1), ("IGRD", 0)), 1)]
        data = read_adf11(write_blocks(self.dir, "plt12_te.dat", blocks, metas=None))["input_file"]
        np.testing.assert_array_equal(data["0"][0, 0], blocks[0][1])

    def test_errors(self):
        blocks = [self.block((("IGRD", 3), ("IPRT", 0)), 1)]   # stage 0 has 2 metastables
        with self.assertRaisesRegex(ValueError, "outside the metastables"):
            read_adf11(write_blocks(self.dir, "plt96r_te.dat", blocks))
        header, values = self.block((("IGRD", 1), ("IPRT", 0)), 1)
        with self.assertRaisesRegex(ValueError, "values, not"):
            read_adf11(write_blocks(self.dir, "plt96r_te.dat", [(header, values[:-1])]))


if __name__ == "__main__":
    unittest.main()

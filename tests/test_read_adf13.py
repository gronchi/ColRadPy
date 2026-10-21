"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the adf13 reader (ionisations per photon, ADAS appxa-13) with files written here in the
layouts of the ADAS libraries.
"""

import os
import tempfile
import unittest

import numpy as np

from colradpy.read_adf13 import read_adf13


def field(value, letter="E"):
    """A value of 9 characters (8e9.2); three-digit exponents lose the letter (1.41+100)."""
    mantissa, exponent = f"{value:.2E}".split("E")
    exponent = int(exponent)
    text = mantissa + (f"{exponent:+04d}" if abs(exponent) > 99 else letter + f"{exponent:+03d}")
    return text.rjust(9), float(mantissa + "e" + str(exponent))


def values_lines(values, letter):
    texts = [field(v, letter)[0] for v in values]
    return ["".join(texts[i:i + 8]) for i in range(0, len(texts), 8)]


class Block:
    def __init__(self, wave, dens, temp, indm="1", header="{wave:9.1f}A{nd:4d}{nt:4d}", letter="E", scale=1.):
        self.wave, self.dens, self.temp, self.indm, self.header, self.letter = wave, dens, temp, indm, header, letter
        rng = np.random.default_rng(len(dens) + 7 * len(temp))
        self.sxb = scale * 10 ** rng.uniform(-1, 3, (len(dens), len(temp)))   # (density, temperature)

    def lines(self, isel):
        head = self.header.format(wave=self.wave, nd=len(self.dens), nt=len(self.temp))
        lines = [head + f" /FILMEM = TEST     /CODE = ADAS208  /INDM = {self.indm}/ISEL = {isel:4d}"]
        lines += values_lines(self.dens, self.letter) + values_lines(self.temp, self.letter)
        for row in self.sxb:
            lines += values_lines(row, self.letter)
        return lines

    def expected(self):
        """The S/XB as written, (temperatures, densities)."""
        return np.array([[field(v, self.letter)[1] for v in row] for row in self.sxb]).T


def write(path, blocks, comments=True):
    lines = [f"{len(blocks):5d}    /C + 1 IONISATIONS PER PHOTON         /"]
    for i, b in enumerate(blocks):
        lines += b.lines(i + 1)
    if comments:
        lines += ["C" + "-" * 76, "C  written for the ColRadPy tests", "C" + "-" * 76]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


DENS = [1e11, 1e12, 1e13, 1e14, 1e15]
TEMP = [2., 5., 10., 20., 50., 100., 200., 500., 1000.]


class ReadAdf13Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def read(self, blocks, comments=True):
        return read_adf13(write(os.path.join(self.tmp.name, "sxb.dat"), blocks, comments))["input_file"]

    def check(self, blocks, data):
        np.testing.assert_array_equal(data["wave_air"], [b.wave for b in blocks])
        for i, b in enumerate(blocks):
            np.testing.assert_array_equal(data["sxb"][i], b.expected())

    def test_layouts(self):
        # sxb96#h of ADAS writes wavelengths of 100000 A and more from the first column
        # ('123665.9A   24  29'), which the reader took as the number of densities, 24; the
        # ionelec files write '  904.0 A', on which it failed.
        blocks = [Block(1215.2, DENS, TEMP, header="{wave:8.1f}A{nd:5d}{nt:4d}"),
                  Block(123665.9, DENS, TEMP, header="{wave:8.1f}A{nd:5d}{nt:4d}"),
                  Block(904.0, DENS, TEMP, header="{wave:7.1f} A{nd:4d}{nt:4d}")]
        data = self.read(blocks)
        self.check(blocks, data)
        self.assertEqual((data["num_trans"], data["element"], data["charge_state"]), (3, "C", 1))
        np.testing.assert_array_equal(data["dens_grid"], DENS)
        np.testing.assert_array_equal(data["temp_grid"], TEMP)
        self.assertEqual(data["sxb"].shape, (3, len(TEMP), len(DENS)))

    def test_d_exponents(self):
        # The example of appxa-13 writes 1.00D+11, which the reader failed on
        blocks = [Block(8254.0, DENS, TEMP, letter="D")]
        self.check(blocks, self.read(blocks))

    def test_metastable_of_each_block(self):
        # Metastable resolved files (pjr, llr) give the S/XB of each metastable in its own blocks;
        # the reader kept the INDM of the first block only.
        blocks = [Block(1335.7, DENS, TEMP, indm="1"), Block(1335.7, DENS, TEMP, indm="2", scale=3.)]
        data = self.read(blocks)
        np.testing.assert_array_equal(data["meta_indx"], ["1", "2"])
        self.check(blocks, data)

    def test_three_digit_exponent(self):
        # sxb96#si of ADAS writes 1.41+100, which the reader failed on
        block = Block(1045.4, DENS, TEMP)
        block.sxb[0, 0] = 1.41e100
        data = self.read([block])
        self.assertEqual(data["sxb"][0, 0, 0], 1.41e100)
        self.check([block], data)

    def test_grids_of_each_block(self):
        # Each block has its own grids (the example of appxa-13 has 5 and 4 densities); the
        # reader used the sizes of the first block for all of them.
        blocks = [Block(8254.0, DENS, TEMP), Block(4573.0, DENS[:4], TEMP[:7])]
        data = self.read(blocks)
        self.assertEqual(data["num_dens"], [5, 4])
        np.testing.assert_array_equal(data["temp_grid"][1], TEMP[:7])
        self.check(blocks, data)

    def test_without_comments(self):
        # The reader looped forever at the end of a file without comments.
        blocks = [Block(8254.0, DENS, TEMP), Block(4573.0, DENS, TEMP)]
        self.check(blocks, self.read(blocks, comments=False))

    def test_errors(self):
        path = os.path.join(self.tmp.name, "short.dat")
        block = Block(8254.0, DENS, TEMP)
        with open(path, "w") as f:
            f.write("\n".join([f"{2:5d}    /C + 1 IONISATIONS PER PHOTON         /"] + block.lines(1)) + "\n")
        with self.assertRaisesRegex(ValueError, "transition 2"):
            read_adf13(path)
        with open(path, "w") as f:
            f.write("\n".join([f"{1:5d}    /C + 1 IONISATIONS PER PHOTON         /"] + block.lines(1)[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "transition 1 has"):
            read_adf13(path)


if __name__ == "__main__":
    unittest.main()

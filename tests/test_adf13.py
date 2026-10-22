"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks adf13 output (S/XB, ionizations per photon) against the fixed-width ADAS format (appxa-13):
the first line is i5,4x,'/',c35,'/' with the ion as a2,'+',i2, each block header f9,'A',2i4 followed
by '/'-delimited keywords, and densities, temperatures and S/XB are 8e9.2 fields; and the values
against the SCD and PECs they are made from.
"""

from pathlib import Path
import contextlib
import io
import os
import re
import tempfile
import unittest
import warnings

import numpy as np

from colradpy import colradpy
from colradpy.read_adf13 import read_adf13
from colradpy.write_adf13 import MAX_SXB, MIN_SXB, write_adf13


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"
KEYWORDS = re.compile(r" /FILMEM = (.{8})/CODE = (.{8}) /INDM =(.{2})/ISEL =(.{5})$")


class Adf13Reader:
    """Strict reader of the fixed-width fields, one dict per block, with the rules of the ADAS
    reader: the ion in the five characters after the first '/', the wavelength before the 'A' of
    the first ten characters, '/' delimited keywords, and the data until a line starting with C."""

    def __init__(self, test, path):
        self.test = test
        with open(path) as f:
            self.lines = f.read().splitlines()
        first = self.lines[0]
        test.assertEqual(first[5:10], "    /")
        test.assertEqual(len(first), 10 + 35 + 1)
        test.assertEqual(first[-1], "/")
        self.nsel = int(first[:5])
        self.ion = first[10:15]
        self.blocks = []
        self.k = 1
        while not self.lines[self.k].startswith("C"):
            header = self.lines[self.k]
            self.k += 1
            test.assertEqual(header[9:11], " A", header)
            match = KEYWORDS.match(header[19:])
            test.assertIsNotNone(match, header)
            block = {"wave": float(header[:9]), "ndens": int(header[11:15]), "nte": int(header[15:19]),
                     "filmem": match.group(1).strip(), "code": match.group(2).strip(),
                     "indm": match.group(3).strip(), "isel": int(match.group(4))}
            block["dens"] = self.values(block["ndens"])
            block["temp"] = self.values(block["nte"])
            block["sxb"] = np.array([self.values(block["nte"]) for _ in range(block["ndens"])])
            self.blocks.append(block)
        self.comments = self.lines[self.k:]

    def values(self, n):
        """Read n values from whole lines of at most eight e9.2 fields."""
        out = []
        while len(out) < n:
            line = self.lines[self.k]
            self.k += 1
            self.test.assertEqual(len(line) % 9, 0, line)
            self.test.assertLessEqual(len(line) // 9, 8, line)
            for i in range(0, len(line), 9):
                field = line[i:i + 9]
                self.test.assertRegex(field, r"^ \d\.\d\dE[+-]\d\d$")
                out.append(float(field))
        self.test.assertEqual(len(out), n)
        return np.array(out)


def solve(metas, **kwargs):
    model = colradpy(str(INPUT / "cpb03_ls#be0.dat"), metas=np.array(metas), temp_grid=np.geomspace(2., 50., 9),
                     electron_den=np.array([1e12, 1e13]), **kwargs)
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.solve_cr()
    return model


def write(args, **kwargs):
    """Text of the adf13 file written by write_adf13(path, *args, **kwargs)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.adf13")
        write_adf13(path, *args, **kwargs)
        with open(path) as f:
            return f.read()


class Adf13Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def written(self, model, **kwargs):
        path = os.path.join(self.dir, "be0.adf13")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.write_sxb_adf13(fil_name=path, **kwargs)
        return Adf13Reader(self, path), path

    def test_bundled_model(self):
        # One block for each line with excitation emission, S/XB = SCD summed over the parents over
        # the excitation PEC, INDM = T for the populations normalized to the whole ion.
        model = solve([0])
        processed = model.data["processed"]
        excitation = processed["pecs"][:, 0]
        lines = np.where((excitation > 0).any(axis=(1, 2)) & np.isfinite(processed["wave_air"]) &
                         (processed["wave_air"] < 1e7))[0]
        adf13, path = self.written(model)
        self.assertEqual(adf13.ion, "Be+ 0")
        self.assertEqual(adf13.nsel, len(lines))
        self.assertEqual([b["isel"] for b in adf13.blocks], list(range(1, len(lines) + 1)))
        ionization = processed["scd"][0].sum(axis=0)
        for line, block in zip(lines, adf13.blocks):
            self.assertEqual((block["filmem"], block["code"], block["indm"]), ("cpb03_ls", "ColRadPy", "T"))
            self.assertAlmostEqual(block["wave"], processed["wave_air"][line] * 10, delta=1e-6 * block["wave"] + 1e-3)
            np.testing.assert_allclose(block["dens"], model.data["user"]["dens_grid"], rtol=5e-3)
            np.testing.assert_allclose(block["temp"], model.data["user"]["temp_grid"], rtol=5e-3)
            with np.errstate(divide="ignore"):
                expected = np.where(excitation[line] > 0, ionization / excitation[line], MIN_SXB)
            np.testing.assert_allclose(block["sxb"], np.clip(expected, MIN_SXB, MAX_SXB).T, rtol=5e-3)
        # ColRadPy's own reader reads the same file
        data = read_adf13(path)["input_file"]
        self.assertEqual((data["element"], data["charge_state"], data["num_trans"]), ("Be", 0, len(lines)))
        np.testing.assert_array_equal(data["meta_indx"], "T")
        np.testing.assert_allclose(data["sxb"][0], adf13.blocks[0]["sxb"].T)
        # the comments list each block with the levels of its line
        table = adf13.comments.index("C  ISEL  WAVELENGTH (A)  TYPE   INDM  UPPER  LOWER") + 2
        row = adf13.comments[table].split()
        self.assertEqual(float(row[2]), adf13.blocks[0]["wave"])
        self.assertEqual(row[1:2] + row[3:], ["1", "EXCIT", "T"] + [str(v) for v in processed["pec_levels"][lines[0]]])

    def test_metastables(self):
        # With two metastables the blocks of each are written in turn, INDM = 1, 2, each with the
        # SCD and excitation PECs of its metastable.
        model = solve([0, 1])
        processed = model.data["processed"]
        lines = np.array([2, 5])
        adf13, _ = self.written(model, sxb_inds=lines)
        self.assertEqual([(b["indm"], b["isel"]) for b in adf13.blocks], [("1", 1), ("1", 2), ("2", 3), ("2", 4)])
        for n, block in enumerate(adf13.blocks):
            m, line = n // 2, lines[n % 2]
            expected = processed["scd"][m].sum(axis=0) / processed["pecs"][line, m]
            np.testing.assert_allclose(block["sxb"], np.clip(expected, MIN_SXB, MAX_SXB).T, rtol=5e-3)

    def test_values_and_limits(self):
        grids = (np.array([1e13]), np.array([5., 10., 20.]), np.array([0]))
        sxb = np.array([[[[0.], [-1.], [1e120]]]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            text = write(([0], np.array([1000.]), sxb, "Ar", 16) + grids)
        messages = " | ".join(str(w.message) for w in caught)
        self.assertIn("negative", messages)
        self.assertIn("above", messages)
        self.assertIn(" 1.00E-74 1.00E-74 9.99E+99", text)
        self.assertTrue(text.startswith("    1    /Ar+16 IONISATIONS PER PHOTON       /\n"))
        self.assertIn("/FILMEM = colradpy/CODE = ColRadPy /INDM = 1/ISEL =    1", text)
        self.assertIn("/INDM = T/", write(([0], np.array([1000.]), np.ones((1, 1, 3, 1)), "Ar", 16) + grids, total=True))
        with self.assertRaisesRegex(ValueError, "not finite"):
            write(([0], np.array([1000.]), np.full((1, 1, 3, 1), np.nan), "Ar", 16) + grids)
        with self.assertRaisesRegex(ValueError, "one metastable"):
            write(([0], np.array([1000.]), np.ones((1, 2, 3, 1)), "Ar", 16, grids[0], grids[1], [0, 1]), total=True)
        with self.assertRaisesRegex(ValueError, "metastable columns"):
            write(([0], np.array([1000.]), np.ones((1, 2, 3, 1)), "Ar", 16) + grids)
        with self.assertWarnsRegex(UserWarning, "8 values on each line"):
            write(([0], np.array([1000.]), np.ones((1, 1, 3, 1)), "Ar", 16) + grids, num=10)
        with self.assertWarnsRegex(UserWarning, "three digits of ISEL"):
            write((np.arange(1000), np.full(1000, 1000.), np.ones((1000, 1, 3, 1)), "Ar", 16) + grids)

    def test_errors_of_the_method(self):
        with self.assertRaisesRegex(ValueError, "use_ionization"):
            solve([0], use_ionization=False).write_sxb_adf13(fil_name=os.devnull)
        paired = colradpy(str(INPUT / "cpb03_ls#be0.dat"), metas=np.array([0]), temp_grid=np.array([10., 20.]),
                          electron_den=np.array([1e12, 1e13]), temp_dens_pair=True)
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            paired.solve_cr()
        with self.assertRaisesRegex(ValueError, "temp_dens_pair"):
            paired.write_sxb_adf13(fil_name=os.devnull)


if __name__ == "__main__":
    unittest.main()

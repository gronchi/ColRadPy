"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks adf15 output against the fixed-width ADAS format (appxa-15): the first
line is i5,4x,'/', each block header f9,'A',2i4 followed by '/'-delimited
keywords, and densities, temperatures and PECs are 8e9.2 fields.
"""

from pathlib import Path
import os
import re
import tempfile
import unittest
import warnings

import numpy as np

from colradpy import colradpy
from colradpy.write_adf15 import MAX_WAVELENGTH, MIN_PEC, write_adf15


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"
KEYWORDS = re.compile(r" /FILMEM = .{8}/TYPE = (.{8}) /INDM =(.{2})/ISEL =(.{5})$")


class Adf15Reader:
    """Strict reader of the fixed-width fields, one dict per block."""

    def __init__(self, test, path):
        self.test = test
        with open(path) as f:
            self.lines = f.read().splitlines()
        first = self.lines[0]
        test.assertEqual(first[5:10], "    /")
        self.nsel = int(first[:5])
        self.blocks = []
        self.k = 1
        while not self.lines[self.k].startswith("C"):
            header = self.lines[self.k]
            self.k += 1
            test.assertEqual(header[9:11], " A", header)
            match = KEYWORDS.match(header[19:])
            test.assertIsNotNone(match, header)
            block = {
                "wave": float(header[:9]), "ndens": int(header[11:15]),
                "nte": int(header[15:19]), "type": match.group(1).strip(),
                "indm": match.group(2).strip(), "isel": int(match.group(3)),
            }
            block["dens"] = self.values(block["ndens"])
            block["temp"] = self.values(block["nte"])
            block["pec"] = np.array([self.values(block["nte"]) for _ in range(block["ndens"])])
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


def write(args, **kwargs):
    """Text of the adf15 file written by write_adf15(path, *args, **kwargs)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.adf15")
        write_adf15(path, *args, **kwargs)
        with open(path) as f:
            return f.read()


class Adf15Tests(unittest.TestCase):
    def read(self, path):
        return Adf15Reader(self, path)

    def test_bundled_model_round_trip(self):
        temperatures = np.geomspace(2., 50., 9)  # more than eight per line
        densities = np.array([1e12, 1e13])
        model = colradpy(str(INPUT / "cpb03_ls#be0.dat"), metas=np.array([0]),
                         temp_grid=temperatures, electron_den=densities)
        # Degenerate levels in bundled data produce a pre-existing
        # zero-energy wavelength warning, outside this check's scope.
        with np.errstate(divide="ignore"):
            model.solve_cr()
        processed = model.data["processed"]
        # Lines between degenerate or nearly degenerate levels (infinite or
        # centimetre wavelengths) do not fit the f9 field and are skipped.
        wave = processed["wave_air"] * 10
        written = np.where((wave > 0) & (wave < MAX_WAVELENGTH))[0]
        self.assertEqual(len(wave) - len(written), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "be0.adf15")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.write_pecs_adf15(fil_name=path)
            adf15 = self.read(path)

        # One block per line and driver (ground excitation, then recombination).
        drivers = processed["pecs"].shape[1]
        self.assertEqual(drivers, 2)
        self.assertEqual(adf15.nsel, len(adf15.blocks))
        self.assertEqual(adf15.nsel, len(written) * drivers)
        self.assertEqual([b["isel"] for b in adf15.blocks], list(range(1, adf15.nsel + 1)))
        for d, (typ, indm) in enumerate((("EXCIT", "1"), ("RECOM", "+1"))):
            blocks = adf15.blocks[d * len(written):(d + 1) * len(written)]
            self.assertEqual({(b["type"], b["indm"]) for b in blocks}, {(typ, indm)})
            for line, block in zip(written, blocks):
                # Wavelengths in A, not ten times larger.
                self.assertAlmostEqual(block["wave"], processed["wave_air"][line] * 10,
                                       delta=1e-6 * block["wave"] + 1e-3)
                np.testing.assert_allclose(block["dens"], densities, rtol=5e-3)
                np.testing.assert_allclose(block["temp"], temperatures, rtol=5e-3)
                expected = np.maximum(processed["pecs"][line, d].T, MIN_PEC)
                np.testing.assert_allclose(block["pec"], expected, rtol=5e-3)

        self.assertTrue(all(line.startswith("C") for line in adf15.comments))
        self.assertIn("C  NUCLEAR CHARGE =  4", adf15.comments)
        self.assertIn("C  ION CHARGE +1  =  1", adf15.comments)

    def test_selection_and_options_of_the_method(self):
        model = colradpy(str(INPUT / "cpb03_ls#be0.dat"), metas=np.array([0]),
                         temp_grid=np.array([10.]), electron_den=np.array([1e13]),
                         use_recombination=False, use_recombination_three_body=False)
        with np.errstate(divide="ignore"):
            model.solve_cr()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "be0.adf15")
            # A one-element array selects that line, not all of them.
            model.write_pecs_adf15(fil_name=path, pec_inds=np.array([3]))
            adf15 = self.read(path)
            self.assertEqual(len(adf15.blocks), 1)
            self.assertEqual(adf15.blocks[0]["type"], "EXCIT")
            model.write_pecs_adf15(fil_name=path, pec_inds=[3, 5])
            self.assertEqual(self.read(path).nsel, 2)

        paired = colradpy(str(INPUT / "cpb03_ls#be0.dat"), metas=np.array([0]),
                          temp_grid=np.array([10., 20.]), electron_den=np.array([1e12, 1e13]),
                          temp_dens_pair=True)
        with np.errstate(divide="ignore"):
            paired.solve_cr()
        with self.assertRaises(ValueError):
            paired.write_pecs_adf15(fil_name=os.devnull)

    def test_driver_types_and_values(self):
        # Two metastables, one recombining parent and one charge exchange parent.
        pecs = np.full((2, 4, 1, 1), 1e-10)
        pecs[1, 1, 0, 0] = 0.
        pecs[1, 2, 0, 0] = 1e-120
        wave = np.array([3.99392, 4548.315])
        args = ([1], wave, pecs, "Be", 0, np.array([1e13]), np.array([10.]),
                np.array([0, 1]), np.array([1e5]))
        lines = write(args, num_recomb=1).splitlines()
        headers = [i for i, line in enumerate(lines) if "/ISEL" in line]
        drivers = [KEYWORDS.match(lines[i][19:]).groups()[:2] for i in headers]
        self.assertEqual([(typ.strip(), indm) for typ, indm in drivers],
                         [("EXCIT", " 1"), ("EXCIT", " 2"), ("RECOM", "+1"), ("CHEXC", "+1")])
        # Zero and values that e9.2 cannot hold are written as the smallest PEC
        # (the PEC line follows the header, density and temperature lines).
        self.assertEqual(lines[headers[1] + 3], " 1.00E-74")
        self.assertEqual(lines[headers[2] + 3], " 1.00E-74")

        # X-ray wavelengths keep their precision in the f9 field.
        text = write(([0],) + args[1:], num_recomb=1)
        self.assertTrue(text.splitlines()[1].startswith(" 3.993920 A"))

        # Without recombination the column after the metastables is charge exchange.
        text = write(args, num_recomb=0)
        self.assertIn("/TYPE = CHEXC    /INDM =+2/ISEL =    4", text)

    def test_values_that_cannot_be_written(self):
        base = np.full((1, 1, 1, 1), 1e-10)
        common = ("Be", 0, np.array([1e13]), np.array([10.]), np.array([0]), np.array([]))
        for wave in (np.inf, -10., 0., 1e9):
            with self.subTest(wave=wave), self.assertRaises(ValueError):
                write(([0], np.array([wave]), base) + common)
        with self.assertRaises(ValueError):
            write(([0], np.array([5000.]), np.full((1, 1, 1, 1), np.nan)) + common)
        with self.assertWarns(UserWarning):
            write(([0], np.array([5000.]), -base) + common)
        # The caller's PECs are not modified by the floor.
        pecs = np.zeros((1, 1, 1, 1))
        write(([0], np.array([5000.]), pecs) + common)
        self.assertEqual(pecs[0, 0, 0, 0], 0.)


if __name__ == "__main__":
    unittest.main()

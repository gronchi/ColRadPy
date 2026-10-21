"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the adf17 reader (condensed projection matrices, ADAS appxa-17) with files written
here in the formats of the cbnm93 and cbnm96 libraries, and that importing it reads no file.
"""

import importlib
import os
import sys
import tempfile
import unittest

import numpy as np


GRIDS = {"dens": [1e3, 1e6, 1e9, 1e12, 1e15], "temp": [5e2, 1e3, 2e3, 5e3, 1e4, 2e4]}


def number(mantissa, exponent, letter):
    """A field of 17 characters as ADAS208 writes it (three-digit exponents lose the letter)."""
    text = f"{mantissa:.10f}" + (f"{exponent:+04d}" if abs(exponent) > 99 else letter + f"{exponent:+03d}")
    return text.rjust(17)


def value_of(field):
    text = field.strip().replace("D", "e")
    if "e" not in text and "E" not in text:
        text = text[:-4] + "e" + text[-4:]
    return float(text)


def place(fields, length):
    """A line with the fields (column -> text) at their columns."""
    line = [" "] * length
    for column, text in fields.items():
        line[column:column + len(text)] = text
    return "".join(line).rstrip()


class Adf17File:
    """An adf17 file whose values follow from their position in the file."""

    def __init__(self, letter="D", parents=(("2S", 2., ((1., 4),)),), with_qc=True, with_prb=False,
                 big_exponent=False):
        self.letter, self.parents, self.with_qc, self.with_prb = letter, parents, with_qc, with_prb
        self.big_exponent = big_exponent
        self.values = {}  # (parent, system, kind, indices) -> value

    def field(self, key):
        index = len(self.values)
        sign = -1 if index % 3 == 0 else 1
        mantissa, exponent = sign * (1 + (index % 89) / 97.), -((index * 7) % 60)
        if self.big_exponent and index == 5:
            exponent = 102
        text = number(mantissa, exponent, self.letter)
        self.values[key] = value_of(text)
        return text

    def write(self, path):
        nd, nt, npar = len(GRIDS["dens"]), len(GRIDS["temp"]), len(self.parents)
        lines = [f" SEQUENCE = BE   NUCCHG = 6   NPARNT = {npar}   MAXD = {nd}   MAXT ={nt:2d}", " ",
                 " IEDMAT = 0   IECION = 0   IETREC = 0   IEDREC = 0   IERREC = 0   IEXREC = 0   IERSYS = 0", " "]
        for label, key in (("NE(CM-3) =", "dens"), ("TE(K)    =", "temp")):
            values = [f"{v:.2E}".replace("E", "D").rjust(10) for v in GRIDS[key]]
            lines.append(" " + label + "".join(values[:3]))      # the grids continue on a second line
            lines.append(" " * 11 + "".join(values[3:]))
        lines.append(" ")
        for p, (term, spin, systems) in enumerate(self.parents):
            lines += [f" IPRT = {p + 1:2d}     TRMPRT = ({term})     SPNPRT = {spin:3.0f}.", " " + "-" * 44]
            for s, (system_spin, shells) in enumerate(systems):
                lines += [" " * 50 + f"SPNSYS = {system_spin:2.0f}.     NSHEL = {shells:2d}", " " * 50 + "-" * 27]
                for d in range(nd):
                    for t in range(nt):
                        for kind in ("proj", "dir"):
                            for i in range(shells):
                                prefix = (f" ID={d + 1:2d} IT={t + 1:2d} " if kind == "proj" and i == 0 else
                                          "       (DIR) " if i == 0 else " " * 13) + f"N={i + 2:2d}  "
                                fields = {0: prefix}
                                for j in range(shells):
                                    fields[19 + 17 * j] = self.field((p, s, kind, "crmat", i, j, t, d))
                                for q in range(npar):
                                    fields[106 + 17 * q] = self.field((p, s, kind, "ion", i, q, t, d))
                                tail = ["rhs"] if kind == "proj" else ["three", "diel", "rad", "cx"]
                                for c, name in enumerate(tail):
                                    fields[176 + 17 * c] = self.field((p, s, kind, name, i, t, d))
                                lines.append(place(fields, 260))
                        if self.with_qc:
                            fields = {0: "       (PRB)  "} if self.with_prb else {}
                            if self.with_prb:
                                fields[19] = self.field((p, s, "prb", t, d))
                            for q in range(npar):
                                fields[106 + 17 * q] = self.field((p, s, "qc", q, t, d))
                            lines.append(place(fields, 200))
                        lines.append(" ")
        lines += ["C" + "-" * 79, "C  written for the ColRadPy tests", "C" + "-" * 79]
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return path


class ReadAdf17Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def check(self, written, data):
        np.testing.assert_allclose(data["dens_grid"], GRIDS["dens"])
        np.testing.assert_allclose(data["temp_grid"], GRIDS["temp"])
        self.assertEqual(len(data["parents"]), len(written.parents))
        names = {"three": "recomb_three_body", "diel": "recomb_diel", "rad": "recomb_rad", "cx": "recomb_cx"}
        for (p, s, kind, *rest), value in written.values.items():
            system = data["parents"][p]["spin_systems"][s]
            if kind == "prb":
                got = system["prb"][tuple(rest)]
            elif kind == "qc":
                got = system["parent_qc"][tuple(rest)]
            else:
                block = system if kind == "proj" else system["dir"]
                name, *index = rest
                got = block[names.get(name, name)][tuple(index)]
            self.assertEqual(got, value, (p, s, kind, rest))

    def read(self, written, name):
        from colradpy.read_adf17 import read_adf17
        return read_adf17(written.write(os.path.join(self.tmp.name, name)))["input_file"]

    def test_cbnm93_format(self):
        # The reader split lines on double spaces: 77 of 86 ADAS adf17 files failed
        # (IndexError or ValueError), and it read only the first parent and spin system.
        written = Adf17File(big_exponent=True)
        data = self.read(written, "cbnm93#be_c2ls.dat")
        self.assertEqual((data["sequence"], data["nuc_chg"], data["num_parent"]), ("BE", 6, 1))
        system = data["parents"][0]["spin_systems"][0]
        np.testing.assert_array_equal(system["n"], [2, 3, 4, 5])
        self.assertEqual(system["crmat"].shape, (4, 4, 6, 5))
        self.assertIsNone(system["prb"])
        self.assertTrue(any(abs(v) > 1e100 for v in written.values.values()))  # written as 1.05...+102
        self.check(written, data)

    def test_cbnm96_format(self):
        # E exponents, a (PRB) line after each temperature, two parents and two spin systems
        written = Adf17File(letter="E", with_prb=True,
                            parents=(("2P", 2., ((1., 4), (3., 3))), ("4P", 4., ((3., 3), (5., 2)))))
        data = self.read(written, "cbnm96#c_n1ls.dat")
        self.assertEqual([p["term"] for p in data["parents"]], ["2P", "4P"])
        self.assertEqual([s["spin"] for s in data["parents"][1]["spin_systems"]], [3., 5.])
        self.check(written, data)

    def test_without_parent_coefficients(self):
        # cbnm93#he files have no line of parent coefficients after each temperature
        written = Adf17File(with_qc=False)
        data = self.read(written, "cbnm93#he_be2ls.dat")
        self.assertIsNone(data["parents"][0]["spin_systems"][0]["parent_qc"])
        self.check(written, data)

    def test_import_reads_no_file(self):
        # Importing the module opened '../cbnm93#h_h0ls.dat' and raised FileNotFoundError.
        cwd = os.getcwd()
        os.chdir(self.tmp.name)  # a working directory without that file
        try:
            sys.modules.pop("colradpy.read_adf17", None)
            module = importlib.import_module("colradpy.read_adf17")
        finally:
            os.chdir(cwd)
        self.assertTrue(callable(module.read_adf17))

    def test_errors(self):
        from colradpy.read_adf17 import read_adf17
        with self.assertRaises(FileNotFoundError):
            read_adf17(os.path.join(self.tmp.name, "missing.dat"))
        path = os.path.join(self.tmp.name, "bnl99pm#he_be4.dat")
        with open(path, "w") as f:  # the bundled-n beam files of ADAS311 are another format
            f.write("      /NRES=  29 /MAXEB=  25 /MAXTE=  20 /MAXNE=  25 /DATE=30/08/99 /CODE=ADAS311\n")
        with self.assertRaisesRegex(ValueError, "not an adf17 file of condensed projection matrices"):
            read_adf17(path)


if __name__ == "__main__":
    unittest.main()

"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks that reduced ionization rates that cannot be physical, placeholders some adf04 files
hold where the rate underflowed at their lowest temperatures, are left out of the rates.
"""

import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from colradpy import colradpy
from colradpy.colradpy_class import MAX_RECOMBINATION, MAX_REDUCED_IONIZATION
from colradpy.ionization_balance_class import ionization_balance


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"


def rates(path, temperatures, metas=(0,)):
    model = colradpy(str(path), metas=np.array(metas), temp_grid=np.asarray(temperatures),
                     electron_den=np.array([1e13]))
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        model.solve_cr()
    return model


class PlaceholderTests(unittest.TestCase):
    def test_bundled_placeholders(self):
        # cpb03_ls#be1.dat gives 1.0, 1.0 and 0.0277 cm3/s for the ground to the second parent
        # at its three lowest temperatures, then 1e-10.
        temperatures = np.array([0.1, 0.5, 1., 2., 5., 20.])
        with self.assertWarnsRegex(UserWarning, r"level 1 to parent 2 at 0\.172, 0\.431, 0\.862 eV"):
            model = rates(INPUT / "cpb03_ls#be1.dat", temperatures, metas=(0, 1))
        ioniz = model.data["rates"]["ioniz"]
        file_temps = model.data["input_file"]["temp_grid"] / 11604.5
        k = int(np.where((ioniz["ion_transitions"] == [1, 2]).all(axis=1))[0][0])
        values = ioniz["ion_excit"][k]
        kept = values <= MAX_REDUCED_IONIZATION
        np.testing.assert_array_equal(kept[:3], False)
        reduced = ioniz["ionization_reduced"][0, 1]
        self.assertTrue((reduced <= MAX_REDUCED_IONIZATION).all())
        # below the lowest kept temperature, a power law through the two lowest kept points
        t0, t1 = file_temps[kept][:2]
        r0, r1 = values[kept][:2]
        below = temperatures < t0
        np.testing.assert_allclose(reduced[below], r0 * (temperatures[below] / t0)**(np.log(r1 / r0) / np.log(t1 / t0)),
                                   rtol=1e-12)
        # the recombination from the second parent stays physical (it was 2.7e-9 cm3/s at 0.5 eV)
        self.assertLess(model.data["processed"]["acd"][:, 1, :, 0].sum(axis=0).max(), 1e-9)

    def test_recombination_placeholder(self):
        # OpenADAS cop98#10_ls#ne1 gives 3.25e-5 cm3/s for 'R  2  +2' at its lowest temperature,
        # then 1e-30: recombination ten million times too fast below 0.4 eV.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        lines = (INPUT / "cpb03_ls#be0.dat").read_text().splitlines()
        k = next(i for i, line in enumerate(lines) if line.startswith("R  2  +1"))
        fields = lines[k].split()
        original = np.array([float(v.replace("-", "e-").replace("+", "e+")) for v in fields[3:]])
        fields[3] = "3.25-05"
        lines[k] = "R  2  +1         " + " ".join(fields[3:])
        path = os.path.join(tmp.name, "be0_recombination_placeholder.dat")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        temperatures = np.array([0.05, 0.1, 1., 10.])
        with self.assertWarnsRegex(UserWarning, r"recombination rates above 1e-07 .* parent 1 to level 2 at 0\.0862 eV"):
            model = rates(path, temperatures)
        reference = rates(INPUT / "cpb03_ls#be0.dat", temperatures)
        file_temps = model.data["input_file"]["temp_grid"] / 11604.5
        recombination = model.data["rates"]["recomb"]["recombination"][1, 0]
        self.assertTrue((recombination <= MAX_RECOMBINATION).all())
        # at and above the second file temperature the rates are those of the original file
        above = temperatures >= file_temps[1]
        np.testing.assert_allclose(recombination[above],
                                   reference.data["rates"]["recomb"]["recombination"][1, 0][above], rtol=1e-12)
        self.assertLess(recombination.max(), 10 * original.max())

    def test_ionization_balance_with_placeholders(self):
        # OpenADAS cop98#10_ls#ne7 holds 2.24e30 at its two lowest temperatures; through three-body
        # recombination this gave ACD of 1e20 cm3/s and steady states off by up to 500%.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        lines = (INPUT / "cpb03_ls#be0.dat").read_text().splitlines()
        k = next(i for i, line in enumerate(lines) if line.startswith("S  1  +1"))
        fields = lines[k].split()
        fields[3:5] = ["2.24+30", "2.24+30"]
        lines[k] = "S  1  +1         " + " ".join(fields[3:])
        path = os.path.join(tmp.name, "be0_placeholder.dat")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            balance = ionization_balance([path, str(INPUT / "cpb03_ls#be1.dat")], np.array([0]),
                                         np.array([0.2, 1., 5.]), np.array([1e13]))
            balance.populate_ion_matrix()
            balance.solve_time_independent()
        for stage in balance.data["cr_data"]["gcrs"].values():
            self.assertLess(np.abs(stage["acd"]).max(), 1e-8)
        pops = balance.data["processed"]["pops_ss"]
        np.testing.assert_allclose(pops.sum(axis=0), 1, rtol=1e-12)
        self.assertGreaterEqual(pops.min(), -1e-15)


if __name__ == "__main__":
    unittest.main()

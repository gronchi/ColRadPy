"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks how the recombination and reduced ionization rates of an adf04 file are interpolated on
the user temperatures (log of the rates on a log temperature axis by default) and continued
outside the file temperatures, and that artifacts at the lowest temperatures are left out.
"""

import contextlib
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from colradpy import colradpy
from colradpy.colradpy_class import KELVIN_PER_EV
from colradpy.colradpy_class import (RECOMBINATION_EXPONENTS, REDUCED_IONIZATION_EXPONENTS_ABOVE,
                                     STEEPEST_RECOMBINATION, STEEPEST_REDUCED_IONIZATION,
                                     low_temperature_artifacts)


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"
FILE = INPUT / "cpb03_ls#be0.dat"


def adf04_number(value):
    """A value as adf04 files write it, 2.56-08."""
    exponent = math.floor(math.log10(value))
    mantissa = round(value / 10**exponent, 2)
    if mantissa >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    return f"{mantissa:.2f}{exponent:+03d}".rjust(8)


def rates(path, temperatures, **kwargs):
    model = colradpy(str(path), metas=np.array([0]), temp_grid=np.asarray(temperatures),
                     electron_den=np.array([1e13]), **kwargs)
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        model.make_ioniz_from_reduced_ionizrates()
        model.make_recombination_rates_from_file()
    return model


def file_temperatures():
    return rates(FILE, [1.]).data["input_file"]["temp_grid"] / KELVIN_PER_EV


class RateInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.temps = file_temperatures()

    def edited(self, prefix, values):
        """The bundled Be I file with the line starting with prefix (R or S) given values."""
        lines = FILE.read_text().splitlines()
        k = next(i for i, line in enumerate(lines) if line.startswith(prefix))
        lines[k] = prefix + " " * 8 + "".join(adf04_number(v) for v in values)
        path = os.path.join(self.tmp.name, "be0.dat")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def test_between_file_temperatures(self):
        # Linear interpolation of the rates gives the arithmetic mean of two neighbouring values
        # at their geometric mean temperature, which overestimates rates that change by orders of
        # magnitude; the log-log interpolation gives their geometric mean.
        values = np.array([2e-12, 1e-14, 9e-15, 8e-15, 7e-15, 6e-15, 5e-15, 4e-15, 3e-15, 2e-15, 1e-15])
        middle = np.sqrt(self.temps[:-1] * self.temps[1:])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # the first value falls faster than any rate
            path = self.edited("R  1  +1", values)
            written = np.array([float(adf04_number(v).replace("-", "e-")) for v in values])
            loglog = rates(path, middle).data["rates"]["recomb"]["recombination"][0, 0]
            linear = rates(path, middle, rate_interp_recomb="log_slinear").data["rates"]["recomb"]["recombination"][0, 0]
        np.testing.assert_allclose(loglog[1:], np.sqrt(written[1:-1] * written[2:]), rtol=1e-12)
        np.testing.assert_allclose(linear[1:], (written[1:-1] + written[2:]) / 2, rtol=1e-12)

    def test_power_laws_outside_file_temperatures(self):
        # Outside the file temperatures linear extrapolation crosses zero: the Be I recombination
        # fell to 1e-21 cm3/s and the ionization of the ground to 2e-12 cm3/s at 1 keV.
        below = self.temps[0] * np.array([0.1, 0.5])
        above = self.temps[-1] * np.array([2., 10.])
        model = rates(FILE, np.concatenate([below, self.temps, above]))
        recombination = model.data["rates"]["recomb"]["recombination"][0, 0]
        reduced = model.data["rates"]["ioniz"]["ionization_reduced"][0, 0]
        r = model.data["rates"]["recomb"]["recomb_excit"][0]
        s = model.data["rates"]["ioniz"]["ion_excit"][0]
        t = self.temps

        def power(values, a, b, limits):
            return np.clip(np.log(values[b] / values[a]) / np.log(t[b] / t[a]), *limits)

        np.testing.assert_allclose(recombination[2:-2], r, rtol=1e-12)
        np.testing.assert_allclose(recombination[-2:], r[-1] * (above / t[-1])**power(r, -2, -1, RECOMBINATION_EXPONENTS), rtol=1e-12)
        np.testing.assert_allclose(recombination[:2], r[0] * (below / t[0])**power(r, 0, 1, RECOMBINATION_EXPONENTS), rtol=1e-12)
        np.testing.assert_allclose(reduced[-2:], s[-1] * (above / t[-1])**power(s, -2, -1, REDUCED_IONIZATION_EXPONENTS_ABOVE), rtol=1e-12)
        self.assertGreater(reduced[-1], 0.3 * s[-1])

    def test_exponents_are_limited(self):
        # A recombination rate falling as T^-3 at the highest temperatures is continued as T^-1.5,
        # and one rising at the lowest is held constant below them.
        values = 1e-13 * (self.temps / self.temps[0])**0.5
        values[-1] = values[-2] * (self.temps[-1] / self.temps[-2])**-3.
        written = np.array([float(adf04_number(v).replace("-", "e-")) for v in values])
        model = rates(self.edited("R  1  +1", values), [self.temps[0] / 10, self.temps[-1] * 10])
        recombination = model.data["rates"]["recomb"]["recombination"][0, 0]
        self.assertAlmostEqual(recombination[0] / written[0], 1., places=12)
        self.assertAlmostEqual(recombination[1] / written[-1], 10.**RECOMBINATION_EXPONENTS[0], places=12)

    def test_low_temperature_artifacts_left_out(self):
        # OpenADAS cop98#10_ls#ne1 gives 'R  2  +3' 2.56e-8 cm3/s at its lowest temperature, then
        # 1e-30: below the 1e-7 bound of placeholders, it was taken as a rate (and extrapolated to
        # 7e-8 cm3/s at 0.05 eV)
        values = list(FILE.read_text().split("R  3  +1")[1].split("\n")[0].split())
        values = [2.56e-8] + [float(v.replace("-", "e-").replace("+", "e+")) for v in values[1:]]
        path = self.edited("R  3  +1", values)
        with self.assertWarnsRegex(UserWarning, r"recombination rates at the lowest temperatures that fall faster "
                                                r"than T\^-2 are not physical .* parent 1 to level 3 at 0\.0862 eV"):
            model = rates(path, self.temps)
        recombination = model.data["rates"]["recomb"]["recombination"][2, 0]
        written = model.data["rates"]["recomb"]["recomb_excit"][2]
        np.testing.assert_allclose(recombination[1:], written[1:], rtol=1e-12)
        # the next two points rise with temperature, so the rate is held constant below them
        self.assertAlmostEqual(recombination[0] / written[1], 1., places=12)

    def test_artifact_rule(self):
        t = np.array([0.17, 0.34, 0.69, 1.7, 3.4, 6.9, 17.])
        physical = np.array([t**-1.5, t**0.5, t**0, np.exp(-2 / t) * t**-1.5, 1e-13 / (1 + (t / 3)**2)])
        np.testing.assert_array_equal(low_temperature_artifacts(t, physical, STEEPEST_RECOMBINATION), False)
        rows = np.array([[2.56e-8, 1e-30, 5.4e-46, 7.9e-29, 2.3e-23, 7.6e-21, 1.1e-19],   # cop98#10_ls#ne1
                         [1.51e-7, 7.04e-8, 8.50e-9, 5.10e-9, 4.89e-9, 5.29e-9, 6.35e-9],  # cop98#10_ls#ne3
                         [1e-13, 1e-11, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13],               # not at the lowest
                         [0., 3e-12, 2e-12, 1.8e-12, 1.5e-12, 1.2e-12, 1e-12]])            # zeros not compared
        artifacts = low_temperature_artifacts(t, rows, STEEPEST_REDUCED_IONIZATION)
        np.testing.assert_array_equal(artifacts.sum(axis=1), [2, 2, 0, 0])
        self.assertTrue(artifacts[:2, :2].all())

    def test_options(self):
        with self.assertRaisesRegex(ValueError, "log-log interpolation is available for ionization and recombination"):
            rates(FILE, [1.], rate_interp_col="loglog_slinear")
        # the other options interpolate the rates themselves, as before
        middle = np.sqrt(self.temps[0] * self.temps[1])
        s = rates(FILE, [middle], rate_interp_ion="slinear").data["rates"]["ioniz"]
        k = 0
        values, t = s["ion_excit"][k], self.temps
        expected = values[0] + (values[1] - values[0]) * (middle - t[0]) / (t[1] - t[0])
        self.assertAlmostEqual(s["ionization_reduced"][0, 0, 0] / expected, 1., places=12)


if __name__ == "__main__":
    unittest.main()

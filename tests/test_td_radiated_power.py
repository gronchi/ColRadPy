"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks that time-dependent line power is the photon energy times the PEC of the
same line, which requires both to use the population of the line's upper level.
"""

from pathlib import Path
import unittest

import numpy as np
from scipy import constants

from colradpy import colradpy


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"
# Photon energy per cm^-1 of transition energy, in J.
J_PER_WAVENUMBER = constants.h * constants.c * 100


def solve_be0(metas, upper):
    """Be I evolving from all of the population in level `upper`."""
    densities = np.array([1e12, 1e13, 1e14])
    model = colradpy(
        str(INPUT / "cpb03_ls#be0.dat"),
        metas=np.array(metas),
        temp_grid=np.array([10.0, 20.0]),
        electron_den=densities,
        use_ionization=False,
        use_recombination=False,
        use_recombination_three_body=False,
        td_t=np.array([0.0, 1e-8, 1e-6, 1e-4]),
    )
    # Degenerate levels in bundled data produce a pre-existing zero-energy
    # wavelength warning, outside this check's scope.
    with np.errstate(divide="ignore"):
        model.solve_cr()
    n0 = np.zeros(len(model.data["cr_matrix"]["cr"]))
    n0[upper] = 1.0
    model.data["user"]["td_n0"] = n0
    model.solve_time_dependent()
    return model, densities


class TimeDependentRadiatedPowerTests(unittest.TestCase):
    def test_line_power_is_photon_energy_times_pec(self):
        for metas in ([0], [0, 2]):
            with self.subTest(metas=metas):
                model, _ = solve_be0(metas, upper=3)
                td = model.data["processed"]["td"]
                # The time-dependent lines are built in the same order as the
                # quasistatic ones, which record their (upper, lower) levels.
                levels = np.asarray(model.data["processed"]["pec_levels"])
                energy = model.data["atomic"]["energy"]
                photon = np.abs(energy[levels[:, 0]] - energy[levels[:, 1]])
                photon = photon * J_PER_WAVENUMBER
                # PEC is in cm^3 s^-1 and power in W m^3.
                expected = photon[:, None, None, None] * td["pecs"] / 1e6
                np.testing.assert_allclose(td["pls"], expected, rtol=1e-4)
                np.testing.assert_allclose(td["plt"], td["pls"].sum(axis=0))

    def test_initially_populated_line_at_time_zero(self):
        # At t=0 only level 2 is populated, so the 2->0 power is
        # photon energy * A / n_e and a line from unpopulated level 3 has
        # no power beyond round-off.
        model, densities = solve_be0([0], upper=2)
        td = model.data["processed"]["td"]
        levels = [tuple(pair) for pair in model.data["processed"]["pec_levels"]]
        energy = model.data["atomic"]["energy"]
        a_value = model.data["cr_matrix"]["A_ji"][2, 0]
        expected = (abs(energy[2] - energy[0]) * J_PER_WAVENUMBER
                    * a_value / densities / 1e6)
        power = td["pls"][levels.index((2, 0))][0]
        for temperature in range(power.shape[0]):
            np.testing.assert_allclose(power[temperature], expected, rtol=1e-4)
        unpopulated = np.abs(td["pls"][levels.index((3, 0))][0]).max()
        self.assertLess(unpopulated, 1e-10 * expected.min())


if __name__ == "__main__":
    unittest.main()

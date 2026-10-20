"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks parent statistical weights (ADF04 specification, ADAS appxa-04) and
three-body recombination from detailed balance with ionization.
"""

import contextlib
import io
from pathlib import Path
import os
import tempfile
import unittest

import numpy as np
from scipy import constants

from colradpy import colradpy
from colradpy.colradpy_class import L_LETTERS, parent_statistical_weight
from colradpy.read_adf04_py3_class import read_adf04
from colradpy.r8yip import r8ecip


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"
# h^3 / (2 (2 pi m_e)^(3/2)) in cm^3 eV^(3/2), the Saha factor without T^(-3/2).
SAHA = (constants.h / np.sqrt(2 * np.pi * constants.m_e * constants.e))**3 / 2 * 1e6


def solve(name, temperatures, **kwargs):
    model = colradpy(str(INPUT / name), metas=np.array([0]), temp_grid=np.array(temperatures),
                     electron_den=np.array([1e13]), **kwargs)
    # Degenerate levels in bundled data produce a pre-existing zero-energy
    # wavelength warning; extreme temperatures overflow the ionization factor.
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        model.solve_cr()
    return model


class ParentWeightTests(unittest.TestCase):
    def test_orbital_letters(self):
        self.assertEqual(L_LETTERS[:9], "SPDFGHIKL")
        self.assertNotIn("J", L_LETTERS)

    def test_ls_parents(self):
        for term, weight in (("1S", 1), ("2S", 2), ("3P", 9), ("4F", 28), ("1I", 13),
                             ("2K", 30), ("2L", 34), ("3p", 9)):
            with self.subTest(term=term):
                self.assertEqual(parent_statistical_weight(term), weight)

    def test_ic_parents(self):
        for term, weight in (("3P0", 1), ("3P1", 3), ("3P2", 5), ("3P2.0", 5),
                             ("2P0.5", 2), ("2P1.5", 4), ("2P3/2", 4), ("6D4.5", 10)):
            with self.subTest(term=term):
                self.assertEqual(parent_statistical_weight(term), weight)

    def test_invalid_parents(self):
        for term in ("", "P", "2J", "2X", "2P*", "2P1.25"):
            with self.subTest(term=term), self.assertRaises(ValueError):
                parent_statistical_weight(term)

    def test_parent_without_term_is_1s(self):
        with open(INPUT / "cpb03_ls#be0.dat") as f:
            lines = f.readlines()
        self.assertIn("75190.0(2S)", lines[0])
        lines[0] = lines[0].replace("75190.0(2S)", "75190.0    ")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "no_term.dat")
            with open(path, "w") as f:
                f.writelines(lines)
            with contextlib.redirect_stdout(io.StringIO()):
                atomic = read_adf04(path)["atomic"]
        self.assertEqual(list(atomic["ion_term"]), ["1S"])
        self.assertEqual(atomic["ion_pot"][0], 75190.0)


class ThreeBodyTests(unittest.TestCase):
    def check_detailed_balance(self, model):
        """Three-body recombination / ionization is the Saha ratio of the level to its parent."""
        atomic, rates = model.data["atomic"], model.data["rates"]
        temps = model.data["user"]["temp_grid"]
        weights = 2 * atomic["w"] + 1
        transitions = rates["ioniz"]["ion_transitions"]
        for p, term in enumerate(atomic["ion_term"]):
            expected = SAHA * weights[:, None] / parent_statistical_weight(term) \
                * rates["ioniz"]["ionization_reduced"][:, p] / temps**1.5
            np.testing.assert_allclose(rates["recomb"]["recomb_three_body"][:, p], expected,
                                       rtol=3e-4)
            # The reduced rate is exp(dE/kT) S, with the exponent of the file
            # reduced rates or, for the other levels, of r8ecip.
            ev_per_wavenumber = np.full(len(atomic["energy"]), 9.11269E-06 * 157890.0 / 11604.5)
            ev_per_wavenumber[transitions[transitions[:, 1] == p + 1, 0] - 1] = 0.00012398774011749576
            x = ((atomic["ion_pot"][p] - atomic["energy"]) * ev_per_wavenumber)[:, None] / temps
            ionization = rates["ioniz"]["ionization"][:, p]
            usable = (ionization > 0) & (x < 700)
            self.assertTrue(usable.any())
            with np.errstate(all="ignore"):
                ratio = rates["recomb"]["recomb_three_body"][:, p] / ionization
                saha = SAHA * weights[:, None] / parent_statistical_weight(term) * np.exp(x) / temps**1.5
            np.testing.assert_allclose(ratio[usable], saha[usable], rtol=3e-4)

    def test_detailed_balance_of_bundled_files(self):
        for name in ("cpb03_ls#be0.dat", "cpb03_ls#be1.dat", "be2_adf04"):
            with self.subTest(file=name):
                self.check_detailed_balance(solve(name, [2., 10., 50.]))

    def test_parent_weight_scales_three_body(self):
        model = solve("cpb03_ls#be0.dat", [2., 10.])
        ls = model.data["rates"]["recomb"]["recomb_three_body"].copy()
        for term, weight in (("2P1.5", 4), ("3P0", 1)):
            with self.subTest(term=term):
                model.data["atomic"]["ion_term"] = np.array([term])
                model.make_three_body_recombination()
                np.testing.assert_allclose(model.data["rates"]["recomb"]["recomb_three_body"],
                                           ls * 2 / weight)

    def test_low_temperature_is_not_zero(self):
        # dE/kT is 1500 to 15000 for the Be2+ ground state, where exp(dE/kT) * S
        # was infinity times zero.
        for ecip in (False, True):
            with self.subTest(suppliment_with_ecip=ecip):
                model = solve("be2_adf04", [0.01, 0.1, 1.], suppliment_with_ecip=ecip)
                three_body = model.data["rates"]["recomb"]["recomb_three_body"][0, 0]
                self.assertTrue((three_body > 0).all() and np.isfinite(three_body).all())
                self.check_detailed_balance(model)

    def test_ecip_only_levels_at_low_temperature(self):
        # ECIP rates are zero above dE/kT = 180; three-body recombination into levels
        # whose ionization comes only from ECIP uses the reduced rates, which are not.
        model = solve("be2_adf04", [0.01, 0.1, 1.])
        ioniz = model.data["rates"]["ioniz"]
        exponent = model.ecip_exponent()
        file_levels = ioniz["ion_transitions"][ioniz["ion_transitions"][:, 1] == 1, 0] - 1
        ecip_only = np.setdiff1d(np.where(ioniz["ecip_reduced"][:, 0, 0] > 0)[0], file_levels)
        self.assertGreater(len(ecip_only), 0)
        cut = exponent[ecip_only, 0] > 180
        self.assertTrue(cut.any())
        np.testing.assert_array_equal(ioniz["ionization"][ecip_only, 0][cut], 0.0)
        three_body = model.data["rates"]["recomb"]["recomb_three_body"][ecip_only, 0]
        self.assertTrue((three_body > 0).all())
        np.testing.assert_array_equal(ioniz["ionization_reduced"][ecip_only, 0],
                                      ioniz["ecip_reduced"][ecip_only, 0])
        self.check_detailed_balance(model)

    def test_ecip_rates_from_reduced_rates(self):
        model = solve("cpb03_ls#be1.dat", [0.01, 0.1, 1., 10.])
        ioniz = model.data["rates"]["ioniz"]
        exponent = model.ecip_exponent()
        made = ioniz["ecip_reduced"] > 0
        kept = made & (exponent <= 180)
        np.testing.assert_allclose(ioniz["ecip"][kept],
                                   (ioniz["ecip_reduced"] * np.exp(-exponent))[kept], rtol=1e-12)
        np.testing.assert_array_equal(ioniz["ecip"][made & (exponent > 180)], 0.0)
        # The same as r8ecip without the reduced option.
        atomic = model.data["atomic"]
        level = np.where(made[:, 0, 0])[0][0]
        rates = r8ecip(atomic["charge_state"], atomic["ion_pot"][0], atomic["energy"][[level]],
                       atomic["zpla"][[level], 0], model.data["user"]["temp_grid"] * 11604.5)
        np.testing.assert_allclose(ioniz["ecip"][level, 0], rates[0], rtol=1e-12)

    def test_file_ionization_kept_below_its_grid(self):
        # The lowest temperature is below the file grid, where the ground-state
        # rate underflows. ECIP must not replace the file rates at every temperature.
        kwargs = dict(rate_interp_ion="log_slinear")
        with_ecip = solve("cpb03_ls#be0.dat", [0.01, 0.1, 1.], suppliment_with_ecip=True, **kwargs)
        file_only = solve("cpb03_ls#be0.dat", [0.01, 0.1, 1.], suppliment_with_ecip=False, **kwargs)
        np.testing.assert_array_equal(with_ecip.data["rates"]["ioniz"]["ionization"][0],
                                      file_only.data["rates"]["ioniz"]["ionization"][0])
        np.testing.assert_array_equal(with_ecip.data["rates"]["recomb"]["recomb_three_body"][0],
                                      file_only.data["rates"]["recomb"]["recomb_three_body"][0])


if __name__ == "__main__":
    unittest.main()

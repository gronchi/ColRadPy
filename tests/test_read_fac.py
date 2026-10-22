"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the reader of the Flexible Atomic Code's ASCII tables (colradpy.read_fac) and the averages of
cross-sections over a thermal distribution (colradpy.convolve_eedf), with files written here in the
formats of FAC's PrintTable (dbase.c) for a toy He-like ion and its H-like parents: their layout,
the rates against exact Maxwellian averages, the folding of autoionization into the other rates
against a model with explicit autoionization, and the model ColRadPy makes from them.
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
from scipy import constants
from scipy.special import exp1

from colradpy import colradpy
from colradpy.convolve_eedf import maxwellian, rate_coefficients, speed
from colradpy.read_fac import (ALPHA, BOHR_AREA, HARTREE_EV, RYDBERG_EV, UPSILON_RATE, FacCrossSection, fac_cross_sections,
                               read_fac, read_fac_ce, read_fac_ci, read_fac_en, read_fac_rr, two_photon_rate)

#atomic unit of rate (s-1) and FAC's area unit (a0^2 in 1e-20 cm2)
RATE_AU = 4.13413733E16
AREA_AU20 = BOHR_AREA/1e-20

#levels of the toy ion (FAC index, electrons, energy in eV, 2J, sname, name), out of energy order
LEVELS = [(0, 2, 0., 0, '1s2', '1s+2(0)0'),
          (1, 2, 12., 2, '1s1.2p1', '1s+1(1)1.2p+1(3)1'),
          (2, 2, 10., 2, '1s1.2s1', '1s+1(1)1.2s+1(1)1'),
          (3, 2, 11., 0, '1s1.2s1', '1s+1(1)1.2s+1(1)0'),
          (4, 2, 60., 0, '2s2', '2s+2(0)0'),
          (8, 2, 70., 0, '2s1.3s1', '2s+1(1)1.3s+1(1)0'),
          (5, 1, 55., 1, '2s1', '2s+1(1)1'),
          (6, 1, 50., 1, '1s1', '1s+1(1)1'),
          (7, 3, -5., 1, '1s2.2s1', '1s+2(0)0.2s+1(1)1')]
TEMPS = np.array([2., 5., 10., 20., 50., 100.])


def header(table_type, blocks, z=6., symbol='C'):
    return (f"FAC 1.1.5\nEndian\t= 0\nTSess\t= 1700000000\nType\t= {table_type}\nVerbose\t= 1\n"
            f"{symbol} Z\t= {z:5.1f}\nNBlocks\t= {blocks}\n")


def grid(key, values):
    return f"{key}\t= {len(values)}\n" + "".join(f"\t {v:15.8E}\n" for v in values)


def write_en(path, levels=LEVELS, verbose=1):
    text = header(1, len({l[1] for l in levels})).replace("Verbose\t= 1", f"Verbose\t= {verbose}")
    text += "E0\t= 7, -5.00000000E+00\n"
    for nele in sorted({l[1] for l in levels}, reverse=True):
        rows = [l for l in levels if l[1] == nele]
        text += f"\nNELE\t= {nele}\nNLEV\t= {len(rows)}\n  ILEV  IBASE    ENERGY       P   VNL         2J\n"
        for index, _, energy, two_j, sname, name in rows:
            text += f"{index:6d} {-1:6d} {energy + 5:15.8E} {0:1d} {201:5d} {two_j:10d} {'1*2':<32s} {sname:<48s} {name}\n"
    with open(path, "w") as f:
        f.write(text)


def two_j(index):
    return next(l[3] for l in LEVELS if l[0] == index)


def energy(index):
    return next(l[2] for l in LEVELS if l[0] == index)


def write_tr(path, transitions):
    """transitions: {multipole: [(upper, lower, rate)]}"""
    text = header(2, len(transitions))
    for multipole, rows in transitions.items():
        text += f"\nNELE\t= 2\nNTRANS\t= {len(rows)}\nMULTIP\t= {multipole}\nGAUGE\t= 2\nMODE\t= 1\n"
        for upper, lower, rate in rows:
            text += (f"{upper:6d} {two_j(upper):10d} {lower:6d} {two_j(lower):10d} {energy(upper) - energy(lower):13.6E} "
                     f"{0.1:13.6E} {rate:13.6E} {0.5:13.6E}\n")
    with open(path, "w") as f:
        f.write(text)


def write_ce(path, transitions, utype=1, qkmode=1):
    """transitions: [(lower, upper, scattered energies, collision strengths, bethe, born, born energy)]"""
    usr = transitions[0][2]
    text = header(3, 1) + f"\nNELE\t= 2\nNTRANS\t= {len(transitions)}\nQKMODE\t= {qkmode}\nNPARAMS\t= 0\nMSUB\t= 0\nPWTYPE\t= 0\n"
    text += grid("NTEGRID", [10.]) + "TE0\t= 1.00000000E+01\nETYPE\t= 1\n" + grid("NEGRID", usr)
    text += f"UTYPE\t= {utype}\n" + grid("NUSR", usr)
    for lower, upper, scattered, omega, bethe, born, born_energy in transitions:
        de = energy(upper) - energy(lower)
        text += f"{lower:6d} {two_j(lower):10d} {upper:6d} {two_j(upper):10d} {de:11.4E} 1\n"
        text += f"{bethe:11.4E} {born:11.4E} {born_energy:11.4E}\n"
        if qkmode == 2:
            text += "1.0000E+00 1.0000E+00 1.0000E+00 1.0000E+00 \n"
        for e, o in zip(scattered, omega):
            incident = e + de
            k2 = 2 * incident / HARTREE_EV * (1 + 0.5 * ALPHA**2 * incident / HARTREE_EV)
            sigma = np.pi * AREA_AU20 / k2 / (two_j(lower) + 1) * o
            text += f"{e + (de if utype == 0 else 0.):11.4E} {o:11.4E} {sigma:11.4E}\n"
    with open(path, "w") as f:
        f.write(text)


def write_ci(path, transitions):
    """transitions: [(bound, free, ejected energies, collision strengths, params)]"""
    usr = transitions[0][2]
    text = header(6, 1) + f"\nNELE\t= 2\nNTRANS\t= {len(transitions)}\nQKMODE\t= 5\nNPARAMS\t= 4\nPWTYPE\t= 0\n"
    text += grid("NTEGRID", [40.]) + "ETYPE\t= 1\n" + grid("NEGRID", usr) + "UTYPE\t= 1\n" + grid("NUSR", usr)
    for bound, free, ejected, omega, params in transitions:
        de = energy(free) - energy(bound)
        text += f"{bound:6d} {two_j(bound):10d} {free:6d} {two_j(free):10d} {de:11.4E} {0:2d}\n"
        text += "".join(f"{p:11.4E} " for p in params) + "\n"
        for e, o in zip(ejected, omega):
            incident = e + de
            a = incident / HARTREE_EV * (1 + 0.5 * ALPHA**2 * incident / HARTREE_EV)
            text += f"{e:11.4E} {o:11.4E} {AREA_AU20 / (2 * a * (two_j(bound) + 1)) * o:11.4E}\n"
    with open(path, "w") as f:
        f.write(text)


def write_rr(path, transitions, qkmode=2):
    """transitions: [(bound, free, n*1000 + l, photoelectron energies, strengths, params)]"""
    usr = transitions[0][3]
    text = header(4, 1) + f"\nNELE\t= 2\nNTRANS\t= {len(transitions)}\nQKMODE\t= {qkmode}\nMULTIP\t= -1\nNPARAMS\t= 4\n"
    text += grid("NTEGRID", [40.]) + "ETYPE\t= 1\n" + grid("NEGRID", usr) + "UTYPE\t= 1\n" + grid("NUSR", usr)
    for bound, free, kl, photoelectron, strength, params in transitions:
        de = energy(free) - energy(bound)
        text += f"{bound:6d} {two_j(bound):10d} {free:6d} {two_j(free):10d} {de:11.4E} {kl:6d}\n"
        if qkmode == 2:
            text += "".join(f"{p:11.4E} " for p in params) + "\n"
        for e, s in zip(photoelectron, strength):
            ee, eph = e / HARTREE_EV, (e + de) / HARTREE_EV
            phi = 2 * np.pi * ALPHA * s * AREA_AU20
            rr = phi * (ALPHA * eph)**2 / (2 * ee) / (1 + 0.5 * ALPHA**2 * ee) / (two_j(free) + 1)
            text += f"{e:11.4E} {rr:11.4E} {phi / (two_j(bound) + 1):11.4E} {s:11.4E}\n"
    with open(path, "w") as f:
        f.write(text)


def dc_strength(bound, free, rate):
    """FAC's dielectronic capture strength (1e-20 cm2 eV) of an autoionization rate (s-1)."""
    er = (energy(bound) - energy(free)) / HARTREE_EV
    return 0.5 * (two_j(bound) + 1) * np.pi**2 * (rate / RATE_AU) / (er * (two_j(free) + 1)) * AREA_AU20 * HARTREE_EV


def write_ai(path, transitions):
    """transitions: [(bound, free, rate)]"""
    text = header(5, 1) + f"\nNELE\t= 2\nNTRANS\t= {len(transitions)}\nEMIN\t= 0.00000000E+00\n" + grid("NEGRID", [0.])
    for bound, free, rate in transitions:
        text += (f"{bound:6d} {two_j(bound):10d} {free:6d} {two_j(free):10d} {energy(bound) - energy(free):11.4E} "
                 f"{rate:11.4E} {dc_strength(bound, free, rate):11.4E}\n")
    with open(path, "w") as f:
        f.write(text)


SCATTERED = np.geomspace(1e-3, 30., 60)              #in units of the threshold


def ci_fit(x, p):
    y = 1 - 1 / x
    return p[0] * np.log(x) + p[1] * y**2 + p[2] * y / x + p[3] * y / x**2


CI_PARAMS = [0.8, -0.3, 0.2, 0.1]


def write_ion(directory, prefix="C02", ce=True, rr=True, ci=True, ai=True):
    """The FAC files of the toy ion, with exact forms: a constant collision strength (2 -> 3, i.e.
    1s2s 3S1 -> 1s2s 1S0), a Bethe form (0 -> 1), exchange (2 -> 1), FAC's ionization fit, a constant
    bound-free strength, and two autoionizing levels above the parents (4 and 8, which decays to 4),
    excited from the ground."""
    fil = os.path.join(directory, prefix)
    write_en(fil + "a.en")
    write_tr(fil + "a.tr", {-1: [(1, 0, 1e9), (4, 1, 1e10), (8, 4, 5e9), (8, 1, 2e9)], 1: [(2, 0, 1e3)],
                            2: [(1, 0, 1e2)]})
    if ce:
        de = {(2, 3): 1., (0, 1): 12., (2, 1): 2.}
        omega = {(2, 3): np.full(60, 0.1), (0, 1): 0.2 * np.log(1 + SCATTERED) + 0.3,
                 (2, 1): 0.05 * (1 + SCATTERED)**-2.}
        write_ce(fil + "a.ce", [(2, 3, SCATTERED * de[(2, 3)], omega[(2, 3)], 0., 0.1, 100.),
                                (0, 1, SCATTERED * de[(0, 1)], omega[(0, 1)], 0.2, 0.3, 40. * 12.),
                                (2, 1, SCATTERED * de[(2, 1)], omega[(2, 1)], -1., 0., 0.),
                                (0, 4, SCATTERED * 60., np.full(60, 0.02), 0., 0.02, 6000.),
                                (0, 8, SCATTERED * 70., np.full(60, 0.01), 0., 0.01, 7000.)])
    if ci:
        ejected = np.geomspace(1e-2, 30., 40)
        write_ci(fil + "a.ci", [(b, 6, ejected * (50. - energy(b)), ci_fit(1 + ejected, CI_PARAMS), CI_PARAMS)
                                for b in (0, 2, 3, 1)])
    if rr:
        photoelectron = np.geomspace(1e-2, 30., 40) * 50.
        write_rr(fil + "a.rr", [(0, 6, 1000, photoelectron, np.full(40, 0.02), [0., 0., 0., 0.]),
                                (2, 6, 2000, photoelectron, np.full(40, 0.01), [0., 0., 0., 0.])], qkmode=1)
    if ai:
        write_ai(fil + "a.ai", [(4, 6, 1e13), (4, 5, 2e12), (8, 6, 3e12)])
    return fil


#stabilizing fractions and probabilities of reaching each parent of the autoionizing levels 4 and 8
B4 = 1e10 / (1e10 + 1.2e13)
P4 = np.array([1e13, 2e12]) / (1e10 + 1.2e13)
B8 = (2e9 + 5e9 * B4) / (7e9 + 3e12)
P8 = (np.array([3e12, 0.]) + 5e9 * P4) / (7e9 + 3e12)


def model(fil, temps=TEMPS, dens=np.array([1e8, 1e12]), fold_autoionization=True, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = colradpy(read_fac(fil, 2, temps, fold_autoionization=fold_autoionization), np.array([0]), temps,
                     dens, **kwargs)
        m.solve_cr()
    return m


class FilesTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def test_levels(self):
        fil = write_ion(self.dir)
        en = read_fac_en(fil + "a.en")
        self.assertEqual((en["symbol"], en["Z"]), ("C", 6.))
        np.testing.assert_array_equal(en["nele"], [3, 2, 2, 2, 2, 2, 2, 1, 1])
        data = read_fac(fil, 2, TEMPS)
        atomic = data["atomic"]
        #the ion's levels sorted by energy, from its ground; the parents after them; nele 3 left out
        np.testing.assert_array_equal(atomic["fac_index"], [0, 2, 3, 1, 4, 8])
        np.testing.assert_allclose(atomic["energy"] * constants.h * constants.c * 100 / constants.e,
                                   [0., 10., 11., 12., 60., 70.], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(atomic["w"], [0., 1., 0., 1., 0., 0.])
        np.testing.assert_array_equal(atomic["config"], ["1s2", "1s1.2s1", "1s1.2s1", "1s1.2p1", "2s2", "2s1.3s1"])
        np.testing.assert_array_equal(atomic["L"], -1)
        np.testing.assert_array_equal(atomic["ion_fac_index"], [6, 5])
        np.testing.assert_allclose(atomic["ion_pot"] * constants.h * constants.c * 100 / constants.e, [50., 55.])
        np.testing.assert_array_equal(atomic["ion_w"], [0.5, 0.5])
        self.assertEqual((atomic["element"], atomic["charge_state"], atomic["iz0"], atomic["iz1"]), ("C", 4, 6, 5))
        np.testing.assert_allclose(data["input_file"]["temp_grid"] / 11604.5, TEMPS, rtol=1e-15)
        self.assertTrue(data["input_file"]["temp_grid"][-1] / 11604.5 >= TEMPS[-1])
        self.assertEqual(data["user"]["file_loc"], fil)

    def test_transitions(self):
        fil = write_ion(self.dir)
        data = read_fac(fil, 2, TEMPS, fold_autoionization=False)
        rates = data["rates"]
        #(upper, lower) of the sorted levels, 1 based: radiative and excitation pairs
        pairs = [tuple(p) for p in rates["excit"]["col_transitions"]]
        self.assertEqual(pairs, [(2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 4), (6, 1), (6, 4), (6, 5)])
        a_val = dict(zip(pairs, rates["a_val"]))
        self.assertEqual(a_val[(4, 1)], 1e9 + 1e2)                                 #E1 + M2
        self.assertEqual(a_val[(2, 1)], 1e3)
        #the two-photon decay of 1s2s 1S0 to the ground, which has no radiative rate in a.tr
        self.assertAlmostEqual(a_val[(3, 1)], two_photon_rate(6, 2))
        np.testing.assert_array_equal(rates["two_photon"]["transition"], [3, 1])
        upsilon = dict(zip(pairs, rates["excit"]["col_excit"]))
        np.testing.assert_array_equal(upsilon[(4, 1)] > 0, True)
        np.testing.assert_array_equal(upsilon[(5, 4)], 0.)
        #ionization to the lowest parent, recombination and dielectronic capture
        self.assertEqual([tuple(p) for p in rates["ioniz"]["ion_transitions"]], [(1, 1), (2, 1), (3, 1), (4, 1)])
        self.assertEqual([tuple(p) for p in rates["recomb"]["recomb_transitions"]],
                         [(1, 1), (1, 2), (1, 5), (1, 6), (2, 5)])
        np.testing.assert_array_equal(data["input_file"]["autoionization"], [[0, 0]] * 4 + [[1e13, 2e12], [3e12, 0]])
        np.testing.assert_array_equal(data["input_file"]["stabilizing"], 1.)

    def test_cross_sections_as_fac_prints_them(self):
        # The cross-sections from FAC's collision strengths and bound-free strengths are those it prints.
        fil = write_ion(self.dir)
        for t in read_fac_ce(fil + "a.ce")["transitions"]:
            section = FacCrossSection.excitation(t, t["energy"])
            np.testing.assert_allclose(section(t["energy"] + t["scattered"]), t["sigma"], rtol=1e-3)
        for t in read_fac_rr(fil + "a.rr")["transitions"]:
            section = FacCrossSection.recombination(t, t["energy"])
            np.testing.assert_allclose(section(t["photoelectron"]), t["sigma"], rtol=1e-3)
        for t in read_fac_ci(fil + "a.ci")["transitions"]:
            section = FacCrossSection.ionization(t, t["energy"])
            np.testing.assert_allclose(section(t["energy"] + t["ejected"]), t["sigma"], rtol=1e-3)
        self.assertEqual([s.threshold for s in fac_cross_sections(fil, "ce", nele=2)], [1., 12., 2., 60., 70.])

    def test_errors(self):
        fil = write_ion(self.dir)
        with self.assertRaisesRegex(ValueError, "no levels with 4 electrons"):
            read_fac(fil, 4, TEMPS)
        with self.assertRaisesRegex(ValueError, "parents"):
            read_fac(fil, 1, TEMPS)
        with self.assertRaisesRegex(FileNotFoundError, "nonea.en"):
            read_fac(os.path.join(self.dir, "none"), 2, TEMPS)
        with self.assertRaisesRegex(ValueError, "en\\) and radiative rates"):
            read_fac(fil, 2, TEMPS, processes=["en", "ce"])
        with self.assertRaisesRegex(ValueError, "Maxwellian"):
            read_fac(fil, 2, TEMPS, eedf="Gaussian")
        os.remove(fil + "a.ci")
        with self.assertRaises(FileNotFoundError):
            read_fac(fil, 2, TEMPS, processes=["en", "tr", "ci"])
        with self.assertWarnsRegex(UserWarning, "a.ci not found"):
            read_fac(fil, 2, TEMPS)
        write_ion(self.dir)
        write_en(fil + "a.en", verbose=0)
        with self.assertRaisesRegex(ValueError, "verbose"):
            read_fac(fil, 2, TEMPS)
        write_ion(self.dir)
        write_ai(fil + "a.ai", [(4, 6, 1e13), (4, 6, 1e13)])
        with self.assertRaisesRegex(ValueError, "twice"):
            read_fac(fil, 2, TEMPS)
        write_ion(self.dir)
        write_en(fil + "a.en", LEVELS + LEVELS[-1:])
        with self.assertRaisesRegex(ValueError, "level 7 twice"):
            read_fac(fil, 2, TEMPS)

    def test_no_fac_needed(self):
        self.assertNotIn("pfac", sys.modules)


class RatesTests(unittest.TestCase):
    """Rates of the toy ion against exact Maxwellian averages (non-relativistic, at temperatures where
    FAC's relativistic factor 1 + alpha^2 E/2 of the cross-sections is below 1e-5)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = read_fac(write_ion(tmp.name), 2, TEMPS, relativistic=False, fold_autoionization=False)["rates"]
        self.pairs = [tuple(p) for p in self.data["excit"]["col_transitions"]]

    def upsilon(self, pair):
        return self.data["excit"]["col_excit"][self.pairs.index(pair)]

    def test_constant_collision_strength(self):
        # A constant collision strength is its own effective collision strength.
        np.testing.assert_allclose(self.upsilon((3, 2)), 0.1, rtol=1e-3)

    def test_bethe_form(self):
        # Omega = b ln(E/E0) + c on the tabulated energies and as FAC's Born limit above them:
        # Upsilon = c + b exp(y) E1(y), y = E0/kT.
        y = 12. / TEMPS
        np.testing.assert_allclose(self.upsilon((4, 1)), 0.3 + 0.2 * np.exp(y) * exp1(y), rtol=1e-3)

    def test_exchange(self):
        # Without a Born limit (b < 0) the collision strength follows the power law of its last two
        # values above the last energy (here exchange, E^-2).
        t = read_fac_ce(self.files() + "a.ce")["transitions"][2]
        section = FacCrossSection.excitation(t, 2.)
        energies = 2. + 2. * np.array([40., 80.])
        omega = section.collision_strength(energies)
        np.testing.assert_allclose(omega, 0.05 * (energies / 2.)**-2, rtol=1e-3)

    def files(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return write_ion(tmp.name)

    def test_ionization(self):
        # The reduced ionization rate of FAC's fit, against a direct integration over E.
        from scipy.integrate import quad
        reduced = self.data["ioniz"]["ion_excit"][0]
        threshold = 50.
        for t, value in zip(TEMPS, reduced):
            def integrand(e):
                x = e / threshold
                k2 = 2 * e / HARTREE_EV * (1 + 0.5 * ALPHA**2 * e / HARTREE_EV)
                sigma = ci_fit(x, CI_PARAMS) / k2 * BOHR_AREA
                return sigma * speed(e, False) * maxwellian(e, t, False, reduced=True) * np.exp(-(e - threshold) / t)
            exact = quad(integrand, threshold, threshold + 60 * t, limit=200)[0]
            self.assertAlmostEqual(value / exact, 1., delta=2e-3)

    def test_radiative_recombination(self):
        # A constant bound-free strength S: alpha = K T^-1/2 (Eb^2 + 2 Eb T + 2 T^2), with
        # sigma = pi alpha^3 S a0^2 (Eb + E)^2/(g_free Hartree E) (E in eV) and v f from a Maxwellian.
        recomb = dict(zip([tuple(p) for p in self.data["recomb"]["recomb_transitions"]], self.data["recomb"]["recomb_excit"]))
        binding = 50.
        k = np.pi * ALPHA**3 * 0.02 * BOHR_AREA / (2 * HARTREE_EV)            #free parent 1s, g = 2
        speed_factor = constants.c * 100 * np.sqrt(2 / (constants.m_e * constants.c**2 / constants.e))
        exact = k * speed_factor * 2 / np.sqrt(np.pi) * TEMPS**-0.5 * (binding**2 + 2 * binding * TEMPS + 2 * TEMPS**2)
        np.testing.assert_allclose(recomb[(1, 1)], exact, rtol=1e-3)

    def test_dielectronic_capture(self):
        # The capture rate of FAC's strength is the Saha-Boltzmann detailed balance of autoionization:
        # (g_b/2 g_f) (h^2/(2 pi m kT))^1.5 A exp(-E/kT).
        recomb = dict(zip([tuple(p) for p in self.data["recomb"]["recomb_transitions"]], self.data["recomb"]["recomb_excit"]))
        kt = TEMPS * constants.e
        thermal = (constants.h**2 / (2 * np.pi * constants.m_e * kt))**1.5 * 1e6
        np.testing.assert_allclose(recomb[(1, 5)], 1 / 4 * thermal * 1e13 * np.exp(-10. / TEMPS), rtol=1e-3)
        np.testing.assert_allclose(recomb[(2, 5)], 1 / 4 * thermal * 2e12 * np.exp(-5. / TEMPS), rtol=1e-3)

    def test_relativistic(self):
        # The Maxwell-Juttner averages are those of a Maxwellian at low temperature.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        relativistic = read_fac(write_ion(tmp.name), 2, TEMPS, fold_autoionization=False)["rates"]
        np.testing.assert_allclose(relativistic["excit"]["col_excit"], self.data["excit"]["col_excit"], rtol=1e-3)
        #at 50 keV the relativistic speeds and distribution lower the rate of a constant cross-section
        rel = rate_coefficients(lambda e: np.where(e > 0, 1e-20, 0.), 0., [5e4])[0]
        nonrel = rate_coefficients(lambda e: np.where(e > 0, 1e-20, 0.), 0., [5e4], relativistic=False)[0]
        self.assertLess(rel, 0.95 * nonrel)


class MaxwellRateTests(unittest.TestCase):
    def test_maxwell_rate_files(self):
        # The rates of pfac.fac.MaxwellRate on their temperatures: upsilons and reduced ionization from
        # the collision strength columns, recombination from the rate column.
        with tempfile.TemporaryDirectory() as tmp:
            fil = write_ion(tmp, ce=False, rr=False, ci=False)
            temps = np.array([5., 10., 20.])
            with open(fil + "ce.mr", "w") as f:
                f.write("# \t0\t 0\t\t1\t 2\t1.2000E+01\t3\t1\n")
                for t, cs in zip(temps, (0.3, 0.4, 0.5)):
                    f.write(f"{t:11.4E}\t{cs:11.4E}\t{1e-3:11.4E}\t{9.:11.4E}\t{1e-3:11.4E}\n")
                f.write("\n\n")
            with open(fil + "ci.mr", "w") as f:
                f.write("#     0\t 0\t    6\t 1\t5.0000E+01\t    3\n")
                for t in temps:
                    f.write(f"{t:11.4E} {2.:11.4E} {1e-9:11.4E}\n")
                f.write("\n\n")
            with open(fil + "rr.mr", "w") as f:
                f.write("#     0\t 0\t    6\t 1\t5.0000E+01\t    3\n")
                for t in temps:
                    f.write(f"{t:11.4E} {2.:11.4E} {3e-2:11.4E}\n")
                f.write("\n\n")
            data = read_fac(fil, 2, TEMPS, eedf="Maxwellian_mr")
        np.testing.assert_allclose(data["input_file"]["temp_grid"] / 11604.5, temps)
        rates = data["rates"]
        pairs = [tuple(p) for p in rates["excit"]["col_transitions"]]
        np.testing.assert_allclose(rates["excit"]["col_excit"][pairs.index((4, 1))], [0.3, 0.4, 0.5])
        np.testing.assert_allclose(rates["ioniz"]["ion_excit"][0],
                                   UPSILON_RATE / np.pi * np.sqrt(HARTREE_EV / (2 * temps)) * 2.)
        recomb = dict(zip([tuple(p) for p in rates["recomb"]["recomb_transitions"]], rates["recomb"]["recomb_excit"]))
        np.testing.assert_allclose(recomb[(1, 1)], 3e-12)
        self.assertEqual(recomb[(1, 5)].shape, (3,))                  #capture on the same temperatures


def table(data, group, pairs, values):
    """{pair: values} of a group of rates ('excit', 'recomb', 'ioniz') of read_fac."""
    rates = data["rates"] if group is None else data["rates"][group]
    return dict(zip(map(tuple, data["rates"][group or "excit"][pairs]), rates[values]))


class FoldingTests(unittest.TestCase):
    """The autoionization folded into the other rates (read_fac's default), against the rates of the
    files and a model with the autoionization in its CR matrix."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.fil = write_ion(tmp.name)
        self.folded = read_fac(self.fil, 2, TEMPS)
        self.files = read_fac(self.fil, 2, TEMPS, fold_autoionization=False)

    def test_folded_rates(self):
        f, r = self.folded, self.files
        np.testing.assert_allclose(f["input_file"]["stabilizing"], [1., 1., 1., 1., B4, B8])
        #the cascade of 8 into the autoionizing 4 keeps the part that stabilizes
        a_f, a_r = table(f, None, "col_transitions", "a_val"), table(r, None, "col_transitions", "a_val")
        self.assertAlmostEqual(a_f[(6, 5)] / (5e9 * B4), 1.)
        self.assertEqual(a_f[(6, 4)], a_r[(6, 4)])
        #flows into the autoionizing levels times their stabilizing fractions
        ups_f, ups_r = table(f, "excit", "col_transitions", "col_excit"), table(r, "excit", "col_transitions", "col_excit")
        np.testing.assert_allclose(ups_f[(5, 1)], ups_r[(5, 1)] * B4, rtol=1e-12)
        np.testing.assert_allclose(ups_f[(6, 1)], ups_r[(6, 1)] * B8, rtol=1e-12)
        np.testing.assert_array_equal(ups_f[(4, 1)], ups_r[(4, 1)])
        rec_f, rec_r = table(f, "recomb", "recomb_transitions", "recomb_excit"), table(r, "recomb", "recomb_transitions", "recomb_excit")
        for pair, b in (((1, 5), B4), ((2, 5), B4), ((1, 6), B8), ((1, 1), 1.), ((1, 2), 1.)):
            np.testing.assert_allclose(rec_f[pair], rec_r[pair] * b, rtol=1e-12)
        #excitation-autoionization of the ground: its reduced excitation rate into d times the
        #probability of d reaching the parent, exp(-(E_d - I_p)/kT)
        ion_f, ion_r = table(f, "ioniz", "ion_transitions", "ion_excit"), table(r, "ioniz", "ion_transitions", "ion_excit")
        excitation = {pair: ups_r[pair] * UPSILON_RATE * np.sqrt(RYDBERG_EV / TEMPS) for pair in ((5, 1), (6, 1))}
        to_first = excitation[(5, 1)] * P4[0] * np.exp(-10. / TEMPS) + excitation[(6, 1)] * P8[0] * np.exp(-20. / TEMPS)
        to_second = excitation[(5, 1)] * P4[1] * np.exp(-5. / TEMPS) + excitation[(6, 1)] * P8[1] * np.exp(-15. / TEMPS)
        np.testing.assert_allclose(ion_f[(1, 1)], ion_r[(1, 1)] + to_first, rtol=1e-12)
        np.testing.assert_allclose(ion_f[(1, 2)], to_second, rtol=1e-12)
        np.testing.assert_array_equal(f["atomic"]["zpla1"][0], [1, 2])

    def test_explicit_autoionization(self):
        # The folded model has the populations, SCD and ACD of the model with the autoionization in its
        # CR matrix, as a decay of the levels to the parents, while collisions from the autoionizing
        # levels are slow beside their decays.
        dens = np.array([1e8, 1e11])
        options = dict(dens=dens, use_recombination_three_body=False, suppliment_with_ecip=False)
        folded = model(self.fil, **options).data
        explicit = model(self.fil, fold_autoionization=False, **options).data
        cr = explicit["cr_matrix"]["cr"].copy()
        auto = explicit["input_file"]["autoionization"]
        n, num_parents = auto.shape
        cr[n:n + num_parents, :n] += auto.T[:, :, None, None]
        cr[np.arange(n), np.arange(n)] -= auto.sum(axis=1)[:, None, None]
        excited, drivers = np.arange(1, n), [0, n, n + 1]
        ionization = explicit["rates"]["ioniz"]["ionization"]
        for k in range(len(TEMPS)):
            for d, ne in enumerate(dens):
                c = cr[:, :, k, d]
                pops = np.linalg.solve(c[np.ix_(excited, excited)], -c[np.ix_(excited, drivers)])
                np.testing.assert_allclose(folded["processed"]["pops_no_norm"][:, :, k, d], pops, rtol=1e-6)
                #SCD of the ground, per ion: ionization and autoionization of all its levels
                excitation = np.r_[1., pops[:, 0]]
                scd = (ionization[:, :, k] + auto / ne).T @ excitation / excitation.sum()
                np.testing.assert_allclose(folded["processed"]["scd"][0, :, k, d], scd, rtol=1e-6)
                #ACD into the ground: what reaches it of the recombination into each level
                acd = (c[0, excited] @ pops[:, 1:] + c[0, [n, n + 1]]) / ne
                np.testing.assert_allclose(folded["processed"]["acd"][0, :, k, d], acd, rtol=1e-6)

    def test_levels_without_radiative_decays(self):
        # An autoionizing level without radiative decays never stabilizes: it is left out with a warning,
        # and its excitation is ionization.
        extra = (9, 2, 65., 0, '2p2', '2p+2(0)0')
        with tempfile.TemporaryDirectory() as tmp, mock.patch(__name__ + ".LEVELS", LEVELS + [extra]):
            fil = write_ion(tmp)
            write_en(fil + "a.en", LEVELS)                       #LEVELS of the patch
            write_ce(fil + "a.ce", [(0, 9, SCATTERED * 65., np.full(60, 0.03), 0., 0.03, 6500.)])
            write_ai(fil + "a.ai", [(4, 6, 1e13), (4, 5, 2e12), (8, 6, 3e12), (9, 6, 1e13)])
            with self.assertWarnsRegex(UserWarning, "1 autoionizing levels"):
                data = read_fac(fil, 2, TEMPS)
            files = read_fac(fil, 2, TEMPS, fold_autoionization=False)
            m = model(fil)
        np.testing.assert_array_equal(data["atomic"]["fac_index"], [0, 2, 3, 1, 4, 8])
        excitation = table(files, "excit", "col_transitions", "col_excit")[(6, 1)] * UPSILON_RATE * np.sqrt(RYDBERG_EV / TEMPS)
        ionization = table(data, "ioniz", "ion_transitions", "ion_excit")
        np.testing.assert_allclose(ionization[(1, 1)], table(files, "ioniz", "ion_transitions", "ion_excit")[(1, 1)] +
                                   excitation * np.exp(-15. / TEMPS), rtol=1e-12)
        self.assertTrue(np.isfinite(m.data["processed"]["scd"]).all())


class ModelTests(unittest.TestCase):
    """colradpy with the data of read_fac as its file."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.fil = write_ion(tmp.name)

    def test_rates_in_the_cr_matrix(self):
        # The excitation rates of the CR matrix are the averages of the cross-sections.
        m = model(self.fil)
        t = read_fac_ce(self.fil + "a.ce")["transitions"][1]                   #ground -> 1s2p
        section = FacCrossSection.excitation(t, 12.)
        exact = rate_coefficients(section, 12., TEMPS)
        np.testing.assert_allclose(m.data["cr_matrix"]["q_ij"][0, 3], exact, rtol=1e-10)
        cr = m.data["cr_matrix"]["cr"]
        scale = np.abs(np.diagonal(cr, axis1=0, axis2=1)).max(axis=-1)
        self.assertLess((np.abs(cr.sum(axis=0)).max(axis=0) / scale).max(), 1e-13)

    def test_dielectronic_recombination(self):
        # At low density the autoionizing levels stabilize radiatively with their fractions B4, B8: ACD
        # into the ground is the radiative recombination into all levels plus the capture times them.
        m = model(self.fil, dens=np.array([1e2]), use_recombination_three_body=False, use_ionization=False,
                  suppliment_with_ecip=False)
        recomb = table(read_fac(self.fil, 2, TEMPS, fold_autoionization=False), "recomb", "recomb_transitions", "recomb_excit")
        expected = recomb[(1, 1)] + recomb[(1, 2)] + recomb[(1, 5)] * B4 + recomb[(1, 6)] * B8
        np.testing.assert_allclose(m.data["processed"]["acd"][0, 0, :, 0], expected, rtol=1e-4)

    def test_pairs_and_three_body(self):
        # temperature-density pairs, three-body recombination with the parents' weights from J, and the
        # file of the data
        grid = model(self.fil)
        pairs = model(self.fil, dens=np.full(len(TEMPS), 1e12), temp_dens_pair=True)
        np.testing.assert_allclose(pairs.data["processed"]["scd"], grid.data["processed"]["scd"][..., 1], rtol=1e-9)
        np.testing.assert_allclose(grid.data["rates"]["recomb"]["w_ion"], [2., 2.])
        self.assertEqual(grid.data["user"]["file_loc"], self.fil)


if __name__ == "__main__":
    unittest.main()

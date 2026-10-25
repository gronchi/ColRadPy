"""Escape factors of optically thick lines (colradpy.escape_factors) and their use in the CR
matrix and the PECs (the opacity option of colradpy).

Run from the repository root with: python -m unittest discover -s tests -v.
"""

import contextlib
import io
import importlib.util
import tempfile
import sys
import types
from unittest import mock
from pathlib import Path
import unittest
import warnings

import numpy as np
from scipy import constants, integrate
from scipy.special import ndtri
from scipy.stats import qmc

from colradpy import colradpy
from colradpy.escape_factors import (GEOMETRIES, _mean_escape_kernel, line_escape_factor, line_profile,
                                     optical_depth, population_escape_factor, thermal_speed)

INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"
TAUS = np.array([1e-3, 0.1, 1., 10., 1e3, 1e5])


class EscapeFactorTests(unittest.TestCase):
    def test_thin_limit_and_fall_off(self):
        for position in ("centre", "average"):
            for geometry in GEOMETRIES:
                values = population_escape_factor(np.r_[1e-9, TAUS], geometry, position=position)
                self.assertAlmostEqual(values[0], 1., places=6)
                self.assertTrue(np.all(np.diff(values) < 0))
        values = line_escape_factor(np.r_[1e-9, TAUS])
        self.assertAlmostEqual(values[0], 1., places=6)
        self.assertTrue(np.all(np.diff(values) < 0))

    def test_geometry_ordering(self):
        # from the centre the paths to the edge are shortest in a sphere and longest in a slab, and
        # photons emitted away from the centre escape more easily than those emitted at it
        for position in ("centre", "average"):
            slab, cylinder, sphere = (population_escape_factor(TAUS, g, position=position) for g in GEOMETRIES)
            self.assertTrue(np.all((slab < cylinder) & (cylinder < sphere)))
        for geometry in GEOMETRIES:
            self.assertTrue(np.all(population_escape_factor(TAUS, geometry, position="average") >
                                   population_escape_factor(TAUS, geometry, position="centre")))

    def test_against_quadrature(self):
        for tau in (0.5, 10., 1e3):
            sphere = integrate.quad(lambda x: 2*line_profile(x)*np.exp(-tau*np.exp(-x**2)), 0, 12, limit=500)[0]
            line = integrate.quad(lambda x: 2*line_profile(x)*-np.expm1(-2*tau*np.exp(-x**2))/(2*tau*np.exp(-x**2)),
                                  0, 12, limit=500)[0]
            np.testing.assert_allclose(population_escape_factor(tau, "sphere", position="centre"), sphere, rtol=1e-6)
            np.testing.assert_allclose(line_escape_factor(tau), line, rtol=1e-6)

    def test_holstein_asymptotes(self):
        # Doppler profile: 1/(tau sqrt(pi ln tau)); Lorentz profile: 1/sqrt(pi tau)
        tau = 1e6
        doppler = population_escape_factor(tau, "sphere", position="centre")*tau*np.sqrt(np.pi*np.log(tau))
        self.assertTrue(0.95 < doppler < 1.02)
        lorentz = population_escape_factor(tau, "sphere", damping=100., position="centre")*np.sqrt(np.pi*tau)
        self.assertAlmostEqual(lorentz, 1., delta=0.01)

    def test_volume_average_thick_limit(self):
        # an optically thick body emits from its surface: the mean escape probability tends to
        # S/(4 V kappa) = 1/(4 t), 1/(2 t) and 3/(4 t) for the slab, the cylinder and the sphere
        t = np.array([1e3])
        for geometry, coefficient in (("slab", 0.25), ("cylinder", 0.5), ("sphere", 0.75)):
            np.testing.assert_allclose(_mean_escape_kernel(geometry)(t)*t, coefficient, rtol=2e-3)

    def test_optical_depth_from_the_oscillator_strength(self):
        # line centre cross section (e^2/(4 eps0 m_e c)) f phi_nu(0), with the A-value from f
        wavelength, f, g_lower, g_upper, mass, temperature = 121.567, 0.4164, 2., 6., 2.014, 1.
        nu = constants.c/(wavelength*1e-9)
        a_value = 2*np.pi*constants.e**2*nu**2/(constants.epsilon_0*constants.m_e*constants.c**3)*g_lower/g_upper*f
        phi_nu = 1/(np.sqrt(np.pi)*nu*thermal_speed(temperature, mass)/100/constants.c)
        sigma = constants.e**2/(4*constants.epsilon_0*constants.m_e*constants.c)*f*phi_nu*1e4   # cm2
        np.testing.assert_allclose(optical_depth(wavelength, a_value, g_upper, g_lower, 1., 1., temperature, mass),
                                   sigma, rtol=1e-12)
        np.testing.assert_allclose(sigma, 7.74351e-14, rtol=1e-6)

    def test_voigt_thin_limit_and_full_wings(self):
        for damping in (1e-4, 1., 100.):
            for geometry in GEOMETRIES:
                for position in ('centre', 'average'):
                    self.assertEqual(population_escape_factor(0., geometry, damping, position), 1.)
            self.assertEqual(line_escape_factor(0., damping), 1.)
            # Independent integration in scaled frequency over the entire Voigt profile.
            for tau in (1e-8, 1., 100.):
                scale = max(1., damping)
                phi0 = line_profile(0., damping)
                expected = integrate.quad(
                    lambda y: 2*scale*line_profile(scale*y, damping)*
                    np.exp(-tau*line_profile(scale*y, damping)/phi0),
                    0., np.inf, epsabs=1e-12, epsrel=1e-8)[0]
                np.testing.assert_allclose(
                    population_escape_factor(tau, 'sphere', damping, 'centre'), expected, rtol=1e-7)

    def test_shapes_and_invalid_physical_inputs(self):
        for damping in (0., 0.1):
            for values in (np.array(1.), np.ones((2, 3)), np.empty((2, 0))):
                for function in (population_escape_factor, line_escape_factor):
                    self.assertEqual(function(values, damping=damping).shape, values.shape)
            for bad in (-1., np.inf, np.nan):
                for function in (population_escape_factor, line_escape_factor):
                    with self.assertRaises(ValueError):
                        function(bad, damping=damping)
                    with self.assertRaises(ValueError):
                        function(1., damping=bad)
        with self.assertRaises(ValueError):
            thermal_speed(0., 1.)
        with self.assertRaises(ValueError):
            optical_depth(100., 1., 1., 1., -1., 1., 1., 1.)

    def test_boundary_integrated_flux(self):
        # Integrate escaping chord radiances over projected area/direction. This must recover
        # volume-integrated escape, not the radiance of the central chord alone.
        nodes, weights = np.polynomial.legendre.leggauss(128)
        mu, weight = (nodes + 1)/2, weights/2
        for tau in (0.1, 10., 1e4):
            slab = np.dot(weight, line_escape_factor(tau/mu))
            sphere = 3*np.dot(weight, mu**2*line_escape_factor(tau*mu))
            np.testing.assert_allclose(population_escape_factor(tau, 'slab'), slab, rtol=2e-5)
            np.testing.assert_allclose(population_escape_factor(tau, 'sphere'), sphere, rtol=2e-5)
        # The conserved slab can be brighter than thin along the central normal chord.
        self.assertGreater(line_escape_factor(1e4)/population_escape_factor(1e4, 'slab'), 1.9)

    def test_cylinder_against_independent_photon_paths(self):
        # Uniform emission points, isotropic directions and redistributed Doppler frequencies.
        # Seeded Sobol samples test transport independently of the surface-integral table.
        sample = qmc.Sobol(4, scramble=True, seed=901).random_base2(18)
        rho = np.sqrt(sample[:, 0])
        angle = 2*np.pi*sample[:, 1]
        path = (np.sqrt(1-rho**2*np.sin(angle)**2) - rho*np.cos(angle))/np.sqrt(1-sample[:, 2]**2)
        profile_ratio = np.exp(-ndtri(sample[:, 3])**2/2)
        for tau in (0.01, 1., 10.):
            escaped = np.exp(-tau*path*profile_ratio).mean()
            np.testing.assert_allclose(population_escape_factor(tau, 'cylinder'), escaped, rtol=2e-3)


class OpacityOptionTests(unittest.TestCase):
    def run_model(self, opacity=None, dens=(1e12, 1e13), **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = colradpy(str(INPUT / "cpb03_ls#be0.dat"), np.array([0]), np.array([5., 20.]),
                             np.array(dens), opacity=opacity, **kwargs)
            model.solve_cr()
        return model

    def resonance(self, model):
        levels = model.data["processed"]["pec_levels"]
        return np.where((levels[:, 0] == 2) & (levels[:, 1] == 0))[0][0]

    def test_negligible_absorbers_are_thin(self):
        thin = self.run_model()
        thick = self.run_model(dict(length=1., densities={0: 1e-6}, temperature=1., mass=9.012))
        for key in ("pecs", "scd", "acd", "plt"):
            np.testing.assert_allclose(thick.data["processed"][key], thin.data["processed"][key], rtol=1e-9)
        np.testing.assert_allclose(thick.data["processed"]["pecs_los"], thin.data["processed"]["pecs"], rtol=1e-9)

    def test_matrix_uses_reduced_a_values(self):
        model = self.run_model(dict(length=5., densities={0: 1e12}, temperature=1., mass=9.012))
        cr = model.data["cr_matrix"]
        theta = model.data["rates"]["escape"]["theta_p"]
        np.testing.assert_allclose(cr["A_ji"], cr["A_ji_thin"]*theta)
        self.assertLess(theta[2, 0], 0.2)
        np.testing.assert_allclose(theta[2, 0], population_escape_factor(
            model.data['rates']['escape']['tau0'][2, 0]))
        self.assertEqual(theta[2, 1], 1.)            # the lower level 1 has no absorbers
        matrix = cr["cr"]
        np.testing.assert_array_less(np.abs(matrix.sum(axis=0)), 1e-12*np.abs(matrix).max())

    def test_given_escape_factors(self):
        model = self.run_model(dict(theta_p={(2, 0): 0.1}, theta_l={(2, 0): 0.3}))
        cr = model.data["cr_matrix"]
        self.assertAlmostEqual(cr["A_ji"][2, 0], 0.1*cr["A_ji_thin"][2, 0])
        i = self.resonance(model)
        np.testing.assert_allclose(model.data["processed"]["pecs_los"][i], 3*model.data["processed"]["pecs"][i])

    def test_trapping_at_low_density(self):
        # with collisions too slow to quench the trapped level, its population grows as 1/theta_p,
        # every excitation still escapes as a photon (net PEC unchanged), and with the average over
        # the volume the line-of-sight radiance stays close to the thin one
        thin = self.run_model(dens=(1e8, 1e9))
        thick = self.run_model(dict(length=5., densities={0: 1e12}, temperature=1., mass=9.012, position="average"),
                               dens=(1e8, 1e9))
        theta = thick.data["rates"]["escape"]["theta_p"][2, 0]
        i = self.resonance(thin)
        ratio = thick.data["processed"]["pops"][1, 0, 0, 0]/thin.data["processed"]["pops"][1, 0, 0, 0]
        self.assertAlmostEqual(ratio*theta, 1., delta=0.02)
        np.testing.assert_allclose(thick.data["processed"]["pecs"][i, 0, :, 0], thin.data["processed"]["pecs"][i, 0, :, 0],
                                   rtol=0.02)
        los = thick.data["processed"]["pecs_los"][i, 0, :, 0]/thin.data["processed"]["pecs"][i, 0, :, 0]
        self.assertTrue(np.all(np.abs(los - 1) < 0.1))

    def test_voigt_profile_and_invalid_input(self):
        model = self.run_model(dict(length=5., densities={0: 1e12}, temperature=1., mass=9.012, profile="voigt"))
        self.assertLess(model.data["rates"]["escape"]["theta_p"][2, 0], 1.)
        for bad in (dict(length=1.), dict(length=1., densities={0: 1.}, temperature=1., mass=1., geometry="box"),
                    dict(length=1., densities={0: 1.}, temperature=1., mass=1., position="edge")):
            with self.assertRaises(ValueError):
                colradpy(str(INPUT / "cpb03_ls#be0.dat"), np.array([0]), np.array([5.]), np.array([1e12]), opacity=bad)

    def test_invalid_opacity_is_rejected_before_solve(self):
        physical = dict(length=5., densities={0: 1e12}, temperature=1., mass=9.012)
        bad_options = [[], {}, dict(theta_p={(2, 0): 0.1}),
                       dict(theta_p={(2, 0): 0.1}, theta_l={}),
                       dict(physical, theta_p={}, theta_l={}), dict(physical, typo=1)]
        for name in ('temperature', 'mass', 'length'):
            for value in (-1., np.nan, np.inf, [1.]):
                bad_options.append(dict(physical, **{name: value}))
        for name in ('temperature', 'mass'):
            bad_options.append(dict(physical, **{name: 0.}))
        for level in (-1, 999, 0.5, '0', True):
            bad_options.append(dict(physical, densities={level: 1.}))
            bad_options.append(dict(theta_p={(2, level): 0.1}, theta_l={(2, level): 0.1}))
        for value in (-1., np.nan, np.inf, [1.]):
            bad_options.append(dict(physical, densities={0: value}))
        for value in (-0.1, 1.1, np.nan, np.inf, [0.1]):
            bad_options.append(dict(theta_p={(2, 0): value}, theta_l={(2, 0): 0.1}))
            bad_options.append(dict(theta_p={(2, 0): 0.1}, theta_l={(2, 0): value}))
        bad_options.append(dict(theta_p={(0, 2): 0.1}, theta_l={(0, 2): 0.1}))
        for opacity in bad_options:
            with self.subTest(opacity=opacity), self.assertRaises(ValueError):
                colradpy(str(INPUT / 'cpb03_ls#be0.dat'), np.array([0]), np.array([5.]),
                         np.array([1e12]), opacity=opacity)

    def test_zero_absorbers_and_independent_input(self):
        opacity = dict(length=0., densities={0: 1e12}, temperature=1., mass=9.012, profile='voigt')
        model = self.run_model(opacity)
        np.testing.assert_array_equal(model.data['rates']['escape']['theta_p'], 1.)
        np.testing.assert_array_equal(model.data['rates']['escape']['theta_l'], 1.)
        opacity['densities'][0] = -1.
        self.assertEqual(model.data['user']['opacity']['densities'][0], 1e12)

    def test_rebuilding_matrix_updates_opacity_without_reducing_a_twice(self):
        model = self.run_model(dict(theta_p={(2, 0): 0.1}, theta_l={(2, 0): 0.3}))
        thin_a = model.data['cr_matrix']['A_ji_thin'][2, 0]
        model.data['user']['opacity']['theta_p'][2, 0] = 0.2
        model.populate_cr_matrix()
        self.assertAlmostEqual(model.data['cr_matrix']['A_ji'][2, 0], 0.2*thin_a)
        self.assertEqual(model.data['cr_matrix']['A_ji_thin'][2, 0], thin_a)

    def test_los_sxb_exports_and_time_dependence(self):
        model = self.run_model(dict(theta_p={(2, 0): 0.1}, theta_l={(2, 0): 0.3}),
                               td_t=np.array([0., 1e-8]))
        i = self.resonance(model)
        np.testing.assert_allclose(model.get_sxb(emission='los')[i], model.get_sxb()[i]/3)
        with self.assertRaises(ValueError):
            model.get_sxb(emission='unknown')
        model.data['user']['td_n0'] = np.eye(len(model.data['cr_matrix']['cr']))[2]
        model.solve_time_dependent()
        td = model.data['processed']['td']
        np.testing.assert_allclose(td['pecs_los'][i], 3*td['pecs'][i], atol=1e-20)
        # Inspect both metadata and the actual arrays passed to each file writer.
        for emission, key in (('net', 'pecs'), ('los', 'pecs_los')):
            with mock.patch('colradpy.colradpy_class.write_adf15') as writer:
                model.write_pecs_adf15('unused', pec_inds=[i], emission=emission)
                np.testing.assert_array_equal(writer.call_args.args[3], model.data['processed'][key])
                self.assertEqual(writer.call_args.kwargs['user']['emission'], emission)
                self.assertEqual(writer.call_args.kwargs['user']['opacity'], model.data['user']['opacity'])
            with mock.patch('colradpy.colradpy_class.write_adf13') as writer:
                model.write_sxb_adf13('unused', sxb_inds=[i], emission=emission)
                np.testing.assert_allclose(writer.call_args.args[3][i], model.get_sxb(emission=emission)[i])
                self.assertEqual(writer.call_args.kwargs['user']['emission'], emission)

    def test_unsupported_population_combinations(self):
        opacity = dict(theta_p={(2, 0): 0.1}, theta_l={(2, 0): 0.3})
        with self.assertRaisesRegex(ValueError, 'projection'):
            colradpy(str(INPUT / 'cpb03_ls#be0.dat'), np.array([0]), np.array([5.]),
                     np.array([1e12]), opacity=opacity, projection=('unused17', 'unused18'))
        model = self.run_model(opacity)
        with self.assertRaisesRegex(ValueError, 'splitting'):
            model.split_pec_multiplet()

    @unittest.skipUnless(importlib.util.find_spec('h5py'), 'optional h5py is not installed')
    def test_hdf_export_preserves_opacity_and_los_arrays(self):
        model = self.run_model(dict(theta_p={(2, 0): 0.1}, theta_l={(2, 0): 0.3}))
        # hdfdict is optional. Check the payload at its boundary, including real HDF5 group names.
        saved = {}

        def dump(payload, output):
            saved.update(payload)
            for name, array in payload['user']['opacity'].items():
                output.create_dataset('opacity/' + name, data=array)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                sys.modules, hdfdict=types.SimpleNamespace(dump=dump)):
            with contextlib.redirect_stdout(io.StringIO()):
                model.dump_hdf5(str(Path(tmp) / 'opacity.h5'), cut_limit=-1.)
        np.testing.assert_array_equal(saved['processed']['pecs_los'], model.data['processed']['pecs_los'])
        np.testing.assert_array_equal(saved['rates']['escape']['theta_p'], model.data['rates']['escape']['theta_p'])
        np.testing.assert_array_equal(saved['user']['opacity']['theta_p'], [[2., 0., 0.1]])
        self.assertEqual(model.data['user']['opacity']['theta_p'], {(2, 0): 0.1})

    def test_paired_grid(self):
        model = self.run_model(dict(length=5., densities={0: 1e12}, temperature=1., mass=9.012),
                               temp_dens_pair=True)
        self.assertEqual(model.data['processed']['pecs_los'].shape, model.data['processed']['pecs'].shape)
        np.testing.assert_array_equal(model.data['cr_matrix']['A_ji_loss'],
                                      model.data['cr_matrix']['A_ji_thin'].sum(axis=1))


if __name__ == "__main__":
    unittest.main()

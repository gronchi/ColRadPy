"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the matrix-exponential solvers against analytic solutions and scipy's expm:
complex eigenvalue pairs (issue #41), defective rate matrices and slow modes with a
source term.
"""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from colradpy.energy_balance import energy_balance
from colradpy.solve_matrix_exponential import (
    _expm_populations,
    _gth_stationary,
    _stationary_population,
    eval_matrix_exponential_solution,
    solve_matrix_exponential,
    solve_matrix_exponential_source,
    solve_matrix_exponential_steady_state,
)


# Population cycles 1 -> 2 -> 3 -> 1, which gives eigenvalues 0 and -1.5 +- 0.87i.
CYCLE = np.array([[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
# Equal rates in a chain make the matrix defective (no complete eigenvector basis).
CHAIN = np.array([[-1.0, 0.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])


def reference(matrix, n0, times, source=None):
    """Populations from expm of the matrix augmented with a constant source."""
    size = len(n0)
    augmented = np.zeros((size + 1, size + 1))
    augmented[:size, :size] = matrix
    if source is not None:
        augmented[:size, size] = source
    start = np.append(n0, 1.0)
    return np.stack([expm(augmented * t) @ start for t in times], axis=1)[:size]


def solve(matrix, n0, times, source):
    if source is None:
        return solve_matrix_exponential(matrix, n0, times)[0]
    return solve_matrix_exponential_source(matrix, n0, source, times)[0]


class SolverTests(unittest.TestCase):
    def assert_real(self, values):
        self.assertFalse(np.iscomplexobj(values), values.dtype)

    def test_complex_eigenvalue_pairs_give_real_populations(self):
        self.assertTrue(np.iscomplexobj(np.linalg.eigvals(CYCLE)))
        n0 = np.array([1.0, 0.0, 0.0])
        times = np.array([0.0, 0.3, 1.0, 5.0])
        for source in (None, np.array([0.2, 0.0, 0.1])):
            for matrix in (CYCLE[:, :, None, None], CYCLE[:, :, None]):
                with self.subTest(source=source is not None, ndim=matrix.ndim):
                    pops = solve(matrix, n0, times, source)
                    self.assert_real(pops)
                    expected = reference(CYCLE, n0, times, source)
                    np.testing.assert_allclose(pops.reshape(3, len(times)), expected, atol=1e-13)
                    if source is None:
                        np.testing.assert_allclose(pops.sum(axis=0), 1.0, atol=1e-13)

    def test_defective_chain_matches_exact_solution(self):
        n0 = np.array([1.0, 0.0, 0.0])
        times = np.array([0.0, 1.0, 2.0])
        exact = np.array([np.exp(-times), times * np.exp(-times),
                          1 - np.exp(-times) - times * np.exp(-times)])
        for source in (None, np.zeros(3)):
            with self.subTest(source=source is not None):
                pops = solve(CHAIN[:, :, None, None], n0, times, source)[:, :, 0, 0]
                np.testing.assert_allclose(pops, exact, atol=1e-12)
                np.testing.assert_allclose(pops.sum(axis=0), 1.0, atol=1e-12)

    def test_defective_transient_is_checked_between_late_output_times(self):
        rates = np.array([[-1.0, 0.0], [1.0, -1.0]])
        n0 = np.array([1.0, 0.0])
        times = np.r_[0.0, 1.0, np.linspace(100.0, 300.0, 126)]
        for matrix in (rates[:, :, None], rates[:, :, None, None]):
            for source in (None, np.zeros(2), np.array([0.1, 0.0])):
                with self.subTest(ndim=matrix.ndim, source=source):
                    pops = solve(matrix, n0, times, source)
                    np.testing.assert_allclose(
                        pops.reshape(2, len(times)), reference(rates, n0, times, source),
                        rtol=1e-12, atol=1e-14,
                    )

    def test_accurate_ill_conditioned_solution_has_bounded_exponential_checks(self):
        # Nearly equal rates make the basis ill conditioned, but this initial
        # population has an accurate eigen solution. A million output times must
        # not require a million matrix exponentials to establish that.
        rate = 1.0 + 1e-9
        rates = np.array([[-1.0, 0.0], [1.0, -rate]])
        self.assertGreater(np.linalg.cond(np.linalg.eig(rates)[1]), 1e8)
        times = np.r_[0.0, np.geomspace(1e-6, 10.0, 999999)]
        n0 = np.array([0.0, 1.0])
        for source in (None, np.zeros(2)):
            with self.subTest(source=source), patch(
                'colradpy.solve_matrix_exponential._expm_populations', wraps=_expm_populations
            ) as exponential:
                pops = solve(rates[:, :, None], n0, times, source)[:, :, 0]
                np.testing.assert_allclose(pops[0], 0.0, atol=1e-14)
                np.testing.assert_allclose(pops[1], np.exp(-rate*times), rtol=1e-12)
                self.assertEqual(exponential.call_count, 1)
                checked = np.sort(exponential.call_args.args[-1])
                self.assertLessEqual(len(checked), 32)
                np.testing.assert_array_equal(checked[:2], times[:2])
                self.assertEqual(checked[-1], times[-1])
                # Checks span the physical time scales, even on a log-spaced grid.
                self.assertLess(np.diff(np.log(checked[1:])).max(),
                                np.log(times[-1]/times[1])/16)

    def test_closed_cycle_conserves_population_over_many_relaxation_times(self):
        rates = np.array([[-1.0, 0.0, 3.0], [1.0, -2.0, 0.0], [0.0, 2.0, -3.0]])
        equilibrium = np.array([6.0, 3.0, 2.0]) / 11
        n0 = np.array([1.0, 0.0, 0.0])
        times = np.array([0.0, 1e-15, 1.0])
        homogeneous = np.stack([n0, expm(rates) @ n0, equilibrium], axis=1)
        for matrix in ((rates * 1e15)[:, :, None], (rates * 1e15)[:, :, None, None]):
            for source in (None, np.zeros(3), equilibrium * 0.3):
                with self.subTest(ndim=matrix.ndim, source=source):
                    pops = solve(matrix, n0, times, source).reshape(3, len(times))
                    expected = homogeneous.copy()
                    if source is not None:
                        expected += source[:, None] * times
                    np.testing.assert_allclose(pops, expected, rtol=1e-12, atol=1e-13)
                    expected_mass = 1.0 + (0.0 if source is None else source.sum()) * times
                    np.testing.assert_allclose(pops.sum(axis=0), expected_mass, atol=1e-13)

    def test_conservation_fallback_preserves_slow_modes_and_small_populations(self):
        # A fast closed cycle and an independent slow capture chain. Keeping only
        # the equilibrium of the whole matrix would incorrectly delete the slow mode.
        rates = np.zeros((5, 5))
        rates[:3, :3] = np.array([[-1.0, 0.0, 3.0], [1.0, -2.0, 0.0],
                                [0.0, 2.0, -3.0]]) * 1e15
        rates[3:, 3:] = np.array([[-1e-12, 0.0], [1e-12, 0.0]])
        times = np.array([1e12])
        for small in (0.5, 1e-12):
            initial = np.array([1.0 - small, 0.0, 0.0, small, 0.0])
            expected = np.r_[(1.0 - small) * np.array([6.0, 3.0, 2.0]) / 11,
                             small * np.exp(-1), small * (1 - np.exp(-1))]
            with self.subTest(slow_population=small), np.errstate(over='ignore', invalid='ignore'):
                for source in (None, np.zeros(5)):
                    pops = solve(rates[:, :, None, None], initial, times, source).ravel()
                    np.testing.assert_allclose(pops, expected, rtol=1e-12, atol=1e-27)

    def test_empty_time_arrays_preserve_output_shapes(self):
        for rates in (CYCLE, CHAIN):
            for paired in (False, True):
                matrix = np.repeat(rates[:, :, None], 2, axis=2)
                if not paired:
                    matrix = matrix[:, :, None, :]
                for times in (np.empty(0), np.empty((2, 0))):
                    expected_shape = (3, 0, 2) if paired else (3, 0, 1, 2)
                    for source in (None, np.zeros(3)):
                        with self.subTest(defective=rates is CHAIN, paired=paired,
                                          time_shape=times.shape, source=source):
                            pops = solve(matrix, np.array([1.0, 0.0, 0.0]), times, source)
                            self.assertEqual(pops.shape, expected_shape)
                            self.assert_real(pops)

    def test_slow_modes_with_source(self):
        rate = 1e-12
        matrix = np.array([[-rate, 0.0], [rate, 0.0]])[:, :, None, None]
        n0 = np.array([1.0, 0.0])
        t = np.array([1e12])
        # Without a source: n1 = exp(-rate t).
        pops = solve_matrix_exponential_source(matrix, n0, np.zeros(2), t)[0]
        np.testing.assert_allclose(pops[:, 0, 0, 0], [np.exp(-1), 1 - np.exp(-1)], rtol=1e-10)
        # With source s into level 1: n1 = s/rate + (1 - s/rate) exp(-rate t), total = 1 + s t.
        s = 3e-12
        pops = solve_matrix_exponential_source(matrix, n0, np.array([s, 0.0]), t)[0]
        n1 = s / rate + (1 - s / rate) * np.exp(-1)
        np.testing.assert_allclose(pops[:, 0, 0, 0], [n1, 1 + s * t[0] - n1], rtol=1e-10)

    def test_weak_source_and_small_initial_populations_are_preserved(self):
        rates = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
        times = np.array([0.0, 1.0, 10.0])
        cases = (
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1e-12, 0.0])),
            (np.array([0.0, 1e-12, 0.0]), np.zeros(3)),
        )
        for matrix in (rates[:, :, None], rates[:, :, None, None]):
            for n0, source in cases:
                with self.subTest(ndim=matrix.ndim, source=source.any()):
                    pops = solve_matrix_exponential_source(matrix, n0, source, times)[0]
                    np.testing.assert_allclose(
                        pops.reshape(3, len(times)), reference(rates, n0, times, source),
                        rtol=1e-12, atol=1e-27,
                    )

    def test_untrusted_points_on_every_time_grid_layout(self):
        # Two densities: a defective chain and a well-behaved cycle.
        matrix = np.stack([CHAIN, CYCLE], axis=-1)[:, :, None, :]
        n0 = np.array([1.0, 0.0, 0.0])
        layouts = {
            "same times": np.array([0.5, 2.0]),
            "times per density": np.array([[0.5, 2.0], [0.1, 3.0]]),
            "one time per density": np.array([[0.7, 1.5]]),
        }
        for source in (None, np.array([0.0, 0.3, 0.0])):
            for name, td_t in layouts.items():
                with self.subTest(source=source is not None, layout=name):
                    pops = solve(matrix, n0, td_t, source)
                    self.assert_real(pops)
                    for density, rates in enumerate((CHAIN, CYCLE)):
                        if name == "same times":
                            got, times = pops[:, :, 0, density], td_t
                        elif name == "times per density":
                            got, times = pops[:, :, 0, density], td_t[density]
                        else:
                            got, times = pops[:, 0, density, None], [td_t[0, density]]
                        np.testing.assert_allclose(
                            got, reference(rates, n0, times, source), atol=1e-12)

    def test_steady_state_is_real(self):
        pops = solve_matrix_exponential_steady_state(CYCLE[:, :, None, None])[0]
        self.assert_real(pops)
        np.testing.assert_allclose(pops[:, 0, 0], 1 / 3, atol=1e-13)

    def test_steady_state_conserves_population_for_defective_and_slow_chains(self):
        reversible = np.array([[-2.0, 1.0, 0.0], [2.0, -4.0, 2.0], [0.0, 3.0, -2.0]])
        expected = (np.array([0.0, 0.0, 1.0]), np.full(3, 1 / 3),
                    np.array([1.0, 2.0, 3.0]) / 6)
        for paired in (False, True):
            for scale in (1e-12, 1.0, 1e12):
                rates = np.stack([CHAIN, CYCLE, reversible], axis=-1) * scale
                matrix = rates if paired else rates[:, :, None, :]
                with self.subTest(paired=paired, scale=scale):
                    pops = solve_matrix_exponential_steady_state(matrix)[0].reshape(3, 3)
                    self.assert_real(pops)
                    np.testing.assert_allclose(pops, np.stack(expected, axis=1), atol=1e-12)
                    np.testing.assert_allclose(pops.sum(axis=0), 1.0, atol=1e-12)
                    for index in range(3):
                        np.testing.assert_allclose(
                            rates[:, :, index] @ pops[:, index] / scale, 0.0, atol=1e-12
                        )

    def test_steady_state_keeps_accurate_ill_conditioned_eigen_solution(self):
        # An ionization-balance-like chain: its eigenvector condition number is ~3e13,
        # but the eigen projection is accurate to round-off. Replacing it because of the
        # condition number alone made the populations ~6e-12 wrong.
        up = 10.0 ** np.array([4.94, 1.03, -2.94, -6.93, -11.12, -14.1, -18.16, -21.46])
        down = 10.0 ** np.array([-4.73, -4.13, -3.84, -3.5, -3.4, -1.92, -1.92, -1.75])
        size = len(up) + 1
        rates = np.zeros((size, size))
        rates[np.arange(1, size), np.arange(size - 1)] = up      # charge z -> z+1
        rates[np.arange(size - 1), np.arange(1, size)] = down    # charge z+1 -> z
        rates[np.diag_indices(size)] = -rates.sum(axis=0)
        # Exact stationary state of a chain: n_{z+1}/n_z = up_z/down_z.
        log_ratio = np.concatenate([[0.0], np.cumsum(np.log(up) - np.log(down))])
        exact = np.exp(log_ratio - log_ratio.max())
        exact /= exact.sum()
        self.assertGreater(np.linalg.cond(np.linalg.eig(rates)[1]), 1e8)
        pops = solve_matrix_exponential_steady_state(rates[:, :, None, None])[0][:, 0, 0]
        np.testing.assert_allclose(pops, exact, rtol=0, atol=1e-13)

    def test_stored_eigen_solution_is_real_and_can_fall_back(self):
        n0 = np.array([1.0, 0.0, 0.0])
        times = np.array([0.0, 1.0, 2.0])
        for rates in (CYCLE, CHAIN):
            with self.subTest(defective=rates is CHAIN):
                matrix = rates[:, :, None, None]
                _, eigenvalues, eigenvectors = solve_matrix_exponential(matrix, n0, times)
                pops = eval_matrix_exponential_solution(times, n0, eigenvalues, eigenvectors,
                                                        matrix=matrix)
                self.assert_real(pops)
                np.testing.assert_allclose(pops[:, :, 0, 0], reference(rates, n0, times),
                                           atol=1e-12)

    def test_stored_eigen_solution_without_matrix_warns_or_raises(self):
        n0 = np.array([1.0, 0.0, 0.0])
        times = np.array([0.0, 1.0])
        _, eigenvalues, eigenvectors = solve_matrix_exponential(
            CHAIN[:, :, None, None], n0, times
        )
        # Ill-conditioned eigenvectors cannot be checked without the matrix: warn and
        # still return finite, real populations.
        with self.assertWarnsRegex(RuntimeWarning, 'matrix='):
            pops = eval_matrix_exponential_solution(times, n0, eigenvalues, eigenvectors)
        self.assert_real(pops)
        self.assertTrue(np.isfinite(pops).all())
        # A solution that has clearly failed raises.
        with self.assertRaisesRegex(ValueError, 'rate matrix'):
            eval_matrix_exponential_solution(times, n0, np.full_like(eigenvalues, np.nan),
                                             eigenvectors)
        # Well-conditioned complex eigenpairs remain usable without the matrix, silently.
        _, eigenvalues, eigenvectors = solve_matrix_exponential(
            CYCLE[:, :, None, None], n0, times
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pops = eval_matrix_exponential_solution(times, n0, eigenvalues, eigenvectors)
        self.assert_real(pops)
        np.testing.assert_allclose(pops[:, :, 0, 0], reference(CYCLE, n0, times), atol=1e-13)

    def test_energy_derivative_with_complex_eigenvalues(self):
        matrix = CYCLE[:, :, None, None]
        n0 = np.array([0.5, 0.3, 0.2])
        _, eigenvalues, eigenvectors = solve_matrix_exponential(matrix, n0, np.array([0.0]))
        self.assertTrue(np.iscomplexobj(eigenvalues))

        def populations(time):
            return eval_matrix_exponential_solution(time, n0, eigenvalues, eigenvectors,
                                                    matrix=matrix)

        # Only the energy right-hand side is exercised, not the constructor.
        state = energy_balance.__new__(energy_balance)
        state.data = {"user": {
            "mass": 12.0, "ion_mass": 2.0, "ion_dens": 1e13,
            "polarizability": 12.0, "ion_charge_number": 1, "ion_temp": 10.0,
        }}
        state.energy_matrix = CYCLE
        state.debye_length = np.array([1e-5])
        derivative = state._evolve_energy(0.5, np.array([0.75, 0.45, 0.3]), populations)
        self.assertEqual(derivative.shape, (3,))
        self.assertTrue(np.isfinite(derivative).all())


class SteadyStateTests(unittest.TestCase):
    def chain(self, up, down):
        """Birth-death chain: up[k] from state k to k + 1, down[k] from k + 1 to k."""
        n = len(up) + 1
        rates = np.zeros((n, n))
        for k, (u, d) in enumerate(zip(up, down)):
            rates[k + 1, k] += u
            rates[k, k] -= u
            rates[k, k + 1] += d
            rates[k + 1, k + 1] -= d
        exact = np.cumprod(np.r_[1., np.asarray(up) / np.asarray(down)])
        return rates, exact / exact.sum()

    def test_small_populations_keep_their_relative_accuracy(self):
        # The steady state came from the eigenvector of the zero eigenvalue (or a linear
        # solve), accurate only relative to the largest population: for a resolved carbon
        # balance at 10 eV it gave C5+ 2e-5 instead of 3e-16. Populations spanning 300
        # orders of magnitude now keep the relative accuracy of the rates.
        up = np.array([1e-40, 3e-35, 2e-50, 1e-30, 5e-45, 1e-20, 7e-60])
        down = np.array([1., 2., 0.5, 3., 1.5, 0.2, 4.])
        rates, exact = self.chain(up, down)
        self.assertLess(exact.min(), 1e-300 * 1e50)
        pops, _, _ = solve_matrix_exponential_steady_state(rates[:, :, None, None])
        np.testing.assert_allclose(pops[:, 0, 0], exact, rtol=1e-13, atol=0)
        # the same with the largest population in the middle of the chain
        rates, exact = self.chain(down[::-1] * 1e-3, up[::-1] * 1e40)
        pops, _, _ = solve_matrix_exponential_steady_state(rates[:, :, None])
        np.testing.assert_allclose(pops[:, 0], exact, rtol=1e-13, atol=0)

    def test_matrix_that_does_not_conserve(self):
        # A matrix with a loss is solved as before: its eigen projection is not normalized, so
        # the normalized stationary solve is used.
        rates, _ = self.chain(np.array([1., 2.]), np.array([3., 1.]))
        rates[2, 2] -= 1e-3
        self.assertIsNone(_gth_stationary(rates))
        pops, _, _ = solve_matrix_exponential_steady_state(rates[:, :, None, None])
        np.testing.assert_allclose(pops[:, 0, 0], _stationary_population(rates), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()

"""Escape factors for optically thick spectral lines.

A photon of a line from level u to level l can be absorbed again by an atom in level l before it
leaves the plasma. In the escape factor method (Irons, JQSRT 22, 1 (1979); Behringer, IPP reports
10/5 (1997) and 10/11 (1998), as in ADAS214) this is described by two factors of the line:

* the population escape factor, theta_p: the absorption is treated as negative emission, so the
  collisional-radiative populations are calculated with the reduced A-value theta_p * A. It is
  averaged over the volume by default; the centre value is optional.
* the line escape factor, theta_l: the fraction of the line radiance along a line of sight through
  the centre that leaves the plasma, relative to the optically thin radiance of the same
  populations.

Both depend only on the line profile and on the optical depth at the line centre from the centre to
the edge of the plasma, tau0 = kappa(0) R. The absorber density and the line profile are taken as
uniform and the redistribution in frequency as complete; the emission is uniform. Stimulated
emission is neglected. The geometries are an infinite slab of half thickness R (at its mid-plane),
an infinite cylinder of radius R (on its axis) and a sphere of radius R (at its centre).

The default population escape factor is the average over the volume (position='average'), the
mean probability that a photon emitted anywhere in the uniform plasma leaves it. With it a single
zone model conserves photons: the escaping photons theta_p A N_u are those that the excitation
puts in, less the collisional quenching. The value at the centre, as in ADAS214, is smaller,
applies to a local centre closure. Applying it to uniform populations is inconsistent with the
volume-integrated transport. A central-chord radiance above its thin value is not itself a
violation of photon conservation. Neither closure solves the spatial population distribution.

The line profile is a Doppler profile, or a Voigt profile with damping parameter a, the Lorentzian
half width at half maximum in units of the Doppler width dnu_D = nu0 v_th / c, v_th = sqrt(2 k T/m).
"""

import numpy as np
from collections.abc import Mapping
from numbers import Integral
from scipy import constants
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.special import expn, voigt_profile
# Prefer current NumPy APIs, with fallbacks for older releases.
try:
    from numpy import trapezoid as _trapezoid
except ImportError:
    from numpy import trapz as _trapezoid

GEOMETRIES = ('slab', 'cylinder', 'sphere')

#Gauss-Legendre nodes for the angle integral of the cylinder
_THETA_NODES, _THETA_WEIGHTS = np.polynomial.legendre.leggauss(64)


def _physical_array(value, name, positive=False):
    """Validate real finite physical inputs without silently truncating indices or arrays."""
    array = np.asarray(value, dtype=float)
    if(np.any(~np.isfinite(array)) or np.any(array <= 0 if positive else array < 0)):
        raise ValueError(name + (' must be finite and positive' if positive else
                                 ' must be finite and nonnegative'))
    return array


def validate_opacity(opacity, num_levels=None):
    """Return an independent validated opacity mapping. Indices refer to sorted model levels.

    Physical inputs are scalar and fixed across the electron grid. Direct-factor mode requires
    both mappings, with identical transition keys; all unlisted lines remain optically thin.
    """
    if(opacity is None):
        return None
    if(not isinstance(opacity, Mapping)):
        raise ValueError('opacity must be a mapping or None')
    given = 'theta_p' in opacity or 'theta_l' in opacity
    allowed = {'theta_p', 'theta_l'} if given else {
        'length', 'densities', 'temperature', 'mass', 'geometry', 'profile', 'position'}
    if(set(opacity) - allowed):
        raise ValueError('unknown or mixed opacity options: ' + str(sorted(set(opacity) - allowed)))
    required = ('theta_p', 'theta_l') if given else ('length', 'densities', 'temperature', 'mass')
    missing = [key for key in required if key not in opacity]
    if(missing):
        raise ValueError('opacity needs ' + ', '.join(missing))

    def level_index(level):
        if(isinstance(level, (bool, np.bool_)) or not isinstance(level, Integral) or level < 0 or
           (num_levels is not None and level >= num_levels)):
            raise ValueError('opacity level indices must be integers within the model levels')
        return int(level)

    result = dict(opacity)
    if(given):
        for name in required:
            if(not isinstance(opacity[name], Mapping)):
                raise ValueError(name + ' must map (upper, lower) pairs to factors')
            result[name] = {}
            for pair, value in opacity[name].items():
                if(not isinstance(pair, tuple) or len(pair) != 2):
                    raise ValueError(name + ' keys must be (upper, lower) tuples')
                upper, lower = (level_index(level) for level in pair)
                if(upper <= lower):
                    raise ValueError('opacity transitions need upper > lower')
                factor = _physical_array(value, name)
                if(factor.ndim != 0 or factor > 1):
                    raise ValueError(name + ' factors must be scalars between zero and one')
                result[name][upper, lower] = float(factor)
        if(result['theta_p'].keys() != result['theta_l'].keys()):
            raise ValueError('theta_p and theta_l must specify the same transitions')
    else:
        for name in ('length', 'temperature', 'mass'):
            value = _physical_array(opacity[name], name, positive=name != 'length')
            if(value.ndim != 0):
                raise ValueError('opacity ' + name + ' must be scalar')
            result[name] = float(value)
        if(not isinstance(opacity['densities'], Mapping)):
            raise ValueError('opacity densities must map level indices to absolute densities')
        result['densities'] = {}
        for level, density in opacity['densities'].items():
            level = level_index(level)
            value = _physical_array(density, 'absorber density')
            if(value.ndim != 0):
                raise ValueError('absorber densities must be scalar')
            result['densities'][level] = float(value)
        for name, default, choices in (
                ('geometry', 'cylinder', GEOMETRIES),
                ('profile', 'doppler', ('doppler', 'voigt')),
                ('position', 'average', ('centre', 'average'))):
            if(result.get(name, default) not in choices):
                raise ValueError('opacity ' + name + ' must be one of ' + ', '.join(choices))
            result.setdefault(name, default)
    return result


def thermal_speed(temperature, mass):
    """v_th = sqrt(2 k T / m) in cm/s, of the absorbers at temperature (eV) and mass (amu)."""
    return np.sqrt(2*constants.e*_physical_array(temperature, 'temperature', positive=True)/
                   (_physical_array(mass, 'mass', positive=True)*constants.atomic_mass))*100.


def line_profile(x, damping=0.):
    """Line profile per unit x = (nu - nu0)/dnu_D, normalized to one: exp(-x^2)/sqrt(pi) for a
    Doppler profile (damping 0), the Voigt profile with Lorentzian half width damping otherwise."""
    x = np.asarray(x, dtype=float)
    damping = _physical_array(damping, 'damping')
    if(np.any(damping > 0)):
        return voigt_profile(x, 1/np.sqrt(2), damping)
    return np.exp(-x**2)/np.sqrt(np.pi)*np.ones_like(damping)


def optical_depth(wavelength, a_value, g_upper, g_lower, n_lower, length, temperature, mass, damping=0.):
    """Optical depth at the line centre over the length, tau0 = kappa(0) * length.

    kappa(0) = n_lower (g_upper/g_lower) lambda^3 A phi(0) / (8 pi v_th), with phi(0) the line
    profile per unit x at the centre (1/sqrt(pi) for a Doppler profile).

    :param wavelength: wavelength of the line (nm)
    :param a_value: transition probability (s-1)
    :param g_upper: statistical weight of the upper level
    :param g_lower: statistical weight of the lower level
    :param n_lower: density of the absorbing lower level (cm-3)
    :param length: distance from the centre to the edge of the plasma, R (cm)
    :param temperature: temperature of the absorbers (eV)
    :param mass: mass of the absorbers (amu)
    :param damping: Voigt damping parameter, 0 for a Doppler profile
    :returns: tau0
    """
    lam = _physical_array(wavelength, 'wavelength', positive=True)*1e-7
    kappa0 = (_physical_array(n_lower, 'lower-level density')*
              _physical_array(g_upper, 'upper weight', positive=True)/
              _physical_array(g_lower, 'lower weight', positive=True)*
              lam**3*_physical_array(a_value, 'A-value')*line_profile(0., damping)/
              (8*np.pi*thermal_speed(temperature, mass)))
    return kappa0*_physical_array(length, 'length')


def _frequency_grid(tau0):
    """Doppler nodes through the escaping Gaussian tail; Voigt uses infinite-domain quadrature."""
    tau_max = max(float(np.max(tau0)), 1.)
    x_core = np.sqrt(np.log(tau_max)) + 7
    return np.linspace(0., x_core, 4001)


def _profile_integral(tau0, damping, kernel):
    """2 * int_0^inf phi(x) kernel(tau0 phi(x)/phi(0)) dx, for each tau0."""
    tau0 = _physical_array(tau0, 'tau0').ravel()
    damping = _physical_array(damping, 'damping')
    if(damping.ndim != 0):
        raise ValueError('damping must be scalar for escape factors')
    damping = float(damping)
    if(not tau0.size):
        return tau0.copy()
    if(damping > 0):
        # Integrate the complete Voigt wings, mapping [0, infinity) to [0, 1].
        # Breakpoints resolve the core and the optically thick wing even at large tau.
        scale = max(1., damping)
        phi0 = float(line_profile(0., damping))
        values = np.ones_like(tau0)
        for i, tau in enumerate(tau0):
            if(tau == 0):
                continue
            wing = np.sqrt(tau)*np.sqrt(damping/(np.pi*phi0))
            points = sorted(set(x/(scale + x) for x in (scale, max(scale, wing))))

            def integrand(z):
                x = scale*z/(1-z)
                phi = float(line_profile(x, damping))
                return 2*phi*float(kernel(np.asarray(tau*phi/phi0)))*scale/(1-z)**2

            values[i] = quad(integrand, 0., 1., points=points, epsabs=0.,
                             epsrel=2e-8, limit=300)[0]
        return np.clip(values, 0., 1.)
    x = _frequency_grid(tau0)
    phi = line_profile(x, damping)
    ratio = phi/line_profile(0., damping)
    values = kernel(tau0[:, None]*ratio[None, :])
    result = 2*_trapezoid(phi[None, :]*values, x, axis=1)
    return np.where(tau0 == 0, 1., np.clip(result, 0., 1.))


def _escape_kernel(geometry):
    """Probability that a photon emitted at the centre, with optical depth t to the edge along the
    shortest path, leaves the plasma, averaged over the directions."""
    if(geometry == 'sphere'):
        return lambda t: np.exp(-t)
    if(geometry == 'slab'):
        #(1/2) int_-1^1 exp(-t/|mu|) dmu = E2(t)
        return lambda t: expn(2, t)
    if(geometry == 'cylinder'):
        #directions at angle theta to the axis travel t/sin(theta): int_0^pi/2 exp(-t/sin) sin dtheta
        theta = (_THETA_NODES + 1)*np.pi/4
        weights = _THETA_WEIGHTS*np.pi/4
        sin = np.sin(theta)

        def kernel(t):
            return np.tensordot(np.exp(-t[..., None]/sin)*sin, weights, axes=([-1], [0]))
        return kernel
    raise ValueError('geometry must be one of ' + ', '.join(GEOMETRIES) + ', not ' + repr(geometry))


def _mean_escape_kernel(geometry):
    """Escape probability averaged over the uniform plasma, for a monochromatic optical depth t from
    the centre to the edge (half thickness of the slab, radius of the cylinder or sphere)."""
    if(geometry == 'sphere'):
        def kernel(t):
            #3/(4 t^3) [t^2 - 1/2 + (t + 1/2) exp(-2 t)], its series where that cancels
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                exact = 3/(4*t**3)*(t**2 - 0.5 + (t + 0.5)*np.exp(-2*t))
            return np.where(t < 1e-2, 1 - 0.75*t + 0.4*t**2 - t**3/6 + 2*t**4/35, exact)
        return kernel
    if(geometry == 'slab'):
        def kernel(t):
            #(1 - 2 E3(T))/(2 T) for the full thickness T = 2 t
            with np.errstate(divide='ignore', invalid='ignore'):
                exact = (1 - 2*expn(3, 2*t))/(4*t)
            small = 1 + t*(np.log(2*np.maximum(t, 1e-300)) + np.euler_gamma - 1.5) - 2*t**2/3
            return np.where(t < 1e-5, small, exact)
        return kernel
    if(geometry == 'cylinder'):
        log_t, log_k = _cylinder_mean_table()
        interpolation = PchipInterpolator(log_t, log_k, extrapolate=False)

        def kernel(t):
            with np.errstate(divide='ignore'):
                lt = np.log(np.maximum(t, 1e-300))
            inside = np.exp(interpolation(np.clip(lt, log_t[0], log_t[-1])))
            #beyond the table the escape falls as 1/t (from the surface layer)
            beyond = np.exp(log_k[-1] - (lt - log_t[-1]))
            return np.where(t <= np.exp(log_t[0]), 1 - 4*t/3,
                            np.where(lt > log_t[-1], beyond, inside))
        return kernel
    raise ValueError('geometry must be one of ' + ', '.join(GEOMETRIES) + ', not ' + repr(geometry))


_CYLINDER_TABLE = None


def _cylinder_mean_table():
    """Logarithmic cylinder escape table from boundary-integrated monochromatic radiance.

    Integrate along each chord analytically first, then over projected impact parameter
    y=sin(phi) and polar angle theta. This resolves the surface layer even at large t:
    beta(t) = 2/(pi t) int sin(theta)^2 cos(phi) [1-exp(-2t cos(phi)/sin(theta))] dphi dtheta.
    """
    global _CYLINDER_TABLE
    if(_CYLINDER_TABLE is None):
        nodes, weights = np.polynomial.legendre.leggauss(96)
        angle, weight = (nodes + 1)*np.pi/4, weights*np.pi/4
        sin, cos = np.sin(angle), np.cos(angle)
        path = 2*cos[None, :]/sin[:, None]
        weight = weight[:, None]*weight[None, :]*sin[:, None]**2*cos[None, :]
        log_t = np.linspace(np.log(1e-6), np.log(1e4), 401)
        values = np.array([2/(np.pi*np.exp(lt))*np.sum(-np.expm1(-np.exp(lt)*path)*weight)
                           for lt in log_t])
        _CYLINDER_TABLE = (log_t, np.log(values))
    return _CYLINDER_TABLE


def population_escape_factor(tau0, geometry='cylinder', damping=0., position='average'):
    """Population escape factor theta_p: the fraction of the emitted photons that are not absorbed
    again, which multiplies the A-value in the populations.

    :param tau0: optical depth at the line centre from the centre to the edge
    :param geometry: 'slab' (mid-plane), 'cylinder' (axis) or 'sphere' (centre)
    :param damping: Voigt damping parameter, 0 for a Doppler profile
    :param position: 'centre' for the value at the centre (as ADAS214) or 'average' for the
                     average over the volume, which conserves photons in a single zone model
    :returns: theta_p with the shape of tau0
    """
    shape = np.shape(tau0)
    if(position == 'centre'):
        kernel = _escape_kernel(geometry)
    elif(position == 'average'):
        kernel = _mean_escape_kernel(geometry)
    else:
        raise ValueError("position must be 'centre' or 'average', not " + repr(position))
    return _profile_integral(tau0, damping, kernel).reshape(shape)


def line_escape_factor(tau0, damping=0.):
    """Line escape factor theta_l along a line of sight through the centre (length 2 R): the
    escaping radiance of the line relative to its optically thin radiance, for uniform emission.

    :param tau0: optical depth at the line centre from the centre to the edge
    :param damping: Voigt damping parameter, 0 for a Doppler profile
    :returns: theta_l with the shape of tau0
    """
    shape = np.shape(tau0)

    def kernel(t):
        chord = 2*t
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(chord > 1e-8, -np.expm1(-chord)/np.where(chord > 1e-8, chord, 1.), 1. - chord/2)
    return _profile_integral(tau0, damping, kernel).reshape(shape)

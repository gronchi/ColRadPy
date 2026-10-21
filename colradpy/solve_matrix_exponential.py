import warnings

import numpy as np
from scipy.linalg import expm

#Eigenvector matrices with a larger condition number may give a wrong solution (the rate
#matrix is close to defective, e.g. equal rates in a chain). The condition number is only
#a bound, so those temperature/density points are compared with a matrix exponential at
#up to MAX_EXPM_CHECK_TIMES logarithmically spaced requested times. A failed comparison
#recomputes all times in real arithmetic. Sampling is a heuristic, not an accuracy
#guarantee between checks; closed-system conservation is checked at every output time.
EIGENVECTOR_COND_LIMIT = 1.e8
CHECK_TOL = 1.e-6
MAX_EXPM_CHECK_TIMES = 32

#np.linalg.eig returns complex eigenvectors when the rate matrix has complex conjugate
#eigenvalue pairs, which leaves round-off imaginary parts in the populations. Imaginary
#parts up to this fraction of the largest population are dropped; larger ones mean the
#eigen solution failed and it is recomputed with expm.
IMAG_TOL = 1.e-8

#A rate matrix whose columns sum to zero within this fraction of their largest rate conserves
#particles, and its steady state is found by GTH elimination.
CONSERVATION_TOL = 1.e-10


def _inverse_and_condition(eigenvectors):
    """Inverts the eigenvector matrices and estimates their condition numbers.

    The estimate ||V||_F*||V^-1||_F is an upper bound on the 2-norm condition number.
    Exactly singular matrices get a NaN inverse, so they fail the condition check.
    """
    try:
        inverse = np.linalg.inv(eigenvectors)
    except np.linalg.LinAlgError:
        #one singular matrix makes the stacked inverse fail, so invert point by point
        inverse = np.full(np.shape(eigenvectors), np.nan, dtype=np.result_type(eigenvectors, 1.))
        for index in np.ndindex(np.shape(eigenvectors)[:-2]):
            try:
                inverse[index] = np.linalg.inv(eigenvectors[index])
            except np.linalg.LinAlgError:
                pass
    cond = np.linalg.norm(eigenvectors, axis=(-2, -1))*np.linalg.norm(inverse, axis=(-2, -1))
    return inverse, cond


def _is_closed_rate_matrix(matrix):
    """Recognize nonnegative transfer rates with column sums zero to round-off."""
    if np.iscomplexobj(matrix) or not np.isfinite(matrix).all():
        return False
    transfers = matrix.copy()
    np.fill_diagonal(transfers, 0.)
    tolerance = 32*np.finfo(np.result_type(matrix, 1.)).eps*np.abs(matrix).sum(axis=0)
    return (transfers >= 0).all() and (np.abs(matrix.sum(axis=0)) <= tolerance).all()


def _closed_expm_population(matrix, td_n0, source, time):
    """Scale and square a closed-system propagator, preserving its column sums.

    Ordinary expm can accumulate population drift over many relaxation times.
    Normalizing the transition matrix after each square preserves conservation
    without normalizing populations or discarding slow modes. A constant source
    is integrated alongside the propagator using the same doubling steps.
    """
    norm = np.linalg.norm(matrix, 1)
    if time == 0 or norm == 0:
        return td_n0.copy() if source is None else td_n0 + source*time
    squarings = max(0, int(np.ceil(np.log2(norm) + np.log2(abs(time)))))
    step = np.ldexp(float(time), -squarings)
    size = len(td_n0)
    source_scale = 0. if source is None else np.max(np.abs(source))
    if source_scale == 0:
        transition = expm(matrix*step)
        gain = np.zeros(size)
    else:
        augmented = np.zeros((size + 1, size + 1))
        augmented[:size, :size] = matrix*step
        augmented[:size, size] = source/source_scale
        propagated = expm(augmented)
        transition = propagated[:size, :size]
        gain = propagated[:size, size]*(source_scale*step)
    transition /= transition.sum(axis=0)
    for _ in range(squarings):
        gain = np.matmul(transition, gain) + gain
        transition = np.matmul(transition, transition)
        transition /= transition.sum(axis=0)
    return np.matmul(transition, td_n0) + gain


def _expm_populations(matrix, td_n0, source, times):
    """Populations at each time for one rate matrix, computed in real arithmetic.

    A constant source is included by adding one extra state with unit population
    that feeds the source term into the other states. Closed rate matrices use
    conservation-preserving scaling and squaring.
    """
    size = len(td_n0)
    if len(times) == 0:
        return np.empty((size, 0))
    if _is_closed_rate_matrix(matrix):
        return np.stack([_closed_expm_population(matrix, td_n0, source, t)
                         for t in times], axis=1)
    augmented = np.zeros((size + 1, size + 1))
    augmented[:size, :size] = matrix
    if(source is not None):
        augmented[:size, size] = source
    start = np.append(td_n0, 1.)
    return np.stack([np.matmul(expm(augmented*t), start) for t in times], axis=1)[:size]


def _gth_stationary(rates):
    """Steady state of a rate matrix that conserves particles (rates[i, j] >= 0 for i != j is
       the rate from state j to state i, and each column sums to zero) by Grassmann-Taksar-Heyman
       elimination. It uses only the off-diagonal rates and has no subtractions, so every
       population, however small, keeps the relative accuracy of the rates; solutions through
       the eigenvectors or a linear solve keep only that of the largest populations.

       W. K. Grassmann, M. I. Taksar and D. P. Heyman, Regenerative analysis and steady state
       distributions for Markov chains, Operations Research 33, 1107 (1985).

    :param rates: the rate matrix
    :type rates: 2d float array

    :returns: float array or None --the populations, summing to one, or None if the matrix
              does not conserve particles, has a negative rate, has a state without a rate to
              the states before it (which the elimination needs) or overflows
    """
    rates = np.asarray(rates)
    if(np.iscomplexobj(rates)):
        if(np.any(rates.imag != 0)):
            return None
        rates = rates.real
    off = np.array(rates, dtype=float)
    np.fill_diagonal(off, 0.)
    if(np.any(off < 0) or not np.isfinite(rates).all()):
        return None
    column_max = np.max(np.abs(rates), axis=0)
    if(np.any(np.abs(rates.sum(axis=0)) > CONSERVATION_TOL*column_max)):
        return None
    q = off.T.copy() #q[i, j]: rate from state i to state j
    n = len(q)
    with np.errstate(over='ignore', invalid='ignore'):
        for k in range(n - 1, 0, -1):
            total = q[k, :k].sum()
            if(not total > 0):
                return None
            q[:k, k] /= total
            q[:k, :k] += np.outer(q[:k, k], q[k, :k])
        pops = np.zeros(n)
        pops[0] = 1.
        for j in range(1, n):
            pops[j] = pops[:j] @ q[:j, j]
        pops = pops/pops.sum()
    if(not np.isfinite(pops).all()):
        return None
    return pops


def _stationary_population(matrix):
    """Solve a unique stationary distribution without inverting an eigenbasis.

    Scaling each balance equation separately keeps very slow rates in the solve.
    Replace one redundant balance equation with population conservation.
    """
    scales = np.max(np.abs(matrix), axis=1)
    equations = matrix / np.where(scales == 0, 1., scales)[:, None]
    equations[-1] = 1.
    rhs = np.zeros(len(matrix))
    rhs[-1] = 1.
    try:
        return np.linalg.solve(equations, rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError('Cannot determine a unique steady state from this rate matrix.') from error


def _expm_check_indices(times):
    """Select at most 32 requested times, covering zero and nonzero time scales.

    Space targets logarithmically in time, rather than in array index. Include
    both ends of each sign's range and choose the nearest requested times.
    Sorting unique values also supports repeated or unordered output times.
    """
    times = np.asarray(times)
    if len(times) <= MAX_EXPM_CHECK_TIMES:
        return np.arange(len(times))
    values, indices = np.unique(times, return_index=True)
    if len(values) <= MAX_EXPM_CHECK_TIMES:
        return indices
    checks = indices[values == 0].tolist()
    groups = [group for group in (np.flatnonzero(values > 0),
                                  np.flatnonzero(values < 0)[::-1]) if group.size]
    count = (MAX_EXPM_CHECK_TIMES - len(checks)) // len(groups)
    for group in groups:
        logs = np.log(np.abs(values[group]))
        targets = np.linspace(logs[0], logs[-1], min(count, len(group)))
        right = np.minimum(np.searchsorted(logs, targets), len(group) - 1)
        left = np.maximum(right - 1, 0)
        nearest = np.where(np.abs(logs[right] - targets) < np.abs(logs[left] - targets),
                           right, left)
        checks.extend(indices[group[nearest]])
    return np.unique(checks)


def _real_populations(td_pop, matrix, cond, td_n0, source, td_t, td_t_str):
    """Returns real populations, recomputing untrusted temperature/density points with expm.

    A point is untrusted when its populations are not finite, their imaginary parts are
    larger than IMAG_TOL, violate closed-system population conservation, or its
    eigenvector matrix is ill conditioned and the solution differs from a matrix
    exponential at a sampled requested time. Condition checks use at most 32 times
    spanning the time scales; conservation is checked at every requested time.
    """
    if td_pop.size == 0:
        return np.real(td_pop).copy() if np.iscomplexobj(td_pop) else td_pop
    time_axis = td_t_str != 'l'
    for index in np.ndindex(np.shape(matrix)[2:]):
        point = (slice(None), slice(None)) + index if time_axis else (slice(None),) + index
        rates = matrix[(slice(None), slice(None)) + index]
        #td_t is either one set of times, one set per density or one time per density
        if(td_t_str == 't'):
            times = td_t
        elif(td_t_str == 'lt'):
            times = td_t[index[-1]]
        else:
            times = [td_t[index[-1]]]
        pops = td_pop[point]
        scale = np.abs(pops).max() if pops.size else 0.
        trusted = np.isfinite(pops).all() and np.abs(np.imag(pops)).max() <= IMAG_TOL*scale
        eigen = pops if time_axis else pops[:, None]
        if trusted and _is_closed_rate_matrix(rates):
            expected = np.sum(td_n0) + (0. if source is None else np.sum(source))*np.asarray(times)
            mass_scale = np.maximum(np.abs(expected), np.abs(eigen).sum(axis=0))
            trusted = (np.abs(eigen.sum(axis=0) - expected) <= CHECK_TOL*mass_scale).all()
        if trusted and not cond[index] <= EIGENVECTOR_COND_LIMIT:
            checks = _expm_check_indices(times)
            exact = _expm_populations(rates, td_n0, source, np.asarray(times)[checks])
            trusted = (np.abs(eigen[:, checks] - exact).max(axis=0)
                       <= CHECK_TOL*np.abs(exact).max(axis=0)).all()
        if(not trusted):
            exact = _expm_populations(rates, td_n0, source, times)
            td_pop[point] = exact if time_axis else exact[:, 0]
    return np.real(td_pop).copy() if np.iscomplexobj(td_pop) else td_pop


def solve_matrix_exponential(matrix,td_n0,td_t):
    """This definition will solve a 4 dimensional matrix using the matrix expoentiaiation method
       

       R. LeVeque, Finite Difference Methods for Ordinary and Par-
       tial Differential Equations: Steady-State and Time-Dependent
       Problems (Classics in Applied Mathematics Classics in Applied
       Mathemat), Society for Industrial and Applied Mathematics,
       Philadelphia, PA, USA, 2007.



    Args:
      :param matrix: The 4d matrix to be solved
      :type matrix: 4d matrix x,y,temp,dens

      :param td_n0: The initial fractional populations
      :type td_n0: float array

      :param td_t: array of times for the solution
      :type metas: float array


    Returns:
      This returns three arrays the time dependent populations, eigenvals and eigenvectors.
      The populations are real; points whose eigen solution cannot be trusted are
      recomputed with scipy.linalg.expm.

    """
    #calculate the eigenvectors and values for the matrix
    #the last two dimensions have to be the ij parts so we need to transpose matrix
    #string logic for the einsums based on matrix dimensions        
    if(len(np.shape(matrix)) ==4):
        eigenvals, eigenvectors = np.linalg.eig(matrix.transpose(2,3,0,1))

        eval_str = 'klj'
        evec_str = 'klij'
        vt_str1 = 'klj'
        pop_str='ikl'
    if(len(np.shape(matrix)) ==3):
        eigenvals, eigenvectors = np.linalg.eig(matrix.transpose(2,0,1))

        eval_str = 'lj'
        evec_str = 'lij'
        vt_str1 = 'lj'
        pop_str = 'il'

    if(len(np.shape(td_t)) > 1):    
        if(np.shape(td_t)[0] ==1):
            td_t = td_t[0,:]
            td_t_str = 'l'
            vt_str2 = ''
        else:
            td_t_str = 'lt'
            pop_str = pop_str[0:1] +'t' + pop_str[1:]
            vt_str2='t'
    else:
        td_t_str = 't'
        vt_str2 = 't'
        pop_str = pop_str[0:1] +'t' + pop_str[1:]

    inverse, cond = _inverse_and_condition(eigenvectors)
    v0 = np.dot(inverse,td_n0)
    #vt = v0[:,:,:,None]*np.exp(eigenvals[:,:,:,None]*td_t)
    vt = np.einsum(eval_str+ ',' +vt_str1+vt_str2 +'->'+vt_str1+vt_str2,v0,
                   np.exp(np.einsum(eval_str+','+td_t_str+'->'+vt_str1+vt_str2  ,eigenvals,td_t)))

    td_pop = np.einsum(evec_str+','+vt_str1+vt_str2+'->'+pop_str  , eigenvectors,vt)
    td_pop = _real_populations(td_pop, matrix, cond, td_n0, None, td_t, td_t_str)

    return td_pop, eigenvals, eigenvectors


def eval_matrix_exponential_solution(time, n0, eigenvalues, eigenvectors, matrix=None):
    """Evaluates the populations at the given times from a stored eigen solution.

    The populations are real. If the 4d rate matrix is given, points whose eigen
    solution cannot be trusted are recomputed with scipy.linalg.expm. Without that
    matrix, a failed eigen solution (non-finite values or large imaginary parts)
    raises ValueError, and ill-conditioned eigenvectors only give a RuntimeWarning,
    because the condition number is a bound and many such solutions are accurate.
    """
    inverse, cond = _inverse_and_condition(eigenvectors)
    v0 = np.dot(inverse, n0)
    vt = np.einsum('klj,kljt->kljt', v0, np.exp(np.einsum('klj,t->kljt', eigenvalues, time)))
    pops = np.einsum('klij,kljt->itkl', eigenvectors, vt)
    if(matrix is not None):
        pops = _real_populations(pops, matrix, cond, n0, None, time, 't')
    else:
        scale = np.max(np.abs(pops), axis=(0, 1), initial=0.)
        imaginary = np.max(np.abs(pops.imag), axis=(0, 1), initial=0.)
        if(not np.isfinite(pops).all() or np.any(imaginary > IMAG_TOL*scale)):
            raise ValueError('The eigen solution failed; pass the rate matrix so '
                             'these points can be recomputed with expm.')
        if(not np.all(cond <= EIGENVECTOR_COND_LIMIT)):
            warnings.warn('Ill-conditioned eigenvectors: the populations may be inaccurate. '
                          'Pass matrix= to check them against expm.',
                          RuntimeWarning, stacklevel=2)
        if(np.iscomplexobj(pops)):
            pops = pops.real.copy()
    pops[pops < 0] = 0
    return pops


def solve_matrix_exponential_steady_state(matrix):
    """ This definition will solve the steady state solutions of a 4d matrix
       

       R. LeVeque, Finite Difference Methods for Ordinary and Par-
       tial Differential Equations: Steady-State and Time-Dependent
       Problems (Classics in Applied Mathematics Classics in Applied
       Mathemat), Society for Industrial and Applied Mathematics,
       Philadelphia, PA, USA, 2007.

    A rate matrix that conserves particles gets its steady state by GTH elimination
    (_gth_stationary), which keeps the relative accuracy of every population.
    Otherwise only the smallest eigenval survives in steady state,
    so find the smallest eigenval and the corresponding eigenvector
    then solve for the populations with just that eigen value.
    An unreliable eigen projection is replaced by a normalized stationary
    balance solve. If that system is singular, ValueError is raised rather
    than returning an unnormalized or arbitrary population vector.
    The eigenvalues and eigenvectors are returned in every case.



    Args:
      :param matrix: The 4d matrix to be solved
      :type matrix: 4d matrix[x,y,temp,dens]

    Returns:
      This returns three arrays the steady state populations, eigenvals and eigenvectors

    """

    #becuase this is steady state the initial populations don't matter
    td_n0 = np.zeros(np.shape(matrix)[0]) 
    td_n0[0] = 1.
    #sort on the eigenval index to find the longest lived state
    #that will give the equilibrium populuations
    if(len(np.shape(matrix)) ==4):
        
        eigenvals, eigenvectors = np.linalg.eig(matrix.transpose(2,3,0,1))
        axis=2
    if(len(np.shape(matrix)) ==3):
        eigenvals, eigenvectors = np.linalg.eig(matrix.transpose(2,0,1))
        axis=1


    inverse, _ = _inverse_and_condition(eigenvectors)
    v0 = np.dot(inverse,td_n0)

    index = list(np.ix_(*[np.arange(i) for i in eigenvals.shape]))
    index[axis] = np.abs(eigenvals).argsort(axis)
    index = tuple(index)  # Needs to be tuple for multidimensional indexing to work

    if(len(np.shape(matrix)) ==4):    
        ev = eigenvectors.transpose(0,1,3,2)[index]#egienvectors sorted on eigenvals
        ss_pop = np.einsum('kl,klj->klj',v0[index][:,:,0],ev[:,:,0,:]).transpose(2,0,1)
    if(len(np.shape(matrix)) ==3):
        ev = eigenvectors.transpose(0,2,1)[index]#egienvectors sorted on eigenvals
        ss_pop = np.einsum('k,kj->kj',v0[index][:,0],ev[:,0,:]).transpose(1,0)

    #A defective transient subspace can corrupt the projection even when the
    #stationary distribution itself is unique. Check each point independently.
    #A steady state can be verified directly (finite, real, non-negative, normalized and
    #balanced), so the eigenvector condition number, which is only a bound and is large
    #for many accurate ionization balances, is not used here.
    for point in np.ndindex(matrix.shape[2:]):
        index = (slice(None),) + point
        rates = matrix[(slice(None), slice(None)) + point]
        #populations far below the largest are accurate only from the GTH elimination
        gth = _gth_stationary(rates)
        if(gth is not None):
            ss_pop[index] = gth
            continue
        pops = ss_pop[index]
        scale = np.max(np.abs(pops))
        row_scale = np.max(np.abs(rates), axis=1)
        scaled_rates = rates / np.where(row_scale == 0, 1., row_scale)[:, None]
        trusted = (np.isfinite(pops).all()
                   and np.max(np.abs(pops.imag)) <= IMAG_TOL*scale
                   and np.min(pops.real) >= -CHECK_TOL*scale
                   and abs(pops.real.sum() - 1.) <= CHECK_TOL
                   and np.max(np.abs(np.matmul(scaled_rates, pops.real))) <= CHECK_TOL*scale)
        if(not trusted):
            ss_pop[index] = _stationary_population(rates)
    if(np.iscomplexobj(ss_pop)):
        ss_pop = ss_pop.real.copy()

    return ss_pop, eigenvals, eigenvectors






def solve_matrix_exponential_source(matrix, td_n0, source, td_t):
    """This definition will solve a 4 dimensional matrix using the matrix expoentiaiation method
       when a source term is also included
       
       This is a slight modification to R. LeVeque 2007, see Johnson thesis


    Args:
      :param matrix: The 4d matrix to be solved
      :type matrix: 4d matrix x,y,temp,dens

      :param td_n0: The initial fractional populations
      :type td_n0: float array

      :param source: The source of particles into the different states.
      :type source: float array


      :param td_t: array of times for the solution
      :type metas: float array


    Returns:
      This returns three arrays the time dependent populations, eigenvals and eigenvectors.
      The populations are real; points whose eigen solution cannot be trusted are
      recomputed with scipy.linalg.expm.

    """

    #calculate the eigenvectors and values for the matrix
    #the last two dimensions have to be the ij parts so we need to transpose matrix
    #string logic for the einsums based on matrix dimensions        
    if(len(np.shape(matrix)) ==4):
        eigenvals, eigenvectors = np.linalg.eig(matrix.transpose(2,3,0,1))

        eval_str = 'klj'
        evec_str = 'klij'
        vt_str1 = 'klj'
        pop_str='ikl'
    if(len(np.shape(matrix)) ==3):
        eigenvals, eigenvectors = np.linalg.eig(matrix.transpose(2,0,1))

        eval_str = 'lj'
        evec_str = 'lij'
        vt_str1 = 'lj'
        pop_str = 'il'

    if(len(np.shape(td_t)) > 1):    
        if(np.shape(td_t)[0] ==1):
            td_t = td_t[0,:]
            td_t_str = 'l'
            vt_str2 = ''
        else:
            td_t_str = 'lt'
            pop_str = pop_str[0:1] +'t' + pop_str[1:]
            vt_str2='t'
    else:
        td_t_str = 't'
        vt_str2 = 't'
        pop_str = pop_str[0:1] +'t' + pop_str[1:]

    #the math equations for this solution
    #are in Johnson thesis 2020 apprendix A
    inverse, cond = _inverse_and_condition(eigenvectors)
    CC = np.dot(inverse,source)
    V0 = np.dot(inverse,td_n0)

    #each mode follows dv/dt = eigenval*v + CC, so v = V0*exp(eigenval*t) + CC*phi with
    #phi = (exp(eigenval*t) - 1)/eigenval, which is t for a zero eigenvalue. expm1 keeps
    #slow modes accurate, so small eigenvalues no longer have to be treated as zero
    to_vt = eval_str+','+td_t_str+'->'+vt_str1+vt_str2
    eig_t = np.einsum(to_vt, eigenvals, td_t)
    eig_b = np.einsum(to_vt, eigenvals, np.ones_like(td_t))
    t_b = np.einsum(to_vt, np.ones_like(eigenvals), td_t)
    zero = eig_b == 0
    phi = np.where(zero, t_b, np.expm1(eig_t)/np.where(zero, 1., eig_b))

    v_non = np.einsum(eval_str+ ',' +vt_str1+vt_str2 +'->'+vt_str1+vt_str2, V0, np.exp(eig_t)) + \
            np.einsum(eval_str+ ',' +vt_str1+vt_str2 +'->'+vt_str1+vt_str2, CC, phi)

    td_pop = np.einsum(evec_str+','+vt_str1+vt_str2+'->'+pop_str  , eigenvectors,v_non)
    td_pop = _real_populations(td_pop, matrix, cond, td_n0, source, td_t, td_t_str)

    return td_pop, eigenvals,eigenvectors

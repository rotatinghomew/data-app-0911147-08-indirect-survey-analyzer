"""Numerical estimators for the three indirect-response mechanisms.

All functions operate on aggregate answer records or counts. No function
returns a per-respondent latent state.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.optimize import linprog, minimize


# ---------------------------------------------------------------------------
# variance helpers

def cluster_mean_variance(values, clusters):
    """Design-based variance of a mean when whole clusters are resampled.

    Returns ``(variance, number_of_clusters)``. The cluster-robust variance is
    used even when the point estimate is a simple proportion; this keeps the
    variance in line with the register's randomisation unit.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    groups = {}
    for value, cluster in zip(values, clusters):
        total, count = groups.get(cluster, (0.0, 0))
        groups[cluster] = (total + float(value), count + 1)
    g = len(groups)
    n = len(values)
    if g < 2 or n == 0:
        return None, g
    mean = values.mean()
    total = 0.0
    for cluster_total, cluster_n in groups.values():
        total += (cluster_total - cluster_n * mean) ** 2
    return g / (g - 1) * total / (n * n), g


def safe_sqrt(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        return None
    return math.sqrt(value)


# ---------------------------------------------------------------------------
# randomised response (single item)

def rr_marginal(answers, mechanism, clusters=None):
    answers = np.asarray(answers, dtype=float)
    n = int(answers.size)
    force_yes = float(mechanism['force_yes'])
    truth = float(mechanism['truth'])
    if n == 0:
        return {'n': 0, 'q': None, 'estimate': None, 'se': None, 'identified': False,
                'reason': '没有可用回答', 'df': None}
    q = float(answers.mean())
    if truth <= 0:
        return {'n': n, 'q': q, 'estimate': None, 'se': None, 'identified': False,
                'reason': 'truth 概率为零，回答不含敏感比例信息', 'df': None}
    estimate = (q - force_yes) / truth
    variance = q * (1.0 - q) / n
    source = 'binomial'
    df = None
    if clusters is not None:
        cluster_variance, g = cluster_mean_variance(answers, clusters)
        if cluster_variance is not None:
            variance = cluster_variance
            source = 'cluster'
            df = g - 1
    se = safe_sqrt(variance) / truth if variance is not None else None
    return {'n': n, 'q': q, 'estimate': estimate, 'se': se, 'identified': True,
            'variance_source': source, 'df': df,
            'in_parameter_space': 0.0 <= estimate <= 1.0}


def rr_identification_bounds(q_low, q_high, se, force_yes, truth, mechanism_error):
    """Interval implied by the declared kernel and an error radius on q."""
    if truth <= 0:
        return None
    low = (q_low - force_yes - mechanism_error) / truth
    high = (q_high - force_yes + mechanism_error) / truth
    return max(0.0, low), min(1.0, high)


def nonresponse_q_bounds(yes_count, observed, invited):
    """Worst-case bounds for the observed yes-rate when nonresponders differ."""
    if invited is None or invited <= observed:
        return None
    missing = invited - observed
    return yes_count / invited, (yes_count + missing) / invited


# ---------------------------------------------------------------------------
# list experiment

def list_marginal(treated_counts, control_counts, treated_clusters=None, control_clusters=None):
    treated = np.asarray(treated_counts, dtype=float)
    control = np.asarray(control_counts, dtype=float)
    n_t, n_c = int(treated.size), int(control.size)
    if n_t == 0 or n_c == 0:
        return {'n': n_t + n_c, 'estimate': None, 'se': None, 'identified': False,
                'reason': '实验臂或对照臂没有回答', 'n_treated': n_t, 'n_control': n_c,
                'df': None}
    estimate = float(treated.mean() - control.mean())
    var_t = treated.var(ddof=1) / n_t if n_t > 1 else None
    var_c = control.var(ddof=1) / n_c if n_c > 1 else None
    source = 'welch'
    df = min(n_t, n_c) - 1 if min(n_t, n_c) > 1 else None
    if treated_clusters is not None and control_clusters is not None:
        cv_t, g_t = cluster_mean_variance(treated, treated_clusters)
        cv_c, g_c = cluster_mean_variance(control, control_clusters)
        if cv_t is not None and cv_c is not None:
            var_t, var_c = cv_t, cv_c
            source = 'cluster'
            df = min(g_t, g_c) - 1
    se = safe_sqrt(None if var_t is None or var_c is None else var_t + var_c)
    return {'n': n_t + n_c, 'estimate': estimate, 'se': se, 'identified': True,
            'variance_source': source, 'n_treated': n_t, 'n_control': n_c,
            'mean_treated': float(treated.mean()), 'mean_control': float(control.mean()),
            'df': df}


def list_nonresponse_bounds(treated, control, missing):
    """Conservative bounds when missing arm membership is unknown."""
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)
    if treated.size == 0 or control.size == 0 or missing <= 0:
        return None
    t_low = treated.sum() / (treated.size + missing)
    t_high = (treated.sum() + missing) / (treated.size + missing)
    c_low = control.sum() / (control.size + missing)
    c_high = (control.sum() + missing) / (control.size + missing)
    return t_low - c_high, t_high - c_low


# ---------------------------------------------------------------------------
# joint randomised response

def joint_em(counts, transition, max_iter=5000, tolerance=1e-12):
    """Maximum-likelihood latent distribution for an rr_joint mechanism."""
    counts = np.asarray(counts, dtype=float)
    transition = np.asarray(transition, dtype=float)
    latent, _ = transition.shape
    total = counts.sum()
    if total <= 0:
        return np.full(latent, 1.0 / latent)
    pi = np.full(latent, 1.0 / latent)
    for _ in range(max_iter):
        joint = pi[:, None] * transition
        denominator = joint.sum(axis=0, keepdims=True)
        weights = joint / np.where(denominator > 0, denominator, 1.0)
        updated = (weights * counts[None, :]).sum(axis=1) / total
        if np.max(np.abs(updated - pi)) < tolerance:
            return updated
        pi = updated
    return pi


def joint_fit(counts, transition, pi):
    predicted = np.asarray(transition, dtype=float).T @ np.asarray(pi, dtype=float)
    observed = np.asarray(counts, dtype=float)
    total = observed.sum()
    observed = observed / total if total else observed
    return {'predicted': predicted, 'observed': observed,
            'max_abs_deviation': float(np.max(np.abs(predicted - observed))) if total else None}


def joint_gof(counts, transition, pi, tolerance=1e-8):
    """Pearson goodness-of-fit at the MLE, with boundary-aware degrees of freedom.

    The transition matrix is declared by the platform, not estimated.  When the
    MLE sits on the simplex boundary (some latent cells are zero), those cells
    are not free parameters, so the degrees of freedom shrink accordingly.
    """
    counts = np.asarray(counts, dtype=float)
    transition = np.asarray(transition, dtype=float)
    pi = np.asarray(pi, dtype=float)
    total = counts.sum()
    if total <= 0:
        return {'chi_square': None, 'df': None, 'p_value': None}
    predicted = transition.T @ pi * total
    mask = predicted > 1e-9
    if not mask.any():
        return {'chi_square': None, 'df': None, 'p_value': None}
    statistic = float(np.sum((counts[mask] - predicted[mask]) ** 2 / predicted[mask]))
    free_cells = int(np.sum(pi > tolerance))
    free_parameters = max(free_cells - 1, 0)
    df = int(mask.sum()) - 1 - free_parameters
    if df <= 0:
        return {'chi_square': statistic, 'df': df, 'p_value': None,
                'note': '饱和或边界解，无剩余自由度可供拟合检验'}
    return {'chi_square': statistic, 'df': df,
            'p_value': float(stats.chi2.sf(statistic, df))}


def minimum_tv_deviation(counts, transition):
    """Smallest total-variation distortion making the plug-in counts feasible."""
    counts = np.asarray(counts, dtype=float)
    transition = np.asarray(transition, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    observed = counts / total
    latent, responses = transition.shape
    variables = latent + 2 * responses
    objective = np.r_[np.zeros(latent), np.ones(2 * responses)]
    equality = np.hstack([transition.T, np.eye(responses), -np.eye(responses)])
    equality = np.vstack([equality, np.r_[np.ones(latent), np.zeros(2 * responses)]])
    result = linprog(objective, A_eq=equality, b_eq=np.r_[observed, 1.0],
                     bounds=[(0, None)] * variables, method='highs')
    if not result.success:
        return None
    return float(result.fun) / 2.0


def joint_bounds(counts, transition, coefficients, delta=0.0, nonresponse=0):
    """Linear-programming bounds for ``coefficients @ pi``.

    ``delta`` is a total-variation error radius for the declared response
    kernel; ``nonresponse`` adds unknown response counts for missing people.
    """
    counts = np.asarray(counts, dtype=float)
    transition = np.asarray(transition, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    observed = counts.sum()
    total = observed + float(nonresponse)
    latent, responses = transition.shape
    if total <= 0:
        return {'feasible': False, 'bounds': None, 'minimum_delta': None}
    variables = latent + 3 * responses
    # total * T^T pi - missing + u - v = counts
    equality = np.hstack([total * transition.T, -np.eye(responses),
                          np.eye(responses), -np.eye(responses)])
    equality = np.vstack([equality, np.r_[np.ones(latent), np.zeros(3 * responses)]])
    equality = np.vstack([equality, np.r_[np.zeros(latent), np.ones(responses),
                                          np.zeros(2 * responses)]])
    rhs = np.r_[counts, 1.0, float(nonresponse)]
    slack = 2.0 * delta * total
    inequality = np.r_[np.zeros(latent), np.zeros(responses), np.ones(2 * responses)][None, :]
    limits = [(0, None)] * variables
    results = []
    for sign in (1.0, -1.0):
        objective = np.r_[sign * coefficients, np.zeros(3 * responses)]
        results.append(linprog(objective, A_eq=equality, b_eq=rhs, A_ub=inequality,
                               b_ub=[slack], bounds=limits, method='highs'))
    minimum = minimum_tv_deviation(counts, transition)
    if not all(result.success for result in results):
        return {'feasible': False, 'bounds': None, 'minimum_delta': minimum}
    low = max(0.0, float(results[0].fun))
    high = min(1.0, float(-results[1].fun))
    if high < low:
        high = low
    return {'feasible': True, 'bounds': [low, high], 'minimum_delta': minimum}


# ---------------------------------------------------------------------------
# combining version-specific estimates

def combine_estimates(estimates):
    """Inverse-variance pooling with a Cochran Q heterogeneity check."""
    usable = [row for row in estimates if row.get('estimate') is not None
              and row.get('se') not in (None, 0) and math.isfinite(row['se'])]
    if not usable:
        return {'estimate': None, 'se': None, 'q': None, 'df': None, 'p_value': None,
                'heterogeneous': None, 'weights': []}
    weights = [1.0 / (row['se'] ** 2) for row in usable]
    total_weight = sum(weights)
    estimate = sum(w * row['estimate'] for w, row in zip(weights, usable)) / total_weight
    se = math.sqrt(1.0 / total_weight)
    if len(usable) < 2:
        return {'estimate': estimate, 'se': se, 'q': None, 'df': None, 'p_value': None,
                'heterogeneous': None, 'weights': weights}
    q = sum(w * (row['estimate'] - estimate) ** 2 for w, row in zip(weights, usable))
    df = len(usable) - 1
    p_value = float(stats.chi2.sf(q, df))
    return {'estimate': estimate, 'se': se, 'q': q, 'df': df, 'p_value': p_value,
            'heterogeneous': p_value < 0.05, 'weights': weights}


def normal_interval(estimate, se, level=0.95, df=None):
    if estimate is None or se is None or not math.isfinite(se):
        return None
    if df is not None and df > 0:
        quantile = stats.t.ppf(0.5 + level / 2.0, df)
    else:
        quantile = stats.norm.ppf(0.5 + level / 2.0)
    return [estimate - quantile * se, estimate + quantile * se]


def percentile_interval(values, level=0.95):
    clean = np.asarray([value for value in values if value is not None
                        and math.isfinite(float(value))], dtype=float)
    if clean.size < 10:
        return None
    alpha = (1.0 - level) / 2.0
    return [float(np.quantile(clean, alpha)), float(np.quantile(clean, 1.0 - alpha))]


def _profile_nll(pi, counts, transition):
    predicted = transition.T @ pi
    if np.any(predicted <= 0):
        return 1e12
    return -float(np.sum(counts * np.log(predicted)))


def joint_profile_interval(counts, transition, coefficients, level=0.95, grid=41):
    """Profile-likelihood interval for ``coefficients @ pi``.

    The interval inverts a one-degree-of-freedom likelihood-ratio test.  It is
    preferable to a Wald interval here because the maximum-likelihood latent
    distribution often sits on the boundary of the simplex, where the usual
    standard error and symmetric interval are not reliable.
    """
    counts = np.asarray(counts, dtype=float)
    transition = np.asarray(transition, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    key = (counts.tobytes(), transition.tobytes(), coefficients.tobytes(), level, grid)
    cached = _PROFILE_CACHE.get(key)
    if cached is not None:
        return cached
    result = _joint_profile_interval(counts, transition, coefficients, level, grid)
    if len(_PROFILE_CACHE) > 512:
        _PROFILE_CACHE.clear()
    _PROFILE_CACHE[key] = result
    return result


_PROFILE_CACHE = {}


def _joint_profile_interval(counts, transition, coefficients, level, grid):
    counts = np.asarray(counts, dtype=float)
    transition = np.asarray(transition, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    latent = coefficients.size
    if counts.sum() <= 0:
        return None, None
    start = joint_em(counts, transition)
    full = _profile_nll(start, counts, transition)
    threshold = float(stats.chi2.ppf(level, 1))
    bounds = [(0.0, 1.0)] * latent
    constraints = [{'type': 'eq', 'fun': lambda pi: pi.sum() - 1.0}]
    cache = {}

    def statistic(value):
        key = round(float(value), 10)
        if key in cache:
            return cache[key]
        constraint = {'type': 'eq',
                      'fun': lambda pi, target=value: coefficients @ pi - target}
        best = minimize(_profile_nll, start, args=(counts, transition),
                        method='SLSQP', bounds=bounds,
                        constraints=constraints + [constraint],
                        options={'ftol': 1e-10, 'maxiter': 200})
        if not best.success:
            for initial in (np.full(latent, 1.0 / latent), np.zeros(latent)):
                alternative = minimize(_profile_nll, initial, args=(counts, transition),
                                       method='SLSQP', bounds=bounds,
                                       constraints=constraints + [constraint],
                                       options={'ftol': 1e-10, 'maxiter': 200})
                if alternative.success and (not best.success or alternative.fun < best.fun):
                    best = alternative
        if not best.success:
            cache[key] = None
            return None
        value_stat = 2.0 * (best.fun - full)
        result = value_stat if math.isfinite(value_stat) else None
        cache[key] = result
        return result

    targets = np.linspace(0.0, 1.0, grid)
    accepted = []
    for target in targets:
        value = statistic(float(target))
        if value is not None and value <= threshold:
            accepted.append(float(target))
    if not accepted:
        return None, float(coefficients @ start)

    def refine(inside, outside):
        for _ in range(18):
            middle = (inside + outside) / 2.0
            value = statistic(middle)
            if value is not None and value <= threshold:
                inside = middle
            else:
                outside = middle
        return inside

    low = min(accepted)
    high = max(accepted)
    below = [value for value in targets if value < low]
    above = [value for value in targets if value > high]
    if below:
        low = refine(low, float(max(below)))
    if above:
        high = refine(high, float(min(above)))
    return [max(0.0, low), min(1.0, high)], float(coefficients @ start)

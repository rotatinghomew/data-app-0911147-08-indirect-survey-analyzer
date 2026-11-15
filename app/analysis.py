"""Design-aware joint analysis for indirect survey mechanisms.

Public entry points: :func:`capabilities`, :func:`evaluate`, :func:`sensitivity`.
All return JSON-compatible dictionaries.  Nothing here reconstructs a
respondent's latent status; every calculation starts from aggregate counts or
from the observed randomised answers.

Identification vocabulary
------------------------
``identified``            a point estimate follows from the declared design
``partially_identified``  only an interval is justified (kernel error, boundary
                          solution, non-response, or no joint coding)
``unidentified``          the design contains no information for the query
``heterogeneous``         several versions give incompatible point estimates;
                          they are not pooled behind the user's back
``version_specific``      pooling is not justified or was switched off
``assumption_based``      a point estimate exists only because the user enabled
                          an explicit modelling assumption (e.g. independence)
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

from . import domain
from . import mechanisms as mech

IDENTIFIED = 'identified'
PARTIAL = 'partially_identified'
UNIDENTIFIED = 'unidentified'
HETEROGENEOUS = 'heterogeneous'
VERSION_SPECIFIC = 'version_specific'
ASSUMPTION_BASED = 'assumption_based'

DEFAULT_ASSUMPTIONS = {
    'mechanism_error': 0.0,
    'nonresponse': 'ignore',
    'pooling': 'auto',
    'ceiling_floor': 'ignore',
    'clustering': 'register',
    'independence': False,
    'confidence': 0.95,
    'bootstrap_replicates': 200,
    'seed': 20260101,
}

_NOTE_NO_PERSON = (
    '只使用聚合回答；平台不保存个人随机分支或潜在状态，结果不反推任何个人。'
)
_NOTE_NO_INDEPENDENCE = (
    '没有保留随机化联合编码的版本可用于该问题；结果只给出由边际比例推出的范围，'
    '不把范围中点当作重叠比例，也没有默认经历相互独立。'
)


# ---------------------------------------------------------------------------
# settings and design description

def normalize_assumptions(raw=None):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError('假设必须是对象')
    unknown = set(raw) - set(DEFAULT_ASSUMPTIONS)
    if unknown:
        raise ValueError('未知假设字段：' + ', '.join(sorted(unknown)))
    result = dict(DEFAULT_ASSUMPTIONS)
    result.update(raw)
    error = result['mechanism_error']
    if not isinstance(error, (int, float)) or isinstance(error, bool) or not 0 <= error <= 1:
        raise ValueError('mechanism_error 应在 0 与 1 之间')
    result['mechanism_error'] = float(error)
    for key, allowed in (('nonresponse', ('ignore', 'bound')),
                         ('pooling', ('auto', 'pool', 'none')),
                         ('ceiling_floor', ('ignore', 'bound')),
                         ('clustering', ('register', 'class'))):
        if result[key] not in allowed:
            raise ValueError('%s 只能是 %s' % (key, '、'.join(allowed)))
    if not isinstance(result['independence'], bool):
        raise ValueError('independence 必须为布尔值')
    confidence = result['confidence']
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) \
            or not 0.5 < confidence < 1:
        raise ValueError('confidence 应在 0.5 与 1 之间')
    result['confidence'] = float(confidence)
    replicates = result['bootstrap_replicates']
    if not isinstance(replicates, int) or isinstance(replicates, bool) \
            or not 20 <= replicates <= 2000:
        raise ValueError('bootstrap_replicates 应在 20 与 2000 之间')
    if not isinstance(result['seed'], int) or isinstance(result['seed'], bool):
        raise ValueError('seed 必须为整数')
    return result


def _design(data):
    definitions = domain.versions(data)
    items = list(data['sensitive_items'])
    if not items or len(set(items)) != len(items):
        raise ValueError('经历项目应非空且不重复')
    allocation = data.get('allocation') or {}
    invited = data.get('invited_counts') or {}
    for version_id, entry in allocation.items():
        if version_id not in definitions:
            raise ValueError('分配登记引用了未知问卷版本')
        if entry.get('unit') not in ('student', 'class'):
            raise ValueError('分配层级只能是 student 或 class')
    for version_id, value in invited.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError('邀请数必须为非负整数')
        if version_id not in definitions:
            # The invitation register may include versions that returned no
            # records at all. Keep them visible as a design note rather than
            # rejecting otherwise usable data.
            continue
    for version in definitions.values():
        if version['kind'] in ('rr', 'list') and version.get('target') not in items:
            raise ValueError('问卷目标不在经历项目中')
    rows = domain.valid_responses(data)
    groups = sorted({row['group'] for row in rows})
    if not groups:
        raise ValueError('没有可分析的群体')

    by_class = defaultdict(set)
    for row in rows:
        by_class[row['class_id']].add(row['version'])
    mixed_classes = sorted(class_id for class_id, versions in by_class.items()
                           if len(versions) > 1)
    observed = Counter(row['version'] for row in rows)
    version_classes = defaultdict(set)
    for row in rows:
        version_classes[row['version']].add(row['class_id'])
    nesting = {}
    for version_id in definitions:
        classes = version_classes[version_id]
        nested = bool(classes) and not mixed_classes and all(
            len(by_class[class_id]) == 1 for class_id in classes)
        others = set()
        for other_id, other_classes in version_classes.items():
            if other_id != version_id:
                others |= other_classes
        nesting[version_id] = bool(nested and others and not (classes & others))
    return {
        'definitions': definitions,
        'items': items,
        'allocation': allocation,
        'invited': invited,
        'rows': rows,
        'groups': groups,
        'observed': observed,
        'by_class': by_class,
        'mixed_classes': mixed_classes,
        'version_class_nesting': nesting,
        'version_classes': version_classes,
    }


def _version_items(design, version):
    if version['kind'] in ('rr', 'list'):
        return [version['target']]
    latent = version.get('latent_states') or []
    widths = {len(state) for state in latent}
    if len(widths) != 1 or widths.pop() != len(design['items']):
        return []
    if any(set(state) - {'0', '1'} for state in latent):
        return []
    return list(design['items'])


def _registered_unit(design, version_id):
    return (design['allocation'].get(version_id) or {}).get('unit', 'student')


def _cluster_column(design, version_id, settings):
    if _registered_unit(design, version_id) == 'class':
        return 'class_id'
    if settings['clustering'] == 'class':
        return 'class_id'
    return None


def _assumption_notes(design, version_id, settings):
    notes = []
    unit = _registered_unit(design, version_id)
    if settings['clustering'] == 'class' and unit == 'student':
        notes.append('%s 登记为学生级随机；本次按班级聚类计算区间。' % version_id)
    if design['version_class_nesting'].get(version_id):
        notes.append(
            '%s 的版本与班级完全重合：版本差异与班级差异无法分开，'
            '跨版本合并只能视为在登记分配机制下的近似。' % version_id)
    if design['mixed_classes']:
        notes.append('有班级同时包含多个版本（%s）。' % '、'.join(design['mixed_classes']))
    return notes


def _rows_for(design, version_id, group=None):
    return [row for row in design['rows']
            if row['version'] == version_id and (group is None or row['group'] == group)]


def _clip(interval, low=0.0, high=1.0):
    if interval is None:
        return None
    return [max(low, interval[0]), min(high, interval[1])]


def safe_se(interval, level):
    if interval is None:
        return None
    quantile = stats.norm.ppf(0.5 + level / 2.0)
    width = interval[1] - interval[0]
    return None if width <= 0 else width / (2.0 * quantile)


# ---------------------------------------------------------------------------
# single-version estimators

def _rr_source(design, version, group, settings):
    rows = _rows_for(design, version['id'], group)
    answers = np.array([row['answer'] for row in rows], dtype=float)
    column = _cluster_column(design, version['id'], settings)
    clusters = [row['class_id'] for row in rows] if column else None
    fit = mech.rr_marginal(answers, version['mechanism'], clusters)
    base = {'version': version['id'], 'mechanism': 'rr', 'target': version.get('target'),
            'n': fit['n'], 'registered_unit': _registered_unit(design, version['id'])}
    if not fit['identified']:
        return dict(base, status=UNIDENTIFIED, estimable=False, estimate=None,
                    se=None, interval=None, identification_bounds=None,
                    note=fit['reason'], notes=[fit['reason']])
    level = settings['confidence']
    interval = _clip(mech.normal_interval(fit['estimate'], fit['se'], level, fit['df']))
    floor = version['mechanism']['force_yes']
    truth = version['mechanism']['truth']
    q_low = q_high = fit['q']
    bounds = None
    notes = []
    if settings['nonresponse'] == 'bound':
        invited = None if group is not None else design['invited'].get(version['id'])
        pair = mech.nonresponse_q_bounds(float(answers.sum()), len(answers), invited)
        if pair is None:
            notes.append('该版本组内未登记邀请数，最坏情形非响应界限不可用；'
                         '区间仍以观测回答为准。')
        else:
            q_low, q_high = pair
            notes.append('邀请数 %d、观测 %d：未响应者按最不利情形计入。'
                         % (invited, len(answers)))
    if settings['mechanism_error'] > 0 or (q_low, q_high) != (fit['q'], fit['q']):
        low, high = mech.rr_identification_bounds(
            q_low, q_high, fit['se'], floor, truth, settings['mechanism_error'])
        bounds = [max(0.0, low), min(1.0, high)]
    return dict(base, status=IDENTIFIED if bounds is None else PARTIAL,
                estimable=True, estimate=fit['estimate'], se=fit['se'],
                interval=interval, identification_bounds=bounds,
                answer_rate=fit['q'], variance_source=fit['variance_source'],
                in_parameter_space=fit['in_parameter_space'],
                notes=notes + _assumption_notes(design, version['id'], settings),
                note=None)


def _list_source(design, version, group, settings):
    rows = _rows_for(design, version['id'], group)
    treated_rows = [row for row in rows if row['added_sensitive']]
    control_rows = [row for row in rows if not row['added_sensitive']]
    treated = np.array([row['count'] for row in treated_rows], dtype=float)
    control = np.array([row['count'] for row in control_rows], dtype=float)
    column = _cluster_column(design, version['id'], settings)
    treated_clusters = [row['class_id'] for row in treated_rows] if column else None
    control_clusters = [row['class_id'] for row in control_rows] if column else None
    fit = mech.list_marginal(treated, control, treated_clusters, control_clusters)
    base = {'version': version['id'], 'mechanism': 'list', 'target': version.get('target'),
            'n': fit['n'], 'registered_unit': _registered_unit(design, version['id']),
            'n_treated': fit['n_treated'], 'n_control': fit['n_control']}
    if not fit['identified']:
        return dict(base, status=UNIDENTIFIED, estimable=False, estimate=None,
                    se=None, interval=None, identification_bounds=None,
                    note=fit['reason'], notes=[fit['reason']])
    level = settings['confidence']
    interval = _clip(mech.normal_interval(fit['estimate'], fit['se'], level, fit['df']))
    k = int(version['k'])
    floor_t = float(np.mean(treated == 0)) if treated.size else 0.0
    ceil_t = float(np.mean(treated == k + 1)) if treated.size else 0.0
    floor_c = float(np.mean(control == 0)) if control.size else 0.0
    ceil_c = float(np.mean(control == k)) if control.size else 0.0
    boundary_share = floor_t + ceil_t + floor_c + ceil_c
    low = fit['estimate'] - settings['mechanism_error']
    high = fit['estimate'] + settings['mechanism_error']
    widened = settings['mechanism_error'] > 0
    notes = []
    if settings['ceiling_floor'] == 'bound' and boundary_share > 0:
        low -= boundary_share
        high += boundary_share
        widened = True
        notes.append('顶底效应：%.1f%% 的回答落在计数边界；'
                     '按一个条目的最大移动计入范围。' % (100 * boundary_share))
    if settings['nonresponse'] == 'bound':
        invited = None if group is not None else design['invited'].get(version['id'])
        missing = invited - len(rows) if invited is not None else 0
        pair = mech.list_nonresponse_bounds(treated, control, missing)
        if pair is None:
            notes.append('该版本组内未登记邀请数，最坏情形非响应界限不可用；'
                         '区间仍以观测回答为准。')
        else:
            low = min(low, pair[0] - settings['mechanism_error'])
            high = max(high, pair[1] + settings['mechanism_error'])
            widened = True
            notes.append('邀请数 %d、观测 %d：未响应者按最不利情形计入。'
                         % (invited, len(rows)))
    bounds = None
    if widened:
        bounds = [max(0.0, low), min(1.0, high)]
    return dict(base, status=IDENTIFIED if bounds is None else PARTIAL,
                estimable=True, estimate=fit['estimate'], se=fit['se'],
                interval=interval, identification_bounds=bounds,
                mean_treated=fit['mean_treated'], mean_control=fit['mean_control'],
                floor_share={'treated': floor_t, 'control': floor_c},
                ceiling_share={'treated': ceil_t, 'control': ceil_c},
                variance_source=fit['variance_source'],
                in_parameter_space=0.0 <= fit['estimate'] <= 1.0,
                notes=notes + _assumption_notes(design, version['id'], settings),
                note=None)


def _joint_counts(rows, version):
    states = list(version['response_states'])
    counts = np.array([sum(1 for row in rows if row['coded_answer'] == state)
                       for state in states], dtype=float)
    return states, counts


def _coefficients(version, wanted_items, design):
    states = list(version['latent_states'])
    positions = [design['items'].index(item) for item in wanted_items]
    coefficients = []
    for state in states:
        if len(state) != len(design['items']):
            return None
        coefficients.append(1.0 if all(state[position] == '1'
                                        for position in positions) else 0.0)
    return np.array(coefficients)


def _response_floor(transition):
    transition = np.asarray(transition, dtype=float)
    return [float(transition[:, column].min()) for column in range(transition.shape[1])]


def _cluster_replicates(rows, states, transition, coefficients, levels, replicates, seed):
    """Cluster/student Bayesian bootstrap replicates of a latent functional."""
    if not rows or replicates <= 0:
        return []
    rng = np.random.default_rng(seed)
    classes = sorted({row['class_id'] for row in rows})
    values = []
    for _ in range(replicates):
        if levels == 'class_id' and len(classes) >= 2:
            draw = dict(zip(classes, rng.dirichlet(np.ones(len(classes)))))
            weights = [draw[row['class_id']] for row in rows]
        else:
            weights = list(rng.dirichlet(np.ones(len(rows))))
        resampled = np.array([sum(weight for weight, row in zip(weights, rows)
                                  if row['coded_answer'] == state) for state in states])
        if resampled.sum() <= 0:
            continue
        pi = mech.joint_em(resampled, transition)
        values.append(float(coefficients @ pi))
    return values


def _joint_core(rows, version, wanted_items, design):
    states, counts = _joint_counts(rows, version)
    transition = np.asarray(version['transition'], dtype=float)
    coefficients = _coefficients(version, wanted_items, design)
    if coefficients is None or len(states) != transition.shape[0]:
        return {'error': '联合编码的状态无法对应到声明的经历项目'}
    if counts.sum() <= 0:
        return {'error': '没有可用回答'}
    pi = mech.joint_em(counts, transition)
    estimate = float(coefficients @ pi)
    # Boundary means the *queried functional* is pinned at 0 or 1.  Zero cells
    # elsewhere in the latent vector are normal for a saturated encoding and do
    # not by themselves make the estimate unreliable.
    boundary = bool(estimate <= 1e-6 or estimate >= 1.0 - 1e-6)
    fit = mech.joint_fit(counts, transition, pi)
    gof = mech.joint_gof(counts, transition, pi)
    minimum = mech.minimum_tv_deviation(counts, transition)
    floors = _response_floor(transition)
    observed_rates = counts / counts.sum()
    at_floor = [states[index] for index, floor in enumerate(floors)
                if observed_rates[index] <= floor + 1e-9]
    # An observed response share below the kernel's per-category minimum is a
    # diagnostic, not proof of misfit: sampling noise can do this at moderate n.
    # The formal check is the Pearson goodness-of-fit test below, whose degrees
    # of freedom account for the boundary solution.
    below_floor = [{'state': states[index],
                    'observed': float(observed_rates[index]),
                    'floor': float(floors[index])}
                   for index in range(len(states))
                   if observed_rates[index] < floors[index] - 1e-9]
    incompatible = bool(gof.get('p_value') is not None and gof['p_value'] < 0.05)
    return {'states': states, 'counts': counts, 'transition': transition,
            'coefficients': coefficients, 'pi': pi, 'estimate': estimate,
            'boundary': boundary, 'fit': fit, 'gof': gof, 'minimum': minimum,            'floors': floors, 'observed_rates': observed_rates, 'at_floor': at_floor,
            'below_floor': below_floor, 'incompatible': incompatible}


def _joint_source(design, version, wanted_items, group, settings):
    rows = _rows_for(design, version['id'], group)
    levels = _cluster_column(design, version['id'], settings)
    core = _joint_core(rows, version, wanted_items, design)
    base = {'version': version['id'], 'mechanism': 'rr_joint',
            'items': list(wanted_items), 'n': len(rows),
            'registered_unit': _registered_unit(design, version['id'])}
    if core.get('error'):
        return dict(base, status=UNIDENTIFIED, estimable=False, estimate=None,
                    se=None, interval=None, identification_bounds=None,
                    note=core['error'], notes=[core['error']])
    counts = core['counts']
    transition = core['transition']
    coefficients = core['coefficients']
    minimum = core['minimum']
    notes = []
    level = settings['confidence']
    if levels:
        replicates = _cluster_replicates(rows, core['states'], transition, coefficients,
                                         levels, settings['bootstrap_replicates'],
                                         settings['seed'])
        interval = _clip(mech.percentile_interval(replicates, level))
        se = float(np.std(replicates, ddof=1)) if len(replicates) >= 20 \
            else safe_se(interval, level)
        method = 'cluster_bootstrap'
    else:
        interval, _ = mech.joint_profile_interval(counts, transition, coefficients, level)
        interval = _clip(interval)
        se = safe_se(interval, level)
        method = 'profile_likelihood'
    nonresponse = 0
    if settings['nonresponse'] == 'bound':
        invited = None if group is not None else design['invited'].get(version['id'])
        if invited is None:
            notes.append('该版本组内未登记邀请数，最坏情形非响应界限不可用；'
                         '范围仍以观测回答为准。')
        elif invited > int(counts.sum()):
            nonresponse = int(invited - int(counts.sum()))
            notes.append('邀请数 %d、观测 %d：未响应者按最不利情形计入。'
                         % (invited, int(counts.sum())))
    bounds = None
    conditional = None
    if settings['mechanism_error'] > 0 or nonresponse:
        result = mech.joint_bounds(counts, transition, coefficients,
                                   delta=settings['mechanism_error'],
                                   nonresponse=nonresponse)
        if result['feasible']:
            bounds = result['bounds']
        else:
            notes.append('在失真半径 %.3f 内没有与观测计数相容的潜在分布。'
                         % settings['mechanism_error'])
    if core['boundary']:
        notes.append('最大似然解把该问题的比例钉在 0 或 1 的边界上：'
                     '点估计无法区分零与小幅正值，区间偏宽，须与区间一起阅读。')
    incompatible = core['incompatible']
    if incompatible:
        notes.append('声明机制与观测计数的拟合检验不通过（Pearson p=%.3f）；'
                     '该版本的点估计不应直接采用。' % core['gof']['p_value'])
    elif core['below_floor']:
        notes.append('有响应类别的观测比例低于声明机制的概率下限（%s），'
                     '样本较小时可能只是抽样波动；已用拟合检验判断，'
                     '此处不据此判定机制失真。'
                     % '、'.join(entry['state'] for entry in core['below_floor']))
    status = IDENTIFIED
    if bounds is not None or core['boundary'] or incompatible:
        status = PARTIAL
    fit = core['fit']
    gof = core['gof']
    return dict(base, status=status, estimable=True,
                point_identified=not incompatible,
                compatible=not incompatible,
                boundary=core['boundary'],
                estimate=core['estimate'], se=se, interval=interval,
                identification_bounds=bounds,
                conditional_bounds=conditional,
                minimum_mechanism_error=minimum,
                latent_distribution=[float(value) for value in core['pi']],
                latent_states=list(version['latent_states']),
                coefficient=[float(value) for value in coefficients],
                fit={'observed': [float(value) for value in fit['observed']],
                     'predicted': [float(value) for value in fit['predicted']],
                     'max_abs_deviation': fit['max_abs_deviation'],
                     'chi_square': gof['chi_square'], 'df': gof['df'],
                     'p_value': gof['p_value'],
                     'compatible': not incompatible,
                     'response_floor': dict(zip(core['states'], core['floors'])),
                     'at_floor_states': core['at_floor'],
                     'below_floor_states': core['below_floor']},
                variance_source=method,
                notes=notes + _assumption_notes(design, version['id'], settings),
                note=None)


# ---------------------------------------------------------------------------
# pooling

def _estimable(sources):
    return [source for source in sources if source.get('estimable')
            and source.get('point_identified', True)
            and source.get('estimate') is not None]


def _pool(sources, settings):
    if not sources:
        return {'status': UNIDENTIFIED, 'estimate': None, 'se': None,
                'interval': None, 'identification_bounds': None,
                'note': '没有适用于该问题的问卷版本。', 'basis': 'none',
                'heterogeneity': None, 'conflicting_versions': []}
    estimable = _estimable(sources)
    if settings['pooling'] == 'none':
        return {'status': VERSION_SPECIFIC, 'estimate': None, 'se': None,
                'interval': None, 'identification_bounds': None,
                'note': '按设置不合并版本，只保留各版本估计。',
                'basis': 'version_specific', 'heterogeneity': None,
                'conflicting_versions': []}
    if not estimable:
        bounds = [source['identification_bounds'] for source in sources
                  if source.get('identification_bounds')]
        if bounds:
            low = max(entry[0] for entry in bounds)
            high = min(entry[1] for entry in bounds)
            if low <= high:
                return {'status': PARTIAL, 'estimate': None, 'se': None,
                        'interval': None, 'identification_bounds': [low, high],
                        'note': '只能给出识别范围，不能给出点估计。',
                        'basis': 'bounds_only', 'heterogeneity': None,
                        'conflicting_versions': []}
        return {'status': UNIDENTIFIED, 'estimate': None, 'se': None,
                'interval': None, 'identification_bounds': None,
                'note': '现有版本都没有给出可用于该问题的点估计。',
                'basis': 'none', 'heterogeneity': None, 'conflicting_versions': []}
    usable = [source for source in estimable if source.get('se')]
    combined = mech.combine_estimates([{'estimate': source['estimate'],
                                        'se': source['se']} for source in usable]) \
        if len(usable) >= 2 else None
    if settings['pooling'] == 'auto' and combined is not None and combined['heterogeneous']:
        bounds_entries = [source['identification_bounds'] for source in sources
                          if source.get('identification_bounds')]
        if len(bounds_entries) == len(sources) and bounds_entries:
            low = max(entry[0] for entry in bounds_entries)
            high = min(entry[1] for entry in bounds_entries)
            if low <= high:
                return {'status': PARTIAL, 'estimate': None, 'se': None,
                        'interval': None, 'identification_bounds': [low, high],
                        'note': '版本间点估计不一致，但在声明的失真范围内仍有共同交集；'
                                '只报告交集范围，不合并成单一点估计。',
                        'basis': 'heterogeneity_intersection',
                        'heterogeneity': combined,
                        'conflicting_versions': _conflicting(estimable)}
        return {'status': HETEROGENEOUS, 'estimate': None, 'se': None,
                'interval': None, 'identification_bounds': None,
                'note': '版本间点估计不一致（Cochran Q p=%.4f），未自动合并；'
                        '请先判断差异来自设计、机制失真还是版本不可比。'
                        % combined['p_value'],
                'basis': 'heterogeneity_test', 'heterogeneity': combined,
                'conflicting_versions': _conflicting(estimable)}
    if len(estimable) == 1:
        source = estimable[0]
        return {'status': source['status'], 'estimate': source['estimate'],
                'se': source.get('se'), 'interval': source['interval'],
                'identification_bounds': source.get('identification_bounds'),
                'note': '只有一个版本可点估计，未做跨版本合并。',
                'basis': 'single_version', 'heterogeneity': None,
                'weights': [1.0], 'conflicting_versions': []}
    if combined is None:
        # All point estimates lack comparable uncertainty: report version
        # specific estimates rather than fabricating a pooled precision.
        return {'status': VERSION_SPECIFIC, 'estimate': None, 'se': None,
                'interval': None, 'identification_bounds': None,
                'note': '版本估计缺少可比较的不确定性，未合并。',
                'basis': 'missing_uncertainty', 'heterogeneity': None,
                'conflicting_versions': []}
    interval = _clip(mech.normal_interval(combined['estimate'], combined['se'],
                                          settings['confidence']))
    bounds_entries = [(source['version'], source['identification_bounds'])
                      for source in sources if source.get('identification_bounds')]
    bounds = None
    disjoint = []
    if bounds_entries:
        low = max(entry[1][0] for entry in bounds_entries)
        high = min(entry[1][1] for entry in bounds_entries)
        if low <= high:
            bounds = [low, high]
        else:
            disjoint = [version for version, _ in bounds_entries]
    conflicts = _conflicting(estimable)
    status = PARTIAL if (bounds is not None or conflicts or disjoint) else IDENTIFIED
    note = None
    if combined['heterogeneous']:
        note = '按设置强制合并，但异质性检验仍显著（p=%.4f）。' % combined['p_value']
    if disjoint:
        note = ((note + ' ') if note else '') + (
            '各版本在声明失真半径内的识别范围并不相交（%s）；'
            '这不是一个可合并的共同范围，说明版本或机制登记存在问题。'
            % '、'.join(disjoint))
    if conflicts:
        note = ((note + ' ') if note else '') + (
            '以下版本的估计落在合并区间之外：%s。' % '、'.join(conflicts))
    return {'status': status, 'estimate': combined['estimate'],
            'se': combined['se'], 'interval': interval,
            'identification_bounds': bounds, 'note': note,
            'basis': 'inverse_variance', 'heterogeneity': combined,
            'weights': combined['weights'], 'conflicting_versions': conflicts,
            'disjoint_versions': disjoint}


def _conflicting(estimable):
    """Versions whose estimate lies outside the range of the others' intervals."""
    conflicts = []
    for source in estimable:
        if source.get('interval') is None:
            continue
        others = [other for other in estimable if other is not source
                  and other.get('interval')]
        if not others:
            continue
        low = min(other['interval'][0] for other in others)
        high = max(other['interval'][1] for other in others)
        if source['estimate'] < low - 1e-9 or source['estimate'] > high + 1e-9:
            conflicts.append(source['version'])
    return conflicts


# ---------------------------------------------------------------------------
# marginals

def _sources_for_item(design, item, group, settings):
    sources = []
    for version in design['definitions'].values():
        if item not in _version_items(design, version):
            continue
        if not _rows_for(design, version['id'], group):
            continue
        if version['kind'] == 'rr':
            sources.append(_rr_source(design, version, group, settings))
        elif version['kind'] == 'list':
            sources.append(_list_source(design, version, group, settings))
        else:
            sources.append(_joint_source(design, version, [item], group, settings))
    return sources


def _marginal_result(design, item, group, settings):
    sources = _sources_for_item(design, item, group, settings)
    pooled = _pool(sources, settings)
    notes = [_NOTE_NO_PERSON]
    if pooled.get('identification_bounds') is not None:
        notes.append('识别范围来自声明的机制失真、非响应或边界解，不是抽样置信区间。')
    if pooled['status'] == HETEROGENEOUS:
        notes.append('各版本的单项目比例无法合并；可在假设比较中改用“强制合并”'
                     '或“不合并版本”查看后果。')
    return {'estimand': {'type': 'marginal', 'item': item, 'group': group},
            'status': pooled['status'],
            'point_identified': pooled.get('estimate') is not None,
            'estimate': pooled.get('estimate'), 'se': pooled.get('se'),
            'interval': pooled.get('interval'),
            'identification_bounds': pooled.get('identification_bounds'),
            'sources': sources, 'pooling': _without(pooled, 'sources'),
            'version_estimates': [_compact_source(source) for source in sources],
            'notes': [note for note in [pooled.get('note')] if note] + notes}


def _compact_source(source):
    return {'version': source['version'], 'mechanism': source['mechanism'],
            'status': source['status'], 'estimable': source.get('estimable', False),
            'estimate': source.get('estimate'), 'se': source.get('se'),
            'interval': source.get('interval'),
            'identification_bounds': source.get('identification_bounds'),
            'n': source.get('n'), 'registered_unit': source.get('registered_unit'),
            'notes': source.get('notes') or []}


# ---------------------------------------------------------------------------
# joint overlap

def _sources_for_joint(design, items, group, settings):
    sources = []
    for version in design['definitions'].values():
        if version['kind'] != 'rr_joint':
            continue
        if not all(item in _version_items(design, version) for item in items):
            continue
        if _rows_for(design, version['id'], group):
            sources.append(_joint_source(design, version, items, group, settings))
    return sources


def _joint_result(design, items, group, settings):
    items = _dedupe(items)
    joint_sources = _sources_for_joint(design, items, group, settings)
    marginals = {item: _marginal_result(design, item, group, settings) for item in items}
    if joint_sources:
        pooled = _pool(joint_sources, settings)
        notes = [_NOTE_NO_PERSON,
                 '联合比例直接来自保留随机化联合编码的版本，不依赖独立性假设。']
        for source in joint_sources:
            for note in source.get('notes') or []:
                if note not in notes:
                    notes.append(note)
        conflicts = []
        for item in items:
            marginal_result = marginals[item]
            for source in joint_sources:
                if not source.get('latent_states') or not source.get('latent_distribution'):
                    continue
                position = design['items'].index(item)
                implied = sum(value for state, value
                              in zip(source['latent_states'], source['latent_distribution'])
                              if state[position] == '1')
                anchor = marginal_result.get('interval')
                if anchor is not None and (implied < anchor[0] - 1e-9
                                           or implied > anchor[1] + 1e-9):
                    if item not in conflicts:
                        conflicts.append(item)
        if conflicts:
            notes.append('联合编码版本隐含的边际（%s）与单项目版本的合并区间不重叠；'
                         '联合点估计的可信度受机制拟合与版本可比性限制，'
                         '不能只取联合版本而忽略这一冲突。' % '、'.join(conflicts))
        return {'estimand': {'type': 'joint', 'items': items, 'group': group},
                'status': pooled['status'],
                'point_identified': pooled.get('estimate') is not None,
                'estimate': pooled.get('estimate'), 'se': pooled.get('se'),
                'interval': pooled.get('interval'),
                'identification_bounds': pooled.get('identification_bounds'),
                'sources': joint_sources, 'implied_marginals': marginals,
                'pooling': _without(pooled, 'sources'),
                'version_estimates': [_compact_source(source) for source in joint_sources],
                'conflicts': conflicts,
                'notes': [note for note in [pooled.get('note')] if note] + notes}

    notes = [_NOTE_NO_PERSON, _NOTE_NO_INDEPENDENCE]
    usable = [marginals[item] for item in items if marginals[item].get('estimate') is not None]
    if len(usable) < len(items):
        # No point marginals: try to propagate any identification ranges.
        ranges = {}
        for item in items:
            bounds = marginals[item].get('identification_bounds')
            if bounds:
                ranges[item] = bounds
        frechet = None
        if len(ranges) == len(items):
            lows = [ranges[item][0] for item in items]
            highs = [ranges[item][1] for item in items]
            frechet = [max(0.0, sum(lows) - (len(items) - 1)), min(highs)]
        return {'estimand': {'type': 'joint', 'items': items, 'group': group},
                'status': PARTIAL if frechet is not None else UNIDENTIFIED,
                'point_identified': False,
                'estimate': None, 'se': None, 'interval': None,
                'identification_bounds': frechet, 'frechet_bounds': frechet,
                'sources': [], 'implied_marginals': marginals,
                'version_estimates': [], 'conflicts': [],
                'notes': notes + ['至少一个经历项目的边际比例不可点估计，'
                                  '重叠比例只能按可得的识别范围传播，'
                                  '不能给出点估计。']}
    estimates = [marginals[item]['estimate'] for item in items]
    frechet = [max(0.0, sum(estimates) - (len(items) - 1)), min(estimates)]
    result = {'estimand': {'type': 'joint', 'items': items, 'group': group},
              'sources': [], 'implied_marginals': marginals,
              'frechet_bounds': frechet, 'identification_bounds': frechet,
              'version_estimates': [], 'conflicts': [], 'notes': notes}
    if settings['independence']:
        estimate = float(np.prod(estimates))
        variance = 0.0
        for index, item in enumerate(items):
            others = float(np.prod([value for position, value in enumerate(estimates)
                                    if position != index]))
            source_se = next((source.get('se') for source in marginals[item]['sources']
                              if source.get('se') is not None), None)
            if source_se is None:
                variance = None
                break
            variance += (others * source_se) ** 2
        se = math.sqrt(variance) if variance is not None else None
        interval = _clip(mech.normal_interval(estimate, se, settings['confidence'])) \
            if se is not None else None
        result.update({'status': ASSUMPTION_BASED, 'point_identified': True,
                       'assumption_based': True, 'estimate': estimate, 'se': se,
                       'interval': interval})
        result['notes'] = notes + [
            '当前点估计只在显式启用的独立性假设下成立；'
            '即使启用，也同时保留不依赖该假设的范围。']
    else:
        result.update({'status': PARTIAL, 'point_identified': False,
                       'estimate': None, 'se': None, 'interval': None})
        result['notes'] = notes + [
            '两个边际只能给出 Fréchet 范围；现有问卷没有可用于该问题的联合编码，'
            '不能据此断言共同经历比例。范围来自边际的下界与上界，'
            '不是置信区间，也不能用中点代替重叠比例。']
    return result


# ---------------------------------------------------------------------------
# group differences, computed within each version first

def _version_difference(design, version, item, group_a, group_b, settings):
    base = {'version': version['id'], 'mechanism': version['kind'],
            'group_a': group_a, 'group_b': group_b}
    if version['kind'] == 'rr':
        first = _rr_source(design, version, group_a, settings)
        second = _rr_source(design, version, group_b, settings)
    elif version['kind'] == 'list':
        first = _list_source(design, version, group_a, settings)
        second = _list_source(design, version, group_b, settings)
    else:
        wanted = list(item) if isinstance(item, (list, tuple)) else [item]
        first = _joint_source(design, version, wanted, group_a, settings)
        second = _joint_source(design, version, wanted, group_b, settings)
    base['estimates'] = {group_a: _compact_source(first),
                         group_b: _compact_source(second)}
    if not first.get('estimable') or not second.get('estimable') \
            or first.get('estimate') is None or second.get('estimate') is None:
        base.update({'status': UNIDENTIFIED, 'estimable': False, 'estimate': None,
                     'se': None, 'interval': None, 'identification_bounds': None,
                     'note': '该版本至少一个群体没有点估计。'})
        return base
    estimate = first['estimate'] - second['estimate']
    se = None
    if first.get('se') is not None and second.get('se') is not None:
        se = math.sqrt(first['se'] ** 2 + second['se'] ** 2)
    interval = _clip(mech.normal_interval(estimate, se, settings['confidence']),
                     -1.0, 1.0) if se is not None else None
    bounds = None
    if first.get('identification_bounds') and second.get('identification_bounds'):
        bounds = [first['identification_bounds'][0] - second['identification_bounds'][1],
                  first['identification_bounds'][1] - second['identification_bounds'][0]]
    boundary = bool(first.get('boundary') or second.get('boundary'))
    notes = []
    if boundary:
        notes.append('至少一个群体的最大似然解在单纯形边界；'
                     '群体差异的正态近似不稳定，请主要参考区间。')
    base.update({'status': PARTIAL if (bounds is not None or boundary or se is None)
                 else IDENTIFIED,
                 'estimable': se is not None, 'estimate': estimate, 'se': se,
                 'interval': interval, 'identification_bounds': bounds,
                 'boundary': boundary, 'notes': notes,
                 'note': None if se is not None else '缺少可比较的不确定性。'})
    return base


def _difference_sources(design, item, group_a, group_b, settings, joint=False):
    sources = []
    for version in design['definitions'].values():
        if joint:
            if version['kind'] != 'rr_joint':
                continue
            if not all(each in _version_items(design, version) for each in item):
                continue
        else:
            if item not in _version_items(design, version):
                continue
        if not _rows_for(design, version['id'], group_a) \
                or not _rows_for(design, version['id'], group_b):
            continue
        sources.append(_version_difference(design, version, item, group_a, group_b, settings))
    return sources


def _pool_differences(sources, settings):
    usable = [source for source in sources if source.get('estimable')
              and source.get('estimate') is not None and source.get('se')]
    if not sources:
        return {'status': UNIDENTIFIED, 'estimate': None, 'se': None, 'interval': None,
                'identification_bounds': None, 'basis': 'none',
                'note': '没有版本同时覆盖两个群体。', 'heterogeneity': None,
                'conflicting_versions': []}
    if settings['pooling'] == 'none':
        return {'status': VERSION_SPECIFIC, 'estimate': None, 'se': None,
                'interval': None, 'identification_bounds': None,
                'basis': 'version_specific', 'note': '按设置不合并版本。',
                'heterogeneity': None, 'conflicting_versions': []}
    if not usable:
        return {'status': UNIDENTIFIED, 'estimate': None, 'se': None, 'interval': None,
                'identification_bounds': None, 'basis': 'none',
                'note': '没有可点估计的版本差异。', 'heterogeneity': None,
                'conflicting_versions': []}
    combined = mech.combine_estimates([{'estimate': source['estimate'],
                                        'se': source['se']} for source in usable]) \
        if len(usable) >= 2 else None
    if settings['pooling'] == 'auto' and combined and combined['heterogeneous']:
        return {'status': HETEROGENEOUS, 'estimate': None, 'se': None,
                'interval': None, 'identification_bounds': None,
                'basis': 'heterogeneity_test',
                'note': '各版本的群体差异不一致（Cochran Q p=%.4f），未合并。'
                        % combined['p_value'],
                'heterogeneity': combined, 'conflicting_versions': []}
    if len(usable) == 1 or combined is None:
        source = usable[0] if usable else None
        if source is None:
            return {'status': UNIDENTIFIED, 'estimate': None, 'se': None,
                    'interval': None, 'identification_bounds': None,
                    'basis': 'none', 'note': '没有可点估计的版本差异。',
                    'heterogeneity': None, 'conflicting_versions': []}
        return {'status': source['status'], 'estimate': source['estimate'],
                'se': source['se'], 'interval': source['interval'],
                'identification_bounds': source.get('identification_bounds'),
                'basis': 'single_version', 'note': None,
                'heterogeneity': None, 'conflicting_versions': []}
    interval = _clip(mech.normal_interval(combined['estimate'], combined['se'],
                                          settings['confidence']), -1.0, 1.0)
    return {'status': IDENTIFIED if not combined['heterogeneous'] else PARTIAL,
            'estimate': combined['estimate'], 'se': combined['se'],
            'interval': interval, 'identification_bounds': None,
            'basis': 'inverse_variance',
            'note': ('强制合并；异质性检验 p=%.4f。' % combined['p_value'])
                    if combined['heterogeneous'] else None,
            'heterogeneity': combined,
            'conflicting_versions': _conflicting(usable)}


def _group_difference_result(design, query, settings):
    if query['estimand'] == 'marginal':
        sources = _difference_sources(design, query['item'],
                                      query['group_a'], query['group_b'], settings)
        pooled = _pool_differences(sources, settings)
        estimand = {'type': 'group_difference', 'estimand': 'marginal',
                    'item': query['item'], 'group_a': query['group_a'],
                    'group_b': query['group_b']}
        notes = [_NOTE_NO_PERSON,
                 '群体差异在每个版本内部计算，使同一版本的机制偏倚在相减时抵消；'
                 '跨版本合并仍受异质性检验约束。']
    else:
        sources = _difference_sources(design, query['items'],
                                      query['group_a'], query['group_b'], settings,
                                      joint=True)
        pooled = _pool_differences(sources, settings)
        estimand = {'type': 'group_difference', 'estimand': 'joint',
                    'items': query['items'], 'group_a': query['group_a'],
                    'group_b': query['group_b']}
        notes = [_NOTE_NO_PERSON,
                 '共同经历比例的群体差异只在保留联合编码的版本内计算。']
        if not sources:
            # No joint coding: propagate Fréchet ranges from both groups.
            first = _joint_result(design, query['items'], query['group_a'], settings)
            second = _joint_result(design, query['items'], query['group_b'], settings)
            bounds = None
            if first.get('identification_bounds') and second.get('identification_bounds'):
                bounds = [first['identification_bounds'][0] - second['identification_bounds'][1],
                          first['identification_bounds'][1] - second['identification_bounds'][0]]
            return {'estimand': estimand, 'status': PARTIAL if bounds else UNIDENTIFIED,
                    'point_identified': False, 'estimate': None, 'se': None,
                    'interval': None, 'identification_bounds': bounds,
                    'version_differences': [], 'groups': {
                        query['group_a']: _compact_joint(first),
                        query['group_b']: _compact_joint(second)},
                    'notes': notes + ['没有联合编码版本，群体差异只能按各组的'
                                      '不依赖独立性范围相减，不能给出点估计。']}
    notes.extend(_pool_note(pooled))
    return {'estimand': estimand, 'status': pooled['status'],
            'point_identified': pooled.get('estimate') is not None,
            'estimate': pooled.get('estimate'), 'se': pooled.get('se'),
            'interval': pooled.get('interval'),
            'identification_bounds': pooled.get('identification_bounds'),
            'version_differences': [_compact_difference(source) for source in sources],
            'pooling': pooled, 'notes': notes}


def _pool_note(pooled):
    notes = []
    if pooled.get('note'):
        notes.append(pooled['note'])
    if pooled.get('conflicting_versions'):
        notes.append('以下版本的差异落在其他版本区间之外：%s。'
                     % '、'.join(pooled['conflicting_versions']))
    return notes


def _compact_joint(result):
    return {'status': result['status'], 'estimate': result.get('estimate'),
            'interval': result.get('interval'),
            'identification_bounds': result.get('identification_bounds'),
            'frechet_bounds': result.get('frechet_bounds')}


def _compact_difference(source):
    return {'version': source['version'], 'mechanism': source['mechanism'],
            'status': source['status'], 'estimable': source.get('estimable', False),
            'estimate': source.get('estimate'), 'se': source.get('se'),
            'interval': source.get('interval'),
            'identification_bounds': source.get('identification_bounds'),
            'estimates': source.get('estimates'),
            'notes': source.get('notes') or []}


def _without(mapping, key):
    return {name: value for name, value in mapping.items() if name != key}


def _dedupe(items):
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# validation

def _validate_query(design, query):
    if not isinstance(query, dict):
        raise ValueError('查询必须是对象')
    kind = query.get('type')
    if kind not in ('marginal', 'joint', 'group_difference'):
        raise ValueError('查询类型只能是 marginal、joint 或 group_difference')
    if kind == 'marginal':
        item = query.get('item', design['items'][0])
        if item not in design['items']:
            raise ValueError('未知经历项目')
        group = query.get('group')
        if group is not None and group not in design['groups']:
            raise ValueError('未知群体')
        return {'type': 'marginal', 'item': item, 'group': group}
    if kind == 'joint':
        items = _dedupe(query.get('items') or list(design['items'][:2]))
        if len(items) < 2:
            raise ValueError('联合查询需要两个不同经历项目')
        if any(item not in design['items'] for item in items):
            raise ValueError('未知经历项目')
        group = query.get('group')
        if group is not None and group not in design['groups']:
            raise ValueError('未知群体')
        return {'type': 'joint', 'items': items, 'group': group}
    estimand = query.get('estimand')
    if estimand not in ('marginal', 'joint'):
        raise ValueError('群体差异的估计对象只能是 marginal 或 joint')
    groups = design['groups']
    if len(groups) < 2:
        raise ValueError('资料只有一个群体，无法提出群体差异问题')
    group_a = query.get('group_a', groups[0])
    group_b = query.get('group_b', groups[1])
    if group_a not in groups or group_b not in groups or group_a == group_b:
        raise ValueError('请选择两个不同的已知群体')
    if estimand == 'marginal':
        item = query.get('item', design['items'][0])
        if item not in design['items']:
            raise ValueError('未知经历项目')
        return {'type': 'group_difference', 'estimand': 'marginal', 'item': item,
                'group_a': group_a, 'group_b': group_b}
    items = _dedupe(query.get('items') or list(design['items'][:2]))
    if len(items) < 2:
        raise ValueError('联合查询需要两个不同经历项目')
    if any(item not in design['items'] for item in items):
        raise ValueError('未知经历项目')
    return {'type': 'group_difference', 'estimand': 'joint', 'items': items,
            'group_a': group_a, 'group_b': group_b}


# ---------------------------------------------------------------------------
# public API

def evaluate(data, query, assumptions=None):
    design = _design(data)
    settings = normalize_assumptions(assumptions)
    clean = _validate_query(design, query)
    if clean['type'] == 'marginal':
        result = _marginal_result(design, clean['item'], clean['group'], settings)
    elif clean['type'] == 'joint':
        result = _joint_result(design, clean['items'], clean['group'], settings)
    else:
        result = _group_difference_result(design, clean, settings)
    result['assumptions'] = settings
    result['design'] = _design_summary(design)
    return _json_clean(result)


def _design_summary(design):
    versions = []
    for version in design['definitions'].values():
        observed = int(design['observed'].get(version['id'], 0))
        invited = design['invited'].get(version['id'])
        versions.append({
            'id': version['id'], 'kind': version['kind'],
            'targets': _version_items(design, version),
            'registered_unit': _registered_unit(design, version['id']),
            'class_nested': design['version_class_nesting'].get(version['id'], False),
            'observed': observed, 'invited': invited,
            'response_rate': observed / invited if invited else None,
        })
    return {'items': design['items'], 'groups': design['groups'],
            'versions': versions, 'mixed_classes': design['mixed_classes'],
            'caveats': [
                '联合编码的潜在状态按敏感项目登记顺序逐位解释为 0/1；'
                '该顺序是声明的一部分，不是从数据估计的。',
                '分配层级以登记为准；数据中版本与班级完全重合时，'
                '版本差异与班级差异不可分离，已作为设计提示给出。',
            ]}


def capabilities(data, assumptions=None):
    design = _design(data)
    settings = normalize_assumptions(assumptions)
    entries = []
    for item in design['items']:
        entries.append(_capability(
            {'type': 'marginal', 'item': item},
            lambda query, item=item: _marginal_result(design, item, None, settings)))
    for first, second in itertools.combinations(design['items'], 2):
        entries.append(_capability(
            {'type': 'joint', 'items': [first, second]},
            lambda query, pair=(first, second): _joint_result(design, list(pair), None,
                                                              settings)))
    if len(design['groups']) >= 2:
        group_a, group_b = design['groups'][0], design['groups'][1]
        for item in design['items']:
            query = {'type': 'group_difference', 'estimand': 'marginal',
                     'item': item, 'group_a': group_a, 'group_b': group_b}
            entries.append(_capability(
                query, lambda query, item=item: _group_difference_result(
                    design, {'type': 'group_difference', 'estimand': 'marginal',
                             'item': item, 'group_a': group_a, 'group_b': group_b},
                    settings)))
        for first, second in itertools.combinations(design['items'], 2):
            query = {'type': 'group_difference', 'estimand': 'joint',
                     'items': [first, second], 'group_a': group_a, 'group_b': group_b}
            entries.append(_capability(
                query, lambda query, pair=(first, second): _group_difference_result(
                    design, {'type': 'group_difference', 'estimand': 'joint',
                             'items': list(pair), 'group_a': group_a, 'group_b': group_b},
                    settings)))
    return _json_clean({'assumptions': settings, 'design': _design_summary(design),
                        'queries': entries,
                        'diagnostics': _diagnostics(design),
                        'recommendations': _recommendations(design)})


def _capability(query, builder):
    result = builder(query)
    sources = result.get('sources') or []
    if not sources and result.get('version_differences'):
        sources = result['version_differences']
    estimand = result.get('estimand', {})
    needs_joint = estimand.get('type') == 'joint' or estimand.get('estimand') == 'joint'
    mechanisms = {source.get('mechanism') for source in sources if source.get('mechanism')}
    if sources and needs_joint and 'rr_joint' in mechanisms:
        basis = 'joint_coding'
    elif sources:
        basis = 'single_item_versions'
    elif result.get('frechet_bounds') is not None:
        basis = 'marginals_only'
    else:
        basis = 'none'
    versions = sorted({source['version'] for source in sources})
    version_estimates = [
        {'version': source['version'],
         'mechanism': source.get('mechanism'),
         'status': source['status'],
         'estimate': source.get('estimate'),
         'interval': source.get('interval'),
         'identification_bounds': source.get('identification_bounds')}
        for source in sources if source.get('estimate') is not None]
    return {'query': query, 'status': result['status'],
            'point_identified': result.get('point_identified', False),
            'estimate': result.get('estimate'), 'interval': result.get('interval'),
            'identification_bounds': result.get('identification_bounds'),
            'frechet_bounds': result.get('frechet_bounds'),
            'versions': versions,
            'version_estimates': version_estimates,
            'mechanisms': sorted(mechanisms),
            'basis': basis,
            'summary': _summarize_status(result['status'], basis,
                                         bool(result.get('identification_bounds')),
                                         bool(result.get('frechet_bounds'))),
            'notes': result.get('notes') or [],
            'blocking_versions': [
                {'version': source['version'], 'status': source['status'],
                 'notes': source.get('notes') or []}
                for source in sources if source['status'] != IDENTIFIED]}


def _summarize_status(status, basis, has_bounds=False, has_frechet=False):
    if status == IDENTIFIED:
        return '可点估计'
    if status == PARTIAL:
        if basis == 'marginals_only' or has_frechet:
            return '只能给出重叠比例的范围'
        if has_bounds:
            return '有点估计，同时给出识别范围'
        return '有点估计，但须结合边界/机制失真解读'
    if status == HETEROGENEOUS:
        return '版本间不一致，不能合并'
    if status == VERSION_SPECIFIC:
        return '只能按版本分别报告'
    if status == ASSUMPTION_BASED:
        return '仅在显式假设下可点估计'
    return '现有问卷不能回答'


def _diagnostics(design):
    diagnostics = []
    for version in design['definitions'].values():
        observed = design['observed'].get(version['id'], 0)
        invited = design['invited'].get(version['id'])
        if invited and observed < invited:
            diagnostics.append('%s 回收 %d/%d（%.1f%%）；班级名单是否完整未单独登记，'
                               '无法区分班级整体缺失与个人未响应。'
                               % (version['id'], observed, invited,
                                  100.0 * observed / invited))
        if design['version_class_nesting'].get(version['id']):
            diagnostics.append('%s 的版本与班级完全重合，版本差异与班级差异不可分离。'
                               % version['id'])
    if design['mixed_classes']:
        diagnostics.append('有班级同时包含多个版本（%s）：按班级聚类时'
                           '版本效应与班级效应会混在一起。'
                           % '、'.join(design['mixed_classes']))
        if version['kind'] == 'list':
            rows = _rows_for(design, version['id'])
            treated = [row['count'] for row in rows if row['added_sensitive']]
            control = [row['count'] for row in rows if not row['added_sensitive']]
            k = int(version['k'])
            total = len(treated) + len(control)
            if total:
                boundary = (sum(count == 0 for count in treated + control) +
                            sum(count == k + 1 for count in treated) +
                            sum(count == k for count in control)) / total
                if boundary:
                    diagnostics.append('%s 有 %.1f%% 的回答落在计数边界，'
                                       '顶底效应可能压缩均值差。'
                                       % (version['id'], 100 * boundary))
        if version['kind'] == 'rr_joint':
            states, counts = _joint_counts(_rows_for(design, version['id']), version)
            if counts.sum():
                core = _joint_core(_rows_for(design, version['id']), version,
                                   _version_items(design, version), design)
                gof = core.get('gof') or {}
                if gof.get('p_value') is not None and gof['p_value'] < 0.05:
                    diagnostics.append(
                        '%s 的声明机制与观测计数不符（Pearson p=%.3f）；'
                        '该版本联合点估计须结合识别范围阅读。'
                        % (version['id'], gof['p_value']))
                if core.get('boundary'):
                    diagnostics.append(
                        '%s 的最大似然解落在单纯形边界（部分潜在状态比例为零），'
                        '点估计接近零不能区分于小幅正值。' % version['id'])
    diagnostics.append('问卷未登记遵从情况；估计按“机制如声明执行”解释，'
                       '无法从现有记录识别直接作答或替换随机分支。')
    return diagnostics


def _recommendations(design):
    recommendations = []
    joint_versions = [version for version in design['definitions'].values()
                      if version['kind'] == 'rr_joint']
    covered = set()
    for version in joint_versions:
        covered |= set(_version_items(design, version))
    missing = [item for item in design['items'] if item not in covered]
    if missing:
        recommendations.append('以下经历没有联合编码版本，重叠比例只能给出范围：%s。'
                               '若下一轮需要点估计，应补充同时包含这些项目的联合编码版本。'
                               % '、'.join(missing))
    for version in joint_versions:
        rows = _rows_for(design, version['id'])
        states, counts = _joint_counts(rows, version)
        if counts.sum() == 0:
            continue
        core = _joint_core(rows, version, _version_items(design, version), design)
        gof = core.get('gof') or {}
        if core.get('boundary'):
            recommendations.append('%s 的联合估计落在单纯形边界，零与小幅正值不可区分。'
                                   '若下一轮需要更稳的共同经历比例，'
                                   '应提高联合编码的区分度或扩大样本。' % version['id'])
        elif gof.get('p_value') is not None and gof['p_value'] < 0.05:
            recommendations.append('%s 的声明机制与观测计数不符（Pearson p=%.3f）。'
                                   '应先核对机制登记或换用更宽松的转移矩阵，'
                                   '再决定是否继续使用该版本的联合估计。'
                                   % (version['id'], gof['p_value']))
    for version in design['definitions'].values():
        invited = design['invited'].get(version['id'])
        observed = design['observed'].get(version['id'], 0)
        if invited and observed / invited < 0.9:
            recommendations.append('%s 回收率为 %.1f%%。若下一轮要比较可合并范围，'
                                   '应按班级登记邀请与回收，并记录未响应原因。'
                                   % (version['id'], 100 * observed / invited))
            break
    if any(version['kind'] == 'list' for version in design['definitions'].values()):
        recommendations.append('名单计数的顶底与不遵从会改变可解释范围；'
                               '下一轮可增加对照条目或记录边界回答，以减少边界假设依赖。')
    return recommendations


# ---------------------------------------------------------------------------
# scenario comparison

_DEFAULT_SCENARIOS = [
    {'label': '基准（按登记机制，忽略未响应）', 'change': {}},
    {'label': '机制失真 2%', 'change': {'mechanism_error': 0.02}},
    {'label': '机制失真 5%', 'change': {'mechanism_error': 0.05}},
    {'label': '机制失真 10%', 'change': {'mechanism_error': 0.10}},
    {'label': '未响应按最不利', 'change': {'nonresponse': 'bound'}},
    {'label': '按班级聚类计算区间', 'change': {'clustering': 'class'}},
    {'label': '不合并版本', 'change': {'pooling': 'none'}},
    {'label': '强制合并版本', 'change': {'pooling': 'pool'}},
    {'label': '顶底按一个条目移动', 'change': {'ceiling_floor': 'bound'}},
    {'label': '显式独立性假设', 'change': {'independence': True}},
]


def sensitivity(data, query, assumptions=None, scenarios=None):
    design = _design(data)
    settings = normalize_assumptions(assumptions)
    clean = _validate_query(design, query)
    if scenarios is None:
        scenarios = _DEFAULT_SCENARIOS
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError('情景必须是至少包含一项的列表')
    rows = []
    for entry in scenarios:
        if not isinstance(entry, dict) or not isinstance(entry.get('label'), str) \
                or not isinstance(entry.get('change', {}), dict):
            raise ValueError('情景需包含 label 与 change 对象')
        merged = dict(settings)
        merged.update(entry.get('change', {}))
        try:
            result = evaluate(data, clean, merged)
            rows.append({'label': entry['label'], 'change': entry.get('change', {}),
                         'status': result['status'],
                         'point_identified': result.get('point_identified', False),
                         'estimate': result.get('estimate'),
                         'interval': result.get('interval'),
                         'identification_bounds': result.get('identification_bounds'),
                         'notes': result.get('notes')})
        except ValueError as error:
            rows.append({'label': entry['label'], 'change': entry.get('change', {}),
                         'status': 'not_applicable', 'point_identified': False,
                         'estimate': None, 'interval': None,
                         'identification_bounds': None, 'notes': [str(error)]})
    return _json_clean({'query': clean, 'assumptions': settings, 'scenarios': rows})


# ---------------------------------------------------------------------------
# JSON hygiene

def _json_clean(value):
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 10)
    return value

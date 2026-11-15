import copy
import json
import math
import unittest
from pathlib import Path

from app import analysis, mechanisms
from app.service import load_study

ROOT = Path(__file__).resolve().parents[1]


def cases():
    return {case['name']: case for case in
            json.loads((ROOT / 'data/analysis-cases.json').read_text(encoding='utf-8'))}


class MechanismTests(unittest.TestCase):
    def test_rr_estimator_inverts_declared_kernel(self):
        fit = mechanisms.rr_marginal([1.0] * 60 + [0.0] * 40,
                                     {'force_yes': 0.2, 'force_no': 0.2, 'truth': 0.6})
        self.assertAlmostEqual(fit['q'], 0.6)
        self.assertAlmostEqual(fit['estimate'], (0.6 - 0.2) / 0.6)
        self.assertTrue(fit['in_parameter_space'])

    def test_rr_without_truth_is_unidentified(self):
        fit = mechanisms.rr_marginal([1.0, 0.0, 1.0, 0.0],
                                     {'force_yes': 0.5, 'force_no': 0.5, 'truth': 0.0})
        self.assertFalse(fit['identified'])
        self.assertIsNone(fit['estimate'])

    def test_list_difference_matches_manual_example(self):
        fit = mechanisms.list_marginal([1.0, 2.0, 2.0, 1.0], [0.0, 1.0, 2.0, 1.0])
        self.assertAlmostEqual(fit['estimate'], 0.5)
        self.assertEqual(fit['mean_treated'], 1.5)
        self.assertEqual(fit['mean_control'], 1.0)

    def test_joint_em_recovers_latent_distribution(self):
        transition = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1],
                      [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]
        latent = [0.4, 0.3, 0.2, 0.1]
        counts = [800 * value for value in
                  mechanisms.np.asarray(transition).T.dot(latent)]
        recovered = mechanisms.joint_em(counts, transition)
        for expected, actual in zip(latent, recovered):
            self.assertAlmostEqual(expected, actual, places=5)

    def test_minimum_tv_deviation_zero_for_compatible_counts(self):
        transition = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1],
                      [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]
        counts = mechanisms.np.asarray(transition).T.dot([0.4, 0.3, 0.2, 0.1]) * 1000
        self.assertLess(mechanisms.minimum_tv_deviation(counts, transition), 1e-9)

    def test_minimum_tv_deviation_positive_for_impossible_counts(self):
        transition = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1],
                      [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]
        # every response state has probability at least 0.1 under this kernel
        counts = [70.0, 20.0, 5.0, 5.0]
        self.assertGreater(mechanisms.minimum_tv_deviation(counts, transition), 0.01)

    def test_joint_bounds_widen_with_distortion(self):
        transition = mechanisms.np.asarray([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1],
                                            [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]])
        counts = transition.T.dot([0.4, 0.3, 0.2, 0.1]) * 400
        coefficient = mechanisms.np.array([0.0, 0.0, 0.0, 1.0])
        narrow = mechanisms.joint_bounds(counts, transition, coefficient, delta=0.02)
        wide = mechanisms.joint_bounds(counts, transition, coefficient, delta=0.10)
        self.assertTrue(narrow['feasible'])
        self.assertTrue(wide['feasible'])
        self.assertLessEqual(wide['bounds'][0], narrow['bounds'][0] + 1e-9)
        self.assertGreaterEqual(wide['bounds'][1], narrow['bounds'][1] - 1e-9)

    def test_combine_flags_heterogeneous_estimates(self):
        combined = mechanisms.combine_estimates([
            {'estimate': 0.2, 'se': 0.01}, {'estimate': 0.6, 'se': 0.01}])
        self.assertTrue(combined['heterogeneous'])
        self.assertLess(combined['p_value'], 0.001)

    def test_combine_accepts_consistent_estimates(self):
        combined = mechanisms.combine_estimates([
            {'estimate': 0.40, 'se': 0.05}, {'estimate': 0.41, 'se': 0.05}])
        self.assertFalse(combined['heterogeneous'])
        self.assertAlmostEqual(combined['estimate'], 0.405)


class MarginalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.study = load_study()

    def test_truthful_rr_degenerates_to_plain_proportion(self):
        study = cases()['truthful_rr']['study']
        result = analysis.evaluate(study, {'type': 'marginal', 'item': 'overload'},
                                   {'pooling': 'none'})
        source = next(entry for entry in result['sources'] if entry['version'] == 'rr-1')
        self.assertAlmostEqual(source['estimate'], 0.4375)

    def test_zero_truth_has_no_point_estimate(self):
        study = cases()['no_truth']['study']
        result = analysis.evaluate(study, {'type': 'marginal', 'item': 'overload'})
        self.assertEqual(result['status'], analysis.UNIDENTIFIED)
        self.assertIsNone(result['estimate'])

    def test_list_manual_example(self):
        study = cases()['list_manual']['study']
        result = analysis.evaluate(study, {'type': 'marginal', 'item': 'overload'})
        self.assertAlmostEqual(result['estimate'], 0.5)

    def test_versions_are_not_averaged_when_heterogeneous(self):
        result = analysis.evaluate(self.study, {'type': 'marginal', 'item': 'overload'})
        self.assertEqual(result['status'], analysis.HETEROGENEOUS)
        self.assertIsNone(result['estimate'])
        self.assertTrue(result['pooling']['heterogeneity']['heterogeneous'])

    def test_forced_pooling_is_marked(self):
        result = analysis.evaluate(self.study, {'type': 'marginal', 'item': 'overload'},
                                   {'pooling': 'pool'})
        self.assertIsNotNone(result['estimate'])
        self.assertIn('强制合并', result['pooling']['note'])

    def test_version_specific_pooling_returns_no_pooled_estimate(self):
        result = analysis.evaluate(self.study, {'type': 'marginal', 'item': 'overload'},
                                   {'pooling': 'none'})
        self.assertEqual(result['status'], analysis.VERSION_SPECIFIC)
        self.assertIsNone(result['estimate'])
        self.assertGreaterEqual(len(result['sources']), 2)


class JointAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.study = load_study()

    def test_joint_estimate_uses_randomized_coding(self):
        result = analysis.evaluate(self.study, {'type': 'joint',
                                                'items': ['overload', 'missed_support']})
        self.assertIsNotNone(result['estimate'])
        self.assertEqual([source['mechanism'] for source in result['sources']],
                         ['rr_joint'])

    def test_no_independence_default_without_joint_version(self):
        study = cases()['marginals_without_joint']['study']
        result = analysis.evaluate(study, {'type': 'joint',
                                           'items': ['overload', 'missed_support']})
        self.assertEqual(result['status'], analysis.PARTIAL)
        self.assertIsNone(result['estimate'])
        low, high = result['frechet_bounds']
        marginals = [result['implied_marginals'][item]['estimate']
                     for item in ('overload', 'missed_support')]
        self.assertAlmostEqual(low, max(0.0, sum(marginals) - 1))
        self.assertAlmostEqual(high, min(marginals))
        self.assertTrue(any('相互独立' in note or '独立性' in note
                            for note in result['notes']))

    def test_explicit_independence_is_labelled_and_keeps_bounds(self):
        study = cases()['marginals_without_joint']['study']
        result = analysis.evaluate(study, {'type': 'joint',
                                           'items': ['overload', 'missed_support']},
                                   {'independence': True})
        self.assertEqual(result['status'], analysis.ASSUMPTION_BASED)
        self.assertTrue(result['assumption_based'])
        marginals = [result['implied_marginals'][item]['estimate']
                     for item in ('overload', 'missed_support')]
        self.assertAlmostEqual(result['estimate'], marginals[0] * marginals[1])
        self.assertIsNotNone(result['frechet_bounds'])

    def test_removing_only_joint_source_changes_the_answer(self):
        """Dropping the joint version must not keep the same certainty."""
        with_joint = analysis.evaluate(
            self.study, {'type': 'joint', 'items': ['overload', 'missed_support']})
        without = copy.deepcopy(self.study)
        without['versions'] = [version for version in without['versions']
                               if version['kind'] != 'rr_joint']
        without['allocation'].pop('joint-1', None)
        without['invited_counts'].pop('joint-1', None)
        without['responses'] = [row for row in without['responses']
                                if row['version'] != 'joint-1']
        alternative = analysis.evaluate(
            without, {'type': 'joint', 'items': ['overload', 'missed_support']})
        self.assertIsNotNone(with_joint['estimate'])
        self.assertIsNone(alternative['estimate'])
        self.assertEqual(alternative['status'], analysis.UNIDENTIFIED)
        self.assertIsNone(alternative['identification_bounds'])
        self.assertIsNone(alternative['frechet_bounds'])

    def test_joint_version_never_returns_personal_state(self):
        result = analysis.evaluate(self.study, {'type': 'joint',
                                                'items': ['overload', 'missed_support']})
        text = json.dumps(result, ensure_ascii=False)
        self.assertNotIn('sensitive_status', text)
        self.assertNotIn('random_branch', text)
        for source in result['sources']:
            self.assertNotIn('response_id', source)

    def test_fitted_joint_kernel_passes_goodness_of_fit(self):
        result = analysis.evaluate(self.study, {'type': 'joint',
                                                'items': ['overload', 'missed_support']})
        fit = result['sources'][0]['fit']
        self.assertGreater(fit['p_value'], 0.05)
        self.assertEqual(fit['df'], 2)
        self.assertTrue(fit['compatible'])

    def test_incompatible_joint_counts_remove_point_estimate(self):
        study = copy.deepcopy(self.study)
        for row in study['responses']:
            if row['version'] == 'joint-1':
                row['coded_answer'] = '00'
        result = analysis.evaluate(study, {'type': 'joint',
                                           'items': ['overload', 'missed_support']})
        self.assertEqual(result['status'], analysis.UNIDENTIFIED)
        self.assertIsNone(result['estimate'])
        self.assertFalse(result['sources'][0]['fit']['compatible'])
        self.assertTrue(any('拟合检验' in note for note in result['notes']))

    def test_boundary_joint_keeps_point_estimate_with_warning(self):
        result = analysis.evaluate(self.study, {'type': 'joint',
                                                'items': ['overload', 'missed_support']})
        source = result['sources'][0]
        self.assertTrue(source['boundary'])
        self.assertEqual(result['status'], analysis.PARTIAL)
        self.assertIsNotNone(result['estimate'])
        self.assertTrue(any('边界' in note for note in result['notes']))


class GroupDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.study = load_study()

    def test_difference_is_computed_within_version(self):
        result = analysis.evaluate(
            self.study, {'type': 'group_difference', 'estimand': 'marginal',
                         'item': 'overload', 'group_a': 'arts', 'group_b': 'science'})
        for source in result['version_differences']:
            estimates = source['estimates']
            self.assertAlmostEqual(
                source['estimate'],
                estimates['arts']['estimate'] - estimates['science']['estimate'])

    def test_unknown_group_is_rejected(self):
        with self.assertRaises(ValueError):
            analysis.evaluate(self.study, {'type': 'group_difference',
                                           'estimand': 'marginal', 'item': 'overload',
                                           'group_a': 'arts', 'group_b': 'nope'})

    def test_joint_difference_without_joint_version_stays_bounded(self):
        study = cases()['marginals_without_joint']['study']
        result = analysis.evaluate(
            study, {'type': 'group_difference', 'estimand': 'joint',
                    'items': ['overload', 'missed_support'],
                    'group_a': 'arts', 'group_b': 'science'})
        self.assertFalse(result['point_identified'])
        self.assertIsNone(result['estimate'])
        self.assertIsNotNone(result['identification_bounds'])


class AssumptionTests(unittest.TestCase):
    def setUp(self):
        self.study = load_study()
        self.query = {'type': 'marginal', 'item': 'overload'}

    def test_nonresponse_widens_identification_bounds(self):
        base = analysis.evaluate(self.study, self.query,
                                 {'pooling': 'none'})
        bounded = analysis.evaluate(self.study, self.query,
                                    {'pooling': 'none', 'nonresponse': 'bound'})
        for source in bounded['sources']:
            self.assertIsNotNone(source['identification_bounds'])
        rr_base = next(source for source in base['sources']
                       if source['version'] == 'rr-1')
        rr_bound = next(source for source in bounded['sources']
                        if source['version'] == 'rr-1')
        self.assertLessEqual(rr_bound['identification_bounds'][0],
                             rr_base['estimate'])
        self.assertGreaterEqual(rr_bound['identification_bounds'][1],
                                rr_base['estimate'])

    def test_mechanism_error_creates_identification_bounds(self):
        result = analysis.evaluate(self.study, self.query,
                                   {'pooling': 'none', 'mechanism_error': 0.05})
        self.assertTrue(any(source.get('identification_bounds')
                            for source in result['sources']))

    def test_ceiling_floor_widens_list_bounds(self):
        query = {'type': 'marginal', 'item': 'overload'}
        base = analysis.evaluate(self.study, query, {'pooling': 'none'})
        bounded = analysis.evaluate(self.study, query,
                                    {'pooling': 'none', 'ceiling_floor': 'bound'})
        list_base = next(source for source in base['sources']
                         if source['version'] == 'list-1')
        list_bound = next(source for source in bounded['sources']
                          if source['version'] == 'list-1')
        self.assertIsNone(list_base['identification_bounds'])
        self.assertIsNotNone(list_bound['identification_bounds'])

    def test_invalid_assumptions_are_rejected(self):
        for bad in ({'mechanism_error': -0.1}, {'mechanism_error': 2},
                    {'nonresponse': 'drop'}, {'pooling': 'sometimes'},
                    {'unknown': 1}, {'confidence': 0.2}):
            with self.assertRaises(ValueError):
                analysis.evaluate(self.study, self.query, bad)

    def test_sensitivity_covers_core_scenarios(self):
        result = analysis.sensitivity(self.study, self.query)
        labels = [row['label'] for row in result['scenarios']]
        self.assertTrue(any('机制失真' in label for label in labels))
        self.assertTrue(any('未响应' in label for label in labels))
        self.assertTrue(any('不合并' in label for label in labels))
        self.assertTrue(any('独立性' in label for label in labels))
        self.assertEqual(len(result['scenarios']), 10)


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.study = load_study()
        self.report = analysis.capabilities(self.study)
        self.cases = cases()

    def test_every_item_has_a_marginal_query(self):
        marginals = [entry for entry in self.report['queries']
                     if entry['query']['type'] == 'marginal']
        self.assertEqual(len(marginals), len(self.study['sensitive_items']))

    def test_joint_query_is_listed(self):
        joints = [entry for entry in self.report['queries']
                  if entry['query']['type'] == 'joint']
        self.assertEqual(len(joints), 1)
        self.assertEqual(joints[0]['query']['items'], ['overload', 'missed_support'])

    def test_group_difference_queries_cover_both_estimands(self):
        estimands = {entry['query']['estimand'] for entry in self.report['queries']
                     if entry['query']['type'] == 'group_difference'}
        self.assertEqual(estimands, {'marginal', 'joint'})

    def test_marginals_without_joint_case_reports_partial_overlap(self):
        study = self.cases['marginals_without_joint']['study']
        report = analysis.capabilities(study)
        joint = next(entry for entry in report['queries']
                     if entry['query']['type'] == 'joint')
        self.assertEqual(joint['status'], analysis.PARTIAL)
        self.assertFalse(joint['point_identified'])
        self.assertIsNotNone(joint['frechet_bounds'])

    def test_no_truth_case_reports_unidentified(self):
        study = self.cases['no_truth']['study']
        report = analysis.capabilities(study)
        marginal = next(entry for entry in report['queries']
                        if entry['query']['type'] == 'marginal')
        self.assertEqual(marginal['status'], analysis.UNIDENTIFIED)

    def test_diagnostics_mention_response_rates(self):
        text = ' '.join(self.report['diagnostics'])
        self.assertIn('回收', text)
        self.assertIn('80', text)

    def test_design_notes_are_not_claims_about_people(self):
        text = json.dumps(self.report, ensure_ascii=False)
        self.assertNotIn('sensitive_status', text)
        self.assertIn('只使用聚合回答', text)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.study = load_study()

    def test_unknown_query_type_rejected(self):
        with self.assertRaises(ValueError):
            analysis.evaluate(self.study, {'type': 'histogram'})

    def test_unknown_item_rejected(self):
        with self.assertRaises(ValueError):
            analysis.evaluate(self.study, {'type': 'marginal', 'item': 'unknown'})

    def test_joint_needs_two_items(self):
        with self.assertRaises(ValueError):
            analysis.evaluate(self.study, {'type': 'joint', 'items': ['overload']})

    def test_joint_rejects_duplicate_items(self):
        with self.assertRaises(ValueError):
            analysis.evaluate(self.study, {'type': 'joint',
                                           'items': ['overload', 'overload']})

    def test_non_finite_numbers_never_serialised(self):
        result = analysis.evaluate(self.study, {'type': 'joint',
                                                'items': ['overload', 'missed_support']})
        text = json.dumps(result, allow_nan=False)
        self.assertNotIn('NaN', text)
        self.assertNotIn('Infinity', text)

    def test_response_validation_still_applies(self):
        data = copy.deepcopy(self.study)
        row = next(row for row in data['responses'] if row['version'] == 'list-1')
        row['count'] = 99
        with self.assertRaises(ValueError):
            analysis.evaluate(data, {'type': 'marginal', 'item': 'overload'})

    def test_duplicate_joint_state_rejected(self):
        data = copy.deepcopy(self.study)
        version = next(row for row in data['versions'] if row['kind'] == 'rr_joint')
        version['latent_states'][1] = version['latent_states'][0]
        with self.assertRaises(ValueError):
            analysis.evaluate(data, {'type': 'marginal', 'item': 'overload'})


if __name__ == '__main__':
    unittest.main()

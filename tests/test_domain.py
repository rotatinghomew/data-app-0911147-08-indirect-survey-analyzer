import copy
import unittest
from app import domain
from app.service import load_study


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.data = load_study()
        self.request = {'operation': 'counts', 'version': 'rr-1'}

    def test_bad_randomization_mechanism(self):
        data = copy.deepcopy(self.data)
        data['versions'][0]['mechanism']['truth'] = 5
        with self.assertRaises(ValueError):
            domain.baseline(data)

    def test_no_personal_random_branch_is_stored(self):
        for row in self.data['responses']:
            self.assertNotIn('sensitive_status', row)
            self.assertNotIn('random_branch', row)

    def test_impossible_list_count(self):
        data = copy.deepcopy(self.data)
        row = next(row for row in data['responses'] if row['version'] == 'list-1')
        row['count'] = 100
        with self.assertRaises(ValueError):
            domain.baseline(data)

    def test_joint_code_remains_observed_randomized_response(self):
        result = domain.act(self.data, {'operation': 'counts', 'version': 'joint-1'})
        self.assertIsNone(result['raw_mean'])
        self.assertEqual(sum(result['histogram'].values()), result['n'])

    def test_bad_joint_transition_rejected(self):
        data = copy.deepcopy(self.data)
        version = next(row for row in data['versions'] if row['kind'] == 'rr_joint')
        version['transition'][0][0] = 2
        with self.assertRaises(ValueError):
            domain.baseline(data)

    def test_unknown_joint_code_rejected(self):
        data = copy.deepcopy(self.data)
        row = next(row for row in data['responses'] if row['version'] == 'joint-1')
        row['coded_answer'] = 'unknown'
        with self.assertRaises(ValueError):
            domain.baseline(data)


if __name__ == '__main__':
    unittest.main()

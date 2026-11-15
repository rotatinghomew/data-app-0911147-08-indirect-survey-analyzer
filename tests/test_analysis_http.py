from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import json
import tempfile
import threading
import unittest

from app.service import create_server
from app.storage import NoteStore


class AnalysisHttpTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.server = create_server(state_dir=self.temporary.name)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = 'http://127.0.0.1:' + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.temporary.cleanup()

    def request(self, path, payload=None):
        request = Request(self.base + path, None if payload is None else json.dumps(payload).encode(),
                          {'Content-Type': 'application/json'})
        with urlopen(request, timeout=60) as response:
            return response.status, json.load(response)

    def test_capabilities_endpoint(self):
        status, report = self.request('/api/capabilities')
        self.assertEqual(status, 200)
        self.assertTrue(report['queries'])
        self.assertTrue(report['diagnostics'])
        for entry in report['queries']:
            self.assertIn('status', entry)
            self.assertIn('query', entry)
            self.assertIn('summary', entry)

    def test_marginal_analysis(self):
        status, result = self.request('/api/analysis', {
            'query': {'type': 'marginal', 'item': 'overload'}})
        self.assertEqual(status, 200)
        self.assertIn('status', result)
        self.assertIn('sources', result)
        self.assertEqual(result['estimand']['item'], 'overload')

    def test_joint_analysis(self):
        status, result = self.request('/api/analysis', {
            'query': {'type': 'joint', 'items': ['overload', 'missed_support']}})
        self.assertEqual(status, 200)
        self.assertIn('notes', result)
        self.assertNotIn('sensitive_status', json.dumps(result))

    def test_group_difference_analysis(self):
        status, result = self.request('/api/analysis', {
            'query': {'type': 'group_difference', 'estimand': 'marginal',
                      'item': 'overload', 'group_a': 'arts', 'group_b': 'science'}})
        self.assertEqual(status, 200)
        self.assertTrue(result['version_differences'])

    def test_assumptions_are_accepted(self):
        status, result = self.request('/api/analysis', {
            'query': {'type': 'marginal', 'item': 'overload'},
            'assumptions': {'pooling': 'none', 'mechanism_error': 0.05}})
        self.assertEqual(status, 200)
        self.assertEqual(result['assumptions']['pooling'], 'none')
        self.assertEqual(result['assumptions']['mechanism_error'], 0.05)

    def test_sensitivity_endpoint(self):
        status, result = self.request('/api/sensitivity', {
            'query': {'type': 'joint', 'items': ['overload', 'missed_support']}})
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(result['scenarios']), 5)
        for row in result['scenarios']:
            self.assertIn('label', row)
            self.assertIn('status', row)

    def test_invalid_query_is_client_error(self):
        for payload in ({'query': {'type': 'nope'}},
                        {'query': {'type': 'marginal', 'item': 'unknown'}},
                        {'query': {'type': 'joint', 'items': ['overload']}},
                        {'query': {'type': 'marginal', 'item': 'overload'},
                         'assumptions': {'mechanism_error': 5}},
                        {}):
            with self.assertRaises(HTTPError) as caught:
                self.request('/api/analysis', payload)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()

    def test_invalid_sensitivity_is_client_error(self):
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/sensitivity', {'query': {'type': 'nope'}})
        self.assertEqual(caught.exception.code, 400)
        caught.exception.close()

    def test_results_never_serialise_non_finite_values(self):
        status, result = self.request('/api/analysis', {
            'query': {'type': 'joint', 'items': ['overload', 'missed_support']}})
        text = json.dumps(result, allow_nan=False)
        self.assertNotIn('NaN', text)
        self.assertNotIn('Infinity', text)

    def test_existing_summary_and_action_still_work(self):
        status, summary = self.request('/api/summary')
        self.assertEqual(status, 200)
        self.assertIn('versions', summary)
        action = json.loads((Path(__file__).resolve().parents[1] /
                             'data/default-action.json').read_text())
        status, result = self.request('/api/action', action)
        self.assertEqual(status, 200)
        self.assertIn('histogram', result)

    def test_notes_still_persist(self):
        status, entry = self.request('/api/notes', {'text': '联合分析复查',
                                                    'selection': {'query': 'joint'}})
        self.assertEqual(status, 201)
        self.assertEqual(NoteStore(Path(self.temporary.name)).list(), [entry])


if __name__ == '__main__':
    unittest.main()

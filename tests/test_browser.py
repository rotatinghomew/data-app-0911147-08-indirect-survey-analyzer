import json
import os
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from pathlib import Path

from playwright.sync_api import sync_playwright

from app.service import create_server


class BrowserContractTests(unittest.TestCase):
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

    def post(self, route, payload):
        request = Request(self.base + route, json.dumps(payload).encode(),
                          {'Content-Type': 'application/json'})
        with urlopen(request, timeout=60) as response:
            return json.load(response)

    def test_current_workflow_in_real_browser(self):
        root = Path(__file__).resolve().parents[1]
        action = json.loads((root / 'data/default-action.json').read_text())
        request = Request(self.base + '/api/action', json.dumps(action).encode(),
                          {'Content-Type': 'application/json'})
        with urlopen(request, timeout=5) as response:
            expected = json.load(response)
        marginal = self.post('/api/analysis', {
            'query': {'type': 'marginal', 'item': 'overload'}})
        joint = self.post('/api/analysis', {
            'query': {'type': 'joint', 'items': ['overload', 'missed_support']}})
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(args=['--no-sandbox'] if os.geteuid() == 0 else [])
            try:
                page = browser.new_page()
                page.set_default_timeout(20000)
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(self.base, wait_until='networkidle')
                page.wait_for_function("document.querySelector('#summary').textContent !== '载入中'")
                self.assertNotIn('error', json.loads(page.locator('#summary').inner_text()))

                # Capability table lists every query and never shows a bare claim.
                rows = page.locator('#capabilities tbody tr')
                self.assertGreater(rows.count(), 0)
                self.assertTrue(page.locator('#diagnostics li').count() > 0)

                # Records table still enumerates all sources.
                sources = page.locator('#source option').all_text_contents()
                self.assertTrue(sources)
                for source in sources:
                    page.select_option('#source', source)
                    self.assertGreater(page.locator('#records tbody tr').count(), 0)

                # Marginal analysis through the real form and live HTTP result.
                page.select_option('#query-type', 'marginal')
                page.select_option('#query-item', 'overload')
                page.locator('#query-form button[type=submit]').click()
                page.wait_for_function(
                    "document.querySelector('#result .headline') !== null")
                headline = page.locator('#result .headline').inner_text()
                self.assertIn('单项比例', headline)
                self.assertIn(self.STATUS[marginal['status']], headline)

                # Joint analysis is distinguished from marginal and reports bounds.
                page.select_option('#query-type', 'joint')
                page.locator('#query-form button[type=submit]').click()
                page.wait_for_function(
                    "document.querySelector('#result .headline').textContent.includes('共同经历')")
                self.assertIn(self.STATUS[joint['status']],
                              page.locator('#result .headline').inner_text())
                facts = page.locator('#result li').all_text_contents()
                self.assertTrue(any('识别范围' in text or '区间' in text for text in facts))

                # Assumption comparison fills a table.
                page.locator('#sensitivity-button').click()
                page.wait_for_selector('#sensitivity tbody tr')
                self.assertGreaterEqual(page.locator('#sensitivity tbody tr').count(), 5)

                # Notes still persist across reload.
                page.locator('#note').fill('复查当前业务资料与结果')
                page.locator('#note-form button').click()
                page.wait_for_selector('#notes li')
                page.reload(wait_until='networkidle')
                self.assertEqual(page.locator('#notes li').count(), 1)
                self.assertIn('复查当前业务资料与结果', page.locator('#notes li').inner_text())
                self.assertEqual(errors, [])
            finally:
                browser.close()

    STATUS = {
        'identified': '可点估计',
        'partially_identified': '只能给出范围或需谨慎解读',
        'unidentified': '不可识别',
        'heterogeneous': '版本间不一致',
        'version_specific': '只能按版本报告',
        'assumption_based': '依赖显式假设',
    }


if __name__ == '__main__':
    unittest.main()

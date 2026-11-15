from pathlib import Path
from urllib.request import Request, urlopen
import json
import tempfile
import threading

from .service import create_server

ROOT = Path(__file__).resolve().parents[1]

QUERIES = (
    ('单项比例', {'type': 'marginal', 'item': 'overload'}),
    ('共同经历比例', {'type': 'joint', 'items': ['overload', 'missed_support']}),
    ('群体差异', {'type': 'group_difference', 'estimand': 'marginal',
                  'item': 'overload'}),
)


def post(base, route, payload):
    request = Request(base + route, json.dumps(payload).encode(),
                      {'Content-Type': 'application/json'})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def summarize(result):
    return {key: result.get(key) for key in
            ('status', 'point_identified', 'estimate', 'interval',
             'identification_bounds', 'frechet_bounds')}


def main():
    request = json.loads((ROOT / 'data/default-action.json').read_text())
    with tempfile.TemporaryDirectory() as directory:
        server = create_server(state_dir=directory)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = 'http://127.0.0.1:' + str(server.server_port)
        try:
            print('本地产品实时运行：' + base)
            for route in ('/health', '/api/summary'):
                with urlopen(base + route, timeout=5) as response:
                    print(route, json.dumps(json.load(response), ensure_ascii=False, indent=2))
            operation = Request(base + '/api/action', json.dumps(request).encode(),
                                {'Content-Type': 'application/json'})
            with urlopen(operation, timeout=5) as response:
                print('当前操作', json.dumps(json.load(response), ensure_ascii=False, indent=2))

            with urlopen(base + '/api/capabilities', timeout=120) as response:
                capabilities = json.load(response)
            print('可回答的查询', json.dumps(
                [{'query': entry['query'], 'status': entry['status'],
                  'summary': entry['summary']} for entry in capabilities['queries']],
                ensure_ascii=False, indent=2))
            print('设计提示', json.dumps(capabilities['diagnostics'], ensure_ascii=False, indent=2))
            print('设计建议', json.dumps(capabilities['recommendations'],
                                       ensure_ascii=False, indent=2))

            for label, query in QUERIES:
                print(label, json.dumps(summarize(post(base, '/api/analysis',
                                                       {'query': query})),
                                        ensure_ascii=False, indent=2))

            scenarios = post(base, '/api/sensitivity',
                             {'query': {'type': 'joint',
                                        'items': ['overload', 'missed_support']}})
            print('假设比较', json.dumps(
                [{'label': row['label'], 'status': row['status'],
                  'estimate': row['estimate'],
                  'identification_bounds': row['identification_bounds']}
                 for row in scenarios['scenarios']], ensure_ascii=False, indent=2))
            print('运行结束；上述为当前版本的实际响应。')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == '__main__':
    main()

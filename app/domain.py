from collections import Counter
from statistics import mean


def versions(data):
    result = {row['id']: row for row in data['versions']}
    if len(result) != len(data['versions']):
        raise ValueError('重复问卷版本')
    for row in result.values():
        if row['kind'] == 'rr':
            mechanism = row['mechanism']
            if abs(sum(mechanism.values()) - 1) > 1e-9 or any(value < 0 for value in mechanism.values()):
                raise ValueError('随机机制不合法')
        elif row['kind'] == 'rr_joint':
            if len(row['transition']) != len(row['latent_states']):
                raise ValueError('联合机制状态数不匹配')
            if len(set(row['latent_states'])) != len(row['latent_states']):
                raise ValueError('联合机制潜在状态重复')
            if len(set(row['response_states'])) != len(row['response_states']):
                raise ValueError('联合机制响应状态重复')
            for probabilities in row['transition']:
                if len(probabilities) != len(row['response_states']) or any(value < 0 for value in probabilities) or abs(sum(probabilities) - 1) > 1e-9:
                    raise ValueError('联合机制概率不合法')
        elif row['kind'] == 'list':
            if not isinstance(row.get('k'), int) or isinstance(row.get('k'), bool) or row['k'] < 1:
                raise ValueError('名单基础条目数不合法')
        else:
            raise ValueError('未知间接问法')
    return result


def valid_responses(data):
    definitions = versions(data)
    seen = set()
    for row in data['responses']:
        if row['response_id'] in seen:
            raise ValueError('重复答卷')
        seen.add(row['response_id'])
        version = definitions[row['version']]
        if version['kind'] == 'rr':
            if row['answer'] not in (0, 1):
                raise ValueError('随机回答超出范围')
        elif version['kind'] == 'rr_joint':
            if row.get('coded_answer') not in version['response_states']:
                raise ValueError('联合编码不在响应空间')
        else:
            maximum = version['k'] + int(row['added_sensitive'])
            if not isinstance(row['count'], int) or not 0 <= row['count'] <= maximum:
                raise ValueError('名单计数超出范围')
    return data['responses']


def baseline(data):
    rows = valid_responses(data)
    return {'versions': list(versions(data)), 'responses': len(rows),
            'counts': dict(Counter(row['version'] for row in rows)), 'allocation': data['allocation']}


def act(data, request):
    if request.get('operation') != 'counts':
        raise ValueError('未知问卷操作')
    definitions = versions(data)
    version = request.get('version')
    if version not in definitions:
        raise ValueError('未知问卷版本')
    rows = [row for row in valid_responses(data) if row['version'] == version]
    field = {'rr': 'answer', 'rr_joint': 'coded_answer', 'list': 'count'}[definitions[version]['kind']]
    return {'version': version, 'n': len(rows),
            'histogram': dict(sorted(Counter(row[field] for row in rows).items())),
            'raw_mean': mean(row[field] for row in rows) if rows and field != 'coded_answer' else None,
            'notice': '原始回答分布，不是敏感经历比例'}

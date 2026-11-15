const sources = ["responses", "versions", "allocation", "invited_counts"];
let study = null;
let capabilities = null;
let latestRequest = null;

async function api(route, payload) {
  const options = payload === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
  };
  const response = await fetch(route, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || String(response.status));
  return result;
}

function display(target, value) {
  document.getElementById(target).textContent = typeof value === 'string'
    ? value : JSON.stringify(value, null, 2);
}

const STATUS_LABELS = {
  identified: '可点估计',
  partially_identified: '只能给出范围或需谨慎解读',
  unidentified: '不可识别',
  heterogeneous: '版本间不一致',
  version_specific: '只能按版本报告',
  assumption_based: '依赖显式假设'
};

function number(value) {
  return value === null || value === undefined ? '—' : Number(value).toFixed(4);
}

function interval(entry) {
  if (!entry) return '—';
  return '[' + number(entry[0]) + ', ' + number(entry[1]) + ']';
}

function describe(entry) {
  const parts = [STATUS_LABELS[entry.status] || entry.status];
  if (entry.estimate !== null && entry.estimate !== undefined) parts.push('点估计 ' + number(entry.estimate));
  if (entry.interval) parts.push('区间 ' + interval(entry.interval));
  if (entry.identification_bounds) parts.push('识别范围 ' + interval(entry.identification_bounds));
  if (entry.frechet_bounds) parts.push('Fréchet 范围 ' + interval(entry.frechet_bounds));
  return parts.join('；');
}

function queryLabel(query) {
  if (query.type === 'marginal') {
    return '单项比例 · ' + query.item + (query.group ? ' · ' + query.group : '');
  }
  if (query.type === 'joint') {
    return '共同经历 · ' + query.items.join(' + ') + (query.group ? ' · ' + query.group : '');
  }
  const target = query.estimand === 'joint'
    ? '共同经历 ' + (query.items || []).join(' + ')
    : '单项 ' + query.item;
  return '群体差异 · ' + target + ' · ' + query.group_a + ' vs ' + query.group_b;
}

function queryPayload() {
  const type = document.getElementById('query-type').value;
  if (type === 'marginal') {
    return {type, item: document.getElementById('query-item').value,
            group: document.getElementById('query-group').value || null};
  }
  if (type === 'joint') {
    const items = [...document.getElementById('query-items').selectedOptions].map(o => o.value);
    if (items.length < 2) throw new Error('共同经历比例需要选择至少两个项目');
    return {type, items, group: document.getElementById('query-group').value || null};
  }
  const estimand = document.getElementById('query-estimand').value;
  const base = {type, estimand,
    group_a: document.getElementById('query-group-a').value,
    group_b: document.getElementById('query-group-b').value};
  if (estimand === 'marginal') base.item = document.getElementById('query-item').value;
  else {
    const items = [...document.getElementById('query-items').selectedOptions].map(o => o.value);
    if (items.length < 2) throw new Error('共同经历比例需要选择至少两个项目');
    base.items = items;
  }
  return base;
}

function assumptionsPayload() {
  return {
    mechanism_error: Number(document.getElementById('assumption-error').value || 0),
    nonresponse: document.getElementById('assumption-nonresponse').value,
    pooling: document.getElementById('assumption-pooling').value,
    clustering: document.getElementById('assumption-clustering').value,
    ceiling_floor: document.getElementById('assumption-ceiling').value,
    independence: document.getElementById('assumption-independence').checked
  };
}

function toggleFields() {
  const type = document.getElementById('query-type').value;
  const estimand = document.getElementById('query-estimand').value;
  const showItem = type === 'marginal' || (type === 'group_difference' && estimand === 'marginal');
  const showItems = type === 'joint' || (type === 'group_difference' && estimand === 'joint');
  document.getElementById('item-field').hidden = !showItem;
  document.getElementById('items-field').hidden = !showItems;
  document.getElementById('estimand-field').hidden = type !== 'group_difference';
  document.getElementById('group-field').hidden = type !== 'group_difference';
  document.getElementById('query-group').closest('label').hidden = type === 'group_difference';
}

function renderCapabilities() {
  const holder = document.getElementById('capabilities');
  holder.replaceChildren();
  const table = document.createElement('table');
  const head = table.createTHead().insertRow();
  for (const label of ['查询', '可识别状态', '点估计', '区间', '识别范围', '来源版本', '说明']) {
    const cell = document.createElement('th'); cell.textContent = label; head.append(cell);
  }
  const body = table.createTBody();
  for (const entry of capabilities.queries) {
    const row = body.insertRow();
    row.insertCell().textContent = queryLabel(entry.query);
    row.insertCell().textContent = (STATUS_LABELS[entry.status] || entry.status) + ' · ' + entry.summary;
    row.insertCell().textContent = number(entry.estimate);
    row.insertCell().textContent = interval(entry.interval);
    row.insertCell().textContent = interval(entry.identification_bounds || entry.frechet_bounds);
    row.insertCell().textContent = entry.versions.length ? entry.versions.join('、') : '—';
    const note = entry.notes.find(text => text.indexOf('只使用聚合回答') === -1) || '';
    const perVersion = (entry.version_estimates || [])
      .filter(item => item.estimate !== null)
      .map(item => item.version + '＝' + number(item.estimate)).join('；');
    row.insertCell().textContent = perVersion ? perVersion + (note ? ' ' + note : '') : note;
    row.className = 'status-' + entry.status;
  }
  holder.append(table);
  const diagnostic = document.getElementById('diagnostics');
  diagnostic.replaceChildren();
  if (capabilities.diagnostics.length) {
    const title = document.createElement('h3'); title.textContent = '设计提示';
    const list = document.createElement('ul');
    for (const text of capabilities.diagnostics) {
      const item = document.createElement('li'); item.textContent = text; list.append(item);
    }
    diagnostic.append(title, list);
  }
  const recommendation = document.getElementById('recommendations');
  recommendation.replaceChildren();
  if (capabilities.recommendations.length) {
    const title = document.createElement('h3'); title.textContent = '对下一轮设计的建议';
    const list = document.createElement('ul');
    for (const text of capabilities.recommendations) {
      const item = document.createElement('li'); item.textContent = text; list.append(item);
    }
    recommendation.append(title, list);
  }
}

function renderResult(result) {
  const holder = document.getElementById('result');
  holder.replaceChildren();
  const heading = document.createElement('p');
  heading.className = 'headline status-' + result.status;
  heading.textContent = (STATUS_LABELS[result.status] || result.status) + '：' + queryLabel(result.estimand);
  holder.append(heading);
  const facts = document.createElement('ul');
  const add = (label, value) => {
    const item = document.createElement('li');
    item.textContent = label + '：' + value;
    facts.append(item);
  };
  add('点估计', number(result.estimate));
  add('区间', interval(result.interval));
  add('识别范围', interval(result.identification_bounds));
  if (result.frechet_bounds) add('Fréchet 范围', interval(result.frechet_bounds));
  if (result.version_estimates && result.version_estimates.length) {
    add('各版本', result.version_estimates.map(
      entry => entry.version + '＝' + number(entry.estimate) + '（' +
               (STATUS_LABELS[entry.status] || entry.status) + '）').join('；'));
  }
  if (result.version_differences && result.version_differences.length) {
    add('各版本差异', result.version_differences.map(
      entry => entry.version + '＝' + number(entry.estimate) + '（' +
               (STATUS_LABELS[entry.status] || entry.status) + '）').join('；'));
  }
  if (result.implicit_marginals) {
    add('联合版本隐含边际', Object.entries(result.implicit_marginals).map(
      ([item, marginal]) => item + '＝' + number(marginal.estimate)).join('；'));
  }
  for (const note of result.notes || []) {
    const item = document.createElement('li'); item.textContent = note;
    item.className = 'note'; facts.append(item);
  }
  holder.append(facts);
  const detail = document.createElement('details');
  const summary = document.createElement('summary'); summary.textContent = '完整结果';
  const pre = document.createElement('pre'); pre.textContent = JSON.stringify(result, null, 2);
  detail.append(summary, pre);
  holder.append(detail);
}

function renderSensitivity(result) {
  const holder = document.getElementById('sensitivity');
  holder.replaceChildren();
  const table = document.createElement('table');
  const head = table.createTHead().insertRow();
  for (const label of ['假设情景', '可识别状态', '点估计', '区间', '识别范围']) {
    const cell = document.createElement('th'); cell.textContent = label; head.append(cell);
  }
  const body = table.createTBody();
  for (const row of result.scenarios) {
    const rendered = body.insertRow();
    rendered.insertCell().textContent = row.label;
    rendered.insertCell().textContent = STATUS_LABELS[row.status] || row.status;
    rendered.insertCell().textContent = number(row.estimate);
    rendered.insertCell().textContent = interval(row.interval);
    rendered.insertCell().textContent = interval(row.identification_bounds);
    rendered.className = 'status-' + row.status;
  }
  holder.append(table);
}

function renderTable(source) {
  const holder = document.getElementById('records');
  holder.replaceChildren();
  const raw = study[source];
  const rows = Array.isArray(raw) ? raw : Object.entries(raw || {}).map(([key, value]) => ({key, value}));
  if (!rows.length) { holder.textContent = '当前集合没有记录'; return; }
  const columns = [...new Set(rows.flatMap(row => Object.keys(row)))];
  const table = document.createElement('table');
  const head = table.createTHead().insertRow();
  for (const column of columns) {
    const cell = document.createElement('th'); cell.textContent = column; head.append(cell);
  }
  const body = table.createTBody();
  for (const row of rows) {
    const rendered = body.insertRow();
    for (const column of columns) {
      const value = row[column];
      rendered.insertCell().textContent = value === null ? '缺失'
        : typeof value === 'object' ? JSON.stringify(value) : String(value ?? '');
    }
  }
  holder.append(table);
  document.getElementById('record-count').textContent = String(rows.length) + ' 条记录';
}

function fillSelect(id, values, multiple = false) {
  const select = document.getElementById(id);
  for (const value of values) {
    const option = document.createElement('option');
    option.value = value; option.textContent = value;
    option.selected = multiple;
    select.append(option);
  }
}

async function refreshNotes() {
  const result = await api('/api/notes');
  const holder = document.getElementById('notes'); holder.replaceChildren();
  for (const note of result.notes) {
    const item = document.createElement('li'); item.textContent = note.text + ' · ' + note.created_at;
    holder.append(item);
  }
}

async function start() {
  study = await api('/api/study');
  capabilities = await api('/api/capabilities');
  display('summary', await api('/api/summary'));
  document.getElementById('summary').textContent = JSON.stringify(
    {responses: study.responses.length, versions: study.versions.map(v => v.id),
     allocation: study.allocation, invited: study.invited_counts}, null, 2);
  fillSelect('query-item', capabilities.design.items);
  fillSelect('query-items', capabilities.design.items, true);
  fillSelect('query-group', capabilities.design.groups);
  const groups = capabilities.design.groups;
  fillSelect('query-group-a', groups);
  fillSelect('query-group-b', groups);
  if (groups.length > 1) {
    document.getElementById('query-group-a').value = groups[0];
    document.getElementById('query-group-b').value = groups[1];
  }
  const selector = document.getElementById('source');
  for (const source of sources) {
    const option = document.createElement('option'); option.value = source; option.textContent = source;
    selector.append(option);
  }
  selector.addEventListener('change', () => renderTable(selector.value));
  renderTable(sources[0]);
  renderCapabilities();
  document.getElementById('query-type').addEventListener('change', toggleFields);
  document.getElementById('query-estimand').addEventListener('change', toggleFields);
  toggleFields();

  document.getElementById('query-form').addEventListener('submit', async event => {
    event.preventDefault();
    latestRequest = {method: 'POST', route: '/api/analysis',
                     payload: {query: queryPayload(), assumptions: assumptionsPayload()}};
    try {
      const result = await api(latestRequest.route, latestRequest.payload);
      renderResult(result);
    } catch (error) {
      document.getElementById('result').textContent = JSON.stringify({error: error.message}, null, 2);
    }
  });

  document.getElementById('sensitivity-button').addEventListener('click', async () => {
    let payload;
    try {
      payload = {query: queryPayload(), assumptions: assumptionsPayload()};
    } catch (error) {
      document.getElementById('sensitivity').textContent = JSON.stringify({error: error.message}, null, 2);
      return;
    }
    try {
      renderSensitivity(await api('/api/sensitivity', payload));
    } catch (error) {
      document.getElementById('sensitivity').textContent = JSON.stringify({error: error.message}, null, 2);
    }
  });

  document.getElementById('note-form').addEventListener('submit', async event => {
    event.preventDefault();
    const field = document.getElementById('note');
    try {
      await api('/api/notes', {text: field.value, selection: latestRequest || {}});
      field.value = ''; await refreshNotes();
    } catch (error) {
      document.getElementById('result').textContent = JSON.stringify({error: error.message}, null, 2);
    }
  });
  await refreshNotes();
}

start().catch(error => display('summary', {error: error.message}));

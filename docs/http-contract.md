# 本地工作台接口

服务默认仅监听回环地址，供本机使用，不是公开多租户系统。所有 JSON 响应使用 UTF-8，出错时有 error 文本；业务输入错误为 400，未知路径为 404。静态资源仅允许根页面和固定资源名。

## 现行描述接口（保持不变）

| 路径 | 方法 | 含义 |
| --- | --- | --- |
| /health | GET | 进程存活 |
| /api/study | GET | 当前完整业务资料 |
| /api/summary | GET | 现行描述与资料核对 |
| /api/action | POST | 执行网页表单对应的现行操作 |
| /api/notes | GET | 本地工作批注 |
| /api/notes | POST | 保存 text 和 selection，返回新增批注 |

`/api/action` 的具体字段在 data/default-action.json。

## 可识别分析接口

| 路径 | 方法 | 含义 |
| --- | --- | --- |
| /api/capabilities | GET | 现有问卷能回答哪些查询，附设计提示与设计建议 |
| /api/analysis | POST | 执行单项、联合或群体差异查询 |
| /api/sensitivity | POST | 同一查询在假设情景下的状态与范围对比 |

### /api/capabilities
返回 `design`（项目、群体、版本、观测/邀请数与层级提示）、`queries`（每个查询的状态、点估计、区间、识别范围、来源版本与说明）、`diagnostics`（回收、顶底、版本与班级重合、联合机制拟合）和 `recommendations`（是否需补充问卷设计）。

### /api/analysis
请求体：
```json
{
  "query": {"type": "marginal", "item": "overload", "group": null},
  "assumptions": {"pooling": "auto", "nonresponse": "ignore"}
}
```
- `query.type` 为 `marginal`、`joint` 或 `group_difference`。
- `marginal` 用 `item`；`joint` 用至少两个不同 `items`；`group_difference` 另需 `estimand`（`marginal`/`joint`）与 `group_a`/`group_b`。
- `assumptions` 可省略，字段见下表。

| 字段 | 取值 | 默认 | 含义 |
| --- | --- | --- | --- |
| mechanism_error | 0–1 | 0 | 声明机制的总变差失真半径 |
| nonresponse | `ignore`/`bound` | `ignore` | 未响应按观测或最不利情形 |
| pooling | `auto`/`pool`/`none` | `auto` | 版本合并策略 |
| ceiling_floor | `ignore`/`bound` | `ignore` | 名单顶底是否放宽范围 |
| clustering | `register`/`class` | `register` | 不确定性按登记层级或班级 |
| independence | true/false | false | 是否对缺失联合信息启用独立性 |
| confidence | 0.5–1 | 0.95 | 区间水平 |
| bootstrap_replicates | 20–2000 | 200 | 班级自助次数 |
| seed | 整数 | 20260101 | 自助随机种子 |

结果包含 `status`、`point_identified`、`estimate`、`se`、`interval`（抽样区间）、`identification_bounds`（识别范围）、`sources`/`version_estimates`、`pooling`（含异质性检验与合并依据）和 `notes`。联合结果另含 `frechet_bounds`、`implied_marginals` 与 `conflicts`；群体差异另含 `version_differences`。

`status` 见 docs/analysis-contract.md。数值未定义时使用 `null` 并解释原因，不以 NaN 或 Infinity 冒充 JSON 数值。

### /api/sensitivity
请求体为 `query`、可选 `assumptions` 与可选 `scenarios`（`[{"label": "...", "change": {...}}]`）。省略 `scenarios` 时使用默认的十种情形：基准、机制失真 2/5/10%、未响应最不利、班级聚类、不合并、强制合并、顶底移动、显式独立性。返回每行的 `status`、`estimate`、`interval`、`identification_bounds` 与说明；不适用情形标为 `not_applicable` 而不伪造数值。

已有接口保持可用；新增分析接口的字段可按产品需要演进。

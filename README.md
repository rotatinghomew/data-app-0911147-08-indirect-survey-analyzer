# 多种间接问卷机制下敏感经历联合比例的可识别分析

这是供本地讨论使用的数据应用。网页可以检查问卷版本与随机机制、比较现有问卷能回答哪些问题、提出单项经历比例 / 共同经历比例 / 群体差异查询，并比较机制失真、未响应、顶底、聚类与版本合并等假设对结果的影响。结果区分可点估计、部分识别与不可识别，不输出任何个人敏感状态。

## 环境与启动

使用 Python 3.11.14，数值依赖锁定在 requirements.txt。Docker 基础镜像按内容摘要固定；构建阶段安装依赖，服务、检查和演示阶段不下载数据、依赖或模型。所有业务记录都是本地构造的课程与研究管理资料，不包含个人或真实实验信息。

```sh
docker build -t indirect-survey-analyzer:local .
docker run --rm -p 8080:8080 indirect-survey-analyzer:local
```

浏览器打开本机 8080 端口。复用已有 Python 环境时，先在准备阶段安装锁定依赖，再运行服务：

```sh
python3 -m pip install --only-binary=:all: -r requirements.txt
python3 -m app.service
```

浏览器回归使用锁定的 Playwright 1.56.0 与其配套 Chromium。镜像构建阶段预装浏览器和系统库，测试与服务运行不下载任何依赖，不依赖宿主机的 Chrome 或 Edge。直接使用 Python 环境时，也需在准备阶段执行 `python3 -m playwright install --with-deps chromium`。

APP_STUDY 可以指定同结构本地资料，APP_STATE_DIR 指定批注目录，默认是 .local。批注只保存人工文字与当时的表单选择，不冻结分析数据或生成统计发布；删除该目录可回到干净本地状态，原始资料不变。

## 分析能力

- **单项经历比例**：随机化回答按声明机制反演；名单实验用实验臂与对照臂均值差。truth 为零的版本判为不可识别。
- **共同经历比例**：只有保留随机化联合编码的版本能给出。对潜在状态分布做最大似然估计，区间用轮廓似然，因为解常落在单纯形边界。没有联合编码时只给 Fréchet 范围，不默认独立性。
- **群体差异**：先在每个版本内部相减，再按异质性检验决定是否合并，避免把机制偏倚混入差值。
- **版本合并**：默认先检验异质性；不一致时不给出合并点估计，只列各版本，并允许显式强制合并或不合并以比较后果。
- **假设比较**：机制失真半径、未响应最不利、名单顶底、班级聚类、版本合并与显式独立性都可逐项比较，结果标明每种设定下的状态与范围。

可识别状态的表示与估计细节见 docs/analysis-contract.md。

## 构建与检查

```sh
python3 -m compileall -q app tests
python3 -m unittest discover -s tests -v
docker run --rm --network none indirect-survey-analyzer:local python -m unittest discover -s tests -v
```

测试覆盖现行领域语义、分析估计量（含 truthful、无信息、名单手算基准、无联合编码、版本混合与越界输入）、HTTP 错误边界与批注重开，并在真实 Chromium 中遍历能力表、运行单项与联合查询、比较假设、保存批注并刷新确认。浏览器缺失或脚本报错会令测试失败，不跳过检查。data/analysis-cases.json 是独立检查资料，docs/analysis-contract.md 说明其统计意义和适用限制；这些资料不要求每次输出某种结论方向。

## 短时间实机查看

```sh
python3 -m app.demo
```

该入口启动真实本地 HTTP 服务，依次打印健康检查、现行操作、能力表、设计提示与建议、三类查询以及假设比较的实际响应，随后关闭服务。默认资料在十秒左右完成，不读取预存日志。网页交互也使用同一服务。

## 代码位置

app/analysis.py 负责设计核对、识别状态与合并判断；app/mechanisms.py 是各机制的数值估计量；app/domain.py 定义业务资料语义与现行操作；app/service.py 是网页和接口共同使用的服务；web/app.js 提交查询并呈现识别状态与范围；app/storage.py 保存工作批注。外部接口与分析边界分别见 docs/http-contract.md 和 docs/analysis-contract.md。

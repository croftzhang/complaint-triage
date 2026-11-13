# Complaint Triage — 银行投诉分诊 Agent

一个练手项目，用公开数据做"银行客户投诉分诊"：

- **LangChain / LangGraph**：有状态的分诊工作流（分类 → 抽取 → 风险判断 → 法规检索 → 起草回复 → 引用校验 → 人工审批）
- **PyTorch + HuggingFace**：微调 DistilBERT 做投诉产品分类；sentence-transformers 做 embedding
- **TensorFlow / Keras**：预测投诉是否会获得金钱赔偿，输出概率用于风险路由
- **MLOps（MLflow）**：实验记录、Model Registry、`champion` 别名、`mlflow models serve` 上线

数据全部公开：CFPB 消费者投诉数据库（含投诉原文和标签），eCFR 上的 Regulation E（12 CFR Part 1005）。

---

## 1. 架构

```
                 ┌──────────── LangGraph (SQLite checkpointer) ────────────┐
complaint ──► classify ──► extract ──► assess_risk ──┬──► retrieve ──► draft ──► verify_citations ──┬──► human_review ──► END
             (PyTorch     (Claude     (TF model +    │   (FAISS +     (Claude   (regex: 引用必须   │   (interrupt,
              classifier)  Haiku)      rules)        │    bge emb.)    Sonnet)   来自检索结果)      │    等人确认)
                                                     └──► out_of_scope ─────────────────────────────┘
                                                          (非 Reg E 范围: 模板回复, 不走 RAG)
```

- `classify`：MLflow registry 里的 PyTorch 分类器（本地加载或 HTTP 调用）。产品组在 Reg E 范围内（支票/储蓄账户、转账）才走 RAG。
- `extract`：Claude Haiku + `with_structured_output(Extraction)`，抽出金额、日期、诉求。
- `assess_risk`：老年人/军人标签、金额阈值、TF 模型赔付概率，任一命中则 `escalate=True`。
- `retrieve`：bge-small embedding + FAISS，检索 Reg E 条款分块。
- `draft`：Claude Sonnet 起草回复，只允许引用检索到的条款。
- `verify_citations`：正则检查 `§ 1005.x`，引用不在检索结果里就打回重写（最多 2 次）。
- `human_review`：`interrupt()` 暂停，等人 approve / edit / reject 后用 `Command(resume=...)` 继续。

查看图结构：`python -m triage.graph`（输出 Mermaid）。

## 2. 目录

```
complaint-triage/
├── config.yaml              # 所有参数（模型名、路径、阈值）
├── config.smoke.yaml        # 冒烟测试配置：合成数据 + 随机小模型 + CPU
├── requirements.txt         # 主环境：torch / transformers / langchain / langgraph / mlflow
├── requirements-tf.txt      # TF 环境：tensorflow / mlflow
├── triage/                  # 包
│   ├── config.py            # 读 config
│   ├── taxonomy.py          # 9 个产品组；哪些在 Reg E 范围
│   ├── data_io.py           # CFPB 字段标准化、均衡采样、分层切分
│   ├── reg_e.py             # eCFR XML 解析、按条款分块
│   ├── retrieval.py         # HuggingFaceEmbeddings + FAISS 检索器
│   ├── features.py          # TF 模型特征（训练和推理共用，只依赖 numpy/pandas）
│   ├── tracking.py          # MLflow 辅助：model_uri / champion 别名
│   ├── models.py            # 分类器和 TF 模型的客户端（local | http）
│   ├── schemas.py           # Pydantic：Extraction、DraftResponse
│   ├── prompts.py           # 两个 LLM 节点的 prompt
│   ├── llm.py               # ChatAnthropic 工厂 + 离线 fake LLM
│   ├── nodes.py             # LangGraph state 和节点
│   ├── graph.py             # 组装图 + SQLite checkpointer
│   ├── synthetic.py         # 合成数据（仅冒烟测试）
│   └── smoke.py             # 随机初始化的小模型（仅冒烟测试）
├── scripts/
│   ├── 00_start_mlflow.sh   # 启动 MLflow server
│   ├── 01_download_cfpb.py  # CFPB 数据 → train/test parquet
│   ├── 02_fetch_reg_e.py    # eCFR → reg_e_chunks.jsonl
│   ├── 03_embed_and_index.py# FAISS 索引 + 投诉 embedding (.npy)
│   ├── 04_train_classifier.py   # PyTorch + HF Trainer + MLflow   [主环境, GPU]
│   ├── 05_train_relief_tf.py    # TensorFlow/Keras + MLflow       [TF 环境]
│   ├── 06_evaluate.py       # 评估 + 设 champion + 写报告
│   ├── 07_serve_models.sh   # mlflow models serve
│   ├── 08_demo.py           # 端到端运行 LangGraph，含人工审批
│   └── run_smoke.sh         # 一键冒烟测试
├── evals/retrieval_questions.jsonl   # 25 条检索评估问题 → 期望条款
├── data/fixtures/reg_e_fixture.xml   # 离线测试用的 eCFR 结构样例（不是法规原文）
└── reports/                 # 评估报告输出
```

## 3. 环境

建议 Python 3.11 或 3.12。Windows 用户请在 **WSL2** 里操作（TensorFlow 2.10 以后不支持 Windows 原生 GPU；shell 脚本也只在 Linux 下测过）。

```bash
# 主环境（PyTorch + HF + LangChain + MLflow）
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # 按 pytorch.org 选对应 CUDA 版本
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"             # 应为 True

# TF 环境（只跑 scripts/05 和 relief 模型的 serve）
python -m venv .venv-tf && .venv-tf/bin/pip install -r requirements-tf.txt

# API key
cp .env.example .env    # 填 ANTHROPIC_API_KEY
```

两个环境分开是为了避开 PyTorch 和 TensorFlow 的依赖冲突。两边共享的只有 `triage/features.py`、`triage/config.py`、`triage/tracking.py`，它们不 import 任何深度学习框架。

## 4. 先跑冒烟测试（5 分钟，不花钱）

合成数据 + 随机初始化的小模型 + fake LLM，确认所有脚本和 LangGraph 都能跑通：

```bash
bash scripts/00_start_mlflow.sh        # 终端 1，保持打开
bash scripts/run_smoke.sh              # 终端 2（主环境）；TF_PYTHON 默认 .venv-tf/bin/python
```

打开 http://127.0.0.1:5000 能看到 `complaint-triage-smoke` 实验下的三个 run（classifier、relief、evaluation）和两个注册模型。**这里的指标没有意义**（模型是随机的），只看流程。

## 5. 正式运行（两天版）

所有命令在仓库根目录，主环境已激活，MLflow server 已启动（终端 1）。

### 第一天：数据和模型

```bash
# 1. 数据：CFPB API，每个产品组 1000 条带原文的投诉，2023 年以后
python scripts/01_download_cfpb.py
#    API 不稳定时的备选：手动下载 https://files.consumerfinance.gov/ccdb/complaints.csv.zip 解压后
#    python scripts/01_download_cfpb.py --from-csv complaints.csv

# 2. Reg E 原文：eCFR API，按 § 分块
python scripts/02_fetch_reg_e.py

# 3. embedding：bge-small 在 GPU 上，建 FAISS 索引，并存投诉 embedding 给 TF 模型用
python scripts/03_embed_and_index.py

# 4. PyTorch 微调 DistilBERT（3060 上几分钟），记录到 MLflow 并注册
python scripts/04_train_classifier.py

# 5. TensorFlow 赔付模型（CPU 即可）——在 TF 环境里
.venv-tf/bin/python scripts/05_train_relief_tf.py

# 晚上：跑通 LangGraph（分类器本地加载，TF 模型先关闭）
python scripts/08_demo.py
```

### 第二天：评估、上线、演示

```bash
# 6. 评估：分类器重新打分 + 检索 recall@5 + 读取 TF 指标，给两个模型打 champion 别名
python scripts/06_evaluate.py           # 结果在 reports/eval_report.md 和 MLflow 的 evaluation run

# 7. 上线：两个终端各起一个服务
bash scripts/07_serve_models.sh classifier      # 主环境 → :5001
source .venv-tf/bin/activate && bash scripts/07_serve_models.sh relief   # TF 环境 → :5002

#    config.yaml 里改：
#      serving.classifier_backend: http
#      serving.relief_backend: http

# 8. 端到端演示（两个模型都通过 HTTP 调用，Claude 抽取和起草，人工审批）
python scripts/08_demo.py                                   # 测试集里第一条 Reg E 范围内的投诉
python scripts/08_demo.py --index 12                        # 指定测试集行
python scripts/08_demo.py --text "Someone withdrew \$800 ..." --tags "Older American"
python scripts/08_demo.py --auto-approve                    # 不交互
```

`08_demo.py` 会逐节点打印输出，在 `human_review` 停下来等你输入 `approve` / `edit` / `reject`，然后从 SQLite checkpoint 恢复继续。换个 `--thread` 可以查看之前的 run 状态。

## 6. 每项要求对应的代码

| 要求 | 在哪里 | 看什么 |
|---|---|---|
| PyTorch + HuggingFace | `scripts/04_train_classifier.py` | `AutoModelForSequenceClassification`、`Trainer`、`report_to=["mlflow"]`、bf16 |
| HuggingFace (embedding) | `triage/retrieval.py`, `scripts/03` | `HuggingFaceEmbeddings(BAAI/bge-small-en-v1.5)` |
| TensorFlow | `scripts/05_train_relief_tf.py` | Keras 函数式模型、`class_weight`、PR-AUC、`EarlyStopping` |
| MLOps (MLflow) | `triage/tracking.py`, `scripts/04/05/06/07` | `log_params/log_metrics`、`log_model(registered_model_name=)`、`set_registered_model_alias`、`models:/name@champion`、`mlflow models serve` |
| LangChain | `triage/llm.py`, `retrieval.py`, `nodes.py` | `ChatAnthropic.with_structured_output(Pydantic)`、`FAISS.as_retriever()` |
| LangGraph | `triage/nodes.py`, `graph.py` | `StateGraph` + TypedDict state、`Annotated[list, operator.add]` reducer、条件边、重写循环、`SqliteSaver`、`interrupt()` / `Command(resume=)` |

## 7. 评估指标

`scripts/06_evaluate.py` 写出 `reports/eval_report.md`，并记录到 MLflow：

- **分类**：accuracy、macro-F1、每个产品组的准确率（标签来自 CFPB 原始 product 字段）
- **检索**：recall@5，用 `evals/retrieval_questions.jsonl` 的 25 条问题（问题 → 期望条款）
- **赔付模型**：PR-AUC、ROC-AUC，以及阈值下的 precision / recall（正样本比例低，看 PR-AUC 不看 accuracy）

想比较不同配置（max_len、base_model、阈值），改 `config.yaml` 再跑一次 04/05/06 即可；每次都是新的 run 和新的模型版本，`06` 会把指标最好的版本设为 `champion`。

## 8. 成本

- 分类、抽取用 Haiku；起草用 Sonnet。一条投诉走完整流程约 $0.01–0.02。
- 开发调试加评估，两天总计十几美元以内。
- 模型训练和 embedding 全在本地 GPU，不花 API 费用。
- 没有 key 或想省钱时：`TRIAGE_FAKE_LLM=1`，两个 LLM 节点返回固定的假输出，其余流程照常。

## 9. 已知局限

- 只覆盖 Reg E；其他产品组走模板回复。
- 检索只用向量，没有 BM25 / rerank。
- `verify_citations` 只检查"引用的条款是否来自检索结果"，不检查引用内容是否真的支持结论。
- 赔付模型的标签（是否金钱赔偿）和回复无因果关系，只作为优先级信号。
- 模型服务没有鉴权、没有限流，仅限本机。

## 10. 常见问题

- **`mlflow models serve` 报 `torchvision` 缺失**：`04` 已显式传 `pip_requirements`，不会触发；如果自己改了 log_model，记得保留。
- **CFPB API 超时**：脚本会重试；仍失败就用 `--from-csv`。
- **eCFR 返回 404**：`--date` 指定一个较早的日期，例如 `--date 2026-09-01`。
- **CUDA 不可用**：`04` 会自动退到 CPU 并关闭 bf16，只是慢。
- **TF 环境 import 报错**：确认 `05` 和 `07 relief` 都用 `.venv-tf` 的 python。
- **想重新开始**：删 `data/processed data/index data/checkpoints.sqlite mlflow.db mlruns/`。

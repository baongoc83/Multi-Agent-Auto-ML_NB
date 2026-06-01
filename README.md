# Multi-Agent AutoML Pipeline

LLM-driven AutoML pipeline gồm 3 agent chuyên trách chạy tuần tự, được tối ưu cho các bài toán **credit risk**, **propensity**, và **fraud detection** trong ngân hàng — nhưng hoạt động được với bất kỳ dataset binary classification nào.

```
Raw data  ─►  Agent 1: DataCleaner  ─►  Agent 2: FeatureEngineer  ─►  Agent 3: TrainModel  ─►  Trained model + reports
              audit + clean             encode + interact +           FLAML → Optuna → RFE →
                                        select features              PSI → Stability → SHAP+PSI
```

---

## Tính năng chính

- **3 agent độc lập** giao tiếp qua object `Handoff` — mỗi agent có prompt, tool registry và logic riêng
- **LLM fallback 3 lớp**: LiteLLM proxy → OpenAI direct → Claude direct (tự xử lý retry, rate-limit, kết nối)
- **Routing model thông minh**: prompt ngắn → local model, prompt dài → cloud model (qua `MODEL_ROUTING_THRESHOLD`)
- **OOT temporal split tự động** từ cột date — tự cân bằng `OOT_MIN_RATIO` / `OOT_MAX_RATIO`
- **Pipeline feature selection 5 tầng**: RFE → PSI drift → Stability Gini → SHAP+PSI iterative pruning → top-N cut
- **Overfitting detection + LLM-guided retrain** khi gap valid/holdout vượt ngưỡng
- **GPU auto-detect** cho LightGBM, XGBoost, CatBoost
- **Sinh code inference standalone** (`outputs/final_model_code.py`) — chạy độc lập với pipeline

---

## Cài đặt

```powershell
# 1. Clone repo & vào thư mục
cd multi-agent-auto-ml-v1.1

# 2. Tạo virtualenv (khuyến nghị)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Cài dependencies
pip install -r requirements.txt

# 4. (Tuỳ chọn) cài LiteLLM nếu dùng proxy
pip install litellm
```

### Cấu hình LLM endpoint

Project hỗ trợ **2 backend** chọn qua `LLM_BACKEND` env var:

#### Backend `gateway` (mới — đơn giản hơn)

Toàn bộ call qua **một LiteLLM gateway duy nhất** dùng Anthropic API convention. Gateway tự xử lý failover.

```env
LLM_BACKEND=gateway
ANTHROPIC_AUTH_TOKEN=sk-xxxx
ANTHROPIC_BASE_URL=https://llm-gateway-dev.example.com
API_TIMEOUT_MS=3000000

# Model aliases — gateway dispatch tới underlying model
ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-haiku-4-5
ANTHROPIC_DEFAULT_SONNET_MODEL=claude-sonnet-4-6
ANTHROPIC_DEFAULT_OPUS_MODEL=claude-opus-4-7

# Routing: low=Haiku, medium=Haiku/Sonnet, high=Sonnet/Opus
EFFORT_LEVEL=medium
```

#### Backend `legacy` (mặc định — fallback 3 lớp)

LiteLLM proxy → OpenAI direct → Claude direct. Cần **ít nhất một**:

```env
LLM_BACKEND=legacy

# Phương án A: LiteLLM proxy (khuyến nghị)
LITELLM_URL=http://localhost:4000
LOCAL_MODEL=local-model
CLOUD_MODEL=cloud-model

# Phương án B: OpenAI trực tiếp (fallback nếu proxy down)
OPENAI_API_KEY=sk-...
OPENAI_DIRECT_MODEL=gpt-4.1-mini

# Phương án C: Claude trực tiếp (fallback cuối)
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_DIRECT_MODEL=claude-sonnet-4-6
```

Khởi động LiteLLM proxy local nếu dùng:
```powershell
.\start_litellm.ps1
```

---

## Sử dụng

### Chạy nhanh nhất

```powershell
python main.py data/train.csv TARGET
```

Hỗ trợ định dạng: `.csv`, `.tsv`, `.parquet`, `.orc`, `.feather`, `.xlsx`, `.xls`, `.xlsm`, `.json`, và remote (`s3://`, `gs://`, `az://`).
Hỗ trợ compression (file kết thúc bằng): `.gz`, `.bz2` (alias `.b2`), `.xz`, `.zst`, `.zip` — auto-detect và decompress (vd. `train.csv.gz`).

### Credit risk (HomeCredit, KAGGLE)

```powershell
python main.py data/application_train.csv TARGET `
    --domain credit_risk --model-type binary_classification `
    --entity-id SK_ID_CURR --keys SK_ID_CURR,MONTH_DT `
    --col-desc data/HomeCredit_columns_description.csv `
    --col-name-field Row --col-desc-field Description --col-group-field Table
```

### Propensity model

```powershell
python main.py data/leads.csv converted `
    --domain propensity --entity-id customer_id --keys customer_id,event_date
```

### Fraud detection

```powershell
python main.py data/txns.parquet is_fraud `
    --domain fraud --entity-id account_id --keys account_id,txn_ts
```

### Với OOT đã tách sẵn

```powershell
python main.py data/train.parquet TARGET --oot data/oot.parquet
```

### Tham số CLI quan trọng

| Flag | Mô tả |
|---|---|
| `--domain` | `credit_risk \| propensity \| fraud \| generic` — định hướng feature engineering |
| `--model-type` | `binary_classification \| regression \| multiclass` |
| `--entity-id COL` | Cột entity ID (e.g. `customer_id`, `SK_ID_CURR`) — bảo vệ khỏi xoá / encode |
| `--keys C1,C2` | Composite key cho dedup check + OOT temporal extraction |
| `--oot PATH` | OOT đã split sẵn — bỏ qua bước extract temporal |
| `--col-desc PATH` | File CSV/JSON mô tả cột — guide LLM tạo interaction có ý nghĩa |

---

## Cấu trúc project

```
multi-agent-auto-ml-v1.1/
├── main.py                          # CLI entry point
├── pipeline.py                      # AutoMLPipeline — orchestrator
├── handoff.py                       # State holder giữa các agent
├── logger.py                        # AgentLogger — log + markdown report
├── config.py                        # Toàn bộ config + LLM client factory
├── config.yaml                      # LiteLLM proxy config
├── start_litellm.ps1                # Helper khởi động proxy
├── requirements.txt
│
├── Agents/
│   ├── BaseAgent/
│   │   └── base_agent.py            # Base class + LLM call + load_dataframe
│   ├── DataCleaner/
│   │   ├── agent_data_cleaner.py    # Agent 1
│   │   └── prompts/                 # system.txt, user.txt
│   ├── FeatureEngineer/
│   │   ├── agent_feature_engineer.py # Agent 2
│   │   └── prompts/
│   └── TrainModel/
│       ├── agent_train_model.py     # Agent 3 (FLAML → Optuna → ...)
│       └── prompts/
│
├── data/
│   ├── process_home_data.py         # Script chuẩn bị HomeCredit dataset
│   └── HomeCredit_columns_description.csv
│
├── docs/
│   └── config_params.md             # Reference đầy đủ mọi config param
│
├── outputs/                         # Tất cả output của pipeline (auto-tạo)
└── tests/
    ├── test_agent1.py               # Test riêng Agent 1
    ├── test_agent2.py
    └── test_agent3.py
```

---

## Output

Sau khi chạy xong, `outputs/` chứa:

| File | Mô tả |
|---|---|
| `clean_data.csv` | Dataset sau Agent 1 |
| `engineered_data.csv` | Dataset sau Agent 2 |
| `final_model.pkl` | Model + encoders + feature list (joblib) |
| `final_model_code.py` | Code inference standalone |
| `data_cleaner_report.json` | JSON report Agent 1 |
| `feature_engineer_report.json` | JSON report Agent 2 |
| `model_trainer_report.json` | JSON report Agent 3 |
| `psi_report.csv` | PSI drift của từng feature |
| `stability_report.csv` | Mean / std Gini theo tháng |
| `shap_psi_prune_log.csv` | Log từng step pruning |
| `final_report.md` | Markdown report toàn pipeline |
| `agent_execution.log` | Execution log chi tiết |

### Sử dụng model đã train

```python
import pandas as pd
import sys
sys.path.append("outputs")
from final_model_code import predict_proba, predict

df = pd.read_csv("new_data.csv")
scores = predict_proba(df)          # xác suất class dương
preds  = predict(df, threshold=0.5)
```

---

## Test từng agent riêng

```powershell
# Agent 1 — sẽ tự sinh sample data
python tests/test_agent1.py

# Agent 2 — cần outputs/clean_data.csv (chạy Agent 1 trước)
python tests/test_agent2.py

# Agent 3 — cần outputs/engineered_data.csv (chạy Agent 2 trước)
python tests/test_agent3.py
```

---

## Tài liệu chi tiết

- [docs/config_params.md](docs/config_params.md) — Reference đầy đủ ~70 config param (override qua `.env`)

---

## Pipeline chi tiết — Agent 3

```
Train data  ─►  FLAML        ─►  estimator + base hyperparams (AutoML)
            ─►  Optuna       ─►  fine-tune hyperparams (TPE, valid_temporal split)
            ─►  RFE          ─►  cắt xuống MAX_FINAL_FEATURES
            ─►  PSI filter   ─►  loại feature drift > PSI_THRESHOLD (train vs OOT)
            ─►  Stability    ─►  loại feature có std Gini theo tháng > threshold
            ─►  SHAP+PSI     ─►  iterative prune cho đến khi AUC valid ngừng tăng
            ─►  Final train  ─►  CV + valid_temporal + valid_random + OOT + test
            ─►  Overfit check ─► nếu detected → LLM-guided retrain với reg mạnh hơn
```

---

## Yêu cầu môi trường

- Python 3.10+
- Windows / macOS / Linux
- GPU (tuỳ chọn) — auto-detect qua `nvidia-smi`
- RAM khuyến nghị: ≥ 16 GB cho dataset 100k+ rows × 500+ cols

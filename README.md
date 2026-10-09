# Multi-Agent AutoML Pipeline

LLM-driven AutoML pipeline gồm 3 agent chuyên trách chạy tuần tự, được tối ưu cho các bài toán **credit risk**, **propensity**, và **fraud detection** trong ngân hàng — nhưng hoạt động được với bất kỳ dataset binary classification nào.

```
Raw data  ─►  Agent 1: DataCleaner  ─►  Agent 2: FeatureEngineer  ─►  Agent 3: TrainModel  ─►  Trained model + reports
              audit + clean             encode + interact +           FLAML → Optuna → RFE →
                                        select features              PSI → Stability → SHAP+PSI
```

---

## Tính năng chính

- **2 chế độ chạy**:
  - **Split mode** (khi có `--valid` hoặc `--oot`): mỗi agent FIT trên train, capture spec, REPLAY lên valid/oot — không bao giờ load cả 3 partition cùng lúc → peak RAM bằng partition lớn nhất
  - **Single-file mode**: Agent 3 tự auto-split (temporal OOT hoặc 60/20/20)
- **3 agent độc lập** với prompt, tool registry và logic riêng
- **CleaningSpec / FeatureSpec dataclasses** capture mọi transform từ train, replay deterministic lên valid/oot — không re-run LLM, không leakage
- **Stage 0 safety checks** (split mode): schema validation (fail-fast) + PSI distribution drift (advisory)
- **LLM fallback 3 lớp** (legacy backend): LiteLLM proxy → OpenAI direct → Claude direct
- **Gateway backend** cho mạng nội bộ — single source, gateway tự handle failover
- **OOT temporal split tự động** từ cột date — tự cân bằng `OOT_MIN_RATIO` / `OOT_MAX_RATIO`
- **Pipeline feature selection 5 tầng**: RFE → PSI drift → Stability Gini → SHAP+PSI iterative pruning → top-N cut
- **Refit-on-train+valid**: sau feature/hyperparam selection, model deploy được refit trên train+valid gộp (CV tìm `best_iteration` ổn định + calibration trên OOF, không leakage); OOT/test giữ nguyên làm holdout sạch
- **Overfitting detection + LLM-guided retrain** khi gap CV-OOF/holdout (`ref_auc` vs `oot`/`test`) vượt ngưỡng
- **GPU auto-detect** cho LightGBM, XGBoost, CatBoost
- **Smart Excel reader**: magic-byte detection cho file `.xls/.xlsx` bị đặt sai extension
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

# 4. (Tuỳ chọn) cài LiteLLM nếu dùng legacy backend với proxy
pip install litellm
```

### Cấu hình LLM endpoint

Project hỗ trợ **2 backend** chọn qua biến `LLM_BACKEND` trong file `.env`. Chọn backend nào tuỳ vào **vị trí mạng** bạn đang dùng:

| Tình huống | Backend nên dùng | Lý do |
|---|---|---|
| **Đang trong mạng nội bộ** (VPN công ty, văn phòng) — có thể truy cập LLM gateway nội bộ | **`gateway`** | Gateway hosted trong mạng nội bộ, dùng key chung của tổ chức, không cần OpenAI/Claude key cá nhân |
| **Đang ở mạng ngoài** (nhà, quán cafe, công cộng) — KHÔNG truy cập được gateway nội bộ | **`legacy`** | Đi thẳng tới OpenAI / Claude qua Internet công cộng bằng key cá nhân, không cần VPN |

> Switch giữa 2 backend **không cần đổi code** — chỉ cần đổi `LLM_BACKEND` trong `.env` rồi chạy lại.

---

#### Backend `gateway` — dùng khi ở **mạng nội bộ**

Toàn bộ call qua **một LiteLLM gateway nội bộ duy nhất** (Anthropic API convention). Gateway tự xử lý routing và failover phía sau.

**Yêu cầu**: kết nối được tới `ANTHROPIC_BASE_URL` (thường chỉ accessible qua VPN / intranet).

```env
# .env
LLM_BACKEND=gateway

# Endpoint nội bộ — phải reachable từ máy bạn
ANTHROPIC_AUTH_TOKEN=sk-xxxx
ANTHROPIC_BASE_URL=https://llm-gateway-dev.example.com
API_TIMEOUT_MS=3000000

# Model aliases — gateway tự dispatch tới underlying model
ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-haiku-4-5
ANTHROPIC_DEFAULT_SONNET_MODEL=claude-sonnet-4-6
ANTHROPIC_DEFAULT_OPUS_MODEL=claude-opus-4-7

# Routing strategy: low=luôn Haiku | medium=Haiku/Sonnet | high=Sonnet/Opus
EFFORT_LEVEL=medium
```

**Test nhanh trước khi chạy pipeline**:
```powershell
$env:GATEWAY_DEMO_REAL = "1"
python examples/gateway_demo.py
```
Nếu thấy response từ LLM → gateway sống → có thể yên tâm chạy `python main.py ...`.

Nếu lỗi `Connection refused` / `timeout` → kiểm tra VPN, hoặc switch sang backend `legacy`.

---

#### Backend `legacy` — dùng khi ở **mạng ngoài** (mặc định)

3-tier fallback đi qua Internet công cộng: LiteLLM proxy local → OpenAI direct → Claude direct. Cần **ít nhất một** trong 3 phương án.

```env
# .env
LLM_BACKEND=legacy

# Phương án A: LiteLLM proxy chạy local (port 4000)
LITELLM_URL=http://localhost:4000
LOCAL_MODEL=local-model
CLOUD_MODEL=cloud-model

# Phương án B: OpenAI trực tiếp qua Internet công cộng
OPENAI_API_KEY=sk-...
OPENAI_DIRECT_MODEL=gpt-4.1-mini

# Phương án C: Claude trực tiếp qua Internet công cộng (fallback cuối)
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_DIRECT_MODEL=claude-sonnet-4-6
```

Nếu dùng phương án A (proxy local), khởi động proxy ở terminal khác trước:
```powershell
.\start_litellm.ps1
```

Nếu chỉ dùng phương án B hoặc C (OpenAI/Claude trực tiếp), không cần chạy `start_litellm.ps1` — pipeline tự fallback xuống provider trực tiếp.

---

#### Verify đang dùng backend nào

Trước khi chạy pipeline lớn, kiểm tra nhanh backend:

```powershell
python -c "from config import Config; print(f'Backend={Config.BACKEND}')"
```

Sau khi pipeline chạy xong, kiểm tra log:
```powershell
# Nếu thấy nhiều dòng có 'backend=gateway' = đã đi qua gateway
Select-String -Path outputs/agent_execution.log -Pattern "backend=gateway" | Measure-Object | Select-Object Count
```

---

## Sử dụng

### Chạy nhanh nhất

```powershell
python main.py data/train.csv TARGET
```

Hỗ trợ định dạng: `.csv`, `.tsv`, `.parquet`, `.orc`, `.feather`, `.xlsx`, `.xls`, `.xlsm`, `.json`, và remote (`s3://`, `gs://`, `az://`).
Hỗ trợ compression (file kết thúc bằng): `.gz`, `.bz2` (alias `.b2`), `.xz`, `.zst`, `.zip` — auto-detect và decompress (vd. `train.csv.gz`).

Excel files tự động chọn engine từ **magic bytes**, không phải extension — file `.xls` rename thành `.xlsx` vẫn đọc được, và ngược lại.

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

### Với split đã tách sẵn — split mode

Bạn có thể truyền **train / valid / oot** riêng biệt. Khi `--valid` hoặc `--oot` được set, pipeline chuyển sang **split mode**:

1. **Stage 0a** — Schema validation: hard-fail nếu thiếu target / entity_id / composite_keys ở bất kỳ partition nào, log WARN cho dtype mismatch
2. **Stage 0b** — Distribution drift check: sample 50K rows mỗi partition, tính PSI từng numeric feature, log WARN nếu drift > 0.25
3. **Agent 1** FIT trên train → save `clean_train.parquet` + capture `CleaningSpec` → REPLAY spec lên valid/oot → save `clean_valid.parquet`, `clean_oot.parquet`
4. **Agent 2** FIT trên `clean_train.parquet` → capture `FeatureSpec` (encoders + interactions + selected_features) → REPLAY lên `clean_valid/oot.parquet`
5. **Agent 3** load 3 file engineered, gắn marker `_split_`, concat → run training pipeline → user-supplied splits được dựng lại exact

Mỗi agent **chỉ load 1 partition trong RAM tại 1 thời điểm** → peak memory ≈ partition lớn nhất, không phải tổng 3 file. Đây là lý do split mode chạy được trên dataset 1M+ rows × 3000+ features mà concat-mode cũ OOM.

```powershell
# Full split: cả train, valid, oot riêng biệt
python main.py data/train.parquet TARGET `
    --valid data/valid.parquet `
    --oot   data/oot.parquet `
    --domain credit_risk `
    --entity-id customer_id --keys customer_id,snap_dt

# Chỉ có train + valid (không có oot)
python main.py data/train.parquet TARGET --valid data/valid.parquet

# Chỉ có train + oot (không có valid)
# → Agent 3 tự auto-split 20% của train thành valid
python main.py data/train.parquet TARGET --oot data/oot.parquet
```

### Behavior matrix — 5 scenarios

| Input | Mode | Splits Agent 3 tạo |
|---|---|---|
| `train + valid + oot` | split | train + valid + oot (exact, theo marker do Agent 3 gán) |
| `train + valid` | split | train + valid (exact), no oot |
| `train + oot` | split | train (80%) + valid (20% auto từ train) + oot (exact) |
| `train` only (có date_col) | single | train + valid_temporal + valid_random + oot (OOT temporal) |
| `train` only (no date_col) | single | train (60%) + valid (20%) + test (20%) |

> Trong split mode, row-only operations (`drop_duplicates`, `deduplicate_by_key`) **chỉ apply lên train** — valid/oot giữ nguyên row count để evaluation metric không bị bias.

### Tham số CLI quan trọng

| Flag | Mô tả |
|---|---|
| `--domain` | `credit_risk \| propensity \| fraud \| generic` — định hướng feature engineering |
| `--model-type` | `binary_classification \| regression \| multiclass` |
| `--entity-id COL` | Cột entity ID (e.g. `customer_id`, `SK_ID_CURR`) — bảo vệ khỏi drop / clip / encode |
| `--keys C1,C2` | Composite key cho dedup check + OOT temporal extraction |
| `--valid PATH` | Valid đã tách sẵn — kích hoạt split mode |
| `--oot PATH` | OOT đã tách sẵn — kích hoạt split mode |
| `--sample-ratio R` | (Split mode) Stratified sample TRAIN với tỉ lệ R, valid/oot giữ nguyên. VD `--sample-ratio 0.3` |
| `--no-prefilter` | (Split mode) Tắt auto-drop cột rác trong Agent 1. Mặc định: bật |
| `--no-drift-check` | (Split mode) Tắt Stage 0b PSI drift check. Mặc định: bật |
| `--col-desc PATH` | File CSV/JSON/Parquet/Excel mô tả cột — guide LLM tạo interaction có ý nghĩa |

### Memory savers cho dataset lớn (1M+ rows × 1000+ features)

Pipeline áp dụng **4 lớp tối ưu RAM mặc định bật**, đủ để chạy dataset 500k rows × 1500 cols dưới 8 GB:

1. **Agent 1 — Pre-filter cột**: scan train, drop cột null>95% / constant / dominant>99%. Drop list lưu trong `CleaningSpec.drops` → replay xuống valid/oot tự động
2. **Agent 2 — Metadata-only dtype iter**: `select_top_features` đọc `df.dtypes` thay vì slice `df[feature_cols]` (slice trigger block consolidation tạo float64 matrix `(n_cols, n_rows)` → từng spike vài GB). Drop in-place thay slice ở cuối.
3. **Agent 3 — dtype downcast tại fit**: `_fit_prepare_X` rewrite — bỏ `df.copy()` đầu function, build dict từng cột:
   - Object/cat → smallest signed int (int8 nếu ≤127 cats, int16 nếu ≤32k, int32 nếu lớn hơn)
   - Float64 → **float32** (50% RAM)
   - Int64 → int32
   - Bool → int8
4. **Parquet+snappy intermediate files**: ~10× nhỏ hơn CSV, load Agent kế tiếp nhanh hơn
5. **Per-partition release**: `del agent + gc.collect()` sau mỗi agent → peak RAM ≈ partition lớn nhất, không phải tổng 3 file

**Benchmark dtype downcast (50k × 1400 mixed cols)**: 1.82 GB → 190 MB (≈9.6×). Cho case thực 100k × 1455: ~1.08 GB int64 alloc → ~150-450 MB tùy tỉ lệ cat/num.

**Joblib temp folder** — `JOBLIB_TEMP_FOLDER` được auto-set tại `config.py` về `tempfile.gettempdir()` ngay khi import (`/tmp` trên Linux, `%TEMP%` trên Windows). Tránh `BrokenProcessPool: FileNotFoundError: '/dev/shm/joblib_memmapping_folder_...'` khi `/dev/shm` nhỏ (Docker default 64 MB). Override qua `.env` nếu `/tmp` cũng nhỏ:

```env
JOBLIB_TEMP_FOLDER=/data/joblib_tmp
```

Nếu vẫn OOM (>3M rows × 3000+ features cần >24 GB RAM cho riêng train), thêm `--sample-ratio`:

```powershell
# Dataset 3M train x 3500 features → sample 30% train, valid/oot giữ nguyên
python main.py data/train.parquet TARGET `
    --valid data/valid.parquet --oot data/oot.parquet `
    --sample-ratio 0.3 `
    --domain credit_risk --entity-id customer_id --keys customer_id,snap_dt
```

Sampling chỉ áp dụng **train**, stratified theo target → giữ class balance. Valid/oot không bao giờ bị sample (để đánh giá đúng performance trên dữ liệu thật).

---

## Cấu trúc project

```
multi-agent-auto-ml-v1.1/
├── main.py                          # CLI entry point
├── pipeline.py                      # AutoMLPipeline — orchestrator + Stage 0a/0b
├── splitting.py                     # Chia train/valid/oot — dùng chung pipeline + Agent 3
├── replay_driver.py                 # Template replay_pipeline.py cho mỗi run
├── logger.py                        # AgentLogger — log + markdown report
├── config.py                        # Toàn bộ config + LLM client factory
├── config_gateway.py                # GatewayConfig override khi LLM_BACKEND=gateway
├── config.yaml                      # LiteLLM proxy config
├── start_litellm.ps1                # Helper khởi động proxy
├── requirements.txt
├── _e2e_check.py                    # 6-test end-to-end smoke suite (mock LLM)
│
├── Agents/
│   ├── BaseAgent/
│   │   └── base_agent.py            # LLM call + load_dataframe + smart Excel reader
│   ├── DataCleaner/
│   │   ├── agent_data_cleaner.py    # Agent 1 + CleaningSpec
│   │   └── prompts/                 # system.txt, user.txt
│   ├── FeatureEngineer/
│   │   ├── agent_feature_engineer.py # Agent 2 + FeatureSpec
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
│   ├── architecture.md              # Visual walkthrough + mermaid diagrams
│   ├── awareness-pattern.md         # Agentic pattern reference
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

### Layout — mỗi lần chạy 1 run dir, counter reset mỗi ngày

```
outputs/
└── 2026-06-08/                          ← thư mục theo ngày (UTC local)
    ├── run_01/                          ← lần chạy 1 của hôm nay
    │   ├── agent_execution.log
    │   ├── data_cleaner_report.json
    │   ├── feature_engineer_report.json
    │   ├── model_trainer_report.json
    │   ├── psi_report.csv
    │   ├── stability_report.csv
    │   ├── shap_psi_prune_log.csv
    │   ├── shap_summary.png                          ← NEW: bar plot top features final model
    │   ├── shap_feature_explanations.csv             ← NEW: LLM-narrated top N features
    │   ├── final_report.md                              (đã nhúng SHAP section + image)
    │   ├── final_model.pkl
    │   ├── final_model_code.py
    │   ├── pipeline_process_data_cleaner.py        ← NEW: replay Agent 1
    │   ├── pipeline_process_feature_engineer.py    ← NEW: replay Agent 2
    │   ├── feature_spec.pkl                        ← NEW: sidecar cho Agent 2
    │   └── pipeline_process_train_model.py         ← NEW: replay Agent 3
    ├── run_02/                          ← lần chạy 2 của hôm nay
    └── run_03/
└── 2026-06-09/                          ← sang ngày mới → counter reset
    └── run_01/
```

Counter `run_NN` đếm dựa trên `max(NN) + 1` của các thư mục `run_*` đã tồn tại trong `outputs/YYYY-MM-DD/`. Sang ngày mới, thư mục ngày mới rỗng → tự động bắt đầu lại từ `run_01`.

### Files trong mỗi run

| File | Mô tả |
|---|---|
| `agent_execution.log` | Execution log chi tiết (timestamp + agent + action) |
| `data_cleaner_report.json` | JSON Agent 1 — bao gồm `cleaning_spec` |
| `feature_engineer_report.json` | JSON Agent 2 — bao gồm `feature_spec` |
| `model_trainer_report.json` | JSON Agent 3 — best model, params, metrics |
| `psi_report.csv` | PSI drift train↔OOT từng feature |
| `stability_report.csv` | Mean / std Gini theo tháng |
| `shap_psi_prune_log.csv` | Log từng step SHAP+PSI pruning |
| `shap_summary.png` | Bar plot top (`2 × TOP_N`) features của **final model** bằng mean \|SHAP\| |
| `shap_feature_explanations.csv` | Rank + SHAP importance + `description` + LLM-narrated `meaning` / `why_matters` cho top `SHAP_FINAL_EXPLAIN_TOP_N` (default 20) features |
| `final_report.md` | Markdown report tổng hợp — nhúng `shap_summary.png` + table top features |
| `final_model.pkl` | Model + encoders + feature list (joblib) |
| `final_model_code.py` | Code inference standalone |
| `pipeline_process_data_cleaner.py` | Script replay Agent 1 trên data mới — embed `CleaningSpec` (drops + dtype_fixes + clip_bounds) inline, no agent/LLM dependency |
| `pipeline_process_feature_engineer.py` + `feature_spec.pkl` | Script replay Agent 2 — load `feature_spec.pkl` sidecar (chứa fitted LabelEncoders) + apply interactions / encoders / one-hot / selected_features |
| `pipeline_process_train_model.py` | Script replay Agent 3 — content giống `final_model_code.py` (inference dùng artifact đã train) |

### Intermediate files KHÔNG còn lưu mặc định

`clean_*.parquet` + `engineered_*.parquet` (handoff giữa các agent) được ghi vào **system tempdir** (`tempfile.mkdtemp()`) và **xoá khi pipeline kết thúc** (kể cả raise). Không chiếm chỗ đĩa.

Nếu cần inspect intermediate (debugging, kiểm tra split mode), set:

```env
KEEP_INTERMEDIATES=true
```

→ intermediate files ghi thẳng vào `run_NN/` cùng với các artifact khác.

### Sử dụng replay script

```powershell
# Apply lại Agent 1 transforms (drops + dtype fixes + clip) lên data mới
python outputs/2026-06-08/run_01/pipeline_process_data_cleaner.py `
    new_data.csv  cleaned.parquet

# Apply Agent 2 transforms (encoders + interactions + select)
# Đặt cả 2 file pipeline_process_feature_engineer.py + feature_spec.pkl cùng folder
python outputs/2026-06-08/run_01/pipeline_process_feature_engineer.py `
    cleaned.parquet  engineered.parquet

# Inference qua Agent 3 — final_model.pkl đặt cùng folder
python outputs/2026-06-08/run_01/pipeline_process_train_model.py
```

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

## End-to-end smoke test

Project có sẵn `_e2e_check.py` — 6 test với mock LLM, chạy nhanh để verify toàn bộ flow không gãy:

```powershell
python _e2e_check.py
```

Tests cover:
1. SINGLE-FILE mode
2. SPLIT mode (verify exact row count preserve + schema parity 3 files)
3. SPLIT mode + sample_ratio + prefilter (junk col drops replay)
4. Schema validation: missing target → raise trước Agent 1
5. Schema validation: missing entity_id / composite_key → raise; dtype mismatch → warn only
6. Distribution drift: WARN khi shift, không false positive trên stable partition

---

## Test từng agent riêng

```powershell
# Agent 1 — sẽ tự sinh sample data
python tests/test_agent1.py

# Agent 2 — cần outputs/clean_data.parquet (chạy Agent 1 trước)
python tests/test_agent2.py

# Agent 3 — cần outputs/engineered_data.parquet (chạy Agent 2 trước)
python tests/test_agent3.py
```

---

## Tài liệu chi tiết

**Tổng quan & reference**

- [docs/architecture.md](docs/architecture.md) — Visual walkthrough + mermaid diagrams cho từng agent
- [docs/awareness-pattern.md](docs/awareness-pattern.md) — Agentic pattern reference (Plan-Execute + Awareness)
- [docs/config_params.md](docs/config_params.md) — Reference đầy đủ ~75 config param (override qua `.env`)

**Flow chi tiết từng agent** (step-by-step + mermaid + no-leakage rules + config knobs)

- [docs/agent-1-data-cleaner.md](docs/agent-1-data-cleaner.md) — DataCleaner: prefilter → real-stats → LLM → canonical-ordered execute → spec replay
- [docs/agent-2-feature-engineer.md](docs/agent-2-feature-engineer.md) — FeatureEngineer: domain-aware interactions, label/onehot encoders, top-K selection, FeatureSpec replay
- [docs/agent-3-train-model.md](docs/agent-3-train-model.md) — TrainModel 11 bước: split → encode → FLAML → Optuna → RFE → PSI → Stability → SHAP+PSI prune → final train → overfit-aware retrain

---

## Pipeline chi tiết — Agent 3

```
Train data  ─►  FLAML        ─►  estimator + base hyperparams (AutoML)
            ─►  Optuna       ─►  fine-tune hyperparams (TPE, valid_temporal split)
            ─►  RFE          ─►  cắt xuống RFE_TARGET_FEATURES
            ─►  PSI filter   ─►  loại feature drift > PSI_THRESHOLD (train vs OOT)
            ─►  Stability    ─►  loại feature có std Gini theo tháng > threshold
            ─►  SHAP+PSI     ─►  iterative prune về ~MAX_FINAL_FEATURES (FEATURE_CAP_POLICY)
            ─►  Refit         ─►  gộp train+valid, CV best_iteration, calibrate trên OOF
            ─►  Eval          ─►  OOT (temporal) / test (holdout) — giữ sạch, không gộp
            ─►  Overfit check ─► gap ref_auc(CV-OOF) vs holdout → LLM-guided retrain
```

---

## Yêu cầu môi trường

- Python 3.12 (pinned in `.python-version`; numpy 2.3 needs >= 3.11 and the reference runs used 3.12)
- Windows / macOS / Linux
- GPU (tuỳ chọn) — auto-detect qua `nvidia-smi`
- RAM khuyến nghị (sau dtype-downcast tại Agent 3):
  - ≥ 8 GB cho dataset 100k rows × 500 cols
  - ≥ 16 GB cho 500k rows × 1500 cols (đã verify trên sample_data_x3)
  - ≥ 32 GB cho 3M+ rows × 1500+ cols (hoặc dùng `--sample-ratio`)
- Trên Linux/Docker: ưu tiên `/tmp` >= 4 GB free (xem `JOBLIB_TEMP_FOLDER` ở section memory savers)

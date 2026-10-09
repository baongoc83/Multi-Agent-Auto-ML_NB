# Multi-Agent AutoML Pipeline

LLM-driven AutoML pipeline gồm 3 agent chuyên trách chạy tuần tự, được tối ưu cho các bài toán **credit risk**, **propensity**, và **fraud detection** trong ngân hàng — nhưng hoạt động được với bất kỳ dataset binary classification nào.

```
Raw data ─► Stage 0: split ─► Agent 1: DataCleaner ─► Stage 1b: NullProcessor ─► Agent 2: FeatureEngineer ─► Agent 3: TrainModel ─► model + replay bundle
            (OOT / test)      audit + clean            NULL policy, fit on train   encode + interact + select    FLAML → rank → PSI → Stability →
                                                                                                                  batch prune → Optuna → refit
```

---

## Tính năng chính

- **2 chế độ chạy**:
  - **Split mode** (khi có `--valid` hoặc `--oot`): mỗi agent FIT trên train, capture spec, REPLAY lên valid/oot — không bao giờ load cả 3 partition cùng lúc → peak RAM bằng partition lớn nhất
  - **Single-file mode**: Stage 0 chia ngay từ dữ liệu thô (temporal OOT nếu có cột ngày, nếu không thì 60/20/20 với `test` ngẫu nhiên), rồi chạy y như split mode — Agent 1/2 không bao giờ thấy holdout
- **3 agent độc lập** với prompt, tool registry và logic riêng
- **CleaningSpec / FeatureSpec dataclasses** capture mọi transform từ train, replay deterministic lên valid/oot — không re-run LLM, không leakage
- **Stage 1b — NullProcessor** (`preprocessing/`): chính sách xử lý NULL theo từng cột (median / zero / constant / category / indicator), fit trên train, replay mọi nơi; rule do LLM gợi ý sau guardrail tự động, không cần duyệt tay; có guard đổi đơn vị (scale) và artifact JSON có hash
- **Stage 0 safety checks** (split mode): schema validation (fail-fast) + PSI distribution drift (advisory)
- **LLM backend linh hoạt**: proxy / gateway doanh nghiệp / server on-prem chuẩn OpenAI (vLLM, SGLang, Ollama — Qwen, DeepSeek, GLM); `LLM_ALLOW_EXTERNAL_FALLBACK=false` để không bao giờ gọi API public; tự bỏ phần `<think>` của model suy luận trước khi đọc JSON
- **Gateway backend** cho mạng nội bộ (chuẩn Anthropic) — single source, gateway tự handle failover
- **OOT temporal split tự động** từ cột date — tự cân bằng `OOT_MIN_RATIO` / `OOT_MAX_RATIO`
- **Chọn feature nhanh và có kiểm soát**: xếp hạng một lần theo importance + loại feature gần trùng → PSI drift → Stability Gini → prune theo lô với dung sai AUC, trần `MAX_FINAL_FEATURES` (mặc định 100). Optuna chạy một lần trên tập feature cuối, mỗi trial có giới hạn thời gian. Cách cũ (RFE, prune từng feature, re-tune) vẫn chọn được qua config
- **Refit-on-train+valid**: sau feature/hyperparam selection, model deploy được refit trên train+valid gộp (CV tìm `best_iteration` ổn định + calibration trên OOF, không leakage); OOT/test giữ nguyên làm holdout sạch. Metric `valid_*` là in-sample và được gắn nhãn rõ
- **PD floor / cap** (`PD_FLOOR=0.0003`, `PD_CAP=0.9999`) lưu cùng model, áp dụng cả lúc scoring
- **Replay bundle** cho môi trường khác: manifest có provenance (git SHA, code fingerprint, snapshot config, hash dữ liệu đầu vào, phiên bản thư viện) + SHA-256 từng artifact được kiểm trước khi `joblib.load`; scoring chế độ strict (dừng khi thiếu cột, sai kiểu, quá nhiều category lạ, lệch phiên bản thư viện)
- **An toàn biểu thức**: biểu thức feature do LLM viết được kiểm bằng AST allowlist trước khi chạy (thay `eval` trần)
- **Overfitting detection + LLM-guided retrain** khi gap CV-OOF/holdout (`ref_auc` vs `oot`/`test`) vượt ngưỡng
- **GPU auto-detect** cho LightGBM, XGBoost, CatBoost
- **Smart Excel reader**: magic-byte detection cho file `.xls/.xlsx` bị đặt sai extension

---

## Cài đặt

```powershell
# 1. Clone repo & vào thư mục
cd multi-agent-auto-ml-v1.1

# 2. Tạo virtualenv Python 3.12 (khuyến nghị venv/container sạch cho production)
py -3.12 -m venv .venv            # Linux: python3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1      # Linux: source .venv/bin/activate

# 3. Cài dependencies (đã pin chính xác theo các run tham chiếu)
pip install -r requirements.txt

# 4. (Tuỳ chọn) chỉ khi host LiteLLM proxy trên chính máy này — dùng venv RIÊNG
#    (litellm[proxy] kéo ~80 package và pin cứng pydantic; pipeline không import litellm)
py -3.12 -m venv .venv-proxy
.\.venv-proxy\Scripts\pip install -r requirements-proxy.txt
$env:LITELLM_EXE = (Resolve-Path .\.venv-proxy\Scripts\litellm.exe)   # start_litellm.ps1 dùng biến này
```

Mọi đường dẫn trong code đều tương đối: `OUTPUT_DIR` dạng tương đối được neo vào thư mục gốc repo, dữ liệu truyền qua CLI hoặc biến môi trường (`AUTOML_DATA_ROOT`, `AUTOML_TEST_DATA`).

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

Nếu chỉ dùng phương án B hoặc C (OpenAI/Claude trực tiếp), không cần chạy `start_litellm.ps1` — pipeline tự fallback xuống provider trực tiếp. Đặt `LLM_SKIP_PROXY=true` để bỏ hẳn bước thử proxy (tiết kiệm ~14 giây mỗi lời gọi khi không host proxy).

---

#### Backend `legacy` với model on-prem / enterprise cloud — dùng cho **production**

Backend `legacy` dùng API chat chuẩn OpenAI, nên `LITELLM_URL` có thể trỏ vào LiteLLM proxy, gateway doanh nghiệp, hoặc thẳng server vLLM / SGLang / Ollama (thêm `/v1`). Ví dụ cấu hình model trong proxy: xem cuối `config.yaml`.

```env
LLM_BACKEND=legacy
LITELLM_URL=http://llm-gw.internal:4000        # hoặc http://gpu-node:8000/v1 (vLLM)
API_KEY=<key của gateway / server>
LOCAL_MODEL=qwen3-8b                           # prompt ngắn (< MODEL_ROUTING_THRESHOLD ký tự)
CLOUD_MODEL=deepseek-v3                        # prompt dài — cần context >= 64k (Agent 2 ~28k token)
LLM_ALLOW_EXTERNAL_FALLBACK=false              # không bao giờ gọi OpenAI / Anthropic public
TIMEOUT=300
LLM_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}   # Qwen3 trên vLLM/SGLang
NO_PROXY=llm-gw.internal,gpu-node
```

Các tuỳ chọn khác cho server on-prem: `LLM_JSON_MODE_NATIVE`, `LLM_OMIT_PARAMS`, `LLM_STRIP_REASONING`, `LLM_MAX_TOKENS_LARGE` — mô tả trong [docs/config_params.md](docs/config_params.md#on-prem--enterprise-llm-endpoints).

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
| `--sample-ratio R` | Stratified sample TRAIN với tỉ lệ R, valid/oot giữ nguyên. VD `--sample-ratio 0.3`. Chỉ dùng khi thiếu RAM: trên Home Credit, run dùng 10% train (kèm ngân sách rút gọn) thấp hơn run đầy đủ ~0,008 test AUC |
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
├── provenance.py                    # Git SHA, code fingerprint, config snapshot, input hash
├── logger.py                        # AgentLogger — log + markdown report
├── config.py                        # Toàn bộ config + LLM client factory
├── config_gateway.py                # GatewayConfig override khi LLM_BACKEND=gateway
├── config.yaml                      # LiteLLM proxy config (kèm ví dụ model on-prem)
├── start_litellm.ps1                # Helper khởi động proxy (litellm trên PATH hoặc LITELLM_EXE)
├── requirements.txt                 # Pin chính xác — pipeline
├── requirements-proxy.txt           # Chỉ khi host LiteLLM proxy
├── .python-version                  # 3.12
├── _e2e_check.py                    # 6-test end-to-end smoke suite (mock LLM)
│
├── preprocessing/
│   ├── null_processor.py            # Stage 1b — NullProcessor (sklearn transformer, artifact JSON)
│   ├── null_rule_advisor.py         # LLM gợi ý rule NULL + guardrail tự động
│   ├── safe_expr.py                 # AST allowlist cho biểu thức feature do LLM viết
│   └── null_rules.example.yaml
│
├── tools/
│   ├── overnight.py                 # Chạy nhiều dataset liên tiếp + tóm tắt từng run
│   └── run_summary.py               # run_summary.md: metric trung thực + smoke test bundle
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
│   ├── config_params.md             # Reference đầy đủ mọi config param
│   ├── replay.md                    # Replay bundle — chạy lại / scoring ở môi trường khác
│   └── sequence_flow.md             # Sequence diagrams
│
├── outputs/                         # Tất cả output của pipeline (auto-tạo, không commit)
└── tests/
    ├── test_agent1.py ... test_agent3.py   # Test từng agent (gọi LLM thật)
    └── test_*.py                    # Test offline: replay, null processor, scoring contract,
                                     # security, chọn feature, Optuna guard, on-prem LLM ...
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
    │   ├── psi_report.csv / stability_report.csv     (chỉ khi có OOT / cột ngày)
    │   ├── shap_psi_prune_log.csv                     đường AUC theo số feature của bước prune
    │   ├── shap_summary.png / shap_beeswarm.png
    │   ├── shap_feature_explanations.csv / final_model_shap_report.md
    │   ├── charts/                                    ROC, PR, KS, lift, gain, decile, calibration
    │   ├── final_report.md
    │   ├── run_summary.md                             (khi chạy qua tools/overnight.py)
    │   │   ── replay bundle (mang sang môi trường khác) ──
    │   ├── replay_manifest.json                       provenance + SHA-256 từng artifact + metric kỳ vọng
    │   ├── replay_pipeline.py                         retrain / score lại run này
    │   ├── split_assignment.parquet                   key → partition + thứ tự dòng
    │   ├── cleaning_spec.pkl
    │   ├── null_processor.json                        Stage 1b (JSON, có hash)
    │   ├── feature_spec.pkl
    │   ├── final_model.pkl                            ensemble + calibrator + PD floor/cap + versions
    │   ├── final_model_code.py
    │   └── pipeline_process_{data_cleaner,feature_engineer,train_model}.py
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
| `shap_psi_prune_log.csv` | Đường cong của bước prune: số feature, valid AUC (thô + làm mượt), tập được chọn |
| `replay_manifest.json` | Thông tin để chạy lại run: split, spec, tham số, metric kỳ vọng, `provenance` (git SHA, code fingerprint, config snapshot, hash input, phiên bản thư viện), `artifacts` (SHA-256 từng file) |
| `replay_pipeline.py` | Driver replay: `--mode retrain` (tái lập) hoặc `--mode score` (chấm điểm dữ liệu mới) |
| `null_processor.json` | Stage 1b: rule NULL từng cột + thống kê train + giá trị imputation; có hash nội dung |
| `split_assignment.parquet` / `cleaning_spec.pkl` | Phân vùng từng dòng (theo key) / spec làm sạch |
| `shap_summary.png` | Bar plot top (`2 × TOP_N`) features của **final model** bằng mean \|SHAP\| |
| `shap_feature_explanations.csv` | Rank + SHAP importance + `description` + LLM-narrated `meaning` / `why_matters` cho top `SHAP_FINAL_EXPLAIN_TOP_N` (default 20) features |
| `final_report.md` | Markdown report tổng hợp — nhúng `shap_summary.png` + table top features |
| `final_model.pkl` | Ensemble multi-seed + calibrator + PD floor/cap + encoders + feature list + versions (joblib) |
| `final_model_code.py` | Inference trên dữ liệu **đã qua Agent 1 + Stage 1b + Agent 2**; dừng khi thiếu feature hoặc lệch phiên bản thư viện (`AUTOML_ALLOW_VERSION_MISMATCH=1` để bỏ qua) |
| `pipeline_process_data_cleaner.py` | Script replay Agent 1 trên data mới — embed `CleaningSpec` (drops + dtype_fixes + clip_bounds) inline |
| `pipeline_process_feature_engineer.py` + `feature_spec.pkl` | Script replay Agent 2 — gọi thẳng `FeatureSpec.apply(strict=True)` của repo (cần repo trên `sys.path` hoặc `AUTOML_REPO`) |
| `pipeline_process_train_model.py` | Giống `final_model_code.py` |

### Intermediate files KHÔNG còn lưu mặc định

`clean_*.parquet` + `engineered_*.parquet` (handoff giữa các agent) được ghi vào **system tempdir** (`tempfile.mkdtemp()`) và **xoá khi pipeline kết thúc** (kể cả raise). Không chiếm chỗ đĩa.

Nếu cần inspect intermediate (debugging, kiểm tra split mode), set:

```env
KEEP_INTERMEDIATES=true
```

→ intermediate files ghi thẳng vào `run_NN/` cùng với các artifact khác.

### Chấm điểm / tái lập ở môi trường khác (khuyến nghị)

Cách chuẩn là dùng replay bundle — chạy đủ chuỗi Agent 1 → Stage 1b → Agent 2 → model, kiểm hash và phiên bản trước khi load:

```powershell
# Chấm điểm dữ liệu thô mới (strict: dừng nếu thiếu cột, sai kiểu, đổi đơn vị, quá nhiều category lạ)
python outputs/2026-10-08/run_03/replay_pipeline.py new_raw.parquet --mode score --out scores.parquet

# Tái lập toàn bộ run (train lại từ quyết định đã đóng băng) và so metric với run gốc
python outputs/2026-10-08/run_03/replay_pipeline.py original_input.csv --mode retrain
```

Cần repo cùng phiên bản (spec là pickle của class trong repo): driver tự tìm repo, hoặc truyền `--repo` / đặt `AUTOML_REPO`. Lệch phiên bản thư viện thì dừng, trừ khi thêm `--allow-version-mismatch`. Chi tiết: [docs/replay.md](docs/replay.md).

### Sử dụng replay script từng agent

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
sys.path.append("outputs/2026-10-08/run_03")
from final_model_code import predict_proba, predict

df = pd.read_parquet("engineered.parquet")   # dữ liệu ĐÃ qua Agent 1 + Stage 1b + Agent 2
scores = predict_proba(df)          # PD đã calibrate, chặn trong [PD_FLOOR, PD_CAP]
preds  = predict(df, threshold=0.5)
```

Với dữ liệu thô, dùng `replay_pipeline.py --mode score` ở trên.

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

## Test offline (không gọi LLM)

```powershell
# Các test tự tạo run dir khi import → trỏ OUTPUT_DIR sang thư mục tạm để không làm bẩn outputs/
$env:OUTPUT_DIR = "$env:TEMP\automl_tests"
Get-ChildItem tests -Filter "test_*.py" | Where-Object Name -notmatch "^test_agent\d" |
    ForEach-Object { py -3.12 $_.FullName }
```

| Test | Kiểm tra |
|---|---|
| `test_replay_reproducibility.py`, `test_replay_diagnosis.py`, `test_reader_normalization.py` | Replay tái lập chính xác, chẩn đoán khi không tái lập, chuẩn hoá đầu đọc CSV/parquet/Excel/SQL |
| `test_null_processor.py` | Stage 1b: fit-on-train, indicator, scale guard, artifact JSON + hash, guardrail rule LLM |
| `test_feature_contract.py` | `FeatureSpec.apply` không im lặng: strict mode, category lạ, one-hot không phụ thuộc batch, thứ tự replay |
| `test_security.py` | AST allowlist cho biểu thức, hash artifact trong replay, calibrator sigmoid lưu được |
| `test_feature_selection.py`, `test_optuna_guard.py` | Xếp hạng + prune theo lô, giới hạn thời gian từng trial, quy tắc chấp nhận re-tune |
| `test_provenance_pd.py`, `test_review_fixes.py` | Provenance, PD floor/cap, CV tree headroom, nhãn metric in-sample |
| `test_llm_onprem.py` | Đầu ra model suy luận, tham số server on-prem, không gọi API public khi tắt fallback, không có đường dẫn máy, import đúng hoa/thường cho Linux |

## Chạy nhiều dataset liên tiếp

```powershell
py -3.12 tools/overnight.py --only home_credit_application_train --data-root D:\datasets `
    --env LLM_SKIP_PROXY=true OPTUNA_N_TRIALS=60
```

Kết quả: `outputs/overnight_<timestamp>/OVERNIGHT_SUMMARY.md` + `<run>.summary.md` (metric trung thực trên OOT/test, funnel feature, sự kiện chính, provenance, smoke test chấm điểm qua bundle). Script chặn máy ngủ khi để không, nhưng không chặn được Modern Standby khi gập máy — với run dài nên cắm sạc và tắt sleep.

## Test từng agent riêng (gọi LLM thật)

> Đây là script thử nghiệm cũ, đọc/ghi file trực tiếp trong `outputs/` (không qua `run_NN/`). Để kiểm thử toàn trình nên dùng `python main.py ...` hoặc `_e2e_check.py`.

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

**Tổng quan & reference**

- [docs/architecture.md](docs/architecture.md) — Visual walkthrough + mermaid diagrams cho từng agent
- [docs/awareness-pattern.md](docs/awareness-pattern.md) — Agentic pattern reference (Plan-Execute + Awareness)
- [docs/config_params.md](docs/config_params.md) — Reference đầy đủ config param (override qua `.env`), gồm phần on-prem LLM, Stage 1b, chọn feature
- [docs/replay.md](docs/replay.md) — Replay bundle: tái lập và chấm điểm ở môi trường khác
- [docs/sequence_flow.md](docs/sequence_flow.md) — Sequence diagrams

**Flow chi tiết từng agent** (step-by-step + mermaid + no-leakage rules + config knobs)

- [docs/agent-1-data-cleaner.md](docs/agent-1-data-cleaner.md) — DataCleaner: prefilter → real-stats → LLM → canonical-ordered execute → spec replay
- [docs/agent-2-feature-engineer.md](docs/agent-2-feature-engineer.md) — FeatureEngineer: domain-aware interactions, label/onehot encoders, top-K selection, FeatureSpec replay
- [docs/agent-3-train-model.md](docs/agent-3-train-model.md) — TrainModel: split → encode → FLAML → rank → PSI → Stability → batch prune → Optuna → refit → overfit-aware retrain

---

## Pipeline chi tiết — Agent 3 (mặc định)

```
Train data  ─►  FLAML         ─►  estimator + base hyperparams (AutoML, time-budgeted)
            ─►  Rank          ─►  1 lần fit, xếp hạng importance, loại feature gần trùng → RFE_TARGET_FEATURES
            ─►  PSI filter    ─►  loại feature drift > PSI_THRESHOLD (train vs OOT; bỏ qua nếu không có OOT)
            ─►  Stability     ─►  loại feature có std Gini theo tháng > threshold (cần cột ngày)
            ─►  Batch prune   ─►  bỏ 10% yếu nhất mỗi vòng, chọn tập nhỏ nhất trong dung sai AUC, ≤ MAX_FINAL_FEATURES
            ─►  Optuna        ─►  tune một lần trên tập cuối (giới hạn thời gian từng trial, DART tắt)
            ─►  Refit         ─►  gộp train+valid, CV best_iteration (có headroom cây), calibrate trên OOF, PD floor/cap
            ─►  Eval          ─►  OOT (temporal) / test (holdout) — giữ sạch, không gộp
            ─►  Overfit check ─►  gap ref_auc(CV-OOF) vs holdout → LLM-guided retrain
```

Luồng cũ (Optuna trước → RFE → prune từng feature → re-tune) vẫn bật được: `OPTUNA_AFTER_SELECTION=false`, `FEATURE_RANK_METHOD=rfe`, `PRUNE_METHOD=one_by_one`.

Kết quả tham chiếu trên Home Credit (test ngẫu nhiên, không có OOT): mặc định hiện tại cho 79 feature, test AUC 0,7801, CV 0,7795; một run cùng luồng mới không bị máy ngủ mất ~2 giờ. Luồng cũ cho 149 feature, test AUC 0,7804, 5,9 giờ.

---

## Yêu cầu môi trường

- Python 3.12 (ghi trong `.python-version`; numpy 2.3 cần ≥ 3.11, các run tham chiếu dùng 3.12)
- LLM: endpoint chuẩn OpenAI (proxy / gateway / vLLM / SGLang / Ollama) hoặc gateway chuẩn Anthropic; model nhận prompt dài (`CLOUD_MODEL`) cần context ≥ 64k token
- Windows / macOS / Linux
- GPU (tuỳ chọn) — auto-detect qua `nvidia-smi`
- RAM khuyến nghị (sau dtype-downcast tại Agent 3):
  - ≥ 8 GB cho dataset 100k rows × 500 cols
  - ≥ 16 GB cho 500k rows × 1500 cols (đã verify trên sample_data_x3)
  - ≥ 32 GB cho 3M+ rows × 1500+ cols (hoặc dùng `--sample-ratio`)
- Trên Linux/Docker: ưu tiên `/tmp` >= 4 GB free (xem `JOBLIB_TEMP_FOLDER` ở section memory savers)

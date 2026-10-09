# Config Parameters Reference

Tất cả tham số đều đọc từ biến môi trường (`.env`). Nếu không set, giá trị mặc định được dùng.

---

## Backend selector

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `LLM_BACKEND` | `legacy` | Chọn backend LLM: `legacy` (LiteLLM proxy + OpenAI/Claude fallback) hoặc `gateway` (single LiteLLM gateway với Anthropic API). Khi set `gateway`, params Gateway phía dưới được dùng và các params Legacy bị ignore. |

**Chọn backend theo môi trường mạng**:

| Vị trí mạng | `LLM_BACKEND` | Lý do |
|---|---|---|
| Mạng nội bộ (VPN / văn phòng) — reachable tới gateway nội bộ | `gateway` | Key chung của tổ chức, không cần key cá nhân |
| Mạng ngoài (nhà / quán cafe) — KHÔNG có VPN | `legacy` | Đi thẳng OpenAI/Claude public Internet bằng key cá nhân |

Switch không cần đổi code — chỉ đổi `LLM_BACKEND` trong `.env` rồi chạy lại pipeline.

---

## LLM Gateway (khi `LLM_BACKEND=gateway`)

Single LiteLLM gateway speaking Anthropic API. Gateway tự xử lý internal failover — không có application-level fallback.

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `ANTHROPIC_AUTH_TOKEN` | `""` (bắt buộc) | Bearer token để authenticate với gateway |
| `ANTHROPIC_BASE_URL` | `http://localhost:4000` | URL của LiteLLM gateway |
| `API_TIMEOUT_MS` | `3000000` | Per-request timeout (ms). Default 50 phút — chịu được agent reasoning dài |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | `claude-haiku-4-5` | Alias tier nhanh/nhỏ — gateway dispatch tới underlying model |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | `claude-sonnet-4-6` | Alias tier trung |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | `claude-opus-4-7` | Alias tier cao cấp |
| `EFFORT_LEVEL` | `medium` | Routing: `low`=luôn Haiku, `medium`=Haiku/Sonnet theo size, `high`=Sonnet/Opus theo size |

---

## LLM Endpoints (khi `LLM_BACKEND=legacy`)

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `LITELLM_URL` | `http://localhost:4000` | URL của LiteLLM proxy server (backend chính) |
| `API_KEY` | `anything` | API key gửi lên proxy (proxy tự quản lý, không cần thật) |
| `LOCAL_MODEL` | `local-model` | Tên model nhỏ/local dùng cho prompt ngắn (< `MODEL_ROUTING_THRESHOLD`) |
| `CLOUD_MODEL` | `cloud-model` | Tên model lớn/cloud dùng cho prompt dài (≥ `MODEL_ROUTING_THRESHOLD`) |
| `TIMEOUT` | `60` | Timeout (giây) cho mỗi request LLM |

---

## LLM Behaviour

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `LLM_TEMPERATURE` | `0.2` | Độ ngẫu nhiên output (0 = deterministic, 1 = sáng tạo). Thấp → ổn định hơn cho tác vụ phân tích |
| `LLM_MAX_TOKENS` | `2000` | Giới hạn token output cho hầu hết các call |
| `LLM_MAX_TOKENS_LARGE` | `4000` | Token budget lớn hơn cho task output dài (FE decisions, code generation) |
| `LLM_TOP_P` | `0.95` | Nucleus sampling — chỉ lấy top 95% probability mass, loại bỏ token xác suất thấp |
| `LLM_FREQUENCY_PENALTY` | `0.1` | Phạt từ lặp lại trong output (0–2). Tăng nếu LLM hay lặp từ |
| `LLM_PRESENCE_PENALTY` | `0.0` | Khuyến khích đưa ra ý mới (0–2). Giữ 0 để tập trung vào topic gốc |
| `LLM_SEED` | `None` | Fix seed để output LLM reproducible. Không set = tắt |
| `LLM_MAX_RETRIES` | `3` | Số lần retry khi gặp rate-limit (429) hoặc lỗi server (5xx) |
| `LLM_RETRY_DELAY` | `2.0` | Delay cơ sở (giây) cho exponential backoff giữa các retry |
| `MODEL_ROUTING_THRESHOLD` | `6000` | Prompt dài hơn ngưỡng này (chars) sẽ được route sang `CLOUD_MODEL` |

---

## Direct Cloud Fallback

Thứ tự fallback: **LiteLLM proxy → OpenAI → Claude**

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `OPENAI_API_KEY` | `""` | API key OpenAI — fallback 2 khi proxy down |
| `OPENAI_DIRECT_MODEL` | `gpt-4.1-mini` | Model OpenAI dùng khi call trực tiếp |
| `ANTHROPIC_API_KEY` | `""` | API key Anthropic — fallback cuối cùng |
| `CLAUDE_DIRECT_MODEL` | `claude-sonnet-4-6` | Model Claude dùng khi call trực tiếp |

---

## Output Paths

Mọi artifact persisted nằm dưới `outputs/<YYYY-MM-DD>/run_<NN>/` — run_NN counter reset mỗi ngày (xem section "Run directory layout" bên dưới).

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `OUTPUT_DIR` | `outputs` | Thư mục gốc. Path day + run được `Config.init_run()` tạo runtime |
| `KEEP_INTERMEDIATES` | `false` | Khi `true`, các file handoff `clean_*.parquet` / `engineered_*.parquet` cũng giữ trong `run_NN/` (cho debug/test). Mặc định ghi vào system tempdir và xoá khi pipeline kết thúc |

Các path constants dưới đây **được Config.init_run() set lại runtime** trỏ vào `RUN_DIR`. Default value khi Config import là rỗng (chưa allocated run).

| Constant | Tên file trong run_NN/ | Vai trò |
|---|---|---|
| `EXECUTION_LOG_PATH` | `agent_execution.log` | Execution log tổng hợp 3 agent |
| `DATA_CLEANER_REPORT_PATH` | `data_cleaner_report.json` | Report Agent 1 + `cleaning_spec` |
| `FEATURE_ENGINEER_REPORT_PATH` | `feature_engineer_report.json` | Report Agent 2 + `feature_spec` |
| `MODEL_TRAINER_REPORT_PATH` | `model_trainer_report.json` | Report Agent 3 |
| `PSI_REPORT_PATH` | `psi_report.csv` | PSI drift mỗi feature |
| `STABILITY_REPORT_PATH` | `stability_report.csv` | Mean/std Gini theo tháng |
| `SHAP_PSI_PRUNE_LOG_PATH` | `shap_psi_prune_log.csv` | Log iterative pruning |
| `FINAL_REPORT_PATH` | `final_report.md` | Markdown report tổng pipeline |
| `FINAL_MODEL_PATH` | `final_model.pkl` | Trained model + encoders (joblib) |
| `FINAL_MODEL_CODE_PATH` | `final_model_code.py` | Inference code standalone |
| `PIPELINE_PROCESS_DC_PATH` | `pipeline_process_data_cleaner.py` | Script replay Agent 1 |
| `PIPELINE_PROCESS_FE_PATH` | `pipeline_process_feature_engineer.py` | Script replay Agent 2 |
| `PIPELINE_PROCESS_FE_SPEC_PATH` | `feature_spec.pkl` | Sidecar fitted FeatureSpec |
| `PIPELINE_PROCESS_TM_PATH` | `pipeline_process_train_model.py` | Script replay Agent 3 (= inference code) |

Intermediate handoff files (`CLEAN_DATA_PATH`, `ENGINEERED_DATA_PATH`, `CLEAN_TRAIN_PATH`, `CLEAN_VALID_PATH`, `CLEAN_OOT_PATH`, `ENGINEERED_TRAIN_PATH`, `ENGINEERED_VALID_PATH`, `ENGINEERED_OOT_PATH`) trỏ vào `TMP_DIR` (= `tempfile.mkdtemp()` mặc định) và **bị xoá sau khi pipeline kết thúc** (kể cả khi raise — try/finally).

### Run directory layout

```
outputs/YYYY-MM-DD/
├── run_01/   ← lần chạy 1 của ngày
├── run_02/   ← lần chạy 2 của ngày
└── run_NN/   ← max(NN)+1 dùng (không phải len+1 → tránh collision nếu xoá giữa)
```

Counter reset = sang ngày mới, thư mục `YYYY-MM-DD/` mới rỗng → `init_run()` bắt đầu lại từ `run_01`. Không cần config.

---

## S3 / S3-compatible storage

Dùng khi `BaseAgent.load_dataframe` nhận path `s3://...`. Hỗ trợ S3-compatible (MinIO, Wasabi, ...) qua `S3_ENDPOINT_URL`.

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `S3_ACCESS_KEY` | `""` | Access key. Để trống → dùng AWS default credential chain |
| `S3_SECRET_KEY` | `""` | Secret key |
| `S3_ENDPOINT_URL` | `""` | Endpoint override cho S3-compatible (MinIO...). Để trống = AWS S3 |
| `S3_CONNECT_TIMEOUT` | `300` | Timeout kết nối (giây) |
| `S3_REQUEST_TIMEOUT` | `3600` | Timeout request (giây) — tăng cho parquet rất lớn |

---

## Logging

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `LOG_RESULT_PREVIEW_CHARS` | `200` | Số ký tự tối đa của tool result hiển thị trong log (tránh log quá dài) |
| `SUMMARY_ACTION_PREVIEW` | `3` | Số action đầu tiên hiển thị trong chuỗi summary log |

---

## Data Cleaning — Agent 1

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `NULL_DROP_THRESHOLD` | `0.8` | Cột có tỷ lệ null > 80% sẽ bị gợi ý xóa |
| `HIGH_CARDINALITY_THRESHOLD` | `50` | Cột có > 50 giá trị unique sẽ bỏ qua bước value_counts trong log |
| `DUPLICATE_PCT_THRESHOLD` | `1.0` | Tỷ lệ row trùng lặp > 1% → gợi ý drop duplicates |
| `PK_VIOLATION_PCT_THRESHOLD` | `1.0` | Tỷ lệ vi phạm composite key > 1% → gợi ý deduplicate_by_key |
| `ENTITY_ORPHAN_PCT_THRESHOLD` | `5.0` | Tỷ lệ entity không có identity anchor > 5% → cảnh báo data quality |
| `AMBIGUOUS_IDENTITY_PCT_THRESHOLD` | `5.0` | Tỷ lệ identity value dùng chung nhiều entity > 5% → flag fraud ring |
| `SOFT_DUP_PCT_THRESHOLD` | `0.5` | Tỷ lệ soft-duplicate (cùng composite key, khác giá trị) > 0.5% → gợi ý deduplicate_by_key |
| `APP_DUP_PCT_THRESHOLD` | `10.0` | Tỷ lệ entity có nhiều application > 10% → gợi ý deduplicate_by_key |
| `OUTLIER_PCT_THRESHOLD` | `5.0` | Tỷ lệ outlier (IQR×3) > 5% → gợi ý clip |
| `IMBALANCE_RATIO_THRESHOLD` | `20.0` | Tỷ lệ max/min class count > 20 → cảnh báo class imbalance |
| `OUTLIER_NUMERIC_COLS_LIMIT` | `10` | Tối đa số cột numeric chạy outlier detection (tránh chậm với dataset rộng) |

---

## Feature Engineering — Agent 2

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `HIGH_CORRELATION_THRESHOLD` | `0.8` | Cặp feature có correlation > 0.8 → gợi ý bỏ một cái (multicollinearity) |
| `LOW_CORRELATION_THRESHOLD` | `0.04` | Correlation với target < 0.04 → feature yếu, ưu tiên loại |
| `MIN_CORRELATION_THRESHOLD` | `0.001` | Correlation với target < 0.001 → feature gần như vô nghĩa, flag xóa |
| `TOP_K_FEATURES_CAP` | `800` | Trần tối đa số feature được chọn (dù `TOP_K_RATIO` tính ra nhiều hơn vẫn bị cap) |
| `TOP_K_RATIO` | `0.70` | Tỷ lệ feature giữ lại khi `select_top_features` (0 < ratio ≤ 1) |
| `FEATURE_META_NUMERIC_RATIO` | `0.6` | Tỷ lệ cột numeric đưa metadata vào prompt LLM (0 < ratio ≤ 1) |
| `FEATURE_META_MAX_NUMERIC_COLS` | `300` | Trần cứng số cột numeric đưa vào prompt (an toàn cho dataset rất rộng) |
| `FEATURE_META_MAX_CATEGORICAL_COLS` | `80` | Tối đa số cột categorical đưa metadata đầy đủ vào prompt |
| `MAX_DESC_PER_GROUP` | `50` | Tối đa số description hiển thị mỗi group trong khối column-description của prompt |
| `TARGET_NEW_FEATURE_COUNT` | `25` | Số interaction feature mục tiêu LLM sinh mỗi run (Agent 2 dùng để chia phân bổ FAMILY A–H) |
| `FE_CREATE_INTERACTIONS_ENABLED` | `true` | Công tắc bước tạo interaction feature. `false` → bỏ qua toàn bộ action `create_interaction` (chỉ dùng cột gốc + encoded; IV/WoE/selection vẫn chạy). Override per-run từ pipeline: `pipeline.run(..., create_interactions=False)`; hoặc gọi agent trực tiếp: `process(...)` / `process_splits(...)` / `fit_transform(..., create_interactions=False)` |

### IV / WoE feature selection (chuẩn vàng credit-risk)

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `IV_BINS_DEFAULT` | `10` | Số quantile bin của `compute_iv` (chuẩn Siddiqi); tăng 20 cho phân phối dày, giảm 5 nếu phần lớn feature < 50 unique |
| `IV_LEAKAGE_THRESHOLD` | `0.50` | IV ≥ ngưỡng này → flag leakage-suspect; cũng là cận trên của band "strong" (useless < 0.02 ≤ weak < 0.10 ≤ medium < 0.30 ≤ strong < ngưỡng ≤ suspect) |
| `MULTICOLLINEARITY_THRESHOLD` | `0.85` | \|Pearson\| vượt ngưỡng → `select_top_features(criterion='iv')` bỏ feature IV thấp hơn trong cặp; siết 0.75 cho scorecard nghiêm |
| `IV_MAX_NULL_RATIO` | `0.50` | Bỏ qua compute IV cho cột missing trên mức này (IV bị NaN-bin chi phối, vô nghĩa) |
| `WOE_MIN_IV` | `0.02` | IV tối thiểu để được `apply_woe_transform`; dưới mức này WoE thêm noise hơn là tuyến tính hóa |

### Replay bundle

Mỗi run ghi `replay_pipeline.py` + `replay_manifest.json` + `split_assignment.parquet` + `cleaning_spec.pkl` vào RUN_DIR, đủ để chạy lại toàn trình (split → transform → retrain/score) ở môi trường khác. Không có tham số bật/tắt — bundle luôn được ghi, và lỗi khi ghi bundle không làm hỏng run. Chi tiết: [docs/replay.md](replay.md).

### IV stability train ↔ valid/oot (diagnostic)

Sau khi transform mỗi partition holdout, Agent 2 tính lại IV trên partition đó bằng **bin edges đóng băng từ train**, rồi so với IV train. Bắt được thứ PSI không thấy: feature có phân phối X ổn định nhưng quan hệ với target suy yếu hoặc **đảo chiều**. Kết quả ghi vào `feature_engineer_report.json` key `iv_stability`. Thuần diagnostic — không drop feature; Agent 3 PSI / SHAP+PSI prune vẫn là nơi duy nhất loại feature.

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `FE_IV_STABILITY_ENABLED` | `true` | Bật/tắt toàn bộ bước kiểm tra. Tự skip nếu `compute_iv` hoặc `apply_woe_transform` không chạy (không có bin đóng băng để so) |
| `FE_IV_STABILITY_MAX_DROP` | `0.30` | % sụt IV so với train vượt ngưỡng → flag `UNSTABLE` (reason `iv_drop`) |
| `FE_IV_STABILITY_MIN_WOE_CORR` | `0.50` | Pearson giữa vector WoE train và WoE holdout (theo bin) dưới ngưỡng → flag (reason `woe_reversal`). Âm = quan hệ đảo chiều hẳn |
| `FE_IV_STABILITY_MIN_BIN_COUNT` | `30` | Bin có ít hơn số dòng này ở holdout bị loại khỏi phép tính sign-flip / corr (đuôi quá nhiễu) |
| `FE_IV_STABILITY_TOP_N` | `20` | Số feature kém ổn định nhất liệt kê trong dòng log WARN |

Feature có `iv_train < WOE_MIN_IV` được bỏ qua (`n_skipped_weak`) — vốn đã vô dụng trên train thì không có gì để "ổn định".

---

## Model Training — Agent 3

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `TRAIN_TEST_SPLIT_SIZE` | `0.2` | Tỷ lệ valid trong pool sau khi tách OOT (20% pool → valid) |
| `RANDOM_STATE` | `42` | Seed cố định cho tất cả random operations (split, model, Optuna) |
| `CLASSIFICATION_UNIQUE_THRESHOLD` | `10` | Target có ≤ 10 giá trị unique → classification, ngược lại → regression |
| `CV_N_SPLITS` | `5` | Số fold StratifiedKFold cho đánh giá CV cuối cùng của model |

---

## Probability Calibration (IFRS9 / scorecard PD)

Bọc model cuối bằng isotonic/sigmoid calibration fit trên held-out valid split → raw ranking score thành xác suất hiệu chỉnh (well-calibrated PD). **Là lựa chọn, không bị force** — bật mặc định; tắt thì agent log `disabled by CALIBRATION_ENABLED=false` và `self.calibrator=None`, mọi downstream (metrics, save model, charts) bỏ qua sạch.

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `CALIBRATION_ENABLED` | `true` | `true` → fit calibrator trên split đầu tiên đủ điều kiện (`valid_random` → `valid_temporal` → `valid` → `oot`, cần ≥ 100 rows & 2 class). `false` → tắt hoàn toàn, model trả raw probability. AUC neutral; cải thiện lớn Brier/log-loss |
| `CALIBRATION_METHOD` | `isotonic` | `isotonic` (non-parametric, banking default, cần ≥ ~1000 rows calibration) hoặc `sigmoid` (Platt scaling, rẻ hơn, hợp calibration set nhỏ) |

---

## Multi-seed Bagging (variance reduction)

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `MULTI_SEED_ENABLED` | `true` | `true` → train `MULTI_SEED_N` bản best-config với seed khác nhau, average `predict_proba`. `false` → single-model legacy |
| `MULTI_SEED_N` | `5` | Số seed khi bật. +0.1–0.3% AUC + PD ổn định hơn giữa các lần retrain |

---

## OOT / Temporal Split

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `TEMPORAL_FREQ` | `auto` | Độ phân giải mọi lát cắt thời gian (OOT / valid_temporal / stability). `auto` = bucket theo **tháng** (nhận `monthly_snapshot` vs `intra_month`, **không bao giờ tự đoán weekly**); `weekly` = bucket theo **tuần** (phải khai báo rõ); `monthly` = ép tháng. |
| `WEEK_CLOSING_DAY` | `` (auto) | Ngày chốt tuần (snapshot/cutoff) khi `TEMPORAL_FREQ=weekly`: `MON`..`SUN`. Để trống → tự lấy weekday phổ biến nhất trong `date_col`. |
| `OOT_INIT_MONTHS` | `2` | Sàn tối thiểu số **kỳ** (tháng/tuần) luôn đưa vào OOT (dù tỷ lệ chưa đủ) |
| `OOT_MIN_RATIO` | `0.15` | Mở rộng OOT cho đến khi đạt ≥ 15% tổng data |
| `OOT_MAX_RATIO` | `0.20` | Trần cứng OOT — thu hẹp nếu vượt 20% |
| `VALID_TEMPORAL_RATIO` | `0.20` | Tỷ lệ valid_temporal trong tổng valid set (20% valid gần OOT boundary nhất) |

> **Weekly model**: chỉ bật bằng param (không auto). Có thể override per-run thay vì env:
> `pipeline.run(..., temporal_freq="weekly", week_closing_day="FRI")`.
> Khi đó OOT lấy trailing whole-weeks, valid_temporal snap nguyên tuần (tránh leak trong cùng tuần), stability Gini group theo tuần; report ghi `temporal.cadence="weekly_snapshot"` + số tuần mỗi split. `OOT_INIT_MONTHS` / `STABILITY_MIN_MONTHS` được hiểu là **số kỳ** (tuần) trong chế độ weekly — cân nhắc tăng nếu lịch sử tuần dài.

---

## FLAML AutoML

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `FLAML_TIME_BUDGET` | `3600` | Thời gian tối đa (giây) FLAML được phép tìm kiếm model tốt nhất |
| `FLAML_ESTIMATORS` | `xgboost,lgbm,catboost,rf,extra_tree` | Danh sách estimator FLAML thử |
| `FLAML_N_SPLITS` | `5` | Số fold CV khi FLAML không có validation set riêng |
| `FLAML_MAX_ROWS` | `500000` | Trần số row truyền vào FLAML; sample ngẫu nhiên nếu dataset lớn hơn (tránh OOM khi FLAML copy DataFrame) |

---

## Optuna Hyperparameter Tuning

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `OPTUNA_N_TRIALS` | `300` | Số trial tối đa Optuna chạy để tìm hyperparams tốt nhất |
| `OPTUNA_TIMEOUT` | `7200` | Timeout (giây) cho toàn bộ Optuna study (thường bind trước `N_TRIALS`) |
| `LR_MIN` / `LR_MAX` | `0.005` / `0.3` | Khoảng learning rate Optuna tìm kiếm (tất cả model) |
| `MAX_DEPTH_MIN` / `MAX_DEPTH_MAX` | `3` / `12` | Khoảng max_depth cây (XGBoost, LGBM, RF, ExtraTrees) |
| `N_ESTIMATORS_MIN` / `N_ESTIMATORS_MAX` | `100` / `2000` | Khoảng số cây (cận trên lớn vô hại nhờ early stopping) |
| `SUBSAMPLE_MIN` / `SUBSAMPLE_MAX` | `0.4` / `1.0` | Khoảng tỷ lệ sample row/col khi train mỗi cây |
| `NUM_LEAVES_MIN` / `NUM_LEAVES_MAX` | `15` / `512` | Khoảng num_leaves riêng của LightGBM |
| `CB_DEPTH_MIN` / `CB_DEPTH_MAX` | `4` / `12` | Khoảng depth riêng của CatBoost |

---

## RFE (Recursive Feature Elimination)

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `RFE_TARGET_FEATURES` | `150` | Số feature muốn giữ lại sau RFE |
| `RFE_STEP` | `0.01` | Mỗi vòng RFE loại bỏ 1% số feature còn lại (bước nhỏ → nhiều vòng refit hơn) |
| `RFE_CV_SPLITS` | `3` | Số fold CV dùng trong RFECV (chỉ khi `ENABLE_RFECV=true`) |
| `ENABLE_RFECV` | `false` | Bật RFECV (tự chọn số feature tối ưu qua CV). `false` = dùng RFE cố định |
| `RFE_N_ESTIMATORS` | `200` | n_estimators cho base model fit trong RFE selection |

---

## PSI (Population Stability Index)

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `PSI_THRESHOLD` | `0.35` | PSI > 0.35 → feature drift quá lớn giữa train và OOT → DROP (nâng từ 0.30 vì SHAP+PSI prune đã cân PSI vs importance) |
| `PSI_BINS` | `100` | Số bin tính PSI (nhiều bin → chính xác hơn, cần đủ data) |

---

## Stability Check

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `STABILITY_MIN_MONTHS` | `6` | Cần ít nhất 6 tháng data mới chạy stability check, ít hơn → skip |
| `STABILITY_GINI_STD_THRESHOLD` | `0.20` | Độ lệch chuẩn Gini theo tháng > 0.20 → feature không ổn định → DROP (nâng từ 0.15 để prune có thêm ứng viên cân với SHAP) |
| `STABILITY_MIN_GINI` | `0.01` | Gini trung bình < 0.01 → feature gần như không có predictive power → DROP |

---

## Final Feature Cut

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `MAX_FINAL_FEATURES` | `100` | Sau PSI + Stability, nếu còn > 100 feature thì cut tiếp bằng importance top-N |

---

## SHAP + PSI Iterative Pruning

Loại bỏ feature có SHAP importance thấp + PSI drift cao theo từng bước cho đến khi AUC validation ngừng cải thiện.

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `SHAP_N_ESTIMATORS` | `200` | n_estimators cho model nhanh fit tại mỗi step pruning |
| `SHAP_SAMPLE_SIZE` | `20000` | Tối đa số row sample để tính SHAP value (5K cho rank nhiễu ở 1.5M+ rows; 20K ổn định prune run-to-run) |
| `SHAP_PSI_MAX_NO_IMPROVE` | `3` | Dừng pruning sau N step liên tiếp không cải thiện AUC (nâng từ 2 để vượt plateau cục bộ) |
| `SHAP_PSI_MIN_FEATURES_FLOOR` | `5` | Sàn cứng tối thiểu số feature giữ lại sau pruning |
| `SHAP_PSI_MIN_FEATURES_RATIO` | `0.10` | Sàn tương đối: giữ ít nhất 10% × `MAX_FINAL_FEATURES` |
| `FEATURE_CAP_POLICY` | `auc_first` | Khi chạm streak early-stop mà count vẫn > `MAX_FINAL_FEATURES`: `auc_first` = ưu tiên AUC, dừng prune (cap là target mềm); `cap_first` = prune tiếp tới ≤ cap rồi mới bật lại early-stop |

---

## SHAP Final Visualization + LLM Explanation

Sau khi final model train xong (kể cả sau overfit-retrain), Agent 3 compute SHAP trên final model + LLM explain top N features. Output: PNG bar plot + CSV với narrative.

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `SHAP_FINAL_EXPLAIN_ENABLED` | `true` | Bật/tắt toàn bộ step. Tắt khi chạy batch automation không cần narrative |
| `SHAP_FINAL_EXPLAIN_TOP_N` | `20` | Số features LLM explain. Plot show 2× số này để thấy distribution context |

**Output files**:
- `shap_summary.png` — bar plot top (`2 × TOP_N`) features
- `shap_feature_explanations.csv` — rank, feature, shap_importance, description, meaning, why_matters
- Section nhúng vào `final_report.md`

**Graceful degrade**:
- `shap` chưa cài → fallback `model.feature_importances_`
- `matplotlib` lỗi → skip PNG, vẫn save CSV
- LLM lỗi → CSV không có `meaning`/`why_matters`, chỉ rank + importance

---

## Overfitting Detection

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `OVERFIT_THRESHOLD` | `0.12` | Gap tương đối `(ref_auc - holdout_auc) / ref_auc` > 12% → flag overfit và trigger LLM-guided retrain. `ref_auc` = `cv_auc_mean` (OOF in-time, fallback `valid_auc` → `valid_temporal_auc`); holdout = `oot`/`test`. Đã đổi từ in-sample `valid_auc` sang OOF sau refactor refit-on-train+valid để tránh false trigger |

---

## Model Diagnostic Charts

Agent 3 sinh 9 chart chẩn đoán (`{run_dir}/charts/*.png`) sau khi train model cuối: ROC, PR, KS, Lift, Gain, Bad-rate by decile, Avg-score by decile, Calibration, Score distribution. Nhúng vào `final_report.md` qua `pipeline._render_charts_section`.

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `CHART_N_DECILES` | `10` | Số quantile-bucket cho các chart Lift / Gain / Bad-rate / Avg-score **và** calibration reliability plot. `10` = decile cổ điển. Tăng (vd `20`) để mịn hơn trên eval split lớn; giảm (vd `5`) trên split nhỏ để tránh bucket rỗng. Khi `≠ 10`, nhãn trục/title tự đổi "Decile" → "Bucket (of N)" |

---

## Parallel Worker Staging (joblib)

Joblib (sklearn/FLAML/Optuna nội bộ) cần serialize `X_train` ra file `.pkl` để worker process đọc memory-map. Path mặc định trên Linux là `/dev/shm` — tmpfs rất nhỏ (Docker default **64 MB**) → file bị truncate → `BrokenProcessPool: FileNotFoundError`.

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `JOBLIB_TEMP_FOLDER` | `tempfile.gettempdir()` | Thư mục stage `.pkl` cho joblib workers. Auto-set tại `config.py` import time (Linux: `/tmp`, Windows: `%TEMP%`). Override khi `/tmp` cũng nhỏ — point vào disk-backed path có vài GB free, ví dụ `/data/joblib_tmp` |

**Khi nào cần override**:
- Container Linux có `/tmp` mount tmpfs nhỏ
- Dataset train > 5 GB sau dtype downcast
- Thấy lỗi `BrokenProcessPool` mid-CV / mid-Optuna

`setdefault` được dùng → user override trong `.env` luôn được respect.

---

## Memory Optimization Notes

Pipeline áp 4 lớp tối ưu RAM mặc định (không có env knob — đã built-in):

1. **Agent 1 prefilter** (`PREFILTER_MAX_NULL_RATIO`, `PREFILTER_MAX_DOMINANT_RATIO` ở section trên) — drop cột rác trước khi vào Agent 2
2. **Agent 2 metadata-only dtype iter** — `select_top_features` đọc `df.dtypes` thay vì `df[feature_cols].select_dtypes()` (slice cũ trigger block consolidation → alloc float64 matrix `(n_cols, n_rows)` → spike vài GB)
3. **Agent 3 dtype downcast tại fit** — `_fit_prepare_X` rewrite:
   - Object/cat → smallest signed int (`int8` ≤127 cats, `int16` ≤32k, `int32` lớn hơn) — chọn theo `len(le.classes_)` tự động
   - Float64 → `float32` (50% RAM)
   - Int64 → `int32`
   - Bool → `int8`
4. **Per-partition release** — `del agent + gc.collect()` giữa các stage

**Benchmark** (50k × 1400 mixed cols):
- Trước downcast: 1.82 GB
- Sau downcast: 190 MB (≈9.6×)

Cộng dồn 3 lớp + `JOBLIB_TEMP_FOLDER` đủ chạy 500k × 1500 cols dưới 8 GB RAM trên Linux/Docker.

## Null processing (Stage 1b)

| Param | Default | Meaning |
|---|---|---|
| `NULL_PROCESSOR_ENABLED` | `true` | Fit a per-feature missing-value policy on TRAIN after Agent 1 and replay it on all partitions and in the replay bundle (`null_processor.json`, SHA-256 verified on load). |
| `NULL_PROCESSOR_USE_LLM` | `true` | Rules proposed by the LLM behind deterministic guardrails (stats only, no row values); `false` = heuristic only. No human review step. |
| `NULL_DRIFT_WARN_THRESHOLD` | `0.10` | Absolute missing-rate gap vs train that triggers a warning. |

| `NULL_SCALE_WARN_RATIO` | `3.0` | Unit-change guard: median / p95 of the non-zero \|x\| of a batch vs the frozen train value; any single ratio (either direction) ≥ this warns (`distribution shifted`). |
| `NULL_SCALE_FAIL_RATIO` | `10.0` | `critical` only when median **and** p95 shift the same way by ≥ this (a real unit change rescales every quantile; ×1000 VND→thousand-VND is caught). One quantile moving alone — heavy-tailed / sparse columns — is a `warn`. `0` disables. Batches under 100 non-null rows are never judged. |
| `DATA_GUARD_ACTION` | `warn` | What a `critical` data-quality finding does (scale / unit change, unparseable values in a numeric column). `warn`: the run continues; every finding of every stage (Stage 0b PSI drift, Stage 1b scale / missing-rate / unparseable, agents' WARN / SKIP notices) goes to `data_quality_report.json` and the **Data quality warnings** table at the top of `final_report.md`, each with a hint of what to check. `fail`: stop with `NullContractError` (also at scoring, since the choice is stored in `null_processor.json`). Structural errors — absent required column, tampered artifact, unsafe expression — always stop. |

## Feature ranking / pruning (Agent 3)

Every option has a legacy value, so the old behaviour stays available. Unknown values fall back to the default with a `Config WARN` log line.

| Param | Default | Options / meaning |
|---|---|---|
| `FEATURE_RANK_METHOD` | `importance` | `importance`: one model fit, rank, drop near-duplicates (minutes). `rfe`: legacy sklearn RFE (hours; `RFE_STEP`, `RFE_N_ESTIMATORS`, `ENABLE_RFECV` apply). The number of candidates kept is `RFE_TARGET_FEATURES`. |
| `FEATURE_RANK_IMPORTANCE` | `gain` | `gain` (from the fitted booster) or `shap` (mean \|SHAP\|, slower). Used by ranking and by the batch prune. |
| `FEATURE_RANK_CORR_THRESHOLD` | `0.90` | Of two candidates with \|corr\| above this, keep the more important. `0` disables. |
| `FEATURE_RANK_N_ESTIMATORS` / `FEATURE_RANK_LEARNING_RATE` | `500` / `0.05` | Tree budget and learning-rate floor of the ranking & prune models (early stopping on valid). |
| `PRUNE_METHOD` | `batch_tolerance` | `batch_tolerance`: remove `PRUNE_BATCH_FRACTION` of the weakest per round, keep the smallest set within `PRUNE_AUC_TOLERANCE` of the best AUC. `one_by_one`: legacy SHAP+PSI loop (`SHAP_PSI_MAX_NO_IMPROVE`, `SHAP_N_ESTIMATORS`). |
| `PRUNE_BATCH_FRACTION` | `0.10` | Share removed per round (min 1 feature). |
| `PRUNE_AUC_TOLERANCE` | `0.001` | Absolute valid-AUC loss accepted for a smaller set; `0` never trades AUC. |
| `PRUNE_STOP_DROP` | `0.010` | Stop shrinking once AUC is this far below the best seen. |
| `PRUNE_MAX_ROUNDS` | `40` | Safety bound on rounds. |
| `FEATURE_CAP_POLICY` | `cap_first` | `cap_first`: the final set has at most `MAX_FINAL_FEATURES` (best-AUC set under the cap if the tolerance rule alone would leave more). `auc_first`: the cap is only a soft target. |
| `OPTUNA_AFTER_SELECTION` | `true` | `true`: FLAML → rank/prune with FLAML's params → Optuna once on the final features (no re-tune). `false`: legacy order (Optuna → rank → prune → optional re-tune, `ENABLE_RETUNE_AFTER_PRUNE`). |
| `OPTUNA_TRIAL_TIMEOUT` / `OPTUNA_ENABLE_DART` / `TREE_ENSEMBLE_MAX_ESTIMATORS` / `RETUNE_MIN_GAIN` | `0` (auto) / `false` / `500` / `0.0005` | Per-trial wall-clock cap, DART in the LightGBM space, tree cap for rf/extra_tree, and the AUC gain a re-tune must show to replace tuned params. |
| `PRUNE_SMOOTH_WINDOW` | `3` | Odd window of the moving average applied to the AUC-vs-size curve before the tolerance rule (1 = raw). Stops the rule latching onto a lucky spike on small valid sets. |
| `PRUNE_SE_MULTIPLIER` | `0.0` | Widen the tolerance to this multiple of the Hanley-McNeil standard error of the valid AUC (0 = off; ~0.5 for small / noisy validation sets). |
| `FINAL_ES_TREE_HEADROOM` | `2.0` | Final-model CV folds get this multiple of the tuned tree count so early stopping (not the ceiling) picks `best_iteration`. `1` = legacy (reproduce bundles written before this option). |


## On-prem / enterprise LLM endpoints

The legacy backend (`LLM_BACKEND=legacy`) uses the OpenAI chat API, so `LITELLM_URL` can be a LiteLLM proxy, an enterprise gateway, or a vLLM / SGLang / Ollama server directly (append `/v1` for those). `LLM_BACKEND=gateway` is only for Anthropic-compatible gateways.

| Param | Default | Meaning |
|---|---|---|
| `LLM_ALLOW_EXTERNAL_FALLBACK` | `true` | `false` = never call OpenAI / Anthropic directly; proxy errors (incl. connection errors) are retried, then the run fails with a clear error. Use `false` on-prem. |
| `LLM_JSON_MODE_NATIVE` | `true` | Send `response_format={"type":"json_object"}`; `false` moves the JSON instruction into the system prompt for servers that reject it. |
| `LLM_EXTRA_BODY` | empty | JSON merged into every request body, e.g. `{"chat_template_kwargs": {"enable_thinking": false}}` (Qwen3 on vLLM/SGLang). |
| `LLM_OMIT_PARAMS` | empty | Comma list of sampling params to drop (`frequency_penalty,presence_penalty,seed`). |
| `LLM_STRIP_REASONING` | `true` | Remove `<think>…</think>` / unterminated reasoning / GLM box tokens before parsing. JSON is extracted with a real decoder, so braces in reasoning or prose no longer break parsing. |
| `TIMEOUT` | `60` | Seconds per call; raise (e.g. 300) for local models. |
| `LLM_MAX_TOKENS_LARGE` | `4000` | Raise (e.g. 8000) for thinking models that are not switched off. |

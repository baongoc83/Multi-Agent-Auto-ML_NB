# Config Parameters Reference

Tất cả tham số đều đọc từ biến môi trường (`.env`). Nếu không set, giá trị mặc định được dùng.

---

## Backend selector

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `LLM_BACKEND` | `legacy` | Chọn backend LLM: `legacy` (LiteLLM proxy + OpenAI/Claude fallback) hoặc `gateway` (single LiteLLM gateway với Anthropic API). Khi set `gateway`, params Gateway phía dưới được dùng và các params Legacy bị ignore. |

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

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `OUTPUT_DIR` | `outputs` | Thư mục gốc lưu tất cả kết quả |
| `CLEAN_DATA_PATH` | `outputs/clean_data.csv` | File CSV sau khi Agent 1 (DataCleaner) xử lý xong |
| `ENGINEERED_DATA_PATH` | `outputs/engineered_data.csv` | File CSV sau khi Agent 2 (FeatureEngineer) xử lý xong |
| `FINAL_MODEL_CODE_PATH` | `outputs/final_model_code.py` | File Python chứa code model cuối cùng |
| `FINAL_MODEL_PATH` | `outputs/final_model.pkl` | Artifact model đã train (joblib pickle) cùng encoders + feature list |
| `FINAL_REPORT_PATH` | `outputs/final_report.md` | Báo cáo tổng hợp toàn pipeline dạng Markdown |
| `EXECUTION_LOG_PATH` | `outputs/agent_execution.log` | Log toàn bộ quá trình chạy các agent |
| `DATA_CLEANER_REPORT_PATH` | `outputs/data_cleaner_report.json` | JSON report của Agent 1 |
| `FEATURE_ENGINEER_REPORT_PATH` | `outputs/feature_engineer_report.json` | JSON report của Agent 2 |
| `MODEL_TRAINER_REPORT_PATH` | `outputs/model_trainer_report.json` | JSON report của Agent 3 |

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
| `SAMPLE_STATS_PREVIEW` | `3` | Số cột sample lấy stats để đưa vào prompt LLM |
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
| `TOP_K_FEATURES_CAP` | `350` | Trần tối đa số feature được chọn (dù `TOP_K_RATIO` tính ra nhiều hơn vẫn bị cap) |
| `TOP_K_RATIO` | `0.70` | Tỷ lệ feature giữ lại khi `select_top_features` (0 < ratio ≤ 1) |
| `FEATURE_META_NUMERIC_RATIO` | `0.6` | Tỷ lệ cột numeric đưa metadata vào prompt LLM (0 < ratio ≤ 1) |
| `FEATURE_META_MAX_NUMERIC_COLS` | `200` | Trần cứng số cột numeric đưa vào prompt (an toàn cho dataset rất rộng) |
| `FEATURE_META_MAX_CATEGORICAL_COLS` | `40` | Tối đa số cột categorical đưa metadata đầy đủ vào prompt |
| `MAX_DESC_PER_GROUP` | `30` | Tối đa số description hiển thị mỗi group trong khối column-description của prompt |

---

## Model Training — Agent 3

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `TRAIN_TEST_SPLIT_SIZE` | `0.2` | Tỷ lệ valid trong pool sau khi tách OOT (20% pool → valid) |
| `RANDOM_STATE` | `42` | Seed cố định cho tất cả random operations (split, model, Optuna) |
| `MAX_TRAINING_ITERATIONS` | `3` | Số vòng lặp feedback tối đa của training loop |
| `MODELS_TO_COMPARE` | `XGBoost,RandomForest,ExtraTrees,LightGBM,CatBoost` | Danh sách model FLAML sẽ thử (comma-separated) |
| `CLASSIFICATION_UNIQUE_THRESHOLD` | `10` | Target có ≤ 10 giá trị unique → classification, ngược lại → regression |
| `CV_N_SPLITS` | `5` | Số fold StratifiedKFold cho đánh giá CV cuối cùng của model |
| `TARGET_ROC_AUC` | `0.85` | Ngưỡng AUC mục tiêu để pipeline coi là "đạt" |
| `TARGET_F1` | `0.80` | Ngưỡng F1 mục tiêu |
| `TARGET_R2` | `0.75` | Ngưỡng R² mục tiêu (bài toán regression) |
| `TARGET_ACCURACY` | `0.85` | Ngưỡng Accuracy mục tiêu |

---

## OOT / Temporal Split

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `OOT_INIT_MONTHS` | `2` | Sàn tối thiểu số tháng luôn đưa vào OOT (dù tỷ lệ chưa đủ) |
| `OOT_MIN_RATIO` | `0.15` | Mở rộng OOT cho đến khi đạt ≥ 15% tổng data |
| `OOT_MAX_RATIO` | `0.20` | Trần cứng OOT — thu hẹp nếu vượt 20% |
| `VALID_TEMPORAL_RATIO` | `0.20` | Tỷ lệ valid_temporal trong tổng valid set (20% valid gần OOT boundary nhất) |

---

## FLAML AutoML

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `FLAML_TIME_BUDGET` | `600` | Thời gian tối đa (giây) FLAML được phép tìm kiếm model tốt nhất |
| `FLAML_ESTIMATORS` | `xgboost,lgbm,catboost,rf,extra_tree` | Danh sách estimator FLAML thử |
| `FLAML_N_SPLITS` | `5` | Số fold CV khi FLAML không có validation set riêng |
| `FLAML_MAX_ROWS` | `100000` | Trần số row truyền vào FLAML; sample ngẫu nhiên nếu dataset lớn hơn (tránh OOM khi FLAML copy DataFrame) |

---

## Optuna Hyperparameter Tuning

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `OPTUNA_N_TRIALS` | `50` | Số trial tối đa Optuna chạy để tìm hyperparams tốt nhất |
| `OPTUNA_TIMEOUT` | `600` | Timeout (giây) cho toàn bộ Optuna study |
| `LR_MIN` / `LR_MAX` | `0.01` / `0.3` | Khoảng learning rate Optuna tìm kiếm (tất cả model) |
| `MAX_DEPTH_MIN` / `MAX_DEPTH_MAX` | `3` / `10` | Khoảng max_depth cây (XGBoost, LGBM, RF, ExtraTrees) |
| `N_ESTIMATORS_MIN` / `N_ESTIMATORS_MAX` | `100` / `1000` | Khoảng số cây |
| `SUBSAMPLE_MIN` / `SUBSAMPLE_MAX` | `0.4` / `1.0` | Khoảng tỷ lệ sample row/col khi train mỗi cây |
| `NUM_LEAVES_MIN` / `NUM_LEAVES_MAX` | `15` / `256` | Khoảng num_leaves riêng của LightGBM |
| `CB_DEPTH_MIN` / `CB_DEPTH_MAX` | `4` / `10` | Khoảng depth riêng của CatBoost |

---

## RFE (Recursive Feature Elimination)

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `RFE_TARGET_FEATURES` | `50` | Số feature muốn giữ lại sau RFE |
| `RFE_STEP` | `0.05` | Mỗi vòng RFE loại bỏ 5% số feature còn lại |
| `RFE_CV_SPLITS` | `3` | Số fold CV dùng trong RFECV (chỉ khi `ENABLE_RFECV=true`) |
| `ENABLE_RFECV` | `false` | Bật RFECV (tự chọn số feature tối ưu qua CV). `false` = dùng RFE cố định |
| `RFE_N_ESTIMATORS` | `200` | n_estimators cho base model fit trong RFE selection |

---

## PSI (Population Stability Index)

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `PSI_THRESHOLD` | `0.3` | PSI > 0.3 → feature drift quá lớn giữa train và OOT → DROP |
| `PSI_BINS` | `100` | Số bin tính PSI (nhiều bin → chính xác hơn, cần đủ data) |

---

## Stability Check

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `STABILITY_MIN_MONTHS` | `6` | Cần ít nhất 6 tháng data mới chạy stability check, ít hơn → skip |
| `STABILITY_GINI_STD_THRESHOLD` | `0.15` | Độ lệch chuẩn Gini theo tháng > 0.15 → feature không ổn định → DROP |
| `STABILITY_MIN_GINI` | `0.02` | Gini trung bình < 0.02 → feature gần như không có predictive power → DROP |

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
| `SHAP_N_ESTIMATORS` | `100` | n_estimators cho model nhanh fit tại mỗi step pruning |
| `SHAP_SAMPLE_SIZE` | `2000` | Tối đa số row sample để tính SHAP value (cân bằng tốc độ vs độ chính xác) |
| `SHAP_PSI_MAX_NO_IMPROVE` | `2` | Dừng pruning sau N step liên tiếp không cải thiện AUC |
| `SHAP_PSI_MIN_FEATURES_FLOOR` | `5` | Sàn cứng tối thiểu số feature giữ lại sau pruning |
| `SHAP_PSI_MIN_FEATURES_RATIO` | `0.10` | Sàn tương đối: giữ ít nhất 10% × `MAX_FINAL_FEATURES` |

---

## Overfitting Detection

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `OVERFIT_THRESHOLD` | `0.12` | Gap tương đối `(valid_auc - holdout_auc) / valid_auc` > 12% → flag overfit và trigger LLM-guided retrain |

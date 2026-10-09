# Agent 3 — TrainModel

> **Vai trò**: chọn estimator, chọn feature, tune, refit trên train+valid, calibrate, đánh giá trên holdout sạch, rồi ghi model + bundle replay + báo cáo.
> Code: [`Agents/TrainModel/agent_train_model.py`](../Agents/TrainModel/agent_train_model.py). Tham chiếu theo **tên hàm** (số dòng thay đổi theo thời gian).

Luồng **mặc định**:

```
Split → Encode → FLAML → Rank → PSI → Stability → Batch prune → Optuna → Refit (CV + multi-seed + calibration + PD floor/cap)
      → Eval OOT/test → Overfit check → SHAP + LLM explain → Save
```

Luồng **cũ** vẫn chọn được qua config: `OPTUNA_AFTER_SELECTION=false` (Optuna trước, re-tune sau), `FEATURE_RANK_METHOD=rfe`, `PRUNE_METHOD=one_by_one`.

---

## Entry points

| Mode | Hàm | Input |
|---|---|---|
| Pipeline (cả split mode lẫn single-file mode) | `process_splits()` | các file `engineered_{train,valid,oot,test}.parquet` → concat + cột marker `_split_` → `process()` |
| Gọi trực tiếp | `process()` | 1 DataFrame + report của Agent 2 |
| Replay | `replay_fit()` | splits đã dựng lại + estimator / params / feature list đã ghi → chỉ chạy bước refit |

Ở single-file mode, việc chia dữ liệu đã làm ở **Stage 0 của pipeline** (trước Agent 1); Agent 3 chỉ dựng lại các partition từ marker, không tự chia lại.

---

## Step 0–1 — Setup + split

- `target`, `date_col` (thành viên composite key không phải entity id), `id_col` lấy từ report trước; feature = mọi cột trừ target / date / id / marker.
- `_tool_split_data()`: có marker → dựng lại đúng partition. Không marker → dùng chung `splitting.auto_split` với pipeline: có cột ngày thì OOT theo tháng/tuần (`OOT_INIT_MONTHS`, `OOT_MIN_RATIO`, `OOT_MAX_RATIO`) + `valid_temporal` (`VALID_TEMPORAL_RATIO`) + `valid_random`; không có thì 60/20/20 với `test` ngẫu nhiên.
- Không có cột ngày → **không có OOT**; PSI và Stability bị bỏ qua (log ghi rõ).

## Step 2 — Encode + dtype downcast

`_fit_prepare_X` fit LabelEncoder trên train (có sentinel `__NA__` cho giá trị lạ), float → float32, int → int32, bool → int8, NaN số → `-999`. `_transform_X` dùng lại encoder cho valid/OOT/test.

## Step 3 — FLAML

Chọn estimator (`FLAML_ESTIMATORS`) + tham số gốc, metric ROC AUC, eval holdout trên valid. Ngân sách `FLAML_TIME_BUDGET` được co theo kích thước dữ liệu (`_compute_time_budget`). Train lớn hơn `FLAML_MAX_ROWS` được lấy mẫu.

## Step 4 — Rank (`_tool_rank_features`) — mặc định thay cho RFE

1. Một lần fit model chọn feature (`_selection_model`: tham số FLAML, `FEATURE_RANK_N_ESTIMATORS` cây, learning rate tối thiểu `FEATURE_RANK_LEARNING_RATE`, không DART, early stopping trên valid).
2. Xếp hạng theo importance (`FEATURE_RANK_IMPORTANCE`: `gain` hoặc `shap`), bỏ feature importance = 0.
3. Loại feature gần trùng: giữa 2 feature có |corr| > `FEATURE_RANK_CORR_THRESHOLD` giữ cái quan trọng hơn.
4. Giữ `RFE_TARGET_FEATURES` ứng viên cho các bước sau.

Mất vài phút thay vì vài giờ của RFE (Home Credit: ~1,5 phút so với ~3 giờ). `FEATURE_RANK_METHOD=rfe` → `_tool_run_rfe` (sklearn RFE/RFECV, `RFE_STEP`, `RFE_N_ESTIMATORS`).

## Step 5 — PSI filter (`_tool_run_psi_filter`)

PSI từng feature giữa train và OOT (`PSI_BINS` bucket theo quantile train). PSI > `PSI_THRESHOLD` → loại. Chỉ chạy khi có cột ngày và OOT; dùng phân phối, không dùng nhãn OOT.

## Step 6 — Stability (`_tool_run_stability_check`)

Gini đơn biến theo từng tháng/tuần trên train. Cần ≥ `STABILITY_MIN_MONTHS` kỳ. Loại nếu std Gini > `STABILITY_GINI_STD_THRESHOLD` hoặc mean Gini < `STABILITY_MIN_GINI`.

## Step 7 — Batch prune (`_tool_batch_prune`) — mặc định

```
lặp:
    fit model chọn feature trên tập hiện tại (early stopping trên valid) → ghi (số feature, valid AUC)
    bỏ PRUNE_BATCH_FRACTION feature yếu nhất (importance, trộn 50/50 với PSI nếu có OOT)
dừng khi: chạm sàn | PRUNE_MAX_ROUNDS | AUC thấp hơn đỉnh > PRUNE_STOP_DROP
chọn: làm mượt đường AUC (PRUNE_SMOOTH_WINDOW điểm), lấy TẬP NHỎ NHẤT có AUC trong
      dung sai max(PRUNE_AUC_TOLERANCE, PRUNE_SE_MULTIPLIER × SE Hanley-McNeil) so với đỉnh
cap_first: nếu vẫn > MAX_FINAL_FEATURES → lấy tập AUC cao nhất trong các tập ≤ trần
```

- Sàn: `max(SHAP_PSI_MIN_FEATURES_FLOOR, MAX_FINAL_FEATURES × SHAP_PSI_MIN_FEATURES_RATIO)`.
- Đường cong ghi vào `shap_psi_prune_log.csv` (cột `n_features`, `valid_auc`, `smoothed_auc`, `selected`).
- `PRUNE_SE_MULTIPLIER` mặc định 0: trên Home Credit, 0,5 cắt 98 → 52 feature và mất ~0,003 test AUC. Chỉ dùng cho tập valid nhỏ/nhiễu.
- `PRUNE_METHOD=one_by_one` → `_tool_shap_psi_prune` (bỏ 1 feature/bước, dừng sau `SHAP_PSI_MAX_NO_IMPROVE` bước không cải thiện).

## Step 8 — Optuna (`_tool_run_optuna`)

- Mặc định (`OPTUNA_AFTER_SELECTION=true`): chạy **một lần** trên tập feature cuối, không re-tune.
- TPE (`seed=RANDOM_STATE`), tối đa `OPTUNA_N_TRIALS` trial / `OPTUNA_TIMEOUT` (co theo dữ liệu).
- **Giới hạn từng trial**: `OPTUNA_TRIAL_TIMEOUT` (0 = tự động `max(120 s, timeout/8)`); LightGBM/XGBoost kiểm mỗi vòng boosting và prune trial quá hạn.
- DART tắt mặc định (`OPTUNA_ENABLE_DART`); rf/extra_tree bị chặn `TREE_ENSEMBLE_MAX_ESTIMATORS` cây.
- Class weight tự động khi mất cân bằng (`AUTO_CLASS_WEIGHT_THRESHOLD`, `CLASS_WEIGHT_STRATEGY`); với xgboost/catboost `scale_pos_weight` nằm trong không gian tìm kiếm.
- Luồng cũ: re-tune sau prune chỉ được chấp nhận nếu hơn tham số cũ (chấm lại trên feature cuối) ít nhất `RETUNE_MIN_GAIN`.
- Log ghi số trial hoàn thành / bị prune. Lưu ý: timeout đo theo giờ thực, máy ngủ cũng bị tính.

## Step 9 — Refit-on-train+valid (`_tool_train_final_model`)

```
X_final = train + valid_temporal + valid_random (hoặc valid) — OOT / test KHÔNG gộp
CV StratifiedKFold(CV_N_SPLITS) trên X_final:
    trần cây mỗi fold = n_estimators đã tune × FINAL_ES_TREE_HEADROOM (boosters, không DART)
    early stopping (EARLY_STOPPING_ROUNDS) → best_iteration từng fold + OOF predictions
    cảnh báo "CV WARN" nếu fold vẫn dừng ở trần
best_iteration = median các fold
Multi-seed (MULTI_SEED_N) refit trên toàn bộ X_final với n_estimators = best_iteration
Calibrator (CALIBRATION_METHOD) fit trên OOF — không leakage
PD trả ra = clip(calibrated, PD_FLOOR, PD_CAP) — lưu trong artifact, scoring dùng cùng giá trị
```

- Metric: `cv_auc_mean/std` (OOF, ước lượng in-time trung thực), `oot_*` / `test_*` (holdout sạch), `valid_*` = **in-sample**, có cờ `valid_metrics_in_sample=true`; CLI ghi "Valid (IN-SAMPLE)", và các metric này không được đưa cho LLM tóm tắt.
- `FINAL_ES_TREE_HEADROOM=1` = hành vi cũ (cần để replay đúng các bundle tạo trước tuỳ chọn này).
- Calibrator `sigmoid` là class cấp module (`SigmoidCalibrator`) nên lưu/nạp được.

## Step 10 — Overfit check + LLM-guided retrain

Gap = `(cv_auc_mean − holdout_auc) / cv_auc_mean`, holdout = OOT hoặc test. Gap > `OVERFIT_THRESHOLD` → LLM đề xuất tham số regularization (fallback `_apply_conservative_regularization`), refit, giữ bản tốt hơn trên holdout. Lưu ý: khi nhánh này chạy, holdout đã được dùng để chọn nên không còn hoàn toàn "sạch".

## Step 11 — SHAP + giải thích bằng LLM (`_tool_shap_final_explain`)

SHAP trên model cuối (mẫu `SHAP_SAMPLE_SIZE` dòng train), bar + beeswarm, LLM viết `meaning` / `why_matters` cho `SHAP_FINAL_EXPLAIN_TOP_N` feature. Tắt bằng `SHAP_FINAL_EXPLAIN_ENABLED=false`.

## Step 12 — Lưu artifact

| File | Nội dung |
|---|---|
| `final_model.pkl` | ensemble + calibrator + `pd_floor`/`pd_cap` + encoders + feature list + versions |
| `final_model_code.py` / `pipeline_process_train_model.py` | inference trên dữ liệu đã engineered; dừng khi thiếu feature hoặc lệch phiên bản (`AUTOML_ALLOW_VERSION_MISMATCH=1`) |
| `model_trainer_report.json` | estimator, tham số, funnel feature, metric, SHAP top features, tóm tắt LLM |
| `psi_report.csv`, `stability_report.csv` | khi có OOT / cột ngày |
| `shap_psi_prune_log.csv` | đường cong prune |
| `shap_summary.png`, `shap_beeswarm.png`, `shap_feature_explanations.csv`, `final_model_shap_report.md`, `charts/` | giải thích + biểu đồ chẩn đoán |

Bundle replay (manifest, hash, provenance) do pipeline ghi — xem [replay.md](replay.md).

---

## Quy tắc no-leakage

| Bước | Dữ liệu | Ghi chú |
|---|---|---|
| Encode, rank, prune, Optuna | train (+ valid làm ES / chấm điểm) | không chạm OOT/test |
| PSI | phân phối train vs OOT | không dùng nhãn OOT |
| Stability | train | |
| Refit | train + valid | OOT/test giữ nguyên |
| Calibration | OOF của CV | không leakage |
| Overfit retrain | OOT/test để đo | khi chạy thì holdout tham gia chọn model |

## Reproducibility

Mọi bước ngẫu nhiên seed theo `RANDOM_STATE`. FLAML và Optuna phụ thuộc thời gian thực nên chạy lại toàn bộ pipeline có thể ra tham số khác; muốn tái lập đúng một run thì dùng `replay_pipeline.py --mode retrain` (chỉ chạy lại bước refit với quyết định đã đóng băng).

## Config knobs (mặc định hiện tại)

| Nhóm | Param | Mặc định |
|---|---|---|
| FLAML | `FLAML_TIME_BUDGET` / `FLAML_ESTIMATORS` / `FLAML_MAX_ROWS` | 3600 s / `xgboost,lgbm,catboost,rf,extra_tree` / 500 000 |
| Rank | `FEATURE_RANK_METHOD` / `FEATURE_RANK_IMPORTANCE` / `FEATURE_RANK_CORR_THRESHOLD` | `importance` / `gain` / 0,90 |
| | `FEATURE_RANK_N_ESTIMATORS` / `FEATURE_RANK_LEARNING_RATE` / `RFE_TARGET_FEATURES` | 500 / 0,05 / 150 |
| RFE (cũ) | `RFE_STEP` / `RFE_N_ESTIMATORS` / `ENABLE_RFECV` | 0,01 / 200 / false |
| Prune | `PRUNE_METHOD` / `PRUNE_BATCH_FRACTION` / `PRUNE_AUC_TOLERANCE` | `batch_tolerance` / 0,10 / 0,001 |
| | `PRUNE_SMOOTH_WINDOW` / `PRUNE_SE_MULTIPLIER` / `PRUNE_STOP_DROP` / `PRUNE_MAX_ROUNDS` | 3 / 0,0 / 0,01 / 40 |
| | `FEATURE_CAP_POLICY` / `MAX_FINAL_FEATURES` | `cap_first` / 100 |
| Prune (cũ) | `SHAP_N_ESTIMATORS` / `SHAP_PSI_MAX_NO_IMPROVE` | 200 / 3 |
| | `SHAP_PSI_MIN_FEATURES_FLOOR` / `SHAP_PSI_MIN_FEATURES_RATIO` / `SHAP_SAMPLE_SIZE` | 5 / 0,10 / 20 000 |
| PSI / Stability | `PSI_THRESHOLD` / `PSI_BINS` | 0,35 / 100 |
| | `STABILITY_MIN_MONTHS` / `STABILITY_GINI_STD_THRESHOLD` / `STABILITY_MIN_GINI` | 6 / 0,20 / 0,01 |
| Optuna | `OPTUNA_AFTER_SELECTION` / `OPTUNA_N_TRIALS` / `OPTUNA_TIMEOUT` | true / 300 / 7200 s |
| | `OPTUNA_TRIAL_TIMEOUT` / `OPTUNA_ENABLE_DART` / `TREE_ENSEMBLE_MAX_ESTIMATORS` | 0 (auto) / false / 500 |
| | `ENABLE_RETUNE_AFTER_PRUNE` / `RETUNE_FEATURE_REDUCTION_TRIGGER` / `RETUNE_MIN_GAIN` | true / 0,8 / 0,0005 (chỉ luồng cũ) |
| Refit | `CV_N_SPLITS` / `FINAL_ES_TREE_HEADROOM` / `EARLY_STOPPING_ROUNDS` | 5 / 2,0 / 200 |
| | `MULTI_SEED_ENABLED` / `MULTI_SEED_N` | true / 5 |
| Calibration | `CALIBRATION_ENABLED` / `CALIBRATION_METHOD` / `PD_FLOOR` / `PD_CAP` | true / `isotonic` / 0,0003 / 0,9999 |
| Class weight | `AUTO_CLASS_WEIGHT_THRESHOLD` / `CLASS_WEIGHT_STRATEGY` | 3,0 / `auto` |
| Split | `OOT_INIT_MONTHS` / `OOT_MIN_RATIO` / `OOT_MAX_RATIO` / `VALID_TEMPORAL_RATIO` | 2 / 0,15 / 0,20 / 0,20 |
| Khác | `OVERFIT_THRESHOLD` / `RANDOM_STATE` / `SHAP_FINAL_EXPLAIN_TOP_N` | 0,12 / 42 / 20 |

Full reference: [config_params.md](config_params.md).

# Architecture & Visual Walkthrough

Tài liệu mô tả trực quan từng phần của pipeline. Mọi sơ đồ dùng [Mermaid](https://mermaid.js.org/) và render trực tiếp trên GitHub.

---

## 1. Project Structure Overview

```
multi-agent-auto-ml-v1.1/
│
├── main.py                  ◄── CLI entry point
├── pipeline.py              ◄── AutoMLPipeline orchestrator
├── handoff.py               ◄── State holder giữa các agent
├── logger.py                ◄── AgentLogger (log + markdown report)
├── config.py                ◄── Config + LLM client factory
├── config.yaml              ◄── LiteLLM proxy config
│
├── Agents/
│   ├── BaseAgent/
│   │   └── base_agent.py                ◄── LLM call + load_dataframe + ToolRegistry
│   ├── DataCleaner/
│   │   ├── agent_data_cleaner.py        ◄── Agent 1
│   │   └── prompts/{system,user}.txt
│   ├── FeatureEngineer/
│   │   ├── agent_feature_engineer.py    ◄── Agent 2
│   │   └── prompts/{system,user}.txt
│   └── TrainModel/
│       ├── agent_train_model.py         ◄── Agent 3
│       └── prompts/{llm_summary_*}.txt
│
├── data/
│   ├── process_home_data.py             ◄── Script chuẩn bị HomeCredit
│   └── HomeCredit_columns_description.csv
│
├── docs/
│   ├── config_params.md                 ◄── Reference ~70 config params
│   └── architecture.md                  ◄── (file này)
│
├── outputs/                             ◄── Auto-tạo, chứa toàn bộ output
└── tests/
    ├── test_agent1.py
    ├── test_agent2.py
    └── test_agent3.py
```

### Mối quan hệ giữa các module

```mermaid
flowchart TD
    Main[main.py / CLI] --> Pipe[pipeline.AutoMLPipeline]
    Pipe --> A1[DataCleanerAgent]
    Pipe --> A2[FeatureEngineerAgent]
    Pipe --> A3[TrainModelAgent]
    Pipe --> HO[Handoff]

    A1 -.kế thừa.-> Base[BaseAgent]
    A2 -.kế thừa.-> Base
    A3 -.kế thừa.-> Base

    Base --> LLM{LLM Endpoint}
    LLM --> Proxy[LiteLLM Proxy]
    LLM --> OAI[OpenAI Direct]
    LLM --> CLD[Claude Direct]

    A1 --> Logger
    A2 --> Logger
    A3 --> Logger
    HO --> Logger

    Config[(config.py)] --> Base
    Config --> Pipe
    Config --> A1
    Config --> A2
    Config --> A3
```

---

## 2. High-Level Pipeline Flow

```mermaid
flowchart LR
    Raw[(Raw data)] --> A1
    A1[Agent 1<br/>DataCleaner] -->|clean_data.csv<br/>+ report1| HO1[Handoff]
    HO1 --> A2[Agent 2<br/>FeatureEngineer]
    A2 -->|engineered_data.csv<br/>+ report2| HO2[Handoff]
    HO2 --> A3[Agent 3<br/>TrainModel]
    A3 --> Outs[(final_model.pkl<br/>+ inference code<br/>+ reports)]

    style A1 fill:#a8d8ea
    style A2 fill:#ffd3b6
    style A3 fill:#dcedc1
```

Mỗi agent đều theo cùng pattern:

```mermaid
flowchart LR
    In[Input data<br/>+ previous report] --> Collect[Collect stats<br/>tool calls]
    Collect --> Build[Build LLM prompt<br/>system + user]
    Build --> LLM[Call LLM<br/>JSON mode]
    LLM --> Parse[Parse decisions<br/>+ execute tools]
    Parse --> Out[Output data<br/>+ report]

    style LLM fill:#fff2a8
```

---

## 3. Agent 1 — DataCleaner

**Vai trò**: Audit chất lượng dữ liệu, sửa schema, loại bỏ duplicate / cột rác, KHÔNG làm feature selection.

### Sơ đồ hoạt động

```mermaid
flowchart TD
    Start([Load raw data]) --> M[inspect_metadata<br/>shape, dtypes, nulls, duplicates]
    Start --> L[check_label_quality<br/>imbalance, null labels]
    Start --> P[check_pk_uniqueness<br/>exact/soft/app dup<br/>composite key + entity resolution]
    Start --> F[check_column_formats<br/>numeric-as-text, mixed case]
    Start --> O[detect_outliers<br/>IQR×3]
    Start --> T[check_temporal<br/>future dates, leakage]

    M --> Compress[Compress metadata<br/>summary cho LLM]
    L --> Build
    P --> Build
    F --> Build
    O --> Build
    T --> Build
    Compress --> Build[Build user prompt]

    Build --> LLM[Call LLM<br/>JSON mode, max_tokens=4000]
    LLM --> Parse[Parse decisions]

    Parse --> Loop{For each action}
    Loop -->|drop_column| G1{Guard:<br/>null>80% OR constant<br/>OR all-unique?}
    Loop -->|drop_duplicates| Apply
    Loop -->|clip_outliers| G2{Guard:<br/>not PK col?}
    Loop -->|deduplicate_by_key| G3{Guard:<br/>target not in keys?}
    Loop -->|fix_column_dtype| G4{Guard:<br/>not PK col?}

    G1 -->|pass| Apply[Apply action]
    G2 -->|pass| Apply
    G3 -->|pass| Apply
    G4 -->|pass| Apply
    G1 -->|block| Skip[Log SKIP, next]
    G2 -->|block| Skip
    G3 -->|block| Skip
    G4 -->|block| Skip

    Apply --> Loop
    Skip --> Loop

    Loop -->|done| Save[Save clean_data.csv<br/>+ data_cleaner_report.json]
    Save --> End([Output:<br/>path + report])

    style LLM fill:#fff2a8
    style G1 fill:#ffcccc
    style G2 fill:#ffcccc
    style G3 fill:#ffcccc
    style G4 fill:#ffcccc
```

### Tools registry

| Tool | Mục đích |
|---|---|
| `inspect_metadata` | Shape, dtypes, null counts, duplicate stats |
| `get_column_stats` | Distribution / unique values 1 cột |
| `drop_column` | Xoá cột |
| `detect_outliers` | IQR×3 outlier detection |
| `check_label_quality` | Class balance, null labels |
| `check_temporal` | Date range, future dates, leakage |
| `drop_duplicates` | Xoá row trùng exact |
| `clip_outliers` | Clip về bounds IQR×3 |
| `check_pk_uniqueness` | PK / composite key / entity resolution |
| `deduplicate_by_key` | Dedup theo composite key |
| `check_column_formats` | numeric-as-text, mixed case, whitespace |
| `fix_column_dtype` | Cast numeric/datetime/strip/standardize |

### Protected columns

DataCleanerAgent giữ một set "không bao giờ được động vào":

```mermaid
flowchart LR
    PC[_pk_protected_cols] --> E[entity_id_col]
    PC --> C[composite_key_cols]
    PC --> Tg[target_column]

    style PC fill:#ffcccc
```

Mọi action `drop_column`, `clip_outliers`, `fix_column_dtype` đều check set này trước.

### Output

- `outputs/clean_data.csv`
- `outputs/data_cleaner_report.json` — chứa `entity_id_col`, `composite_key_cols`, `target_column`, `actions_taken`, `summary`

---

## 4. Agent 2 — FeatureEngineer

**Vai trò**: Tạo interaction feature có ý nghĩa nghiệp vụ, encode categorical, chọn top-K predictive features.

### Sơ đồ hoạt động

```mermaid
flowchart TD
    Start([Load clean_data.csv<br/>+ report1]) --> Inherit[Inherit protected cols<br/>từ Agent 1]
    Inherit --> Date[Detect date_col<br/>từ composite_key]
    Date --> Analyze[_analyze_features<br/>numeric/categorical metadata]

    Analyze --> CD{col_descriptions<br/>được cung cấp?}
    CD -->|yes| Group[_build_col_desc_section<br/>group by source table]
    CD -->|no| Skip1[skip section]
    Group --> Domain
    Skip1 --> Domain[Inject domain guidance<br/>credit_risk/propensity/fraud/generic]

    Domain --> Build[Build user prompt<br/>analysis + descriptions + target]
    Build --> LLM[Call LLM<br/>JSON mode, max_tokens=4000]
    LLM --> Parse[Parse actions]

    Parse --> Loop{For each action}
    Loop -->|encode_all_categorical| ENC[Encode all object cols<br/>label, onehot fallback nếu nunique>5]
    Loop -->|create_interaction| CI{Expression<br/>refs protected col?}
    Loop -->|correlation_analysis| CORR[Correlation tới target]
    Loop -->|select_top_features| ST[SelectKBest<br/>f_classif/f_regression]

    CI -->|no| EVAL[eval expression<br/>builtins=blocked]
    CI -->|yes| Block[Log SKIP]

    EVAL --> Check{Result constant<br/>or all-NaN?}
    Check -->|yes| Drop[Drop col, raise]
    Check -->|no| Add[Add col to df]

    ST --> Snap[Snapshot protected cols]
    Snap --> Sel["SelectKBest scoring<br/>col-by-col, O(n_rows) mem"]
    Sel --> Restore[Re-add protected cols<br/>nếu bị loại]

    ENC --> Loop
    CORR --> Loop
    Add --> Loop
    Block --> Loop
    Drop --> Loop
    Restore --> Loop

    Loop -->|done| Save[Save engineered_data.csv<br/>+ feature_engineer_report.json]
    Save --> End([Output:<br/>path + report])

    style LLM fill:#fff2a8
    style CI fill:#ffcccc
    style Check fill:#ffcccc
```

### Domain guidance — inject vào system prompt

```mermaid
flowchart LR
    D[domain arg] --> CR[credit_risk<br/>DPD, MOB, debt burden]
    D --> PR[propensity<br/>RFM, engagement]
    D --> FR[fraud<br/>velocity, network, anomaly]
    D --> GE[generic<br/>standard ratios]

    CR --> Guide[DOMAIN_GUIDANCE<br/>injected into system prompt]
    PR --> Guide
    FR --> Guide
    GE --> Guide
```

### Tools

| Tool | Mục đích |
|---|---|
| `create_interaction` | Tạo cột mới qua expression (`df['a'] / df['b']`) |
| `encode_categorical` | Encode 1 cột (label/onehot) |
| `encode_all_categorical` | Encode toàn bộ object cols cùng lúc (preferred) |
| `correlation_analysis` | Pearson corr tới target |
| `select_top_features` | Giữ top-K theo `f_classif` / `f_regression` |

### Action order (rule)

```
encode_all_categorical → create_interaction(s) → correlation_analysis → select_top_features
```

Nếu LLM bỏ qua thứ tự (vd. select trước encode), các cột object sẽ bị loại oan vì SelectKBest không score được.

### Output

- `outputs/engineered_data.csv`
- `outputs/feature_engineer_report.json` — forward `entity_id_col`, `composite_key_cols`, `target_column` cho Agent 3

---

## 5. Agent 3 — TrainModel

**Vai trò**: AutoML pipeline đầy đủ — chọn estimator, fine-tune, lọc feature theo 5 tầng, train final + đánh giá đa split, detect overfit.

### Sơ đồ hoạt động tổng quan

```mermaid
flowchart TD
    Start([Load engineered_data<br/>+ report2]) --> Detect[Detect target / date_col / id_col<br/>ưu tiên report, fallback auto-detect]
    Detect --> Split[Step 1: Split data]

    Split --> S1{date_col<br/>tồn tại?}
    S1 -->|yes| OOT[OOT temporal split<br/>train + valid_temporal + valid_random + oot]
    S1 -->|no| Simple[60/20/20 stratified split<br/>train + valid + test]

    OOT --> Encode[Fit LabelEncoder trên train<br/>__NA__ sentinel cho NaN]
    Simple --> Encode

    Encode --> Budget[Compute time budget<br/>scale theo n_rows × n_cols]
    Budget --> Flaml[Step 2: FLAML AutoML<br/>chọn best estimator]
    Flaml --> Optuna[Step 3: Optuna fine-tune<br/>TPE sampler trên valid_temporal]

    Optuna --> RFE[Step 4: RFE<br/>cắt xuống MAX_FINAL_FEATURES]
    RFE --> PSI[Step 5: PSI filter<br/>drop drift>threshold giữa train và OOT]
    PSI --> Stab[Step 6: Stability check<br/>drop std Gini theo tháng cao]
    Stab --> SHAP[Step 7: SHAP+PSI iterative prune<br/>cho đến khi AUC ngừng tăng]

    SHAP --> Final[Step 8: Train final model<br/>+ eval trên CV, valid, OOT, test]

    Final --> Over{Overfit detected?<br/>gap > OVERFIT_THRESHOLD}
    Over -->|no| Save
    Over -->|yes| LLMReg[LLM-guided retrain<br/>regularization mạnh hơn]
    LLMReg --> Cmp{"Retry holdout<br/>>= original?"}
    Cmp -->|yes| Save
    Cmp -->|no| Revert[Restore original model]
    Revert --> Save

    Save[Save final_model.pkl<br/>+ final_model_code.py<br/>+ reports + CSVs]
    Save --> Summary[Gen LLM summary]
    Summary --> End([Output:<br/>metrics + report])

    style Flaml fill:#a8d8ea
    style Optuna fill:#a8d8ea
    style RFE fill:#dcedc1
    style PSI fill:#dcedc1
    style Stab fill:#dcedc1
    style SHAP fill:#dcedc1
    style LLMReg fill:#fff2a8
    style Over fill:#ffcccc
```

### Step 1 — Data Split chi tiết

```mermaid
flowchart TD
    Df[(DataFrame<br/>+ target + date_col)] --> Has{provided_oot<br/>được truyền?}

    Has -->|yes| Fast[Fast path:<br/>80/20 stratified split của pool]
    Fast --> Out1[train + valid + provided OOT]

    Has -->|no| HasDate{date_col<br/>tồn tại?}
    HasDate -->|no| Fallback[60/20/20 split<br/>train + valid + test]
    HasDate -->|yes| Months[Extract year-month]

    Months --> MinMo{">= 2 distinct<br/>months?"}
    MinMo -->|no| Fallback

    MinMo -->|yes| FindOOT[Find n_oot:<br/>minimum months để đạt OOT_MIN_RATIO]
    FindOOT --> Floor[Apply floor:<br/>n_oot >= OOT_INIT_MONTHS]
    Floor --> Ceil{n_oot ratio<br/>> OOT_MAX_RATIO?}
    Ceil -->|yes| Shrink[Shrink n_oot]
    Ceil -->|no| Materialize
    Shrink --> Materialize[Materialize OOT df]

    Materialize --> VTemp[Valid temporal:<br/>rows gần nhất với OOT boundary]
    VTemp --> VRand[Valid random:<br/>stratified split phần còn lại]
    VRand --> Out2[train + valid_temporal<br/>+ valid_random + oot]

    style HasDate fill:#ffcccc
    style MinMo fill:#ffcccc
    style Ceil fill:#ffcccc
```

### Steps 2-3 — FLAML → Optuna

```mermaid
flowchart LR
    X_train --> FL[FLAML AutoML<br/>time_budget scaled]
    FL --> Est[best_estimator<br/>vd: xgboost, lgbm, ...]
    FL --> P0[base_params]

    Est --> OP[Optuna TPE<br/>n_trials=50, timeout scaled]
    P0 --> OP
    X_valid[X_valid_temporal] --> OP

    OP --> BP[best_params đầy đủ<br/>FixedTrial replay]

    style FL fill:#a8d8ea
    style OP fill:#a8d8ea
```

### Steps 4-7 — Feature Selection 5 tầng

```mermaid
flowchart LR
    All[All features<br/>~n_init] --> RFE
    RFE[Step 4 RFE<br/>cắt xuống MAX_FINAL_FEATURES] --> PSI
    PSI[Step 5 PSI<br/>train vs OOT drift] --> Stab
    Stab[Step 6 Stability<br/>monthly Gini std] --> SHAP
    SHAP[Step 7 SHAP+PSI<br/>iterative prune] --> Final[Final features]

    PSI -.skip nếu.-> NoOOT[no date_col<br/>or no OOT]
    Stab -.skip nếu.-> Few[< STABILITY_MIN_MONTHS]
    SHAP -.skip nếu.-> NoVal[no validation set]

    style RFE fill:#dcedc1
    style PSI fill:#dcedc1
    style Stab fill:#dcedc1
    style SHAP fill:#dcedc1
```

### Step 7 — SHAP+PSI Iterative Prune chi tiết

```mermaid
flowchart TD
    Start([Features after Stability]) --> Base[Compute baseline<br/>valid AUC]
    Base --> Iter{"len(features)<br/>> min_features?"}

    Iter -->|yes| Fit[Fit model fast<br/>SHAP_N_ESTIMATORS]
    Fit --> Shap[Compute SHAP importance<br/>+ PSI lookup]
    Shap --> Score["removal_score =<br/>0.5×(1-shap_norm) + 0.5×psi_norm"]
    Score --> Worst[Pick worst feature]
    Worst --> Try[Train without it<br/>compute candidate AUC]
    Try --> Improve{"candidate_auc<br/>>= best_auc?"}

    Improve -->|yes| Update[best_features = candidate<br/>reset streak]
    Improve -->|no| Streak[no_improve_streak++]

    Update --> Iter
    Streak --> Stop{"streak >=<br/>MAX_NO_IMPROVE?"}
    Stop -->|yes| End
    Stop -->|no| Iter

    Iter -->|no| End([Return best_features])

    style Improve fill:#ffcccc
    style Stop fill:#ffcccc
```

### Step 8 — Overfitting Detection & Retrain

```mermaid
flowchart TD
    Train[Train final model] --> Eval[Eval trên CV + valid +<br/>oot/test]
    Eval --> Check[gap = valid_auc - holdout_auc / valid_auc]
    Check --> Detect{gap > 12%?}

    Detect -->|no| Done([Save model])
    Detect -->|yes| Snap[Pickle current model + params]

    Snap --> Ask[Call LLM:<br/>suggest reg params]
    Ask --> ParseOK{Parsed OK?}
    ParseOK -->|no| Conservative[Apply conservative<br/>regularization defaults]
    ParseOK -->|yes| Override[Merge LLM overrides<br/>vào best_params]

    Conservative --> Retry
    Override --> Retry[Retrain với params mới]

    Retry --> Cmp{"retry_holdout_auc<br/>>= orig_holdout?"}
    Cmp -->|yes| Keep[Keep retrained model]
    Cmp -->|no| Revert[Restore original model<br/>từ pickle]

    Keep --> Done
    Revert --> Done

    style Detect fill:#ffcccc
    style Cmp fill:#ffcccc
    style Ask fill:#fff2a8
```

### Tools nội bộ

| Tool | Mục đích |
|---|---|
| `_tool_split_data` | OOT temporal split (auto / provided OOT / fallback simple) |
| `_tool_run_flaml` | FLAML AutoML chọn estimator |
| `_tool_run_optuna` | Fine-tune hyperparams |
| `_tool_run_rfe` | RFE / RFECV feature selection |
| `_tool_run_psi_filter` | Drop feature theo PSI drift |
| `_tool_run_stability_check` | Drop feature theo monthly Gini std |
| `_tool_shap_psi_prune` | Iterative SHAP+PSI prune |
| `_tool_train_final_model` | Train final + eval đa split |

### Output

| File | Mô tả |
|---|---|
| `outputs/final_model.pkl` | model + cat_encoders + feature_cols + target |
| `outputs/final_model_code.py` | Standalone inference code |
| `outputs/model_trainer_report.json` | JSON report đầy đủ |
| `outputs/psi_report.csv` | PSI score từng feature |
| `outputs/stability_report.csv` | Mean + std Gini theo tháng |
| `outputs/shap_psi_prune_log.csv` | Log từng step pruning |

---

## 6. Handoff & State Flow

```mermaid
sequenceDiagram
    participant P as Pipeline
    participant H as Handoff
    participant A1 as Agent 1
    participant A2 as Agent 2
    participant A3 as Agent 3
    participant FS as Filesystem

    P->>A1: process(input_path)
    A1->>FS: write clean_data.csv
    A1-->>P: (path, report1)
    P->>H: set_data(path, report1, "DataCleaner")
    H->>H: invalidate cache

    P->>H: get_data()
    H->>FS: load_dataframe (cached)
    H-->>P: DataFrame
    P->>H: get_report()
    H-->>P: report1

    P->>A2: process(df, report1, target)
    A2->>FS: write engineered_data.csv
    A2-->>P: (path, report2)
    P->>H: set_data(path, report2, "FeatureEngineer")
    H->>H: invalidate cache

    P->>H: get_data() + get_report()
    H-->>P: DataFrame + report2

    P->>A3: process(df, report2, target, oot_df)
    A3->>FS: write final_model.pkl + reports
    A3-->>P: (metrics, report3)
```

---

## 7. LLM Fallback Strategy

Project hỗ trợ **2 backend** chọn qua env var `LLM_BACKEND`. Mỗi backend có chiến lược khác nhau:

```mermaid
flowchart LR
    Env[LLM_BACKEND env var] --> L{legacy?}
    L -->|yes default| Legacy[3-tier app-level fallback<br/>proxy + OpenAI + Claude]
    L -->|no - gateway| Gateway[Single source<br/>LiteLLM gateway only<br/>gateway tự xử lý failover]

    style Legacy fill:#ffd3b6
    style Gateway fill:#dcedc1
```

### Gateway backend (`LLM_BACKEND=gateway`)

Mỗi `call_llm()` đi thẳng tới gateway, KHÔNG có app-level fallback (gateway tự xử lý nội bộ):

```mermaid
flowchart TD
    Call[call_llm prompt] --> Route{Effort level<br/>+ prompt size}
    Route -->|low/short| H[Haiku tier]
    Route -->|medium/short| H
    Route -->|medium/long| S[Sonnet tier]
    Route -->|high/short| S
    Route -->|high/long| O[Opus tier]

    H --> GW[ANTHROPIC_BASE_URL<br/>via anthropic SDK]
    S --> GW
    O --> GW

    GW --> Retry{Retryable<br/>error?}
    Retry -->|yes| Backoff[Exponential backoff<br/>up to LLM_MAX_RETRIES]
    Backoff --> GW
    Retry -->|no| Done([Return / raise])

    style GW fill:#dcedc1
```

### Legacy backend (`LLM_BACKEND=legacy`, default)

Mỗi `call_llm()` đi qua chain 4 bước:

```mermaid
flowchart TD
    Call[call_llm prompt] --> Route{Prompt length<br/>> MODEL_ROUTING_THRESHOLD?}
    Route -->|yes| Cloud[Use CLOUD_MODEL]
    Route -->|no| Local[Use LOCAL_MODEL]

    Cloud --> P1[Try: LiteLLM proxy]
    Local --> P1

    P1 --> S1{Success?}
    S1 -->|yes| Done([Return result])
    S1 -->|no| Conn{Connection<br/>error?}

    Conn -->|yes| HasKey1{OPENAI_API_KEY<br/>set?}
    Conn -->|no| Swap[Try: swap proxy model<br/>local→cloud / cloud→local]

    Swap --> S2{Success?}
    S2 -->|yes| Done
    S2 -->|no| HasKey1

    HasKey1 -->|yes| OAI[Try: OpenAI direct]
    HasKey1 -->|no| CLD

    OAI --> S3{Success?}
    S3 -->|yes| Done
    S3 -->|no| CLD[Try: Claude direct<br/>last resort]

    CLD --> S4{Success?}
    S4 -->|yes| Done
    S4 -->|no| Fail([Raise])

    style P1 fill:#a8d8ea
    style Swap fill:#a8d8ea
    style OAI fill:#ffd3b6
    style CLD fill:#dcedc1
```

Mỗi attempt còn có **retry exponential backoff** cho lỗi rate-limit (429) và server (5xx), với `LLM_MAX_RETRIES=3`, base delay `LLM_RETRY_DELAY=2s`.

---

## 8. Protected Columns — bảo vệ key columns xuyên suốt

```mermaid
flowchart TD
    Input[Input từ CLI<br/>--entity-id, --keys, --target] --> A1
    A1[Agent 1<br/>auto-detect nếu thiếu] -->|report.entity_id_col<br/>report.composite_key_cols<br/>report.target_column| A2
    A2[Agent 2<br/>inherit + add date_col] -->|protected forwarded| A3
    A3[Agent 3<br/>inherit + add id_col + date_col<br/>cho OOT split]

    A1 -.bảo vệ trong.-> A1G[drop_column / clip / dtype fix]
    A2 -.bảo vệ trong.-> A2G[interaction expressions<br/>encode / select_top]
    A3 -.dùng trong.-> A3G[OOT split + stability check]

    style A1G fill:#ffcccc
    style A2G fill:#ffcccc
    style A3G fill:#ffcccc
```

---

## 9. Output Folder Map

Sau khi pipeline chạy xong, `outputs/` chứa:

```
outputs/
├── clean_data.csv                    ◄── Agent 1 output
├── engineered_data.csv               ◄── Agent 2 output
│
├── data_cleaner_report.json          ◄── Agent 1 actions + decisions
├── feature_engineer_report.json      ◄── Agent 2 actions + decisions
├── model_trainer_report.json         ◄── Agent 3 metrics + feature pipeline
│
├── psi_report.csv                    ◄── PSI drift per feature
├── stability_report.csv              ◄── Monthly Gini stats
├── shap_psi_prune_log.csv            ◄── Pruning step-by-step log
│
├── final_model.pkl                   ◄── Model + encoders (joblib)
├── final_model_code.py               ◄── Standalone inference script
│
├── final_report.md                   ◄── Full markdown report
└── agent_execution.log               ◄── Plain-text execution log
```

---

## 10. Tham chiếu nhanh

- [README.md](../README.md) — Quick start + CLI examples
- [docs/config_params.md](config_params.md) — Reference toàn bộ ~70 config params
- [pipeline.py](../pipeline.py) — Orchestrator chính
- [Agents/BaseAgent/base_agent.py](../Agents/BaseAgent/base_agent.py) — LLM call + tool execution + helpers

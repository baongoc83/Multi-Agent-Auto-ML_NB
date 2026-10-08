# Architecture & Visual Walkthrough

Tài liệu mô tả trực quan từng phần của pipeline. Mọi sơ đồ dùng [Mermaid](https://mermaid.js.org/) và render trực tiếp trên GitHub.

> **Architecture v1.1**

---

## 1. Project Structure Overview

```
multi-agent-auto-ml-v1.1/
│
├── main.py                  ◄── CLI entry point
├── pipeline.py              ◄── AutoMLPipeline orchestrator + Stage 0a/0b
├── splitting.py             ◄── Chia train/valid/oot (dùng chung pipeline + Agent 3)
├── logger.py                ◄── AgentLogger (log + markdown report)
├── config.py                ◄── Config + LLM client factory
├── config_gateway.py        ◄── GatewayConfig override khi LLM_BACKEND=gateway
├── config.yaml              ◄── LiteLLM proxy config
├── _e2e_check.py            ◄── 6-test end-to-end smoke suite
│
├── Agents/
│   ├── BaseAgent/
│   │   └── base_agent.py                ◄── LLM call + load_dataframe + smart Excel reader
│   ├── DataCleaner/
│   │   ├── agent_data_cleaner.py        ◄── Agent 1 + CleaningSpec dataclass
│   │   └── prompts/{system,user}.txt
│   ├── FeatureEngineer/
│   │   ├── agent_feature_engineer.py    ◄── Agent 2 + FeatureSpec dataclass
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
│   ├── architecture.md                  ◄── (file này)
│   ├── awareness-pattern.md             ◄── Agentic pattern reference
│   └── config_params.md                 ◄── Reference ~75 config params
│
├── outputs/                             ◄── Auto-tạo
│   └── YYYY-MM-DD/                          ◄── 1 thư mục / ngày
│       └── run_NN/                              ◄── 1 thư mục / lần chạy (counter reset mỗi ngày)
│           ├── agent_execution.log
│           ├── data_cleaner_report.json, feature_engineer_report.json, model_trainer_report.json
│           ├── psi_report.csv, stability_report.csv, shap_psi_prune_log.csv
│           ├── final_report.md
│           ├── final_model.pkl, final_model_code.py
│           ├── pipeline_process_data_cleaner.py        ◄── replay Agent 1 (embed CleaningSpec)
│           ├── pipeline_process_feature_engineer.py    ◄── replay Agent 2
│           ├── feature_spec.pkl                            ◄── sidecar (fitted LabelEncoders)
│           └── pipeline_process_train_model.py         ◄── replay Agent 3 (= inference code)
└── tests/
    ├── test_agent1.py
    ├── test_agent2.py
    └── test_agent3.py
```

Intermediate `clean_*.parquet` / `engineered_*.parquet` (handoff giữa agents) **không lưu vào `run_NN/`** — ghi vào system tempdir và xoá sau khi pipeline kết thúc (try/finally). Set `KEEP_INTERMEDIATES=true` trong `.env` nếu muốn giữ để debug.

### Mối quan hệ giữa các module

```mermaid
flowchart TD
    Main[main.py / CLI] --> Pipe[pipeline.AutoMLPipeline]
    Pipe --> A1[DataCleanerAgent + CleaningSpec]
    Pipe --> A2[FeatureEngineerAgent + FeatureSpec]
    Pipe --> A3[TrainModelAgent]
    Pipe --> SP[splitting.auto_split<br/>chia train/valid/oot]

    A1 -.kế thừa.-> Base[BaseAgent]
    A2 -.kế thừa.-> Base
    A3 -.kế thừa.-> Base

    Base --> LLM{LLM Endpoint}
    LLM --> Proxy[LiteLLM Proxy]
    LLM --> OAI[OpenAI Direct]
    LLM --> CLD[Claude Direct]
    LLM --> GW[Gateway]

    A1 --> Logger
    A2 --> Logger
    A3 --> Logger

    Config[(config.py)] --> Base
    Config --> Pipe
    Config --> A1
    Config --> A2
    Config --> A3
```

---

## 2. High-Level Pipeline Flow — 2 modes

Pipeline tự chọn mode dựa trên CLI args:

```mermaid
flowchart TD
    Start[pipeline.run] --> Check{valid_path<br/>hoặc oot_path<br/>được set?}
    Check -->|yes| Split[Split mode]
    Check -->|no| Single[Single-file mode]

    style Split fill:#a8d8ea
    style Single fill:#ffd3b6
```

### Split mode — fit-on-train + transform-on-valid/oot

```mermaid
flowchart LR
    Train[(train file)] --> S0a
    Valid[(valid file)] --> S0a
    OOT[(oot file)] --> S0a

    S0a[Stage 0a<br/>Schema validation<br/>fail-fast] --> S0b
    S0b[Stage 0b<br/>PSI drift check<br/>warn only] --> A1

    A1[Agent 1<br/>fit on train<br/>capture CleaningSpec<br/>replay on valid+oot] -->|3x clean_*.parquet| A2
    A2[Agent 2<br/>fit on train<br/>capture FeatureSpec<br/>replay on valid+oot] -->|3x engineered_*.parquet| A3
    A3[Agent 3<br/>concat 3 file<br/>add _split_ marker<br/>train + eval]
    A3 --> Outs[(final_model.pkl<br/>+ inference code<br/>+ reports)]

    style A1 fill:#a8d8ea
    style A2 fill:#ffd3b6
    style A3 fill:#dcedc1
```

### Single-file mode — Agent 3 tự auto-split

```mermaid
flowchart LR
    Raw[(Raw data)] --> SP[Stage 0<br/>splitting.auto_split]
    SP -->|"train / valid / oot or test"| A1
    A1[Agent 1<br/>fit on train<br/>replay on holdouts] -->|clean_*.parquet<br/>+ report1| A2
    A2[Agent 2<br/>fit on train<br/>replay on holdouts] -->|engineered_*.parquet<br/>+ report2| A3
    A3[Agent 3<br/>train + eval<br/>theo split đã chia] --> Outs[(final_model.pkl<br/>+ replay bundle<br/>+ reports)]

    style A1 fill:#a8d8ea
    style A2 fill:#ffd3b6
    style A3 fill:#dcedc1
```

### Pattern mỗi agent (split mode)

```mermaid
flowchart LR
    Load[Load train] --> Stats[Collect stats<br/>tool calls]
    Stats --> Build[Build LLM prompt]
    Build --> LLM[Call LLM<br/>JSON mode]
    LLM --> Exec[Execute decisions<br/>+ capture spec]
    Exec --> Train[Save clean_train.parquet]
    Train --> Replay[Replay spec on<br/>valid + oot]
    Replay --> Out[Save clean_valid.parquet<br/>+ clean_oot.parquet]

    style LLM fill:#fff2a8
    style Exec fill:#a8d8ea
```

---

## 3. Stage 0a — Schema Validation (split mode only)

Trước cả Agent 1, schema được scan từ metadata (không load full data):

- **Parquet**: `pyarrow.parquet.read_schema` — 1 disk seek
- **CSV**: `pd.read_csv(nrows=1000)` để infer dtype
- **Khác**: fallback `BaseAgent.load_dataframe` (Excel/Feather/ORC nhỏ)

```mermaid
flowchart TD
    Start[Read schemas:<br/>train + valid + oot] --> Hard

    Hard{Required cols<br/>existing in all 3?}
    Hard -->|no| Raise[Raise ValueError<br/>liệt kê thiếu cột ở partition nào]
    Hard -->|yes| Soft1

    Soft1{Train có cols<br/>thiếu ở valid/oot?}
    Soft1 -->|yes| W1[Log WARN<br/>spec replay sẽ skip silently]
    Soft1 -->|no| Soft2

    W1 --> Soft2{Valid/oot có cols<br/>thừa không trong train?}
    Soft2 -->|yes| I1[Log INFO<br/>sẽ bị drop khi select feature]
    Soft2 -->|no| Soft3

    I1 --> Soft3{Dtype-kind mismatch<br/>trên shared cols?}
    Soft3 -->|yes| W2[Log WARN<br/>vd: numeric vs object<br/>likely silent NaN downstream]
    Soft3 -->|no| OK

    W2 --> OK[Schema OK<br/>continue Stage 0b]

    style Raise fill:#ffcccc
    style W1 fill:#fff2a8
    style W2 fill:#fff2a8
    style I1 fill:#fff2a8
```

**Hard errors** (raise → abort pipeline):
- `target_column` thiếu ở bất kỳ partition nào
- `entity_id_col` thiếu
- Bất kỳ `composite_key_cols` nào thiếu

**Warnings** (log → tiếp tục):
- Train có cột missing ở valid/oot
- Valid/oot có cột thừa
- Dtype-kind mismatch (numeric vs object)

---

## 4. Stage 0b — Distribution Drift Check (split mode only, optional)

PSI-based drift check, sample 50K rows mỗi partition:

```mermaid
flowchart TD
    Start[Load sample:<br/>~50K rows mỗi partition] --> Skip[Skip cols:<br/>target, entity_id,<br/>composite_keys, date-named]

    Skip --> Numeric[Identify numeric features]
    Numeric --> Pair{For each<br/>train↔valid<br/>train↔oot}

    Pair --> PSI["PSI = sum(actual% - expected%) × ln(actual%/expected%)<br/>over 10 quantile bins"]
    PSI --> Drift{"PSI > 0.25?"}

    Drift -->|no| Info["Log INFO<br/>checked N cols max PSI=x.x"]
    Drift -->|yes| Warn["Log WARN<br/>k cols drift > 0.25<br/>Top 10: col1=psi1..."]

    Info --> Next
    Warn --> Next[Continue<br/>never raises]

    style PSI fill:#a8d8ea
    style Warn fill:#fff2a8
    style Drift fill:#ffcccc
```

**Disable**: `--no-drift-check` hoặc `check_distribution=False` trong `pipeline.run`.

---

## 5. Agent 1 — DataCleaner

**Vai trò**: Audit chất lượng dữ liệu, sửa schema, loại bỏ duplicate / cột rác. Capture mọi transform vào `CleaningSpec` để replay lên valid/oot.

### Sơ đồ hoạt động (split mode)

```mermaid
flowchart TD
    Start([Load train]) --> Sample{train_sample_ratio<br/>được set?}
    Sample -->|yes| Strat[Stratified sample<br/>by target]
    Sample -->|no| Pre
    Strat --> Pre

    Pre{prefilter<br/>= True?}
    Pre -->|yes| Scan["Scan cols rác:<br/>null>95% / constant / dominant>99%<br/>→ spec.drops"]
    Pre -->|no| Stats
    Scan --> Stats

    Stats[Collect stats:<br/>inspect_metadata<br/>check_label_quality<br/>check_pk_uniqueness<br/>check_column_formats<br/>detect_outliers<br/>check_temporal]

    Stats --> Build[Build user prompt]
    Build --> LLM[Call LLM<br/>JSON mode, max_tokens=4000]
    LLM --> Parse[Parse decisions]

    Parse --> Loop{For each action}
    Loop -->|drop_column| G1{"Guard:<br/>not PK +<br/>null>80% OR constant OR all-unique?"}
    Loop -->|drop_duplicates<br/>TRAIN-only| Apply
    Loop -->|clip_outliers| G2{Guard:<br/>not PK + numeric?}
    Loop -->|deduplicate_by_key<br/>TRAIN-only| G3{Guard:<br/>target not in keys?}
    Loop -->|fix_column_dtype| G4{Guard:<br/>not PK?}

    G1 -->|pass| C1[Apply + spec.drops.append]
    G2 -->|pass| C2[Apply + spec.clip_bounds Q1/Q3]
    G3 -->|pass| C3[Apply on train only<br/>NOT in spec]
    G4 -->|pass| C4[Apply + spec.dtype_fixes.append]

    G1 -->|block| Skip[Log SKIP]
    G2 -->|block| Skip
    G3 -->|block| Skip
    G4 -->|block| Skip

    C1 --> Loop
    C2 --> Loop
    C3 --> Loop
    C4 --> Loop
    Apply --> Loop
    Skip --> Loop

    Loop -->|done| SaveT[Save clean_train.parquet<br/>+ spec captured]
    SaveT --> Free[del + gc.collect<br/>free train RAM]
    Free --> RV{valid_path?}
    RV -->|yes| LoadV[Load valid] --> ApplyV[CleaningSpec.apply<br/>drops + dtype + clip] --> SaveV[Save clean_valid.parquet] --> RO
    RV -->|no| RO{oot_path?}
    RO -->|yes| LoadO[Load oot] --> ApplyO[CleaningSpec.apply] --> SaveO[Save clean_oot.parquet] --> End
    RO -->|no| End([Output:<br/>paths + report + spec])

    style LLM fill:#fff2a8
    style G1 fill:#ffcccc
    style G2 fill:#ffcccc
    style G3 fill:#ffcccc
    style G4 fill:#ffcccc
    style C1 fill:#a8d8ea
    style C2 fill:#a8d8ea
    style C3 fill:#dcedc1
    style C4 fill:#a8d8ea
    style ApplyV fill:#a8d8ea
    style ApplyO fill:#a8d8ea
```

### CleaningSpec dataclass

```python
@dataclass
class CleaningSpec:
    drops:       List[str]                       # prefilter + drop_column actions
    dtype_fixes: List[Tuple[str, str]]           # (col, fix_type)
    clip_bounds: Dict[str, Tuple[float, float]]  # col → (lower, upper) từ Q1/Q3 train
    # KHÔNG include drop_duplicates / dedup_by_key → row ops, train-only
```

**Replay** (`spec.apply(df)`):
1. Drop cols trong `spec.drops` nếu còn trong df
2. Áp `dtype_fixes`: `cast_to_numeric` / `cast_to_datetime` / `strip_whitespace` / `standardize_case`
3. Áp `clip_bounds`: `df[col].clip(lower, upper)` với bounds từ train

### Tools registry

| Tool | Mục đích |
|---|---|
| `inspect_metadata` | Shape, dtypes, null counts, duplicate stats |
| `get_column_stats` | Distribution / unique values 1 cột |
| `drop_column` | Xoá cột → capture vào `spec.drops` |
| `detect_outliers` | IQR×3 outlier detection |
| `check_label_quality` | Class balance, null labels |
| `check_temporal` | Date range, future dates, leakage |
| `drop_duplicates` | Xoá row trùng exact (TRAIN-only) |
| `clip_outliers` | Clip về Q1±factor×IQR → capture vào `spec.clip_bounds` |
| `check_pk_uniqueness` | PK / composite key / entity resolution |
| `deduplicate_by_key` | Dedup theo composite key (TRAIN-only) |
| `check_column_formats` | numeric-as-text, mixed case, whitespace |
| `fix_column_dtype` | Cast numeric/datetime/strip/standardize → capture vào `spec.dtype_fixes` |

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

### Mixed-type object cols — `_safe_str_map`

Cột `object` dtype không có nghĩa toàn string — có thể chứa NaN, int, dict trộn lẫn. Helper `_safe_str_map(series, fn)` chỉ apply `fn` lên cells thực sự là string, giữ nguyên NaN/None/numbers:

```python
def _safe_str_map(series, fn):
    return series.map(lambda x: fn(x) if isinstance(x, str) else x)
```

Tránh được crash `AttributeError: Can only use .str accessor with string values!` trên mixed-type cols ở `check_column_formats` + `fix_column_dtype`.

### Output

Intermediate handoff files (`clean_*.parquet`) ghi vào **system tempdir** (`TMP_DIR`) và xoá sau khi pipeline kết thúc — không vào `run_NN/` (set `KEEP_INTERMEDIATES=true` để giữ).

**Persisted vào `RUN_DIR` = `outputs/YYYY-MM-DD/run_NN/`** (cả 2 mode):
- `data_cleaner_report.json` — chứa `entity_id_col`, `composite_key_cols`, `target_column`, `actions_taken`, `summary`, `cleaning_spec` (serialized)
- `pipeline_process_data_cleaner.py` — script Python standalone replay `CleaningSpec` (drops + dtype_fixes + clip_bounds embed inline) trên data mới

---

## 6. Agent 2 — FeatureEngineer

**Vai trò**: Tạo interaction feature có ý nghĩa nghiệp vụ, encode categorical, chọn top-K predictive features. Capture mọi transform vào `FeatureSpec` để replay lên valid/oot.

### Sơ đồ hoạt động (split mode)

```mermaid
flowchart TD
    Start([Load clean_train.parquet<br/>+ report1]) --> Inherit[Inherit protected cols<br/>từ Agent 1]
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
    Loop -->|encode_all_categorical| ENC["Encode all object cols<br/>onehot fallback nếu nunique>5<br/>→ spec.label_encoders<br/>→ spec.onehot_columns"]
    Loop -->|create_interaction| CI{Expression<br/>refs protected col?}
    Loop -->|correlation_analysis| CORR[Correlation tới target<br/>log only, no transform]
    Loop -->|select_top_features| ST[SelectKBest<br/>f_classif/f_regression]

    CI -->|no| EVAL[eval expression<br/>builtins=blocked]
    CI -->|yes| Block[Log SKIP]

    EVAL --> Check{Result constant<br/>or all-NaN?}
    Check -->|yes| Drop[Drop col, raise]
    Check -->|no| Add["Add col + capture<br/>spec.interactions.append (new_col, expr, train_median)"]

    ST --> Snap[Snapshot protected cols]
    Snap --> Sel["SelectKBest scoring<br/>col-by-col, O(n_rows) mem"]
    Sel --> Restore["Re-add protected cols<br/>+ spec.selected_features = final list"]

    ENC --> Loop
    CORR --> Loop
    Add --> Loop
    Block --> Loop
    Drop --> Loop
    Restore --> Loop

    Loop -->|done| SaveT[Save engineered_train.parquet<br/>+ spec]
    SaveT --> Free[del + gc.collect<br/>free train RAM]
    Free --> ReplayV{valid_path?}
    ReplayV -->|yes| ApplyV[Load valid<br/>FeatureSpec.apply<br/>recreate interactions<br/>+ encoders + selection] --> SaveV[Save engineered_valid.parquet] --> ReplayO
    ReplayV -->|no| ReplayO{oot_path?}
    ReplayO -->|yes| ApplyO[Load oot<br/>FeatureSpec.apply] --> SaveO[Save engineered_oot.parquet] --> End
    ReplayO -->|no| End([Output:<br/>paths + report + spec])

    style LLM fill:#fff2a8
    style CI fill:#ffcccc
    style Check fill:#ffcccc
    style Add fill:#a8d8ea
    style ENC fill:#a8d8ea
    style Restore fill:#a8d8ea
    style ApplyV fill:#a8d8ea
    style ApplyO fill:#a8d8ea
```

### FeatureSpec dataclass

```python
@dataclass
class FeatureSpec:
    interactions:      List[Tuple[str, str, float]] # (new_col, expression, train_median_fill)
    label_encoders:    Dict[str, LabelEncoder]      # fitted on train, with __NA__ sentinel
    onehot_columns:    Dict[str, List[str]]         # col → train dummy column names
    selected_features: Optional[List[str]]          # if select_top_features was called
    target_column:     Optional[str]
```

**Replay** (`spec.apply(df)`):
1. **Re-create interactions**: same `eval(expression)`, fill NaN/inf bằng `train_median` đã captured (không phải median của valid/oot — leakage-free)
2. **Label encoders**: `Series.where(isin(known), "__NA__")` rồi `le.transform` — vectorised, ~10× nhanh hơn `.apply(lambda)`
3. **One-hot**: `pd.get_dummies` rồi `reindex(columns=train_dummies, fill_value=0)` — bỏ cols mới, fill 0 cho cols mất
4. **Column selection**: keep only `spec.selected_features` + target

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
| `create_interaction` | Tạo cột mới qua expression (`df['a'] / df['b']`) — capture median fill |
| `encode_categorical` | Encode 1 cột (label/onehot) — capture encoder |
| `encode_all_categorical` | Encode toàn bộ object cols cùng lúc (preferred) — capture batch |
| `correlation_analysis` | Pearson corr tới target (log only, no transform) |
| `select_top_features` | Giữ top-K theo `f_classif` / `f_regression` — capture selected list |

### Action order (rule)

```
encode_all_categorical → create_interaction(s) → correlation_analysis → select_top_features
```

Nếu LLM bỏ qua thứ tự (vd. select trước encode), các cột object sẽ bị loại oan vì SelectKBest không score được.

### Output

Intermediate `engineered_*.parquet` ghi vào `TMP_DIR` và xoá sau khi pipeline kết thúc (giống Agent 1).

**Persisted vào `RUN_DIR`** (cả 2 mode):
- `feature_engineer_report.json` — forward `entity_id_col`, `composite_key_cols`, `target_column`, `feature_spec` (serialized)
- `pipeline_process_feature_engineer.py` — script standalone replay FeatureSpec
- `feature_spec.pkl` — sidecar chứa fitted `LabelEncoder` objects (không embed được inline)

---

## 7. Agent 3 — TrainModel

**Vai trò**: AutoML pipeline đầy đủ — chọn estimator, fine-tune, lọc feature theo 5 tầng, train final + đánh giá đa split, detect overfit.

### Entry points

```mermaid
flowchart LR
    Pipe{Pipeline mode} --> SP[Split mode:<br/>process_splits<br/>train + valid + oot paths]
    Pipe --> SF[Single-file mode:<br/>process<br/>1 DataFrame + oot_df=None]

    SP --> Concat[Load 3 files<br/>add _split_ marker<br/>concat]
    Concat --> Inner[Call process<br/>internally]
    SF --> Inner[process<br/>main training pipeline]

    style SP fill:#a8d8ea
    style SF fill:#ffd3b6
    style Concat fill:#dcedc1
```

**Lưu ý**: `_split_` marker chỉ tồn tại **bên trong Agent 3** (không xuyên Agent 1+2 như kiến trúc cũ). Marker được Agent 3 tự gán khi concat 3 file, rồi `_tool_split_data` đọc marker để reconstruct splits exactly.

### Sơ đồ hoạt động tổng quan

```mermaid
flowchart TD
    Start([Load engineered<br/>1 file hoặc 3 file concat]) --> Detect[Detect target / date_col / id_col<br/>ưu tiên report, fallback auto-detect]
    Detect --> Split[Step 1: Split data]

    Split --> S1{_split_<br/>marker tồn tại?}
    S1 -->|yes| PreSplit[Pre-split branch:<br/>train/valid/oot theo marker]
    S1 -->|no| S2{date_col<br/>tồn tại?}
    S2 -->|yes| OOT[OOT temporal split<br/>train + valid_temporal + valid_random + oot]
    S2 -->|no| Simple[60/20/20 stratified split<br/>train + valid + test]

    PreSplit --> Encode
    OOT --> Encode
    Simple --> Encode[Fit LabelEncoder trên train<br/>__NA__ sentinel cho NaN]

    Encode --> Budget[Compute time budget<br/>scale theo n_rows × n_cols]
    Budget --> Flaml[Step 2: FLAML AutoML<br/>chọn best estimator]
    Flaml --> Optuna[Step 3: Optuna fine-tune<br/>TPE sampler trên valid_temporal]

    Optuna --> RFE[Step 4: RFE<br/>cắt xuống MAX_FINAL_FEATURES]
    RFE --> PSI["Step 5: PSI filter<br/>drop drift>threshold giữa train và OOT"]
    PSI --> Stab[Step 6: Stability check<br/>drop std Gini theo tháng cao]
    Stab --> SHAP[Step 7: SHAP+PSI iterative prune<br/>cho đến khi AUC ngừng tăng]

    SHAP --> Final[Step 8: Train final model<br/>+ eval trên CV, valid, OOT, test]

    Final --> Over{"Overfit detected?<br/>gap > OVERFIT_THRESHOLD"}
    Over -->|no| Save
    Over -->|yes| LLMReg[LLM-guided retrain<br/>regularization mạnh hơn]
    LLMReg --> Cmp{"Retry holdout<br/>>= original?"}
    Cmp -->|yes| Save
    Cmp -->|no| Revert[Restore original model]
    Revert --> Save

    Save[Save final_model.pkl<br/>+ final_model_code.py<br/>+ reports + CSVs]
    Save --> Summary[Gen LLM summary]
    Summary --> End([Output:<br/>metrics + report])

    style PreSplit fill:#a8d8ea
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
    Df[(DataFrame)] --> Marker{_split_<br/>marker tồn tại?}

    Marker -->|yes| Has{Marker values:}
    Has --> M1[train + valid + oot<br/>→ dùng tất cả as-marked]
    Has --> M2[train + valid<br/>→ no OOT, dùng marker]
    Has --> M3["train + oot only<br/>→ auto-split 80/20 train→valid<br/>oot từ marker"]

    Marker -->|no| Has2{provided_oot<br/>được truyền?}
    Has2 -->|yes| Fast[Fast path:<br/>80/20 stratified split của pool]
    Fast --> Out1[train + valid + provided OOT]

    Has2 -->|no| HasDate{date_col<br/>tồn tại?}
    HasDate -->|no| Fallback[60/20/20 split<br/>train + valid + test]
    HasDate -->|yes| Months[Extract year-month]

    Months --> MinMo{">= 2 distinct<br/>months?"}
    MinMo -->|no| Fallback

    MinMo -->|yes| FindOOT[Find n_oot:<br/>minimum months để đạt OOT_MIN_RATIO]
    FindOOT --> Floor["Apply floor:<br/>n_oot >= OOT_INIT_MONTHS"]
    Floor --> Ceil{"n_oot ratio<br/>> OOT_MAX_RATIO?"}
    Ceil -->|yes| Shrink[Shrink n_oot]
    Ceil -->|no| Materialize
    Shrink --> Materialize[Materialize OOT df]

    Materialize --> VTemp[Valid temporal:<br/>rows gần nhất với OOT boundary]
    VTemp --> VRand[Valid random:<br/>stratified split phần còn lại]
    VRand --> Out2[train + valid_temporal<br/>+ valid_random + oot]

    style Marker fill:#ffcccc
    style HasDate fill:#ffcccc
    style MinMo fill:#ffcccc
    style Ceil fill:#ffcccc
    style M1 fill:#a8d8ea
    style M2 fill:#a8d8ea
    style M3 fill:#a8d8ea
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
    Stab -.skip nếu.-> Few["< STABILITY_MIN_MONTHS"]
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
    Eval --> Check["gap = (valid_auc - holdout_auc) / valid_auc"]
    Check --> Detect{"gap > 12%?"}

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
| `_tool_split_data` | Split (pre-split marker / provided OOT / OOT temporal / 60-20-20 fallback) |
| `_tool_run_flaml` | FLAML AutoML chọn estimator |
| `_tool_run_optuna` | Fine-tune hyperparams |
| `_tool_run_rfe` | RFE / RFECV feature selection |
| `_tool_run_psi_filter` | Drop feature theo PSI drift |
| `_tool_run_stability_check` | Drop feature theo monthly Gini std |
| `_tool_shap_psi_prune` | Iterative SHAP+PSI prune |
| `_tool_train_final_model` | Train final + eval đa split |

### Output (persisted vào `RUN_DIR` = `outputs/YYYY-MM-DD/run_NN/`)

| File | Mô tả |
|---|---|
| `final_model.pkl` | model + cat_encoders + feature_cols + target |
| `final_model_code.py` | Standalone inference code |
| `pipeline_process_train_model.py` | Bản copy của `final_model_code.py` dưới naming `pipeline_process_*` cho consistent với Agent 1+2 |
| `model_trainer_report.json` | JSON report đầy đủ |
| `psi_report.csv` | PSI score từng feature |
| `stability_report.csv` | Mean + std Gini theo tháng |
| `shap_psi_prune_log.csv` | Log từng step pruning |
| `final_report.md` + `agent_execution.log` | Markdown report tổng + execution log |

---

## 8. Spec replay — fit-on-train + transform-on-valid/oot

```mermaid
sequenceDiagram
    participant P as Pipeline
    participant A1 as Agent 1
    participant A2 as Agent 2
    participant A3 as Agent 3
    participant FS as Filesystem
    participant CS as CleaningSpec
    participant FSp as FeatureSpec

    Note over P,A3: SPLIT MODE — fit-on-train + transform-on-valid/oot

    P->>P: Stage 0a: validate schema
    P->>P: Stage 0b: PSI drift check

    P->>A1: process_splits(train, valid, oot)
    A1->>FS: load train
    A1->>A1: prefilter + LLM + execute decisions
    A1->>CS: capture drops/dtype_fixes/clip_bounds
    A1->>FS: save clean_train.parquet
    A1->>A1: del train_df + gc

    A1->>FS: load valid
    A1->>CS: apply(valid_df) — replay spec
    A1->>FS: save clean_valid.parquet
    A1->>A1: del + gc

    A1->>FS: load oot
    A1->>CS: apply(oot_df) — replay spec
    A1->>FS: save clean_oot.parquet
    A1-->>P: {paths, report, spec serialized}

    P->>A2: process_splits(clean_train, clean_valid, clean_oot)
    A2->>FS: load clean_train
    A2->>A2: LLM + execute decisions
    A2->>FSp: capture interactions/encoders/selection
    A2->>FS: save engineered_train.parquet
    A2->>A2: del + gc

    A2->>FS: load clean_valid
    A2->>FSp: apply(clean_valid) — recreate features
    A2->>FS: save engineered_valid.parquet
    A2->>A2: del + gc

    A2->>FS: load clean_oot
    A2->>FSp: apply(clean_oot) — recreate features
    A2->>FS: save engineered_oot.parquet
    A2-->>P: {paths, report, spec serialized}

    P->>A3: process_splits(engineered_train, valid, oot)
    A3->>FS: load 3 files
    A3->>A3: add _split_ marker per partition + concat
    A3->>A3: run training pipeline (FLAML → Optuna → RFE → ...)
    A3->>FS: save final_model.pkl + reports
    A3-->>P: (metrics, report)
```

**Memory profile**: ở mỗi thời điểm chỉ có 1 partition trong RAM (Agent 1+2). Agent 3 concat 3 file → cần RAM cho toàn bộ data, nhưng đây là điểm bắt buộc vì FLAML/Optuna/CV cần tất cả slices để compute metrics.

**Tối ưu RAM tại Agent 3** (`_fit_prepare_X`):

```mermaid
flowchart LR
    In[df slice<br/>float64 + object] --> Iter[Iter từng cột<br/>không df.copy đầu]
    Iter --> Cat{Object<br/>or category?}
    Cat -->|yes| LE[LabelEncoder.fit_transform<br/>+ astype int dtype]
    Cat -->|no| Float{is_float_dtype?}
    Float -->|yes| F32[fillna -999<br/>+ to_numpy float32]
    Float -->|no| Int[fillna -999<br/>+ to_numpy int32 / int8]
    LE --> N1[smallest signed int<br/>int8 ≤127, int16 ≤32k, int32 lớn hơn]
    N1 --> Dict[Build dict cột]
    F32 --> Dict
    Int --> Dict
    Dict --> Out[pd.DataFrame from dict<br/>~4-8× nhỏ hơn input]
```

Benchmark (50k × 1400 mixed): 1.82 GB → 190 MB (≈9.6×). Cho case 100k × 1455 features hỗn hợp, alloc int64 matrix ban đầu 1.08 GB → ~150-450 MB sau downcast tùy tỉ lệ cat/num.

**Joblib temp folder**: Khi FLAML / sklearn dùng `n_jobs>1`, joblib stage X_train ra `.pkl` cho worker memory-map. `config.py` auto-set `JOBLIB_TEMP_FOLDER=tempfile.gettempdir()` (Linux: `/tmp`, Windows: `%TEMP%`) ngay khi import — tránh `BrokenProcessPool: FileNotFoundError` trên Docker (`/dev/shm` mặc định 64 MB). User override qua `.env`.

---

## 9. Replay bundle — chạy lại một run ở môi trường khác

Mỗi run ghi 6 file đủ để tái lập toàn trình (split → transform → retrain | score).
Chi tiết đầy đủ: [replay.md](replay.md).

```mermaid
sequenceDiagram
    participant U as User
    participant R as replay_pipeline.py
    participant M as replay_manifest.json
    participant S as split_assignment.parquet
    participant SP as cleaning_spec.pkl<br/>feature_spec.pkl
    participant A3 as TrainModelAgent

    U->>R: replay_pipeline.py <raw> --mode retrain
    R->>M: đọc quyết định đã chốt<br/>(estimator, best_params, final features)
    R->>S: join theo key + _key_occ_<br/>sort theo _split_pos_
    Note over R,S: dựng lại ĐÚNG partition + ĐÚNG thứ tự dòng
    R->>SP: CleaningSpec.apply (mọi partition)<br/>+ apply_row_ops (chỉ train)
    R->>SP: FeatureSpec.apply
    R->>A3: replay_fit(...) — bỏ FLAML/Optuna/RFE/PSI/SHAP
    A3-->>R: metrics
    R->>M: so với expected_metrics
    R-->>U: "Reproduced exactly" hoặc chỉ ra chặng nào sai
```

`--mode score` bỏ bước split và `replay_fit`, chỉ load `final_model.pkl` để chấm điểm dữ liệu mới.

---

## 10. LLM Fallback Strategy

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
    Call[call_llm prompt] --> Route{"Prompt length<br/>> MODEL_ROUTING_THRESHOLD?"}
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

## 11. Smart Excel Reader

Pandas chọn Excel engine theo **extension** (`openpyxl` cho `.xlsx`, `xlrd` cho `.xls`). Khi file bị đặt sai extension (rất phổ biến — file `.xls` rename thành `.xlsx`), engine sai sẽ raise lỗi cryptic như "Can't find workbook in OLE2 compound document".

`BaseAgent._read_excel_smart` sniff 8 byte đầu để chọn engine đúng:

```mermaid
flowchart TD
    Path[File path] --> Read[Read first 8 bytes]
    Read --> Magic{Magic bytes}

    Magic -->|"50 4B 03 04"| ZIP[ZIP container<br/>= .xlsx/.xlsm]
    Magic -->|"D0 CF 11 E0 A1 B1 1A E1"| OLE[OLE2 compound<br/>= .xls]
    Magic -->|other| Bad[Raise RuntimeError<br/>chỉ ra 4 byte đầu]

    ZIP --> EngA["Engine: openpyxl<br/>fallback = xlrd"]
    OLE --> EngB["Engine: xlrd<br/>fallback = openpyxl"]

    EngA --> Try[pd.read_excel<br/>với engine]
    EngB --> Try

    Try --> Result{Success?}
    Result -->|yes| Done([Return DataFrame])
    Result -->|ImportError| Fallback[Try other engine]
    Result -->|other Exception| RealError[Raise with context:<br/>file may be corrupt/encrypted]

    Fallback --> Try

    style Bad fill:#ffcccc
    style RealError fill:#ffcccc
```

Wired vào `BaseAgent.load_dataframe` (raw data files) và `FeatureEngineerAgent._load_col_descriptions` (col descriptions).

---

## 12. Protected Columns — bảo vệ key columns xuyên suốt

```mermaid
flowchart TD
    Input[Input từ CLI<br/>--entity-id, --keys, --target] --> A1
    A1[Agent 1<br/>auto-detect nếu thiếu] -->|report.entity_id_col<br/>report.composite_key_cols<br/>report.target_column| A2
    A2[Agent 2<br/>inherit + add date_col] -->|protected forwarded| A3
    A3[Agent 3<br/>inherit + add id_col + date_col<br/>cho OOT split]

    A1 -.bảo vệ trong.-> A1G[drop_column / clip / dtype fix]
    A2 -.bảo vệ trong.-> A2G[interaction expressions<br/>encode / select_top<br/>+ post-select restoration]
    A3 -.dùng trong.-> A3G[OOT split + stability check]

    style A1G fill:#ffcccc
    style A2G fill:#ffcccc
    style A3G fill:#ffcccc
```

---

## 13. Output Folder Map

Sau khi pipeline chạy xong, `outputs/` chứa **các file phụ thuộc mode**:

### Split mode (`--valid` / `--oot` được set)

```
outputs/
├── clean_train.parquet              ◄── Agent 1: train fit + LLM + spec capture
├── clean_valid.parquet              ◄── Agent 1: spec replay, no row drop
├── clean_oot.parquet                ◄── Agent 1: spec replay, no row drop
│
├── engineered_train.parquet         ◄── Agent 2: train fit
├── engineered_valid.parquet         ◄── Agent 2: FeatureSpec replay
├── engineered_oot.parquet           ◄── Agent 2: FeatureSpec replay
│
├── data_cleaner_report.json         ◄── Agent 1 + cleaning_spec serialized
├── feature_engineer_report.json     ◄── Agent 2 + feature_spec serialized
├── model_trainer_report.json        ◄── Agent 3 metrics + feature pipeline
│
├── psi_report.csv                   ◄── PSI drift per feature (Agent 3 step 5)
├── stability_report.csv             ◄── Monthly Gini stats
├── shap_psi_prune_log.csv           ◄── Pruning step-by-step log
│
├── final_model.pkl                  ◄── Model + encoders (joblib)
├── final_model_code.py              ◄── Standalone inference script
│
├── final_report.md                  ◄── Full markdown report
└── agent_execution.log              ◄── Plain-text execution log
```

### Single-file mode (no `--valid` / `--oot`)

```
outputs/
├── clean_data.parquet               ◄── Agent 1 output (1 file)
├── engineered_data.parquet          ◄── Agent 2 output (1 file)
│
├── data_cleaner_report.json
├── feature_engineer_report.json
├── model_trainer_report.json
│
├── psi_report.csv                   (chỉ nếu Agent 3 split có OOT)
├── stability_report.csv             (chỉ nếu Agent 3 split có date_col)
├── shap_psi_prune_log.csv
│
├── final_model.pkl
├── final_model_code.py
├── final_report.md
└── agent_execution.log
```

---

## 14. End-to-end test suite

`_e2e_check.py` chạy 6 test với mock LLM (không gọi network), verify toàn bộ flow:

| # | Test | Verify |
|---|---|---|
| 1 | SINGLE-FILE mode | Pipeline chạy đến hết, có `cv_auc_mean`, model artifact tồn tại |
| 2 | SPLIT mode | 3 file clean + 3 file engineered tồn tại, valid/oot row count preserved, schema parity giữa 3 file |
| 3 | SPLIT + `--sample-ratio 0.5` + `--no-prefilter`-off | Train sampled ~50%, junk cols (constant/null/dominant) drop từ train được replay sang valid/oot |
| 4 | Schema validation: missing target | RAISE trước Agent 1, không có file output |
| 5 | Schema unit: dtype mismatch + extra col + missing col → WARN; missing entity_id / composite_key → RAISE | Đúng severity classification |
| 6 | Distribution drift: train↔valid heavily shifted → WARN với cột bị shift; train↔oot stable → no false positive | PSI threshold đúng + top-N report đúng |

```powershell
python _e2e_check.py
```

---

## 15. Tham chiếu nhanh

- [README.md](../README.md) — Quick start + CLI examples
- [docs/awareness-pattern.md](awareness-pattern.md) — Agentic pattern (Plan-Execute + Awareness)
- [docs/config_params.md](config_params.md) — Reference toàn bộ ~75 config params
- [pipeline.py](../pipeline.py) — Orchestrator chính + Stage 0a/0b
- [Agents/BaseAgent/base_agent.py](../Agents/BaseAgent/base_agent.py) — LLM call + tool execution + smart Excel reader

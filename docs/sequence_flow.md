# Multi-Agent AutoML v1.1 — Sequence diagrams

Nguồn: `main.py`, `pipeline.py`, `config.py`, `config_gateway.py`, `splitting.py`, `preprocessing/`, `replay_driver.py`, `README.md`.

## 1. Bootstrap + dispatch mode

```mermaid
sequenceDiagram
    autonumber
    actor U as User (CLI)
    participant M as main.py
    participant C as config.py
    participant G as config_gateway.py
    participant P as AutoMLPipeline
    participant L as AgentLogger

    U->>M: python main.py train TARGET [--valid] [--oot] [--domain ...]
    M->>C: import config (module-level)
    C->>C: load_dotenv(); setdefault JOBLIB_TEMP_FOLDER=tempdir
    alt LLM_BACKEND=gateway
        C->>G: from config_gateway import GatewayConfig as Config
        Note over G: subclass Config — override BACKEND, validate(),<br/>get_client() (anthropic SDK + bearer), choose_model(),<br/>get_direct_client/get_claude_client → None
    else legacy (default)
        Note over C: Config gốc — proxy / gateway / server on-prem chuẩn OpenAI;<br/>fallback OpenAI → Claude chỉ khi LLM_ALLOW_EXTERNAL_FALLBACK=true
    end
    M->>C: Config.validate()
    alt gateway
        C-->>M: cần ANTHROPIC_AUTH_TOKEN + BASE_URL http(s) + EFFORT_LEVEL ∈ {low,medium,high}
    else legacy
        C-->>M: cần ≥1 trong LITELLM_URL / OPENAI_API_KEY / ANTHROPIC_API_KEY
    end
    M->>M: argparse → input_path, target, keys, valid/oot, domain, product_type...
    M->>P: AutoMLPipeline()
    P->>C: Config.init_run()
    C-->>P: RUN_DIR = outputs/YYYY-MM-DD/run_NN (mkdir atomic, max+1)<br/>TMP_DIR = mkdtemp("automl_pipeline_") hoặc RUN_DIR nếu KEEP_INTERMEDIATES
    P->>L: AgentLogger() (đọc EXECUTION_LOG_PATH đã rebind)
    P->>P: _check_disk_space() — raise nếu free < MIN_DISK_FREE_GB (2 GB)
    M->>P: pipeline.run(...)
    P->>P: split_mode = valid_path or oot_path
    alt split_mode
        P->>P: _run_split_mode()
    else
        P->>P: _run_single_mode()
    end
    Note over P: finally: Config.cleanup_run() — rmtree TMP_DIR kể cả khi raise
    P-->>M: final_metrics
    M->>U: _print_metrics() + _print_files()
```

## 2. Split mode (`--valid` / `--oot`)

```mermaid
sequenceDiagram
    autonumber
    participant P as AutoMLPipeline
    participant A1 as Agent 1 DataCleaner
    participant A2 as Agent 2 FeatureEngineer
    participant A3 as Agent 3 TrainModel
    participant LLM as LLM (gateway | proxy / on-prem [→ OpenAI → Claude nếu cho phép])
    participant NP as Stage 1b NullProcessor
    participant T as TMP_DIR (parquet)
    participant R as RUN_DIR

    rect rgb(240,240,240)
        Note over P: Stage 0a — schema validation (fail-fast)
        P->>P: _read_schema(train/valid/oot) — parquet: read_schema; csv: nrows=1000
        P->>P: HARD: target / entity_id / composite_keys phải có ở mọi partition → ValueError
        P->>P: SOFT: cols thiếu/thừa, dtype-kind mismatch (numeric vs object) → WARN
    end
    rect rgb(240,240,240)
        Note over P: Stage 0b — drift check (advisory, try/except nuốt lỗi)
        P->>P: _load_sample(first DRIFT_SAMPLE_N=100k rows / partition)
        P->>P: skip target, keys, date-like cols; PSI 10 bins train↔valid, train↔oot
        P->>P: WARN nếu PSI > DRIFT_PSI_THRESHOLD=0.25 (top 10)
    end

    P->>A1: DataCleanerAgent(entity_id, keys, target)
    P->>A1: process_splits(train, valid, oot, prefilter, train_sample_ratio)
    A1->>A1: FIT trên train: prefilter (null>95%, dominant>99%) → stratified sample → real-stats
    A1->>LLM: system.txt + user.txt + stats → cleaning actions
    LLM-->>A1: actions (drop / dtype_fix / clip / dedup...)
    A1->>A1: execute canonical order → capture CleaningSpec
    A1->>T: clean_train.parquet
    loop valid, oot (1 partition / lần trong RAM)
        A1->>A1: REPLAY CleaningSpec (không LLM, không row-drop)
        A1->>T: clean_valid.parquet / clean_oot.parquet
    end
    A1->>R: data_cleaner_report.json + pipeline_process_data_cleaner.py
    A1-->>P: clean_paths, report1
    P->>P: del agent1

    rect rgb(240,240,240)
        Note over P,NP: Stage 1b — null processing (NULL_PROCESSOR_ENABLED)
        P->>NP: suggest_rules(clean_train) — chỉ thống kê, không giá trị dòng
        NP->>LLM: batch 40 cột / lời gọi, song song
        LLM-->>NP: missing_type / strategy / indicator → guardrail tự động
        P->>NP: fit(clean_train, y) → null_processor.json (+ SHA-256)
        loop train, valid, oot|test
            NP->>T: transform → ghi đè clean_*.parquet (scale guard, drift warn)
        end
    end

    P->>A2: FeatureEngineerAgent(col_desc, domain, model_type, product_type)
    P->>P: giữ col_descriptions_loaded cho Agent 3
    P->>A2: null_context (cột đã imputed, indicator có sẵn, cột còn NaN)
    P->>A2: process_splits(clean_train, report1, target, clean_valid, clean_oot, create_interactions)
    A2->>A2: FIT trên clean_train: metadata (FEATURE_META_*) + col desc
    A2->>LLM: system.txt (FAMILY A–H, TARGET_NEW_FEATURE_COUNT) + metadata + mục NULL HANDLING ALREADY APPLIED
    LLM-->>A2: interaction / encoding decisions
    A2->>A2: execute (biểu thức qua AST allowlist; bỏ isna() trên cột đã imputed) → IV/WoE → select_top_features
    A2->>A2: FeatureSpec + step_order (thứ tự fit thật)
    A2->>T: engineered_train.parquet
    loop valid, oot
        A2->>A2: REPLAY FeatureSpec theo step_order (diagnostics: fallback, cột thiếu, tỉ lệ category lạ)
        A2->>T: engineered_valid / engineered_oot.parquet
    end
    A2->>R: feature_engineer_report.json + pipeline_process_feature_engineer.py + feature_spec.pkl
    A2-->>P: eng_paths, report2
    P->>P: del agent2

    P->>A3: TrainModelAgent(col_descriptions, domain)
    P->>A3: process_splits(eng_train, report2, target, eng_valid, eng_oot, calibration, temporal_freq)
    A3->>T: load 3 file → gắn _split_ marker → concat 1 lần duy nhất
    A3->>A3: dtype downcast → FLAML → rank (1 fit, loại gần trùng) → PSI → Stability → batch prune (dung sai AUC, ≤ MAX_FINAL_FEATURES)
    A3->>A3: Optuna 1 lần trên tập cuối (giới hạn từng trial) → refit train+valid (CV best_iter có headroom, calibration OOF, multi-seed, PD floor/cap)
    Note over A3: luồng cũ: OPTUNA_AFTER_SELECTION=false / FEATURE_RANK_METHOD=rfe / PRUNE_METHOD=one_by_one
    A3->>A3: eval OOT/test holdout → overfit check (gap > 0.12)
    opt overfit
        A3->>LLM: đề xuất regularisation → retrain
    end
    A3->>LLM: narrate top-20 SHAP features
    A3->>R: model_trainer_report.json, psi/stability/prune CSV, shap_*.png/csv/md, charts/*.png,<br/>final_model.pkl, final_model_code.py, pipeline_process_train_model.py
    A3-->>P: final_metrics, report3

    P->>R: _write_replay_bundle() → replay_manifest.json (provenance + artifacts SHA-256) + replay_pipeline.py
    P->>R: _generate_final_report() → final_report.md (log markdown + temporal + charts + SHAP + token usage)
    P->>R: logger.save() → agent_execution.log (cũng được lưu khi run lỗi giữa chừng)
```

## 3. Single-file mode (auto-split upfront)

Không còn chuỗi agent riêng. Stage 0 chia raw input ngay rồi uỷ quyền cho split
mode, nên Agent 1 + 2 chỉ fit trên TRAIN và Agent 3 kế thừa đúng tập đã chia —
không tự chia lại.

```mermaid
sequenceDiagram
    autonumber
    participant P as AutoMLPipeline
    participant SP as splitting.auto_split
    participant A1 as Agent 1
    participant A2 as Agent 2
    participant A3 as Agent 3

    P->>SP: _auto_split_input(raw, date_col từ --keys)
    Note over SP: có date_col → train/valid/oot (temporal)<br/>không có → train/valid/test 60/20/20
    SP-->>P: partitions + temporal_meta
    P->>P: ghi split_assignment.parquet<br/>(key + _key_occ_ + _split_pos_)
    Note over P: Stage 0a bỏ qua — các partition cắt từ cùng một frame

    P->>A1: process_splits(train, valid, oot|test)
    Note over A1: fit CleaningSpec trên TRAIN<br/>replay sang holdout; row ops chỉ áp train
    A1-->>P: clean_*.parquet, report1

    Note over P: Stage 1b NullProcessor — như split mode
    P->>A2: process_splits(..., report1)
    Note over A2: fit FeatureSpec trên TRAIN<br/>+ chấm iv_stability trên từng holdout
    A2-->>P: engineered_*.parquet, report2

    P->>A3: process_splits(..., temporal_meta)
    Note over A3: nhận split qua _split_ marker — KHÔNG tự chia lại
    A3-->>P: final_metrics, report3

    P->>P: _write_replay_bundle() → replay_pipeline.py + manifest
    P->>P: _generate_final_report(); logger.save()
```

## 4. LLM routing theo backend

```mermaid
flowchart LR
    B{LLM_BACKEND}
    B -->|gateway| G[anthropic.Anthropic<br/>auth_token + ANTHROPIC_BASE_URL<br/>timeout API_TIMEOUT_MS/1000]
    G --> CM{choose_model<br/>len(prompt) > 6000?}
    CM -->|low| Haiku
    CM -->|medium, short| Haiku
    CM -->|medium, long| Sonnet
    CM -->|high, short| Sonnet
    CM -->|high, long| Opus
    B -->|legacy| P1[OpenAI SDK → LITELLM_URL<br/>proxy / gateway / vLLM / SGLang / Ollama<br/>LOCAL_MODEL ngắn · CLOUD_MODEL dài]
    P1 -->|fail| P1b[model proxy còn lại]
    P1b -->|fail, LLM_ALLOW_EXTERNAL_FALLBACK=false| E[RuntimeError rõ ràng<br/>không gọi API public]
    P1b -->|fail, cho phép| P2[OpenAI direct<br/>api.openai.com]
    P2 -->|fail| P3[Anthropic direct<br/>api.anthropic.com]
    B -->|legacy + LLM_SKIP_PROXY| P2
```

Mọi câu trả lời đều qua `_strip_reasoning` (bỏ `<think>…</think>`, phần suy luận không có thẻ mở, token box của GLM) và `_extract_json` (bộ giải mã JSON thật). Request có thể kèm `LLM_EXTRA_BODY`, bỏ tham số theo `LLM_OMIT_PARAMS`, và tắt `response_format` qua `LLM_JSON_MODE_NATIVE=false`.

## 5. Replay / scoring ở môi trường khác

```mermaid
sequenceDiagram
    autonumber
    actor O as Ops
    participant D as replay_pipeline.py
    participant B as Bundle (run dir)
    participant R as Repo (cùng version)

    O->>D: python replay_pipeline.py input --mode score|retrain [--repo] [--allow-version-mismatch]
    D->>B: đọc replay_manifest.json
    D->>R: thêm repo vào sys.path (--repo / AUTOML_REPO / thư mục cha)
    D->>B: kiểm đủ file
    D->>B: kiểm SHA-256 từng artifact (lệch → dừng)
    D->>R: so code_fingerprint (lệch → cảnh báo)
    D->>B: so phiên bản thư viện + booster (lệch → dừng)
    alt score
        D->>D: CleaningSpec → NullProcessor → FeatureSpec(strict) → ensemble → calibrator → clip PD
        D-->>O: scores.parquet (score + key)
    else retrain
        D->>D: dựng lại split theo key + thứ tự → specs → TrainModelAgent.replay_fit
        D-->>O: so metric với run gốc + chẩn đoán nếu lệch
    end
```

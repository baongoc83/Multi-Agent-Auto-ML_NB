import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import sys
from io import StringIO
import traceback
from base_agent import BaseAgent
from logger import AgentLogger
from config import Config


class ModelTrainerAgent(BaseAgent):

    def __init__(self, logger: AgentLogger):
        super().__init__(name="ModelTrainer", role="ML Model Coder", logger=logger)
        self.df: pd.DataFrame = None
        self.target_column: str = None
        self.iteration: int = 0
        self.training_history: List[Dict] = []
        self.available_models: List[str] = self._detect_available_models()

    def _detect_available_models(self) -> List[str]:
        requested = {m.strip() for m in Config.MODELS_TO_COMPARE.split(",")}
        available = []

        if "XGBoost" in requested:
            try:
                import xgboost  # noqa: F401
                available.append("XGBoost")
            except ImportError:
                pass

        if "RandomForest" in requested:
            try:
                from sklearn.ensemble import RandomForestClassifier  # noqa: F401
                available.append("RandomForest")
            except ImportError:
                pass

        if "ExtraTrees" in requested:
            try:
                from sklearn.ensemble import ExtraTreesClassifier  # noqa: F401
                available.append("ExtraTrees")
            except ImportError:
                pass

        if "LightGBM" in requested:
            try:
                import lightgbm  # noqa: F401
                available.append("LightGBM")
            except ImportError:
                self.logger.log(self.name, "Model Detection", "LightGBM not installed — skipping. Run: pip install lightgbm")

        if "CatBoost" in requested:
            try:
                import catboost  # noqa: F401
                available.append("CatBoost")
            except ImportError:
                self.logger.log(self.name, "Model Detection", "CatBoost not installed — skipping. Run: pip install catboost")

        return available if available else ["XGBoost"]

    def _tool_execute_python_code(self, code: str, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        try:
            import xgboost
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import (
                RandomForestClassifier, RandomForestRegressor,
                ExtraTreesClassifier, ExtraTreesRegressor,
            )
            from sklearn.metrics import (
                roc_auc_score, accuracy_score, precision_score,
                recall_score, f1_score, mean_squared_error, r2_score,
            )
        except ImportError as e:
            return {
                "success": False,
                "metrics": {},
                "stdout": "",
                "stderr": f"Import error: {e}. Please install required packages.",
                "error": str(e),
            }

        # Restrict builtins to a safe subset — prevents LLM-generated code from
        # accessing the filesystem, subprocess, or other dangerous APIs.
        safe_builtins = {
            "print": print,
            "float": float, "int": int, "str": str, "bool": bool,
            "list": list, "dict": dict, "tuple": tuple, "set": set,
            "len": len, "range": range, "enumerate": enumerate, "zip": zip,
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
            "sorted": sorted, "reversed": reversed,
            "isinstance": isinstance, "hasattr": hasattr,
        }

        execution_globals = {
            "pd": pd,
            "np": np,
            "df": df,
            "target_col": target_col,
            "train_test_split": train_test_split,
            "accuracy_score": accuracy_score,
            "roc_auc_score": roc_auc_score,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "f1_score": f1_score,
            "mean_squared_error": mean_squared_error,
            "r2_score": r2_score,
            "XGBClassifier": xgboost.XGBClassifier,
            "XGBRegressor": xgboost.XGBRegressor,
            "RandomForestClassifier": RandomForestClassifier,
            "RandomForestRegressor": RandomForestRegressor,
            "ExtraTreesClassifier": ExtraTreesClassifier,
            "ExtraTreesRegressor": ExtraTreesRegressor,
            "__builtins__": safe_builtins,
        }

        # Optional models: set to None if not installed so generated code can guard with `if X is not None`
        try:
            import lightgbm
            execution_globals["LGBMClassifier"] = lightgbm.LGBMClassifier
            execution_globals["LGBMRegressor"] = lightgbm.LGBMRegressor
        except ImportError:
            execution_globals["LGBMClassifier"] = None
            execution_globals["LGBMRegressor"] = None

        try:
            import catboost
            execution_globals["CatBoostClassifier"] = catboost.CatBoostClassifier
            execution_globals["CatBoostRegressor"] = catboost.CatBoostRegressor
        except ImportError:
            execution_globals["CatBoostClassifier"] = None
            execution_globals["CatBoostRegressor"] = None

        old_stdout, old_stderr = sys.stdout, sys.stderr
        redirected_output = StringIO()
        redirected_error = StringIO()

        result: Dict[str, Any] = {
            "success": False,
            "metrics": {},
            "stdout": "",
            "stderr": "",
            "error": None,
        }

        try:
            sys.stdout = redirected_output
            sys.stderr = redirected_error
            exec(code, execution_globals)  # noqa: S102
            result["success"] = True
            result["metrics"] = execution_globals.get("metrics", {})
            result["stdout"] = redirected_output.getvalue()
            result["stderr"] = redirected_error.getvalue()
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["stderr"] = redirected_error.getvalue() + "\n" + traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return result

    def process(
        self,
        df: pd.DataFrame,
        previous_report: Dict[str, Any],
        target_column: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.logger.log(self.name, "Process Start", f"Received engineered data with shape {df.shape}")
        self.logger.log(self.name, "Available Models", ", ".join(self.available_models))

        self.df = df.copy()
        self.target_column = target_column

        self.logger.log(self.name, "Previous Agent Summary", previous_report.get("summary", "No summary"))

        data_summary = self._analyze_data()
        best_metrics: Dict = None
        final_code: str = None

        self.logger.log(self.name, "Feedback Loop Start", f"Maximum iterations: {Config.MAX_TRAINING_ITERATIONS}")

        for iteration in range(1, Config.MAX_TRAINING_ITERATIONS + 1):
            self.iteration = iteration
            self.logger.log(self.name, f"Iteration {iteration}", "Generating training code...")

            code = self._generate_training_code(data_summary, best_metrics)

            self.logger.log(self.name, f"Iteration {iteration}", "Executing training code...")
            execution_result = self._tool_execute_python_code(code, self.df, self.target_column)

            if not execution_result["success"]:
                self.logger.log(
                    self.name, f"Iteration {iteration} - ERROR",
                    f"Code execution failed: {execution_result['error']}",
                )
                continue

            current_metrics = execution_result["metrics"]
            self.training_history.append({
                "iteration": iteration,
                "metrics": current_metrics,
            })

            # Log without the verbose model_comparison block
            log_metrics = {k: v for k, v in current_metrics.items() if k != "model_comparison"}
            self.logger.log(self.name, f"Iteration {iteration} - Metrics", json.dumps(log_metrics, indent=2))

            if best_metrics is None or self._is_better(current_metrics, best_metrics):
                best_metrics = current_metrics
                final_code = code

            if not self._should_continue_training(current_metrics, iteration):
                self.logger.log(
                    self.name, f"Iteration {iteration} - Decision",
                    "Performance is satisfactory. Stopping training.",
                )
                break
            else:
                self.logger.log(
                    self.name, f"Iteration {iteration} - Decision",
                    "Performance can be improved. Continuing...",
                )

        if best_metrics is None:
            raise RuntimeError("All training iterations failed. Please check the error logs.")

        report = {
            "agent": self.name,
            "available_models": self.available_models,
            "total_iterations": len(self.training_history),
            "final_metrics": best_metrics,
            "training_history": self.training_history,
            "summary": self._generate_summary(best_metrics, len(self.training_history)),
        }
        self.save_report(report, Config.MODEL_TRAINER_REPORT_PATH)

        if final_code:
            with open(Config.FINAL_MODEL_CODE_PATH, "w") as f:
                f.write(final_code)

        self.logger.log(self.name, "Process Complete", f"Best model: {best_metrics.get('best_model', 'unknown')} | Metrics: {json.dumps({k: v for k, v in best_metrics.items() if k not in ('model_comparison',)}, indent=2)}")
        return best_metrics, report

    def _analyze_data(self) -> Dict[str, Any]:
        features = [col for col in self.df.columns if col != self.target_column]
        return {
            "n_samples": len(self.df),
            "n_features": len(features),
            "features": features,
            "target_column": self.target_column,
            "target_distribution": self.df[self.target_column].value_counts().to_dict(),
            "is_classification": self.df[self.target_column].nunique() < Config.CLASSIFICATION_UNIQUE_THRESHOLD,
            "sample_data": self.df.head(3).to_dict(),
        }

    def _build_models_info(self, is_classification: bool) -> str:
        suffix = "Classifier" if is_classification else "Regressor"
        class_map = {
            "XGBoost": f"XGB{suffix}",
            "RandomForest": f"RandomForest{suffix}",
            "ExtraTrees": f"ExtraTrees{suffix}",
            "LightGBM": f"LGBM{suffix}",
            "CatBoost": f"CatBoost{suffix}",
        }
        lines = [f"- {class_map[m]} → key '{m}'" for m in self.available_models if m in class_map]
        return "\n".join(lines)

    def _generate_training_code(self, data_summary: Dict, previous_metrics: Dict = None) -> str:
        is_classification = data_summary.get("is_classification", True)
        suffix = "Classifier" if is_classification else "Regressor"
        primary_metric = "roc_auc_score" if is_classification else "r2"
        models_info = self._build_models_info(is_classification)

        system_prompt = f"""You are an expert Machine Learning engineer. Your task: generate executable Python code that trains multiple ML models, compares them, and reports the best one.

Requirements:
1. Complete, executable code — no import statements (everything is pre-provided)
2. Use train_test_split with test_size={Config.TRAIN_TEST_SPLIT_SIZE}, random_state={Config.RANDOM_STATE}
3. Train EVERY model listed below and store per-model metrics in model_results dict
4. For optional models (LightGBM, CatBoost), guard with `if <Class> is not None:`
5. Set a `metrics` dict at the end with:
   - 'best_model': name of best model (string key matching model_results)
   - Top-level primary metric ('{primary_metric}') and secondary metrics from the best model
   - 'model_comparison': the full model_results dict

Available model classes (DO NOT import — pre-provided, may be None for optional ones):
{models_info}

Available variables (DO NOT import):
- df, target_col, pd, np
- train_test_split
- roc_auc_score, accuracy_score, precision_score, recall_score, f1_score (classification)
- mean_squared_error, r2_score (regression)

Example structure for classification:
```python
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={Config.TRAIN_TEST_SPLIT_SIZE}, random_state={Config.RANDOM_STATE})

model_results = {{}}

xgb = XGB{suffix}(n_estimators=100, learning_rate=0.1, max_depth=5, random_state={Config.RANDOM_STATE}, eval_metric='logloss', verbosity=0)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
model_results['XGBoost'] = {{'roc_auc_score': float(roc_auc_score(y_test, y_pred)), 'accuracy': float(accuracy_score(y_test, y_pred)), 'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))}}

rf = RandomForest{suffix}(n_estimators=100, max_depth=5, random_state={Config.RANDOM_STATE})
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
model_results['RandomForest'] = {{'roc_auc_score': float(roc_auc_score(y_test, y_pred)), 'accuracy': float(accuracy_score(y_test, y_pred)), 'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))}}

if LGBM{suffix} is not None:
    lgbm = LGBM{suffix}(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state={Config.RANDOM_STATE}, verbose=-1)
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    model_results['LightGBM'] = {{'roc_auc_score': float(roc_auc_score(y_test, y_pred)), 'accuracy': float(accuracy_score(y_test, y_pred)), 'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))}}

if CatBoost{suffix} is not None:
    cat = CatBoost{suffix}(iterations=100, learning_rate=0.1, depth=5, random_seed={Config.RANDOM_STATE}, verbose=0)
    cat.fit(X_train, y_train)
    y_pred = cat.predict(X_test)
    model_results['CatBoost'] = {{'roc_auc_score': float(roc_auc_score(y_test, y_pred)), 'accuracy': float(accuracy_score(y_test, y_pred)), 'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))}}

best_model_name = max(model_results, key=lambda m: model_results[m].get('{primary_metric}', -999))
metrics = {{'best_model': best_model_name, **model_results[best_model_name], 'model_comparison': model_results}}
```

Return ONLY executable Python code, no explanations.
"""

        if previous_metrics is None:
            prompt = f"""Generate code to train and compare baseline models.

DATA SUMMARY:
{json.dumps(data_summary, indent=2)}

Models to compare: {', '.join(self.available_models)}
This is iteration {self.iteration}. Use reasonable default hyperparameters for each model.
"""
        else:
            best_model = previous_metrics.get("best_model", "unknown")
            comparison = previous_metrics.get("model_comparison", {})
            comparison_lines = "\n".join(
                f"  {m}: {v.get(primary_metric, 'N/A')}"
                for m, v in comparison.items()
            )
            prompt = f"""Previous iteration results:
Best model: {best_model}
{primary_metric} scores by model:
{comparison_lines}

Generate IMPROVED code with tuned hyperparameters for each model.

DATA SUMMARY:
{json.dumps(data_summary, indent=2)}

Models to compare: {', '.join(self.available_models)}
This is iteration {self.iteration}. Tune hyperparameters:
- learning_rate ({Config.LR_MIN} to {Config.LR_MAX})
- max_depth ({Config.MAX_DEPTH_MIN} to {Config.MAX_DEPTH_MAX})
- n_estimators ({Config.N_ESTIMATORS_MIN} to {Config.N_ESTIMATORS_MAX})
- subsample / colsample_bytree ({Config.SUBSAMPLE_MIN} to {Config.SUBSAMPLE_MAX}) [XGBoost/LightGBM]
- num_leaves ({Config.NUM_LEAVES_MIN} to {Config.NUM_LEAVES_MAX}) [LightGBM]
- depth ({Config.CB_DEPTH_MIN} to {Config.CB_DEPTH_MAX}) [CatBoost]

Focus on improving {primary_metric} across all models.
"""

        llm_response = self.call_llm(prompt, system_prompt)
        code = self._extract_code_from_response(llm_response)

        self.logger.log(
            self.name,
            f"Iteration {self.iteration} - Generated Code",
            code[:300] + "..." if len(code) > 300 else code,
        )
        return code

    def _extract_code_from_response(self, response: str) -> str:
        if "```python" in response:
            return response.split("```python")[1].split("```")[0].strip()
        if "```" in response:
            return response.split("```")[1].split("```")[0].strip()
        return response.strip()

    def _is_better(self, current_metrics: Dict, best_metrics: Dict) -> bool:
        if "roc_auc_score" in current_metrics:
            return current_metrics.get("roc_auc_score", 0) > best_metrics.get("roc_auc_score", 0)
        if "r2" in current_metrics:
            return current_metrics.get("r2", -999) > best_metrics.get("r2", -999)
        return current_metrics.get("f1", 0) > best_metrics.get("f1", 0)

    def _should_continue_training(self, current_metrics: Dict, iteration: int) -> bool:
        if iteration >= Config.MAX_TRAINING_ITERATIONS:
            return False

        # Build a compact history summary without the verbose model_comparison block
        history_summary = []
        for h in self.training_history:
            entry = {
                "iteration": h["iteration"],
                "best_model": h["metrics"].get("best_model", "unknown"),
            }
            entry.update({k: v for k, v in h["metrics"].items() if k not in ("best_model", "model_comparison")})
            history_summary.append(entry)

        decision_prompt = f"""You have trained and compared {len(self.available_models)} ML models with these results:

CURRENT METRICS (Iteration {iteration}):
Best model: {current_metrics.get('best_model', 'unknown')}
{json.dumps({k: v for k, v in current_metrics.items() if k not in ('model_comparison',)}, indent=2)}

TRAINING HISTORY:
{json.dumps(history_summary, indent=2)}

Decision: Should we continue training with different hyperparameters, or is this good enough?

Consider:
- Is {('roc_auc_score' if 'roc_auc_score' in current_metrics else 'r2')} >= {Config.TARGET_ROC_AUC if 'roc_auc_score' in current_metrics else Config.TARGET_R2}?
- Are we seeing improvement across iterations?
- Have we plateaued (no improvement in last 2 iterations)?

Respond with ONLY a JSON object:
{{
  "continue": true or false,
  "reasoning": "Your explanation"
}}
"""

        system_prompt = "You are an ML expert making decisions about model training. Respond with ONLY a JSON object as specified. No other text."

        llm_response = self.call_llm(decision_prompt, system_prompt)

        try:
            response_text = llm_response.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            decision = json.loads(response_text)
            self.logger.log(
                self.name,
                f"Iteration {iteration} - LLM Decision Reasoning",
                decision.get("reasoning", "No reasoning provided"),
            )
            return decision.get("continue", False)

        except json.JSONDecodeError:
            # Deterministic fallback when LLM response cannot be parsed
            if "roc_auc_score" in current_metrics:
                return current_metrics.get("roc_auc_score", 0) < Config.TARGET_ROC_AUC
            if "f1" in current_metrics:
                return current_metrics.get("f1", 0) < Config.TARGET_F1
            if "r2" in current_metrics:
                return current_metrics.get("r2", 0) < Config.TARGET_R2
            return current_metrics.get("accuracy", 0) < Config.TARGET_ACCURACY

    def _generate_summary(self, metrics: Dict, iterations: int) -> str:
        best_model = metrics.get("best_model", "unknown")
        comparison = metrics.get("model_comparison", {})
        scalar_metrics = {
            k: v for k, v in metrics.items()
            if k not in ("best_model", "model_comparison") and isinstance(v, (int, float))
        }
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in scalar_metrics.items()])

        if comparison:
            primary = "roc_auc_score" if any("roc_auc_score" in v for v in comparison.values()) else "r2"
            scores = []
            for m, v in comparison.items():
                val = v.get(primary)
                scores.append(f"{m}: {val:.4f}" if isinstance(val, float) else f"{m}: N/A")
            comparison_str = " | Model comparison (" + primary + "): " + ", ".join(scores)
        else:
            comparison_str = ""

        return f"Best model: {best_model} over {iterations} iterations. Metrics: {metric_str}{comparison_str}"

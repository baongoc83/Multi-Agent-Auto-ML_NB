"""Shared train / valid / oot / test splitting.

Single source of truth for how this pipeline carves a dataset into partitions.
Two callers:

  * `AutoMLPipeline._auto_split_input` — single-file mode. Splits the RAW input
    before Agent 1 runs, so cleaning and feature engineering fit on TRAIN only.
  * `TrainModelAgent._tool_split_data` — the legacy in-agent path, kept for
    direct `TrainModelAgent.process()` callers that hand over one frame.

Keeping one implementation is the point: if the two ever drift, Agent 2 would
be fitting IV / WoE / feature selection on a different notion of "train" than
Agent 3 evaluates on, which is exactly the leakage this module exists to stop.

The logic here is a move of what used to live in TrainModelAgent — same
thresholds, same RANDOM_STATE, same whole-period snapping — so partition row
counts are unchanged from before the hoist.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import Config


# log(stage, message) — lets the caller route lines into its own AgentLogger
# without this module depending on one.
LogFn = Callable[[str, str], None]


def _noop_log(stage: str, message: str) -> None:
    pass


@dataclass
class SplitResult:
    """Partitions plus the temporal metadata detected while cutting them.

    `oot` and `test` are mutually exclusive by construction: a temporal split
    (date_col present, >= 2 periods) yields an OOT tail and an empty `test`;
    the non-temporal fallback yields a random `test` holdout and an empty `oot`.
    Naming them apart keeps PSI and the OOT metrics honest — a random holdout
    reported as "OOT" would show PSI ~ 0 and imply a temporal guarantee that
    was never made.
    """
    train: pd.DataFrame
    valid: pd.DataFrame
    oot: pd.DataFrame
    test: pd.DataFrame
    valid_temporal: pd.DataFrame
    valid_random: pd.DataFrame
    temporal_meta: Dict[str, object] = field(default_factory=lambda: {"cadence": "non_temporal"})
    period_freq: str = "M"
    period_unit: str = "month"
    date_cadence: str = "non_temporal"

    def as_splits_dict(self) -> Dict[str, pd.DataFrame]:
        """The dict shape `TrainModelAgent.splits` expects."""
        return {
            "train":          self.train,
            "valid_temporal": self.valid_temporal,
            "valid_random":   self.valid_random,
            "valid":          self.valid,
            "oot":            self.oot,
            "test":           self.test,
        }


def stratified_split(df: pd.DataFrame, target: str, test_size: float, rs: int):
    """Stratified split with automatic fallback to unstratified when a class is too small."""
    from sklearn.model_selection import train_test_split as _split
    try:
        return _split(df, test_size=test_size, stratify=df[target], random_state=rs)
    except ValueError:
        return _split(df, test_size=test_size, stratify=None, random_state=rs)


def resolve_period_freq(
    dt_series: pd.Series,
    temporal_freq: Optional[str],
    week_closing_day: Optional[str],
    log: LogFn = _noop_log,
) -> Tuple[str, str]:
    """Resolve (pandas period freq, human unit) from temporal_freq.

    "weekly" → ("W-<ANCHOR>", "week"); the anchor is the snapshot/closing
    weekday — taken from week_closing_day, else auto-detected as the most
    common weekday present in date_col. Everything else → ("M", "month").
    Weekly is opt-in only (never auto-inferred) to avoid mis-detection.
    """
    mode = (temporal_freq or "auto").lower()
    if mode != "weekly":
        return "M", "month"
    valid = {"MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"}
    anchor = (week_closing_day or "").strip().upper()
    if anchor not in valid:
        wd = dt_series.dropna().dt.day_name().str[:3].str.upper()
        anchor = wd.mode().iloc[0] if not wd.empty else "SUN"
        log("Temporal freq",
            f"weekly cadence — week_closing_day auto-detected = {anchor} "
            "(most common weekday in date_col; set week_closing_day to override)")
    return f"W-{anchor}", "week"


def auto_split(
    df: pd.DataFrame,
    *,
    target_column: str,
    date_col: Optional[str] = None,
    temporal_freq: Optional[str] = None,
    week_closing_day: Optional[str] = None,
    log: LogFn = _noop_log,
) -> SplitResult:
    """Cut `df` into train / valid / (oot | test).

    Temporal path (date_col present with >= 2 distinct periods):
      OOT  = trailing whole periods, sized by OOT_INIT_MONTHS / OOT_MIN_RATIO
             / OOT_MAX_RATIO.
      valid = valid_temporal (pool periods nearest the OOT boundary, snapped to
             whole periods) + valid_random (stratified from what remains).
      test  = empty.

    Fallback (no date_col, < 2 periods, or empty pool):
      60 / 20 / 20 stratified train / valid / test, oot empty.
    """
    target = target_column
    rs = Config.RANDOM_STATE
    total = len(df)
    empty = pd.DataFrame(columns=df.columns)

    if date_col and date_col in df.columns:
        # Bucket each row into a period (month or week) as a standalone Series —
        # avoids df.copy() and the later .drop(columns=["_ym"]) which each
        # allocate a full copy of the frame. Frequency is resolved from
        # temporal_freq: monthly by default, weekly only when opted in.
        _dt = pd.to_datetime(df[date_col], errors="coerce")
        period_freq, period_unit = resolve_period_freq(
            _dt, temporal_freq, week_closing_day, log)
        ym = _dt.dt.to_period(period_freq)

        nat_count = int(_dt.isna().sum())
        if nat_count:
            log("WARN",
                f"{nat_count} rows have unparseable dates in '{date_col}' — "
                "treated as non-OOT (assigned to train pool)")
        sorted_months = sorted(ym.dropna().unique())   # periods (month or week)
        valid_rows = total - nat_count

        # ── Detect temporal cadence on the FULL series (authoritative) ──────
        # "*_snapshot": exactly one distinct calendar date per period (e.g.
        # every row stamped 2023-01-31, or every Friday) → the model operates
        # at that granularity, so every temporal slice (OOT, valid_temporal,
        # stability) must move in whole-period steps. Otherwise "intra_month" /
        # "weekly" (multiple dates per period). Recorded so the report is
        # explicit about the granularity rather than a blind row-count split.
        n_dates_total = int(_dt.dropna().nunique())
        n_periods_total = len(sorted_months)
        is_snapshot = (n_dates_total == n_periods_total)
        if period_unit == "week":
            date_cadence = "weekly_snapshot" if is_snapshot else "weekly"
        else:
            date_cadence = "monthly_snapshot" if is_snapshot else "intra_month"
        temporal_meta: Dict[str, object] = {
            "cadence": date_cadence,
            "period_unit": period_unit,
            "period_freq": period_freq,
            "date_col": date_col,
            "n_periods_total": n_periods_total,
            "n_distinct_dates": n_dates_total,
            "first_period": str(sorted_months[0]) if sorted_months else None,
            "last_period": str(sorted_months[-1]) if sorted_months else None,
        }
        if n_periods_total >= 2:
            runs_at = period_unit.upper()
            log("Temporal cadence",
                f"{date_cadence} | {n_periods_total} {period_unit}s "
                f"[{temporal_meta['first_period']}..{temporal_meta['last_period']}] "
                f"| {n_dates_total} distinct dates"
                + (f" — model runs {runs_at}LY; all temporal splits snap to whole {period_unit}s"
                   if is_snapshot else ""))

        # B2 fix: need >= 2 distinct periods to create a meaningful OOT set
        if len(sorted_months) < 2:
            log("WARN",
                f"Only {len(sorted_months)} distinct {period_unit}(s) in '{date_col}' — "
                "cannot create OOT split, falling back to simple split")
            # fall through to simple split below
            ym = None
        else:
            # Step 1: find minimum trailing months to reach OOT_MIN_RATIO (start from 1)
            n_oot = 1
            while n_oot < len(sorted_months):
                if ym.isin(sorted_months[-n_oot:]).sum() / valid_rows >= Config.OOT_MIN_RATIO:
                    break
                n_oot += 1

            # Step 2: floor — always include at least OOT_INIT_MONTHS
            n_oot = max(n_oot, Config.OOT_INIT_MONTHS)

            # Clamp to leave at least 1 month for pool
            n_oot = min(n_oot, len(sorted_months) - 1)

            # Step 3: ceiling — shrink if OOT exceeds OOT_MAX_RATIO
            oot_ratio = ym.isin(sorted_months[-n_oot:]).sum() / valid_rows
            if oot_ratio > Config.OOT_MAX_RATIO:
                while n_oot > 1:
                    candidate_ratio = ym.isin(sorted_months[-(n_oot - 1):]).sum() / valid_rows
                    if candidate_ratio <= Config.OOT_MAX_RATIO:
                        n_oot -= 1
                        break
                    n_oot -= 1
                oot_ratio = ym.isin(sorted_months[-n_oot:]).sum() / valid_rows
                log("WARN",
                    f"OOT exceeded cap {Config.OOT_MAX_RATIO:.0%} — "
                    f"reduced to {n_oot} {period_unit}(s) = {oot_ratio:.1%}. "
                    f"Data may have coarse {period_unit} granularity or heavy recency bias.")

            mask_oot  = ym.isin(set(sorted_months[-n_oot:])).values
            mask_pool = ~mask_oot
            pool_size = int(mask_pool.sum())

            # B2 fix: guard against empty pool after OOT split
            if pool_size == 0:
                log("WARN", "Pool is empty after OOT split — falling back to simple split")
                ym = None
            else:
                # Materialise oot_df first (smaller subset)
                oot_df = df[mask_oot].reset_index(drop=True)

                # Valid-temporal: pool rows closest to the OOT boundary.
                # Selection is snapped to WHOLE PERIOD boundaries (month or
                # week), not a raw row-count slice. Without snapping, snapshot
                # data (all rows in a period share the same date, e.g.
                # 2023-01-31 or a Friday) would have valid_temporal and train
                # contain rows from the SAME snapshot period — no real temporal
                # separation. Whole-period snapping is safe for any cadence: it
                # picks the minimum trailing pool periods whose combined row
                # count covers n_temporal, keeping period boundaries intact.
                valid_size = max(1, int(round(pool_size * Config.TRAIN_TEST_SPLIT_SIZE)))
                n_temporal = max(1, int(round(valid_size * Config.VALID_TEMPORAL_RATIO)))

                pool_ym_s   = ym[mask_pool]          # Series: pool rows → Period
                pool_months = sorted(pool_ym_s.dropna().unique())

                # Cadence already detected on the full series above. For a
                # snapshot cadence, snapping valid_temporal to whole periods is
                # what prevents train + valid_temporal sharing the same snapshot
                # period (within-period leakage).
                if date_cadence in ("monthly_snapshot", "weekly_snapshot"):
                    log("Split temporal",
                        f"{date_cadence} — snapping valid_temporal to "
                        f"whole {period_unit}s to avoid within-{period_unit} leakage")

                n_val_months = 1
                while n_val_months < len(pool_months):
                    covered = int(pool_ym_s.isin(pool_months[-n_val_months:]).sum())
                    if covered >= n_temporal:
                        break
                    n_val_months += 1
                # Always leave at least 1 pool period for train
                n_val_months = min(n_val_months, max(1, len(pool_months) - 1))

                temporal_orig_idx = set(
                    pool_ym_s[pool_ym_s.isin(set(pool_months[-n_val_months:]))].index
                )
                log("Split temporal",
                    f"valid_temporal = {n_val_months} whole {period_unit}(s) | "
                    f"rows={len(temporal_orig_idx)} | target_rows={n_temporal}")
                del _dt, ym  # free temporary Series before creating large DataFrames

                mask_temp_orig = pd.Series(False, index=df.index)
                mask_temp_orig[list(temporal_orig_idx)] = True

                # Materialise each split from the original df using boolean masks
                valid_temp_df = df[mask_temp_orig.values].reset_index(drop=True)
                mask_remain   = mask_pool & ~mask_temp_orig.values
                remain_df     = df[mask_remain].reset_index(drop=True)

                # Valid-random: stratified from remaining pool (B3 fix: fallback)
                n_rand     = valid_size - len(valid_temp_df)
                rand_ratio = n_rand / len(remain_df) if len(remain_df) > 0 else 0.0
                if 0 < rand_ratio < 1:
                    train_df, valid_rand_df = stratified_split(remain_df, target, rand_ratio, rs)
                    train_df      = train_df.reset_index(drop=True)
                    valid_rand_df = valid_rand_df.reset_index(drop=True)
                else:
                    train_df      = remain_df
                    valid_rand_df = empty.copy()

                valid_df = pd.concat([valid_temp_df, valid_rand_df], ignore_index=True)

                # Degenerate-split warning — usually means date_col is a
                # mislabelled feature (e.g. days_birth coerced as
                # nanoseconds-since-epoch). The downstream eval_set / CV
                # paths will fall back to `valid` or skip ES entirely,
                # but the user almost certainly wants to know.
                target_in = lambda d: target in d.columns and len(d) > 0
                bad_temp = (target_in(valid_temp_df)
                            and pd.to_numeric(valid_temp_df[target], errors="coerce").nunique() < 2)
                bad_oot  = (target_in(oot_df)
                            and pd.to_numeric(oot_df[target], errors="coerce").nunique() < 2)
                if bad_temp or bad_oot or len(valid_temp_df) < 10 or len(oot_df) < 10:
                    log("SPLIT WARN",
                        f"Degenerate temporal split — valid_temp={len(valid_temp_df)} rows "
                        f"(bad_classes={bad_temp}), oot={len(oot_df)} rows "
                        f"(bad_classes={bad_oot}). date_col='{date_col}' may not be a "
                        "real snapshot column; downstream will fall back to `valid` for ES.")

                # Record how many whole periods each temporal split spans so
                # the report states the cadence (monthly/weekly) explicitly.
                temporal_meta.update({
                    "oot_periods": int(n_oot),
                    "valid_temporal_periods": int(n_val_months),
                    "train_periods": int(len(pool_months) - n_val_months),
                })

                log("Split (OOT)",
                    f"oot_{period_unit}s={n_oot} | oot_ratio={len(oot_df)/total:.1%} | "
                    f"train={len(train_df)} | valid_temp={len(valid_temp_df)} | "
                    f"valid_rand={len(valid_rand_df)} | oot={len(oot_df)}")
                return SplitResult(
                    train=train_df, valid=valid_df, oot=oot_df, test=empty.copy(),
                    valid_temporal=valid_temp_df, valid_random=valid_rand_df,
                    temporal_meta=temporal_meta, period_freq=period_freq,
                    period_unit=period_unit, date_cadence=date_cadence,
                )

    # Fallback: no date col or fell through from OOT path. No temporal OOT was
    # produced, so the splits are NOT month-based — mark cadence accordingly
    # (overrides any cadence detected before the fall-through).
    # 3-way stratified split: 60% train / 20% valid / 20% test (no temporal OOT)
    train_valid_df, test_df = stratified_split(df, target, 0.20, rs)
    train_df, valid_df      = stratified_split(train_valid_df, target, 0.25, rs)
    log("Split (simple)",
        f"train={len(train_df)} | valid={len(valid_df)} | test={len(test_df)} | no OOT")
    return SplitResult(
        train=train_df.reset_index(drop=True),
        valid=valid_df.reset_index(drop=True),
        oot=empty.copy(),
        test=test_df.reset_index(drop=True),
        valid_temporal=empty.copy(),
        valid_random=empty.copy(),
        temporal_meta={"cadence": "non_temporal"},
        period_freq="M",
        period_unit="month",
        date_cadence="non_temporal",
    )

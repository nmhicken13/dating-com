"""
Scoring, ML training, and pipeline helpers (extracted for clarity).
Uses the same string tokens as app.py for outing types and physical milestones.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from supabase import Client

import supabase_db as sb

# --- Mirror app.py canonical values ---
OUTING_TYPE_DATE = "date"
OUTING_TYPE_CASUAL = "casual"
PHYSICAL_NONE = "None"
PHYSICAL_HELD = "Held Hands / Cuddled"
PHYSICAL_KISSED = "Kissed"
ACTIVE_PIPELINE_TD = ("Talking", "Dating")
MEMORY_HALF_LIFE_DAYS = 30.0
MOMENTUM_HALF_LIFE_DAYS = 14.0

ML_FEATURE_NAMES: tuple[str, ...] = (
    "avg_rating",
    "has_kiss",
    "has_hands",
    "latest_is_double",
    "latest_is_group",
    "latest_duration",
    "missing_ty_text",
    "rating_delta",
    "dates_so_far",
    "date_ratio",
)

PIPELINE_EMPTY_SLOT_LABEL = (
    "⬜ Empty slot — add a date to promote a Lead"
)


def _normalize_outing_type(raw: str | None) -> str:
    if raw == OUTING_TYPE_CASUAL:
        return OUTING_TYPE_CASUAL
    return OUTING_TYPE_DATE


def _normalize_company_type(raw: str | None) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "One-on-one"
    s = str(raw).strip()
    if s == "one_on_one":
        return "One-on-one"
    if s == "double":
        return "Double Date"
    if s == "group":
        return "Group Date"
    if s in ("One-on-one", "Double Date", "Group Date"):
        return s
    return "One-on-one"


def _company_type_to_sql(ui_label: str) -> str:
    if ui_label == "Double Date":
        return "double"
    if ui_label == "Group Date":
        return "group"
    return "one_on_one"


def normalize_physical_milestone(raw: str | None) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return PHYSICAL_NONE
    s = str(raw).strip()
    if not s:
        return PHYSICAL_NONE
    if s in (PHYSICAL_NONE, PHYSICAL_HELD, PHYSICAL_KISSED):
        return s
    low = s.lower()
    if "kiss" in low:
        return PHYSICAL_KISSED
    if any(k in low for k in ("cuddle", "held", "hand", "hug", "physical")):
        return PHYSICAL_HELD
    return PHYSICAL_NONE


def merge_pipeline_stats(people_slice: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    if people_slice.empty:
        return people_slice
    m = people_slice.merge(stats_df, left_on="id", right_on="person_id", how="left")
    m["n_romantic_dates"] = (
        pd.to_numeric(m["n_romantic_dates"], errors="coerce").fillna(0).astype(int)
    )
    m["total_spent"] = pd.to_numeric(m["total_spent"], errors="coerce").fillna(0.0)
    m["last_occurred"] = pd.to_datetime(m["last_occurred"], errors="coerce")
    return m.drop(columns=["person_id"], errors="ignore")


def sort_pipeline_people(df: pd.DataFrame, *, by_date_count: bool) -> pd.DataFrame:
    if df.empty or "n_romantic_dates" not in df.columns:
        return df
    out = df.copy()
    out["_sn"] = out["name"].astype(str).str.lower()
    if by_date_count:
        out = out.sort_values(["n_romantic_dates", "_sn"], ascending=[False, True])
    else:
        out = out.sort_values(["last_occurred", "_sn"], ascending=[False, True], na_position="last")
    return out.drop(columns=["_sn"], errors="ignore")


def filter_completed_interactions(dates_df: pd.DataFrame) -> pd.DataFrame:
    if dates_df.empty:
        return dates_df
    d = dates_df.copy()
    if "is_planned" in d.columns:
        d = d[d["is_planned"].fillna(0).astype(int) == 0]
    return d


def filter_dates_for_potential(dates_df: pd.DataFrame) -> pd.DataFrame:
    if dates_df.empty:
        return dates_df
    d = filter_completed_interactions(dates_df)
    if d.empty:
        return d
    if "outing_type" in d.columns:
        d = d[d["outing_type"].map(_normalize_outing_type) == OUTING_TYPE_DATE]
    return d


def compute_ml_date_ratio(
    prefix_romantic: pd.DataFrame, all_completed_interactions: pd.DataFrame
) -> float:
    if prefix_romantic.empty:
        return 0.0
    pid = int(prefix_romantic.iloc[0]["person_id"])
    ac = all_completed_interactions
    if ac.empty or "person_id" not in ac.columns:
        return 1.0 if len(prefix_romantic) > 0 else 0.0
    sub = ac[ac["person_id"].astype(int) == pid]
    n_rom = len(filter_dates_for_potential(prefix_romantic))
    n_all = len(sub)
    if n_all <= 0:
        return 0.0
    return float(n_rom) / float(n_all)


def _rating_numeric(val: object) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 5.5
    try:
        return float(val)
    except (TypeError, ValueError):
        return 5.5


def extract_ml_feature_vector(
    prefix: pd.DataFrame,
    *,
    all_completed_interactions: pd.DataFrame | None = None,
) -> list[float]:
    d = filter_dates_for_potential(prefix)
    if d.empty:
        return [0.0] * len(ML_FEATURE_NAMES)
    d = d.sort_values(
        by=["occurred_on", "id"] if "id" in d.columns else ["occurred_on"],
        kind="mergesort",
    ).reset_index(drop=True)
    n = len(d)
    ratings = pd.to_numeric(d["rating"], errors="coerce")
    avg_rating = float(ratings.mean()) if ratings.notna().any() else 5.5
    last = d.iloc[-1]
    pe_last = normalize_physical_milestone(last.get("physical_escalation"))
    has_kiss = 1.0 if pe_last == PHYSICAL_KISSED else 0.0
    has_hands = 1.0 if pe_last in (PHYSICAL_HELD, PHYSICAL_KISSED) else 0.0
    co = _company_type_to_sql(_normalize_company_type(last.get("company_type")))
    latest_is_double = 1.0 if co == "double" else 0.0
    latest_is_group = 1.0 if co == "group" else 0.0
    dh = last.get("duration_hours")
    try:
        latest_duration = (
            0.0
            if dh is None or (isinstance(dh, float) and pd.isna(dh))
            else float(dh)
        )
    except (TypeError, ValueError):
        latest_duration = 0.0
    missing_ty_text = 0.0
    for i in range(min(3, n)):
        ty = d.iloc[i].get("thank_you")
        if ty is None or (isinstance(ty, float) and pd.isna(ty)):
            missing_ty_text = 1.0
            break
    if n >= 2:
        r_latest = _rating_numeric(d.iloc[-1].get("rating"))
        r_prev = _rating_numeric(d.iloc[-2].get("rating"))
        rating_delta = r_latest - r_prev
    else:
        rating_delta = 0.0
    if all_completed_interactions is not None and not all_completed_interactions.empty:
        date_ratio = compute_ml_date_ratio(d, all_completed_interactions)
    else:
        date_ratio = 1.0 if n > 0 else 0.0
    return [
        float(avg_rating),
        has_kiss,
        has_hands,
        latest_is_double,
        latest_is_group,
        latest_duration,
        missing_ty_text,
        float(rating_delta),
        float(n),
        float(date_ratio),
    ]


def potential_score_breakdown(
    dates_df: pd.DataFrame,
    *,
    reference_date: date | None = None,
    peak_potential_only: bool = False,
) -> dict[str, Any]:
    d0 = dates_df.copy()
    if reference_date is not None and "occurred_on" in d0.columns:
        od = pd.to_datetime(d0["occurred_on"], errors="coerce").dt.date
        d0 = d0[od.notna() & (od <= reference_date)]
    d = filter_dates_for_potential(d0)
    n = len(d)
    if n == 0:
        return {
            "score": 42,
            "n_dates": 0,
            "base_score": 40.0,
            "max_potential": 42.0,
            "momentum_factor": 1.0,
        }
    last = d.iloc[-1]
    last_day = pd.to_datetime(last["occurred_on"], errors="coerce")
    ref: date = reference_date or (
        last_day.date() if pd.notna(last_day) else date.today()
    )
    ratings: list[float] = []
    weights: list[float] = []
    for _, row in d.iterrows():
        od = pd.to_datetime(row["occurred_on"], errors="coerce")
        if pd.isna(od):
            continue
        days_ago = (pd.Timestamp(ref) - od).days
        w = float(np.exp(-np.log(2) * max(0, days_ago) / MEMORY_HALF_LIFE_DAYS))
        r = row.get("rating")
        if r is not None and not (isinstance(r, float) and pd.isna(r)):
            try:
                ratings.append(float(r))
                weights.append(w)
            except (TypeError, ValueError):
                pass
    avg_r = float(np.average(ratings, weights=weights)) if ratings else 5.5
    base = 25.0 + (avg_r / 10.0) * 40.0
    pe = normalize_physical_milestone(last.get("physical_escalation"))
    if pe == PHYSICAL_KISSED:
        base += 18.0
    elif pe == PHYSICAL_HELD:
        base += 10.0
    ty = last.get("thank_you")
    try:
        if ty is not None and int(ty) == 1:
            base += 5.0
    except (TypeError, ValueError):
        pass
    dur = last.get("duration_hours")
    if dur is not None and not (isinstance(dur, float) and pd.isna(dur)):
        try:
            base += min(10.0, float(dur) * 1.5)
        except (TypeError, ValueError):
            pass
    max_potential = min(100.0, base)
    days_since = int((pd.Timestamp(ref) - last_day).days) if pd.notna(last_day) else 0
    mom = float(np.exp(-np.log(2) * max(0, days_since) / MOMENTUM_HALF_LIFE_DAYS))
    if peak_potential_only:
        score_f = max_potential
    else:
        score_f = min(100.0, max_potential * mom)
    return {
        "score": int(round(score_f)),
        "n_dates": n,
        "base_score": base,
        "max_potential": max_potential,
        "momentum_factor": mom,
    }


def calculate_potential(person_row: pd.Series, dates_df: pd.DataFrame) -> int:
    _ = person_row
    return int(potential_score_breakdown(dates_df)["score"])


def _heuristic_potential_for_ml_prefix(prefix: pd.DataFrame) -> int:
    return int(potential_score_breakdown(prefix)["score"])


def _reliability_calibration_save_dict(
    y_arr: np.ndarray, proba: np.ndarray, *, n_bins: int, strategy: str
) -> dict[str, object] | None:
    try:
        prob_true, mean_pred = calibration_curve(
            y_arr, proba, n_bins=n_bins, strategy=strategy
        )
        return {
            "n_bins": n_bins,
            "strategy": strategy,
            "prob_true": [float(x) for x in prob_true],
            "mean_predicted_value": [float(x) for x in mean_pred],
        }
    except ValueError:
        return None


def calculate_ml_probability(
    client: Client, user_id: str, dates_df: pd.DataFrame
) -> int | None:
    try:
        cfg = sb.fetch_ml_config_dict(client, user_id)
    except Exception:
        return None
    if cfg is None:
        return None
    try:
        coef = np.asarray(cfg["coef"], dtype=float).ravel()
        intercept = float(cfg["intercept"])
        mean = np.asarray(cfg["mean"], dtype=float).ravel()
        scale = np.asarray(cfg["scale"], dtype=float).ravel()
    except (KeyError, TypeError, ValueError):
        return None
    if coef.size != len(ML_FEATURE_NAMES) or mean.size != coef.size or scale.size != coef.size:
        return None
    all_completed = filter_completed_interactions(dates_df)
    d = filter_dates_for_potential(dates_df)
    if d.empty:
        return None
    x = np.array(
        extract_ml_feature_vector(d, all_completed_interactions=all_completed),
        dtype=float,
    )
    scale_safe = np.where(scale == 0, 1.0, scale)
    x_s = (x - mean) / scale_safe
    z = float(intercept + np.dot(coef, x_s))
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
    return int(max(0, min(100, round(p * 100.0))))


def retrain_ml_model(client: Client, user_id: str, *, n_bins: int, strategy: str) -> tuple[bool, str]:
    people = sb.load_people_df(client, user_id)
    if people.empty:
        return False, "No people in database."
    dates_raw = sb.load_all_dates_for_user(client, user_id)
    if dates_raw.empty:
        return False, "No dates logged yet."
    status_by_pid = {int(r["id"]): str(r["status"]) for _, r in people.iterrows()}

    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    h_rows: list[int] = []

    for pid, g in dates_raw.groupby("person_id", sort=False):
        pid_i = int(pid)
        st = status_by_pid.get(pid_i, "Archived")
        g_completed = filter_completed_interactions(g)
        g2 = filter_dates_for_potential(g)
        if g2.empty:
            continue
        g2 = g2.reset_index(drop=True)
        n = len(g2)
        for i in range(n):
            prefix = g2.iloc[: i + 1].copy()
            last = prefix.iloc[-1]
            uw = last.get("user_wanted_next_date")
            if uw is not None and not (isinstance(uw, float) and pd.isna(uw)):
                if int(uw) != 1:
                    continue
            feats = extract_ml_feature_vector(
                prefix, all_completed_interactions=g_completed
            )
            X_rows.append(feats)
            h_rows.append(_heuristic_potential_for_ml_prefix(prefix))
            if i < n - 1:
                y_rows.append(1)
            else:
                y_rows.append(1 if st in ACTIVE_PIPELINE_TD else 0)

    if len(X_rows) < 8:
        return (
            False,
            "Need more step-wise examples where you **wanted another date** (several dated people) to train the model.",
        )
    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=int)
    if len(np.unique(y)) < 2:
        return (
            False,
            "Labels are all the same; add mix of Archived and Active people with dates.",
        )

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(Xs, y)

    y_proba = model.predict_proba(Xs)[:, 1]
    y_hat = (y_proba >= 0.5).astype(int)
    acc = float(accuracy_score(y, y_hat))
    prec = float(precision_score(y, y_hat, zero_division=0))
    try:
        auc_val = float(roc_auc_score(y, y_proba))
    except ValueError:
        auc_val = None

    cal_payload = _reliability_calibration_save_dict(
        y, y_proba, n_bins=n_bins, strategy=strategy
    )

    h_arr = np.asarray(h_rows, dtype=float)
    h_proba = np.clip(h_arr / 100.0, 0.0, 1.0)
    h_hat = (h_proba >= 0.5).astype(int)
    h_acc = float(accuracy_score(y, h_hat))
    h_prec = float(precision_score(y, h_hat, zero_division=0))
    try:
        h_auc = float(roc_auc_score(y, h_proba))
    except ValueError:
        h_auc = None

    heur_cal_payload = _reliability_calibration_save_dict(
        y, h_proba, n_bins=n_bins, strategy=strategy
    )

    coef = model.coef_[0].astype(float)
    intercept = float(model.intercept_[0])
    mean = scaler.mean_.astype(float)
    scale = scaler.scale_.astype(float)

    sb.upsert_ml_config(
        client,
        user_id,
        coef=coef.tolist(),
        mean=mean.tolist(),
        scale=scale.tolist(),
        intercept=intercept,
        feature_names=list(ML_FEATURE_NAMES),
        metrics={"accuracy": acc, "precision": prec, "roc_auc": auc_val},
        heuristic_metrics={"accuracy": h_acc, "precision": h_prec, "roc_auc": h_auc},
        calibration=cal_payload,
        heuristic_calibration=heur_cal_payload,
    )

    return True, f"Trained on {len(y_rows)} history steps."


def get_pipeline_top5_slot_person_ids(
    people_df: pd.DataFrame,
    scores_by_id: dict[int, int],
) -> list[int | None]:
    active = people_df[people_df["status"].isin(ACTIVE_PIPELINE_TD)].copy()
    if active.empty:
        return [None] * 5
    active_ids = set(active["id"].astype(int).tolist())
    active["roster_slot"] = pd.to_numeric(active["roster_slot"], errors="coerce")
    by_slot: dict[int, int] = {}
    for _, r in active.iterrows():
        rs = r.get("roster_slot")
        if pd.notna(rs):
            try:
                s = int(rs)
                if 1 <= s <= 5:
                    pid = int(r["id"])
                    if pid in active_ids:
                        by_slot[s] = pid
            except (TypeError, ValueError):
                pass
    slots: list[int | None] = [None] * 5
    used: set[int] = set()
    for s in range(1, 6):
        pid = by_slot.get(s)
        if pid is not None and pid in active_ids:
            slots[s - 1] = pid
            used.add(pid)
    rest = [pid for pid in active["id"].astype(int).tolist() if pid not in used]
    rest.sort(key=lambda p: -scores_by_id.get(p, 0))
    ri = 0
    for i in range(5):
        if slots[i] is None and ri < len(rest):
            slots[i] = rest[ri]
            ri += 1
    return slots


def apply_pipeline_slots_from_person_ids(
    client: Client, user_id: str, slots: list[int | None]
) -> None:
    sb.apply_pipeline_slots_from_person_ids(client, user_id, slots)


def swap_pipeline_ranks(
    client: Client,
    user_id: str,
    people_df: pd.DataFrame,
    scores_by_id: dict[int, int],
    rank_a: int,
    rank_b: int,
) -> None:
    if rank_a < 1 or rank_a > 5 or rank_b < 1 or rank_b > 5:
        return
    slots = get_pipeline_top5_slot_person_ids(people_df, scores_by_id)
    ia, ib = rank_a - 1, rank_b - 1
    slots[ia], slots[ib] = slots[ib], slots[ia]
    apply_pipeline_slots_from_person_ids(client, user_id, slots)


def ensure_default_pipeline_slots(
    client: Client,
    user_id: str,
    people_df: pd.DataFrame,
    scores_by_id: dict[int, int],
) -> None:
    sb.ensure_default_pipeline_slots(client, user_id, people_df, scores_by_id)


def _occurred_on_to_date(val: object) -> date | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return pd.to_datetime(val).date()
    except (TypeError, ValueError):
        return None


def augment_export_df_with_snapshot_potential(exp: pd.DataFrame) -> pd.DataFrame:
    if exp.empty:
        return exp
    out = exp.copy()
    buffers: dict[int, list[pd.Series]] = {}
    scores: list[int] = []
    for _, row in exp.iterrows():
        pid = int(row["person_id"])
        if pid not in buffers:
            buffers[pid] = []
        buffers[pid].append(row)
        cum = pd.DataFrame(buffers[pid])
        cum = cum.rename(columns={"date_id": "id"})
        as_of = _occurred_on_to_date(row["occurred_on"])
        if as_of is None:
            scores.append(0)
        else:
            bd = potential_score_breakdown(
                cum, reference_date=as_of, peak_potential_only=True
            )
            scores.append(int(bd["score"]))
    out["snapshot_potential"] = scores
    return out

"""
Supabase/PostgREST data access — every query scopes by user_id.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
from supabase import Client


def _rows(resp: Any) -> list:
    """PostgREST execute() should return an object with .data; guard None / missing."""
    if resp is None:
        return []
    data = getattr(resp, "data", None)
    if data is None:
        return []
    return data if isinstance(data, list) else [data]


def _company_type_for_postgres(raw: str | None) -> str:
    """Map UI labels or legacy strings to CHECK (one_on_one | double | group)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "one_on_one"
    s = str(raw).strip()
    if s in ("one_on_one", "double", "group"):
        return s
    if s in ("One-on-one", "Double Date", "Group Date"):
        if s == "Double Date":
            return "double"
        if s == "Group Date":
            return "group"
        return "one_on_one"
    low = s.lower().replace("_", " ").replace("-", " ")
    if low in ("one on one", "oneonone", "one on one date"):
        return "one_on_one"
    if low in ("double", "double date", "doubledate"):
        return "double"
    if low in ("group", "group date", "groupdate"):
        return "group"
    return "one_on_one"


def _outing_type_for_postgres(raw: str | None) -> str:
    """CHECK (date | casual)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "date"
    s = str(raw).strip()
    if s == "casual":
        return "casual"
    return "date"


def load_people_df(client: Client, user_id: str) -> pd.DataFrame:
    r = client.table("people").select("*").eq("user_id", user_id).order("name").execute()
    rows = _rows(r)
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "user_id",
                "name",
                "status",
                "initial_met_via",
                "profile_image",
                "roster_slot",
                "created_at",
            ]
        )
    return pd.DataFrame(rows)


def add_person(
    client: Client,
    user_id: str,
    *,
    name: str,
    status: str,
    initial_met_via: str | None,
    profile_image: str | None = None,
) -> int:
    row = {
        "user_id": user_id,
        "name": name.strip(),
        "status": status,
        "initial_met_via": initial_met_via,
        "profile_image": (profile_image or "").strip() or None,
    }
    # postgrest-py: insert() returns QueryRequestBuilder (no .select chain); representation is default.
    r = client.table("people").insert(row).execute()
    got = _rows(r)
    if not got or got[0].get("id") is None:
        raise RuntimeError("Failed to insert person")
    return int(got[0]["id"])


def update_person_status(
    client: Client, user_id: str, person_id: int, status: str
) -> None:
    client.table("people").update({"status": status}).eq("user_id", user_id).eq(
        "id", person_id
    ).execute()


def update_person(
    client: Client,
    user_id: str,
    person_id: int,
    *,
    name: str,
    status: str,
    initial_met_via: str | None,
    profile_image: str | None,
) -> None:
    client.table("people").update(
        {
            "name": name.strip(),
            "status": status,
            "initial_met_via": initial_met_via,
            "profile_image": (profile_image or "").strip() or None,
        }
    ).eq("user_id", user_id).eq("id", person_id).execute()


def delete_person(client: Client, user_id: str, person_id: int) -> None:
    client.table("people").delete().eq("user_id", user_id).eq("id", person_id).execute()


def add_date_event(
    client: Client,
    user_id: str,
    *,
    person_id: int,
    occurred_on: date,
    activity: str | None,
    notes: str | None,
    rating: int | None,
    physical_escalation: str | None,
    outing_type: str,
    company_type: str,
    thank_you: int | None,
    cost: float | None,
    is_planned: bool | int = False,
    scheduled_at: datetime | None = None,
    initiator: str | None = None,
    duration_hours: float | None = None,
    user_wanted_next_date: int = 1,
) -> int:
    ip = 1 if is_planned else 0
    sched_sql: str | None = None
    if scheduled_at is not None:
        sched_sql = scheduled_at.isoformat(timespec="minutes")
    uw_sql = 1 if int(user_wanted_next_date) else 0
    co_sql = _company_type_for_postgres(company_type)
    ot_sql = _outing_type_for_postgres(outing_type)
    row: dict[str, Any] = {
        "user_id": user_id,
        "person_id": person_id,
        "occurred_on": occurred_on.isoformat(),
        "activity": (activity or "").strip() or None,
        "notes": (notes or "").strip() or None,
        "rating": rating,
        "physical_escalation": physical_escalation,
        "outing_type": ot_sql,
        "company_type": co_sql,
        "thank_you": thank_you,
        "cost": cost,
        "is_planned": ip,
        "scheduled_date": sched_sql,
        "initiator": initiator,
        "duration_hours": duration_hours,
        "user_wanted_next_date": uw_sql,
    }
    r = client.table("dates").insert(row).execute()
    got = _rows(r)
    if not got or got[0].get("id") is None:
        raise RuntimeError("Failed to insert date")
    return int(got[0]["id"])


def update_date_event(
    client: Client,
    user_id: str,
    date_id: int,
    *,
    person_id: int,
    occurred_on: date,
    activity: str | None,
    notes: str | None,
    rating: int | None,
    physical_escalation: str | None,
    outing_type: str,
    company_type: str,
    thank_you: int | None,
    cost: float | None,
    is_planned: bool | int = False,
    scheduled_at: datetime | None = None,
    initiator: str | None = None,
    duration_hours: float | None = None,
    user_wanted_next_date: int = 1,
) -> None:
    ip = 1 if is_planned else 0
    sched_sql: str | None = None
    if scheduled_at is not None:
        sched_sql = scheduled_at.isoformat(timespec="minutes")
    uw_sql = 1 if int(user_wanted_next_date) else 0
    co_sql = _company_type_for_postgres(company_type)
    ot_sql = _outing_type_for_postgres(outing_type)
    client.table("dates").update(
        {
            "person_id": person_id,
            "occurred_on": occurred_on.isoformat(),
            "activity": (activity or "").strip() or None,
            "notes": (notes or "").strip() or None,
            "rating": rating,
            "physical_escalation": physical_escalation,
            "outing_type": ot_sql,
            "company_type": co_sql,
            "thank_you": thank_you,
            "cost": cost,
            "is_planned": ip,
            "scheduled_date": sched_sql,
            "initiator": initiator,
            "duration_hours": duration_hours,
            "user_wanted_next_date": uw_sql,
        }
    ).eq("user_id", user_id).eq("id", date_id).execute()


def delete_date_event(client: Client, user_id: str, date_id: int) -> None:
    client.table("dates").delete().eq("user_id", user_id).eq("id", date_id).execute()


def _dates_df_from_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if not df.empty and "occurred_on" in df.columns:
        df["occurred_on"] = (
            pd.to_datetime(df["occurred_on"], errors="coerce").dt.date.astype(str)
        )
    return df


def load_dates_for_person(
    client: Client, user_id: str, person_id: int
) -> pd.DataFrame:
    r = (
        client.table("dates")
        .select(
            "id, person_id, occurred_on, activity, notes, rating, physical_escalation, "
            "outing_type, company_type, thank_you, cost, is_planned, scheduled_date, "
            "initiator, duration_hours, user_wanted_next_date, created_at"
        )
        .eq("user_id", user_id)
        .eq("person_id", person_id)
        .execute()
    )
    df = _dates_df_from_rows(_rows(r))
    if df.empty:
        return df
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.sort_values(["occurred_on", "id"], ascending=[False, False])
    return df


def load_all_dates_for_user(client: Client, user_id: str) -> pd.DataFrame:
    r = (
        client.table("dates")
        .select(
            "id, person_id, occurred_on, rating, physical_escalation, company_type, "
            "thank_you, duration_hours, is_planned, outing_type, user_wanted_next_date"
        )
        .eq("user_id", user_id)
        .execute()
    )
    rows = _rows(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["person_id", "occurred_on", "id"], kind="mergesort")
    return df


def count_completed_outings_for_person(
    client: Client, user_id: str, person_id: int
) -> int:
    r = (
        client.table("dates")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("person_id", person_id)
        .eq("is_planned", 0)
        .limit(0)
        .execute()
    )
    return int(getattr(r, "count", None) or 0) if r is not None else 0


def load_person_pipeline_stats(client: Client, user_id: str) -> pd.DataFrame:
    r = (
        client.table("dates")
        .select("person_id, is_planned, outing_type, occurred_on, cost")
        .eq("user_id", user_id)
        .execute()
    )
    df = pd.DataFrame(_rows(r))
    if df.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "n_romantic_dates",
                "last_occurred",
                "total_spent",
            ]
        )
    df["is_planned"] = pd.to_numeric(df["is_planned"], errors="coerce").fillna(0).astype(int)
    done = df[df["is_planned"] == 0]
    if done.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "n_romantic_dates",
                "last_occurred",
                "total_spent",
            ]
        )
    romantic = done[done["outing_type"].astype(str) == "date"]
    n_rom = romantic.groupby("person_id").size().rename("n_romantic_dates")
    last = done.groupby("person_id")["occurred_on"].max().rename("last_occurred")
    spent = (
        done.assign(cost=pd.to_numeric(done["cost"], errors="coerce").fillna(0.0))
        .groupby("person_id")["cost"]
        .sum()
        .rename("total_spent")
    )
    out = pd.concat([n_rom, last, spent], axis=1).reset_index()
    out["n_romantic_dates"] = (
        pd.to_numeric(out["n_romantic_dates"], errors="coerce").fillna(0).astype(int)
    )
    out["total_spent"] = pd.to_numeric(out["total_spent"], errors="coerce").fillna(0.0)
    return out


def total_spent_for_person(client: Client, user_id: str, person_id: int) -> float:
    r = (
        client.table("dates")
        .select("cost")
        .eq("user_id", user_id)
        .eq("person_id", person_id)
        .eq("is_planned", 0)
        .execute()
    )
    rows = _rows(r)
    if not rows:
        return 0.0
    s = 0.0
    for row in rows:
        c = row.get("cost")
        if c is not None:
            try:
                s += float(c)
            except (TypeError, ValueError):
                pass
    return s


def avg_cost_per_outing(client: Client, user_id: str) -> float | None:
    r = (
        client.table("dates")
        .select("cost")
        .eq("user_id", user_id)
        .eq("is_planned", 0)
        .execute()
    )
    rows = _rows(r)
    if not rows:
        return None
    costs: list[float] = []
    for row in rows:
        c = row.get("cost")
        if c is not None:
            try:
                costs.append(float(c))
            except (TypeError, ValueError):
                costs.append(0.0)
        else:
            costs.append(0.0)
    if not costs:
        return None
    return sum(costs) / len(costs)


def load_dates_for_trends(client: Client, user_id: str) -> pd.DataFrame:
    r = (
        client.table("dates")
        .select(
            "id, person_id, occurred_on, cost, rating, outing_type, physical_escalation, "
            "initiator, people(initial_met_via)"
        )
        .eq("user_id", user_id)
        .eq("is_planned", 0)
        .order("occurred_on")
        .order("id")
        .execute()
    )
    rows = _rows(r)
    if not rows:
        return pd.DataFrame()
    flat: list[dict] = []
    for row in rows:
        p = row.pop("people", None) or {}
        row["met_via_raw"] = p.get("initial_met_via")
        flat.append(row)
    return pd.DataFrame(flat)


def load_dates_export_df(client: Client, user_id: str) -> pd.DataFrame:
    r = (
        client.table("dates")
        .select(
            "id, person_id, occurred_on, activity, notes, rating, physical_escalation, "
            "outing_type, company_type, thank_you, cost, is_planned, scheduled_date, "
            "initiator, duration_hours, user_wanted_next_date, created_at, "
            "people(name, status, initial_met_via, profile_image, created_at)"
        )
        .eq("user_id", user_id)
        .order("occurred_on")
        .order("id")
        .execute()
    )
    rows = _rows(r)
    if not rows:
        return pd.DataFrame()
    out_rows: list[dict] = []
    for row in rows:
        p = row.pop("people", None) or {}
        out_rows.append(
            {
                "date_id": row.get("id"),
                "person_id": row.get("person_id"),
                "occurred_on": row.get("occurred_on"),
                "activity": row.get("activity"),
                "notes": row.get("notes"),
                "rating": row.get("rating"),
                "physical_escalation": row.get("physical_escalation"),
                "outing_type": row.get("outing_type"),
                "company_type": row.get("company_type"),
                "thank_you": row.get("thank_you"),
                "cost": row.get("cost"),
                "is_planned": row.get("is_planned"),
                "scheduled_date": row.get("scheduled_date"),
                "initiator": row.get("initiator"),
                "Duration (hours)": row.get("duration_hours"),
                "user_wanted_next_date": row.get("user_wanted_next_date"),
                "date_row_created_at": row.get("created_at"),
                "person_name": p.get("name"),
                "person_status": p.get("status"),
                "initially_met_via": p.get("initial_met_via"),
                "person_profile_image": p.get("profile_image"),
                "person_created_at": p.get("created_at"),
            }
        )
    return pd.DataFrame(out_rows)


def load_planned_dates_df(client: Client, user_id: str) -> pd.DataFrame:
    r = (
        client.table("dates")
        .select(
            "id, person_id, scheduled_date, occurred_on, activity, notes, people(name)"
        )
        .eq("user_id", user_id)
        .eq("is_planned", 1)
        .order("scheduled_date")
        .order("occurred_on")
        .order("id")
        .execute()
    )
    rows = _rows(r)
    if not rows:
        return pd.DataFrame()
    flat: list[dict] = []
    for row in rows:
        p = row.pop("people", None) or {}
        row["person_name"] = p.get("name")
        flat.append(row)
    df = pd.DataFrame(flat)
    return df


def load_activity_roi_df(client: Client, user_id: str) -> pd.DataFrame:
    r = (
        client.table("dates")
        .select("activity, cost, rating, physical_escalation")
        .eq("user_id", user_id)
        .eq("is_planned", 0)
        .eq("outing_type", "date")
        .execute()
    )
    df = pd.DataFrame(_rows(r))
    empty_cols = [
        "Activity Name",
        "Number of Times Done",
        "Average Cost",
        "Average Rating",
        "% ≥ Held Hands / Cuddled",
    ]
    if df.empty:
        return pd.DataFrame(columns=empty_cols)
    df = df[df["activity"].notna() & (df["activity"].astype(str).str.strip() != "")]
    if df.empty:
        return pd.DataFrame(columns=empty_cols)
    df["Activity Name"] = df["activity"].astype(str).str.strip()
    milestones = {"Held Hands / Cuddled", "Kissed"}
    rows_out: list[dict] = []
    for act_name, g in df.groupby("Activity Name", sort=False):
        n = len(g)
        if n == 0:
            continue
        hit = (g["physical_escalation"].map(lambda x: str(x or "") in milestones)).sum()
        rows_out.append(
            {
                "Activity Name": act_name,
                "Number of Times Done": n,
                "Average Cost": round(
                    float(pd.to_numeric(g["cost"], errors="coerce").fillna(0).mean()),
                    2,
                ),
                "Average Rating": round(
                    float(pd.to_numeric(g["rating"], errors="coerce").mean()), 2
                ),
                "% ≥ Held Hands / Cuddled": round(100.0 * float(hit) / n, 1),
            }
        )
    out = pd.DataFrame(rows_out)
    if out.empty:
        return pd.DataFrame(columns=empty_cols)
    return out.sort_values("Average Rating", ascending=False)


def kpi_date_counts(client: Client, user_id: str) -> tuple[int, int, int]:
    r = (
        client.table("dates")
        .select("occurred_on")
        .eq("user_id", user_id)
        .eq("is_planned", 0)
        .eq("outing_type", "date")
        .execute()
    )
    rows = _rows(r)
    today = date.today()
    ym = today.strftime("%Y-%m")
    y = str(today.year)
    total = len(rows)
    month = sum(1 for row in rows if str(row.get("occurred_on") or "")[:7] == ym)
    year = sum(
        1 for row in rows if str(row.get("occurred_on") or "").startswith(f"{y}-")
    )
    return total, month, year


def apply_pipeline_slots_from_person_ids(
    client: Client, user_id: str, slots: list[int | None]
) -> None:
    client.table("people").update({"roster_slot": None}).eq("user_id", user_id).in_(
        "status", ["Talking", "Dating"]
    ).execute()
    for pos, pid in enumerate(slots, start=1):
        if pid is None:
            continue
        client.table("people").update({"roster_slot": pos}).eq("user_id", user_id).eq(
            "id", pid
        ).in_("status", ["Talking", "Dating"]).execute()


def count_active_with_roster_slot(client: Client, user_id: str) -> int:
    r = (
        client.table("people")
        .select("id, roster_slot")
        .eq("user_id", user_id)
        .in_("status", ["Talking", "Dating"])
        .execute()
    )
    return sum(1 for row in _rows(r) if row.get("roster_slot") is not None)


def ensure_default_pipeline_slots(
    client: Client,
    user_id: str,
    people_df: pd.DataFrame,
    scores_by_id: dict[int, int],
) -> None:
    if count_active_with_roster_slot(client, user_id) > 0:
        return
    active = people_df[people_df["status"].isin(["Talking", "Dating"])]
    ids = sorted(
        [int(x) for x in active["id"].tolist()],
        key=lambda p: -scores_by_id.get(p, 0),
    )
    for pos, pid in enumerate(ids[:5], start=1):
        client.table("people").update({"roster_slot": pos}).eq("user_id", user_id).eq(
            "id", pid
        ).in_("status", ["Talking", "Dating"]).execute()


def count_roster_assigned_friend_included(client: Client, user_id: str) -> int:
    r = (
        client.table("people")
        .select("id, roster_slot")
        .eq("user_id", user_id)
        .in_("status", ["Talking", "Dating", "Friend"])
        .execute()
    )
    return sum(1 for row in _rows(r) if row.get("roster_slot") is not None)


def list_people_ids_roster_candidates(client: Client, user_id: str) -> list[int]:
    r = (
        client.table("people")
        .select("id")
        .eq("user_id", user_id)
        .in_("status", ["Talking", "Dating", "Friend"])
        .order("name")
        .execute()
    )
    return [int(row["id"]) for row in _rows(r)]


def ensure_default_roster_slots(client: Client, user_id: str) -> None:
    if count_roster_assigned_friend_included(client, user_id) > 0:
        return
    ids = list_people_ids_roster_candidates(client, user_id)
    for i, pid in enumerate(ids[:5], start=1):
        client.table("people").update({"roster_slot": i}).eq("user_id", user_id).eq(
            "id", pid
        ).execute()


def load_completed_analytics_df(client: Client, user_id: str) -> pd.DataFrame:
    r = (
        client.table("dates")
        .select(
            "occurred_on, cost, rating, company_type, physical_escalation, "
            "outing_type, person_id, people(initial_met_via)"
        )
        .eq("user_id", user_id)
        .eq("is_planned", 0)
        .execute()
    )
    rows = _rows(r)
    if not rows:
        return pd.DataFrame()
    flat: list[dict] = []
    for row in rows:
        p = row.pop("people", None) or {}
        row["met_via"] = p.get("initial_met_via")
        flat.append(row)
    return pd.DataFrame(flat)


def fetch_profile_image_for_person(
    client: Client, user_id: str, person_id: int
) -> str | None:
    r = (
        client.table("people")
        .select("profile_image")
        .eq("user_id", user_id)
        .eq("id", person_id)
        .limit(1)
        .execute()
    )
    got = _rows(r)
    if not got:
        return None
    v = got[0].get("profile_image")
    return None if v is None or str(v).strip() == "" else str(v)


def upsert_ml_config(
    client: Client,
    user_id: str,
    *,
    coef: list[float],
    mean: list[float],
    scale: list[float],
    intercept: float,
    feature_names: list[str],
    metrics: dict,
    heuristic_metrics: dict,
    calibration: dict | None,
    heuristic_calibration: dict | None,
) -> None:
    row: dict[str, Any] = {
        "user_id": user_id,
        "coef": coef,
        "mean": mean,
        "scale": scale,
        "intercept": intercept,
        "feature_names": feature_names,
        "metrics": metrics,
        "heuristic_metrics": heuristic_metrics,
        "calibration": calibration,
        "heuristic_calibration": heuristic_calibration,
    }
    client.table("ml_configs").upsert(row, on_conflict="user_id").execute()


def fetch_ml_config_dict(client: Client, user_id: str) -> dict | None:
    r = (
        client.table("ml_configs")
        .select("*")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    got = _rows(r)
    if not got:
        return None
    row = got[0]
    return {
        "feature_names": row.get("feature_names") or [],
        "coef": row.get("coef") or [],
        "intercept": row.get("intercept"),
        "mean": row.get("mean") or [],
        "scale": row.get("scale") or [],
        "metrics": row.get("metrics") or {},
        "heuristic_metrics": row.get("heuristic_metrics") or {},
        "calibration": row.get("calibration"),
        "heuristic_calibration": row.get("heuristic_calibration"),
    }

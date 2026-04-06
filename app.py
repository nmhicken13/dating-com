"""
Streamlit dating tracker — multi-tenant SaaS via Supabase (Postgres + Auth).
"""

from __future__ import annotations

import io
import os
import re
import uuid
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from supabase import Client, create_client

import supabase_db as sb
from dating_restore import (
    MOMENTUM_HALF_LIFE_DAYS,
    PIPELINE_EMPTY_SLOT_LABEL,
    augment_export_df_with_snapshot_potential,
    calculate_ml_probability,
    calculate_potential,
    ensure_default_pipeline_slots,
    filter_dates_for_potential,
    get_pipeline_top5_slot_person_ids,
    merge_pipeline_stats,
    potential_score_breakdown,
    retrain_ml_model,
    swap_pipeline_ranks,
)
# People tab: which person’s detail panel is open (int | None).
PEOPLE_TAB_SELECTED_PID_KEY = "people_tab_selected_pid"
# Main nav (replaces st.tabs so we can clear state when switching sections).
APP_NAV_SECTIONS: tuple[str, ...] = ("Dashboard", "People", "Analytics")
APP_NAV_SECTION_KEY = "app_nav_section"
APP_NAV_PREV_KEY = "app_nav_section_prev"
SESSION_USER_ID_KEY = "user_id"
SESSION_ACCESS_TOKEN_KEY = "sb_access_token"
SESSION_REFRESH_TOKEN_KEY = "sb_refresh_token"
PROFILE_PHOTOS_DIR = Path(__file__).resolve().parent / "profile_photos"
ALLOWED_PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic"}

# Heuristic / potential (Engine 1) vs ML / logistic (Engine 2) — pipeline, charts, calibration.
COLOR_HEURISTIC = "#00e8a8"
COLOR_ML = "#38bdf8"
CALIBRATION_N_BINS = 5
CALIBRATION_STRATEGY = "uniform"

STATUSES = ("Leads", "Talking", "Dating", "Friend", "Archived")
PIPELINE_STATUSES = ("Talking", "Dating", "Friend")
STATUS_LEADS = "Leads"
ACTIVE_PIPELINE_TD = ("Talking", "Dating")
VIP_MIN_ROMANTIC_DATES = 3

RATING_CHOICES = ["—"] + [str(i) for i in range(1, 11)]
THANK_YOU_CHOICES = ("—", "Yes", "No")


def _thank_you_to_db(sel: str) -> int | None:
    if sel == "Yes":
        return 1
    if sel == "No":
        return 0
    return None


def _thank_you_to_ui(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    try:
        return "Yes" if int(val) == 1 else "No"
    except (TypeError, ValueError):
        return "—"


def _thank_you_select_index(val: object) -> int:
    s = _thank_you_to_ui(val)
    return THANK_YOU_CHOICES.index(s) if s in THANK_YOU_CHOICES else 0


def _user_wanted_next_from_stored(val: object) -> int:
    """DB / row → 0/1; missing defaults to 1 (legacy rows)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 1
    try:
        return 1 if int(val) else 0
    except (TypeError, ValueError):
        return 1


OUTING_TYPE_DATE = "date"
OUTING_TYPE_CASUAL = "casual"
OUTING_TYPES = (OUTING_TYPE_DATE, OUTING_TYPE_CASUAL)
OUTING_TYPE_LABELS = {
    OUTING_TYPE_DATE: "Date — intentional / romantic outing",
    OUTING_TYPE_CASUAL: "Casual — study, hang out, came over to chill, etc.",
}


def _normalize_outing_type(raw: str | None) -> str:
    if raw == OUTING_TYPE_CASUAL:
        return OUTING_TYPE_CASUAL
    return OUTING_TYPE_DATE


def _outing_display_label(stored: str | None) -> str:
    return OUTING_TYPE_LABELS.get(_normalize_outing_type(stored), "Date")


COMPANY_ONE_ON_ONE = "One-on-one"
COMPANY_DOUBLE = "Double Date"
COMPANY_GROUP = "Group Date"
COMPANY_TYPES: tuple[str, ...] = (COMPANY_ONE_ON_ONE, COMPANY_DOUBLE, COMPANY_GROUP)
COMPANY_LABELS = {x: x for x in COMPANY_TYPES}


def _normalize_company_type(raw: str | None) -> str:
    """UI labels for radios/display (One-on-one, Double Date, Group Date).

    Accepts SQLite tokens (`one_on_one`, `double`, `group`) or legacy fuzzy strings.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return COMPANY_ONE_ON_ONE
    s = str(raw).strip()
    if s == "one_on_one":
        return COMPANY_ONE_ON_ONE
    if s == "double":
        return COMPANY_DOUBLE
    if s == "group":
        return COMPANY_GROUP
    if s in COMPANY_TYPES:
        return s
    low = s.lower().replace("_", " ").replace("-", " ")
    if low in ("one on one", "oneonone", "one on one date"):
        return COMPANY_ONE_ON_ONE
    if low in ("double", "double date", "doubledate"):
        return COMPANY_DOUBLE
    if low in ("group", "group date", "groupdate"):
        return COMPANY_GROUP
    return COMPANY_ONE_ON_ONE


def _company_type_to_sql(ui_label: str) -> str:
    """Values allowed by SQLite CHECK on dates.company_type."""
    if ui_label == COMPANY_DOUBLE:
        return "double"
    if ui_label == COMPANY_GROUP:
        return "group"
    return "one_on_one"


def _company_display_label(stored: str | None) -> str:
    return _normalize_company_type(stored)


def _company_short_tag(stored: str | None) -> str:
    m = {
        COMPANY_ONE_ON_ONE: "1:1",
        COMPANY_DOUBLE: "Double",
        COMPANY_GROUP: "Group",
    }
    return m.get(_normalize_company_type(stored), "1:1")


MET_VIA_DATING_APPS = "Dating Apps"
MET_VIA_MUTUAL_FRIENDS = "Mutual Friends"
MET_VIA_ORGANIC = "Organic / In-Person"
MET_VIA_SCHOOL_WORK = "School/Work"
MET_VIA_WARD = "Ward (Church)"
MET_VIA_OPTIONS: tuple[str, ...] = (
    MET_VIA_DATING_APPS,
    MET_VIA_MUTUAL_FRIENDS,
    MET_VIA_ORGANIC,
    MET_VIA_SCHOOL_WORK,
    MET_VIA_WARD,
)

POST_DATE_TEXTING_OPTIONS: tuple[str, ...] = (
    "—",
    "She texted first",
    "I texted first",
    "Mutual / both",
    "Little or none",
)

PHYSICAL_NONE = "None"
PHYSICAL_HELD = "Held Hands / Cuddled"
PHYSICAL_KISSED = "Kissed"
PHYSICAL_MILESTONE_OPTIONS: tuple[str, ...] = (
    PHYSICAL_NONE,
    PHYSICAL_HELD,
    PHYSICAL_KISSED,
)


def normalize_physical_milestone(raw: str | None) -> str:
    """Canonical milestone stored in `dates.physical_escalation`."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return PHYSICAL_NONE
    s = str(raw).strip()
    if not s:
        return PHYSICAL_NONE
    if s in PHYSICAL_MILESTONE_OPTIONS:
        return s
    low = s.lower()
    if "kiss" in low:
        return PHYSICAL_KISSED
    if any(k in low for k in ("cuddle", "held", "hand", "hug", "physical")):
        return PHYSICAL_HELD
    return PHYSICAL_NONE


def _physical_milestone_select_index(val: object) -> int:
    s = normalize_physical_milestone(
        None
        if val is None or (isinstance(val, float) and pd.isna(val))
        else str(val).strip()
    )
    return PHYSICAL_MILESTONE_OPTIONS.index(s)


INITIATOR_I_ASKED = "I asked her"
INITIATOR_SHE_ASKED = "She asked me"
INITIATOR_MUTUAL = "Mutual/Spontaneous"
INITIATOR_OPTIONS: tuple[str, ...] = (
    INITIATOR_I_ASKED,
    INITIATOR_SHE_ASKED,
    INITIATOR_MUTUAL,
)


def normalize_initiator(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s in INITIATOR_OPTIONS:
        return s
    return None


def _initiator_select_index(val: object) -> int:
    s = normalize_initiator(
        None
        if val is None or (isinstance(val, float) and pd.isna(val))
        else str(val).strip()
    )
    if s is None:
        return 0
    return INITIATOR_OPTIONS.index(s)


def normalize_initial_met_via(raw: str | None) -> str | None:
    """Map free text or legacy values onto MET_VIA_OPTIONS; None if empty."""
    if raw is None:
        return None
    t = str(raw).strip()
    if not t or t in ("(unknown)", "—"):
        return None
    if t in MET_VIA_OPTIONS:
        return t
    low = t.lower()
    app_kw = (
        "hinge",
        "bumble",
        "tinder",
        "okcupid",
        "grindr",
        "grinder",
        "feeld",
        "lex",
        "her ",
        "dating app",
        "dating apps",
        "the apps",
        "on apps",
        "app ",
        "apps",
        "online",
        "match.com",
        "eharmony",
        "cmb",
        "coffee meets",
        "raya",
        "plenty of fish",
        "pof",
    )
    mutual_kw = (
        "mutual",
        "friend intro",
        "friend set",
        "set up",
        "set-up",
        "through a friend",
        "introduced by",
        "fix up",
        "blind date",
    )
    school_kw = (
        "school",
        "coworker",
        "colleague",
        "classmate",
        "campus",
        "office",
        "workplace",
        "university",
        "college",
        "study group",
        "boss",
        "full-time",
        "part-time job",
        "at work",
        "from work",
    )
    org_kw = (
        "organic",
        "in person",
        "in-person",
        "in real life",
        "irl",
        "met at",
        "at a bar",
        "at the gym",
        "gym",
        "party",
        "coffee shop",
        "ran into",
        "chance",
        "random",
    )
    if any(k in low for k in app_kw):
        return MET_VIA_DATING_APPS
    if any(k in low for k in mutual_kw):
        return MET_VIA_MUTUAL_FRIENDS
    if any(k in low for k in school_kw):
        return MET_VIA_SCHOOL_WORK
    if any(k in low for k in org_kw):
        return MET_VIA_ORGANIC
    return MET_VIA_ORGANIC


def _met_via_select_index(stored: object) -> int:
    s = normalize_initial_met_via(
        None
        if stored is None or (isinstance(stored, float) and pd.isna(stored))
        else str(stored)
    )
    if s in MET_VIA_OPTIONS:
        return MET_VIA_OPTIONS.index(s)
    return MET_VIA_OPTIONS.index(MET_VIA_ORGANIC)


def _coerce_secret_str(val: object) -> str | None:
    if val is None:
        return None
    if isinstance(val, (list, dict)):
        return None
    s = str(val).strip().strip('"').strip("'")
    return s or None


def _secrets_key_names(sec: object) -> list[str]:
    try:
        keys = getattr(sec, "keys", None)
        if callable(keys):
            return [str(k) for k in keys()]
    except Exception:
        pass
    return []


def _secret_flat_get(sec: object, *candidates: str) -> str | None:
    """Exact key, attribute, or case-insensitive match."""
    if sec is None:
        return None
    want = {c.upper().replace("-", "_") for c in candidates}
    for name in candidates:
        try:
            if name in sec:
                v = _coerce_secret_str(sec[name])
                if v:
                    return v
        except Exception:
            pass
        try:
            v = getattr(sec, name, None)
            if v is not None:
                s = _coerce_secret_str(v)
                if s:
                    return s
        except Exception:
            pass
    try:
        for k in sec:
            kn = str(k).upper().replace("-", "_")
            if kn in want:
                v = _coerce_secret_str(sec[k])
                if v:
                    return v
    except Exception:
        pass
    return None


def _is_placeholder_url(u: str) -> bool:
    return "YOUR_PROJECT" in u or u == "https://xxxx.supabase.co"


def _is_placeholder_key(k: str) -> bool:
    low = k.lower()
    return low.startswith("your-anon") or low == "eyj..." or len(k) < 20


def _supabase_from_toml_dict(data: dict[str, object]) -> tuple[str | None, str | None]:
    url = _secret_flat_get(
        data,
        "SUPABASE_URL",
        "supabase_url",
        "SUPABASE_URI",
    )
    key = _secret_flat_get(
        data,
        "SUPABASE_ANON_KEY",
        "supabase_anon_key",
        "ANON_KEY",
        "SUPABASE_KEY",
    )
    sub = data.get("supabase")
    if isinstance(sub, dict):
        if not url:
            for nk in ("url", "SUPABASE_URL"):
                if nk in sub:
                    url = _coerce_secret_str(sub.get(nk))
                    break
        if not key:
            for nk in ("anon_key", "anon", "key", "SUPABASE_ANON_KEY"):
                if nk in sub:
                    key = _coerce_secret_str(sub.get(nk))
                    break
    if url and _is_placeholder_url(url):
        url = None
    if key and _is_placeholder_key(key):
        key = None
    return url, key


def _supabase_from_project_secrets_file() -> tuple[str | None, str | None]:
    """When Streamlit cwd is wrong, st.secrets may miss `.streamlit/secrets.toml` next to app.py."""
    path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if not path.is_file():
        return None, None
    try:
        import tomllib
    except ImportError:
        return None, None
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return None, None
    if not isinstance(data, dict):
        return None, None
    return _supabase_from_toml_dict(data)


def _supabase_url_and_key() -> tuple[str | None, str | None]:
    """URL + anon: st.secrets, env, then `app_dir/.streamlit/secrets.toml` (local cwd fix)."""
    url: str | None = None
    key: str | None = None
    try:
        sec = st.secrets
    except Exception:
        sec = None
    if sec is not None:
        url = _secret_flat_get(sec, "SUPABASE_URL", "supabase_url", "SUPABASE_URI")
        key = _secret_flat_get(
            sec,
            "SUPABASE_ANON_KEY",
            "supabase_anon_key",
            "ANON_KEY",
            "SUPABASE_KEY",
        )
        if (not url or not key) and sec is not None:
            try:
                if "supabase" in sec:
                    sub = sec["supabase"]
                    if not url:
                        for nk in ("url", "SUPABASE_URL"):
                            try:
                                if nk in sub:
                                    url = _coerce_secret_str(sub[nk])
                                    break
                            except Exception:
                                pass
                    if not key:
                        for nk in ("anon_key", "anon", "key", "SUPABASE_ANON_KEY"):
                            try:
                                if nk in sub:
                                    key = _coerce_secret_str(sub[nk])
                                    break
                            except Exception:
                                pass
            except Exception:
                pass
    if url and _is_placeholder_url(url):
        url = None
    if key and _is_placeholder_key(key):
        key = None
    if not url:
        u = (os.environ.get("SUPABASE_URL") or "").strip()
        if u and not _is_placeholder_url(u):
            url = u
    if not key:
        k = (os.environ.get("SUPABASE_ANON_KEY") or "").strip()
        if k and not _is_placeholder_key(k):
            key = k
    if not url or not key:
        fu, fk = _supabase_from_project_secrets_file()
        url = url or fu
        key = key or fk
    return url, key


def _render_supabase_missing_help() -> None:
    st.error(
        "Missing Supabase URL or anon key. "
        "**Streamlit Cloud:** [App settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/app-settings) "
        "→ **Secrets** — paste TOML with `SUPABASE_URL` and `SUPABASE_ANON_KEY`, **Save**, then **Reboot app**. "
        "**Local:** `.streamlit/secrets.toml` next to `app.py`, or run `streamlit run app.py` from the repo folder."
    )
    with st.expander("Troubleshooting (no secret values shown)"):
        try:
            sec = st.secrets
            names = _secrets_key_names(sec)
            st.caption(
                f"**Top-level secret keys Streamlit loaded:** {len(names)}"
                + (f" — `{', '.join(sorted(names))}`" if names else " (none — Cloud Secrets empty or TOML parse error?)")
            )
        except Exception as e:
            st.caption(f"Could not read `st.secrets`: `{type(e).__name__}`")
        p = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
        fu, fk = _supabase_from_project_secrets_file()
        if fu or fk:
            fmsg = "loaded Supabase keys from file"
        elif p.is_file():
            fmsg = "file exists but no usable `SUPABASE_URL` / `SUPABASE_ANON_KEY`"
        else:
            fmsg = "file not on server (expected on Cloud — use **Secrets** in settings)"
        st.caption(f"**Fallback** `{p.name}` next to app: {fmsg}")
        env_u = bool((os.environ.get("SUPABASE_URL") or "").strip())
        env_k = bool((os.environ.get("SUPABASE_ANON_KEY") or "").strip())
        st.caption(f"**Environment variables:** `SUPABASE_URL` set={env_u}, `SUPABASE_ANON_KEY` set={env_k}")


def _supabase_secrets_ok() -> bool:
    u, k = _supabase_url_and_key()
    return bool(u and k)


def _supabase_client() -> Client:
    url, key = _supabase_url_and_key()
    if not url or not key:
        raise RuntimeError("Supabase URL or anon key missing")
    client = create_client(url, key)
    at = st.session_state.get(SESSION_ACCESS_TOKEN_KEY)
    rt = st.session_state.get(SESSION_REFRESH_TOKEN_KEY)
    if at and rt:
        client.auth.set_session(at, rt)
    return client


def _require_authenticated_user(client: Client) -> str:
    try:
        u = client.auth.get_user()
    except Exception:
        u = None
    if u is None or getattr(u, "user", None) is None:
        for k in (
            SESSION_USER_ID_KEY,
            SESSION_ACCESS_TOKEN_KEY,
            SESSION_REFRESH_TOKEN_KEY,
        ):
            st.session_state.pop(k, None)
        st.rerun()
    uid = str(u.user.id)
    st.session_state[SESSION_USER_ID_KEY] = uid
    return uid


def render_auth_screen() -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] { display: none !important; }
        div[data-testid="stSidebarCollapsedControl"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Dating tracker")
    st.caption("Sign in to your account — data is private to you.")
    if not _supabase_secrets_ok():
        _render_supabase_missing_help()
        return
    client = _supabase_client()
    t_log, t_sign = st.tabs(["Log in", "Sign up"])
    with t_log:
        le = st.text_input("Email", key="auth_login_email")
        lp = st.text_input("Password", type="password", key="auth_login_pw")
        if st.button("Log in", key="auth_login_btn"):
            if not le.strip() or not lp:
                st.warning("Enter email and password.")
            else:
                try:
                    res = client.auth.sign_in_with_password(
                        {"email": le.strip(), "password": lp}
                    )
                    if res.session:
                        st.session_state[SESSION_ACCESS_TOKEN_KEY] = res.session.access_token
                        st.session_state[SESSION_REFRESH_TOKEN_KEY] = (
                            res.session.refresh_token
                        )
                        st.session_state[SESSION_USER_ID_KEY] = str(res.user.id)
                        st.rerun()
                    else:
                        st.warning("Check your email to confirm your account, then try again.")
                except Exception as e:
                    st.error(str(e))
    with t_sign:
        se = st.text_input("Email", key="auth_sign_email")
        sp = st.text_input("Password", type="password", key="auth_sign_pw")
        sp2 = st.text_input("Confirm password", type="password", key="auth_sign_pw2")
        if st.button("Create account", key="auth_sign_btn"):
            if not se.strip() or not sp:
                st.warning("Enter email and password.")
            elif sp != sp2:
                st.warning("Passwords do not match.")
            else:
                try:
                    client.auth.sign_up({"email": se.strip(), "password": sp})
                    st.success(
                        "Account created. If email confirmation is enabled in Supabase, "
                        "check your inbox, then log in."
                    )
                except Exception as e:
                    st.error(str(e))


def profile_ref_is_url(ref: str | None) -> bool:
    if not ref or not str(ref).strip():
        return False
    return str(ref).strip().lower().startswith(("http://", "https://"))


def unlink_profile_image_file(ref: str | None) -> None:
    if not ref or profile_ref_is_url(ref):
        return
    p = PROFILE_PHOTOS_DIR / str(ref).lstrip("/")
    if p.is_file():
        p.unlink()
    try:
        par = p.parent
        if par != PROFILE_PHOTOS_DIR and par.is_dir() and not any(par.iterdir()):
            par.rmdir()
    except OSError:
        pass


def save_profile_photo_upload(person_id: int, uf) -> str:
    ext = Path(uf.name).suffix.lower()
    if ext not in ALLOWED_PHOTO_EXT:
        ext = ".jpg"
    rel = f"{person_id}/{uuid.uuid4().hex}{ext}"
    full = PROFILE_PHOTOS_DIR / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(uf.getvalue())
    return rel


def delete_person_with_files(client: Client, user_id: str, person_id: int) -> None:
    ref = sb.fetch_profile_image_for_person(client, user_id, person_id)
    if ref:
        unlink_profile_image_file(ref)
    sb.delete_person(client, user_id, person_id)


def _parse_stored_date(val: object) -> date:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return date.today()
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    ts = pd.to_datetime(val, errors="coerce")
    if pd.isna(ts):
        return date.today()
    return ts.date()


def _parse_stored_datetime(val: object) -> datetime | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, datetime):
        return val
    ts = pd.to_datetime(val, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _outings_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        c
        for c in (
            "occurred_on",
            "activity",
            "outing_type",
            "company_type",
            "rating",
            "cost",
            "physical_escalation",
            "duration_hours",
            "thank_you",
            "is_planned",
            "initiator",
            "user_wanted_next_date",
        )
        if c in df.columns
    ]
    out = df[cols].copy()
    if "outing_type" in out.columns:
        out["outing_type"] = out["outing_type"].map(_outing_display_label)
    if "company_type" in out.columns:
        out["company_type"] = out["company_type"].map(_company_display_label)
    if "thank_you" in out.columns:
        out["thank_you"] = out["thank_you"].map(_thank_you_to_ui)
    if "is_planned" in out.columns:
        out["is_planned"] = out["is_planned"].map(lambda x: "Planned" if int(x or 0) == 1 else "")
    if "user_wanted_next_date" in out.columns:
        out["user_wanted_next_date"] = out["user_wanted_next_date"].map(
            lambda x: "Yes" if int(x or 1) == 1 else "No"
        )
    return out


def render_profile_avatar(ref: object, *, width: int = 64) -> None:
    if ref is None or (isinstance(ref, float) and pd.isna(ref)):
        st.caption("No photo")
        return
    s = str(ref).strip()
    if not s:
        st.caption("No photo")
        return
    if profile_ref_is_url(s):
        st.image(s, width=width)
        return
    p = PROFILE_PHOTOS_DIR / s.lstrip("/")
    if p.is_file():
        st.image(str(p), width=width)
    else:
        st.caption("Photo missing")


def render_log_outing_form_fixed_person(
    client: Client,
    user_id: str,
    person_id: int,
    people_df: pd.DataFrame,
    *,
    key_prefix: str,
) -> None:
    """Log a completed outing for a fixed person (no person picker)."""
    row = people_df[people_df["id"].astype(int) == int(person_id)]
    if row.empty:
        st.warning("Person not found.")
        return
    name = str(row.iloc[0]["name"])
    st.markdown("#### Log a new date")
    st.caption(f"With **{name}**")
    lo = st.session_state.log_outing_seq
    with st.form(f"{key_prefix}_log_outing_{lo}"):
        log_ot = st.radio(
            "Kind of meetup",
            OUTING_TYPES,
            format_func=lambda x: OUTING_TYPE_LABELS[x],
            horizontal=True,
            key=f"{key_prefix}_{lo}_meetup",
        )
        log_co = st.radio(
            "Who was there",
            COMPANY_TYPES,
            format_func=lambda x: COMPANY_LABELS[x],
            horizontal=True,
            key=f"{key_prefix}_{lo}_company",
        )
        log_init = st.selectbox(
            "Who asked who out?",
            INITIATOR_OPTIONS,
            key=f"{key_prefix}_{lo}_init",
        )
        when = st.date_input(
            "Day",
            value=date.today(),
            key=f"{key_prefix}_{lo}_day",
        )
        activity = st.text_input(
            "What you did",
            placeholder="Dinner, library study session…",
            key=f"{key_prefix}_{lo}_act",
        )
        log_cost = st.number_input(
            "Cost ($)",
            min_value=0.0,
            value=0.0,
            step=5.0,
            format="%.2f",
            key=f"{key_prefix}_{lo}_cost",
            help="Use 0 for a free hangout.",
        )
        log_dur = st.number_input(
            "Duration (hours)",
            min_value=0.0,
            max_value=336.0,
            value=None,
            step=0.25,
            help="Optional. How long the outing lasted.",
            key=f"{key_prefix}_{lo}_dur",
        )
        st.caption("Optional rating **1–10** (often most useful for dates).")
        rating_raw = st.selectbox(
            "Rating (optional)",
            RATING_CHOICES,
            key=f"{key_prefix}_{lo}_rate",
        )
        log_phys = st.selectbox(
            "Highest Physical Milestone",
            PHYSICAL_MILESTONE_OPTIONS,
            key=f"{key_prefix}_{lo}_phys",
        )
        log_ty = st.selectbox(
            "Did you get a thank-you text after? (optional)",
            THANK_YOU_CHOICES,
            key=f"{key_prefix}_{lo}_thanks",
        )
        log_want_next = st.checkbox(
            "Did you want to see her again?",
            value=True,
            key=f"{key_prefix}_{lo}_want_next",
        )
        dnotes = st.text_area("Notes", height=68, key=f"{key_prefix}_{lo}_notes")
        if st.form_submit_button("Save outing"):
            rid = None if rating_raw == "—" else int(rating_raw)
            new_oid = sb.add_date_event(
                client,
                user_id,
                person_id=int(person_id),
                occurred_on=when,
                activity=activity,
                notes=dnotes,
                rating=rid,
                physical_escalation=log_phys,
                outing_type=log_ot,
                company_type=log_co,
                thank_you=_thank_you_to_db(log_ty),
                cost=float(log_cost),
                initiator=log_init,
                duration_hours=None if log_dur is None else float(log_dur),
                user_wanted_next_date=1 if log_want_next else 0,
            )
            st.session_state.log_outing_seq += 1
            st.success("Outing saved.")
            st.rerun()


def _history_outing_expander_label(row_d: pd.Series) -> str:
    act = str(row_d.get("activity") or "").strip() or "Outing"
    raw_ot = row_d.get("outing_type")
    ot = _normalize_outing_type(
        str(raw_ot)
        if raw_ot is not None and str(raw_ot) != "" and not pd.isna(raw_ot)
        else None
    )
    tag = "Date" if ot == OUTING_TYPE_DATE else "Casual"
    pl = row_d.get("is_planned")
    pre = (
        "Planned · "
        if pl is not None
        and not (isinstance(pl, float) and pd.isna(pl))
        and int(pl) == 1
        else ""
    )
    return f"Edit · {pre}{row_d['occurred_on']} · {act} · {tag}"


def render_date_edit_form(
    client: Client, user_id: str,
    people_df: pd.DataFrame,
    row_d: pd.Series,
    *,
    key_prefix: str,
    lock_person_id: int | None = None,
) -> None:
    """Edit or delete one outing row. If lock_person_id is set, person is fixed (History tab)."""
    did = int(row_d["id"])
    all_ids = people_df["id"].astype(int).tolist()
    all_lbls = people_df["name"].tolist()
    pi0 = (
        all_ids.index(int(row_d["person_id"]))
        if int(row_d["person_id"]) in all_ids
        else 0
    )

    with st.form(f"{key_prefix}_edit_date_{did}"):
        if lock_person_id is not None:
            sp = people_df[people_df["id"].astype(int) == int(lock_person_id)].iloc[0]
            st.caption(f"With **{sp['name']}**")
        else:
            new_pi = st.selectbox(
                "Who was it with?",
                list(range(len(all_ids))),
                index=pi0,
                format_func=lambda i: f"{all_lbls[i]} · {people_df.iloc[i]['status']}",
                key=f"{key_prefix}_ed_who_{did}",
            )
        ed_when = st.date_input(
            "Date",
            value=_parse_stored_date(row_d["occurred_on"]),
            key=f"{key_prefix}_ed_day_{did}",
        )
        raw_row_ot = row_d.get("outing_type")
        ed_ot_ix = OUTING_TYPES.index(
            _normalize_outing_type(
                str(raw_row_ot)
                if raw_row_ot is not None
                and str(raw_row_ot) != ""
                and not pd.isna(raw_row_ot)
                else None
            )
        )
        ed_ot = st.radio(
            "Kind of meetup",
            OUTING_TYPES,
            index=ed_ot_ix,
            format_func=lambda x: OUTING_TYPE_LABELS[x],
            horizontal=True,
            key=f"{key_prefix}_ed_ot_{did}",
        )
        raw_row_co = row_d.get("company_type")
        ed_co_ix = COMPANY_TYPES.index(
            _normalize_company_type(
                str(raw_row_co)
                if raw_row_co is not None
                and str(raw_row_co) != ""
                and not pd.isna(raw_row_co)
                else None
            )
        )
        ed_co = st.radio(
            "Who was there",
            COMPANY_TYPES,
            index=ed_co_ix,
            format_func=lambda x: COMPANY_LABELS[x],
            horizontal=True,
            key=f"{key_prefix}_ed_co_{did}",
        )
        ed_init = st.selectbox(
            "Who asked who out?",
            INITIATOR_OPTIONS,
            index=_initiator_select_index(row_d.get("initiator")),
            key=f"{key_prefix}_ed_init_{did}",
        )
        ed_act = st.text_input(
            "What you did",
            value=str(row_d.get("activity") or ""),
            key=f"{key_prefix}_ed_act_{did}",
        )
        _ed_c = row_d.get("cost")
        _ed_cost_val = (
            0.0
            if _ed_c is None
            or (isinstance(_ed_c, float) and pd.isna(_ed_c))
            else float(_ed_c)
        )
        ed_cost = st.number_input(
            "Cost ($)",
            min_value=0.0,
            value=_ed_cost_val,
            step=5.0,
            format="%.2f",
            key=f"{key_prefix}_ed_cost_{did}",
        )
        _dh_raw = row_d.get("duration_hours") if "duration_hours" in row_d.index else None
        _ed_dur_val: float | None = None
        if _dh_raw is not None and not (isinstance(_dh_raw, float) and pd.isna(_dh_raw)):
            try:
                _ed_dur_val = float(_dh_raw)
            except (TypeError, ValueError):
                _ed_dur_val = None
        ed_dur = st.number_input(
            "Duration (hours)",
            min_value=0.0,
            max_value=336.0,
            value=_ed_dur_val,
            step=0.25,
            help="Optional. Leave empty to clear.",
            key=f"{key_prefix}_ed_dur_{did}",
        )
        rv = row_d.get("rating")
        rv_s = "—" if pd.isna(rv) or rv is None else str(int(rv))
        ri = RATING_CHOICES.index(rv_s) if rv_s in RATING_CHOICES else 0
        ed_rate = st.selectbox(
            "Rating (optional, 1–10)",
            RATING_CHOICES,
            index=ri,
            key=f"{key_prefix}_ed_rate_{did}",
        )
        ed_phys = st.selectbox(
            "Highest Physical Milestone",
            PHYSICAL_MILESTONE_OPTIONS,
            index=_physical_milestone_select_index(row_d.get("physical_escalation")),
            key=f"{key_prefix}_ed_phys_{did}",
        )
        ed_ty = st.selectbox(
            "Thank-you text after?",
            THANK_YOU_CHOICES,
            index=_thank_you_select_index(row_d.get("thank_you")),
            key=f"{key_prefix}_ed_ty_{did}",
        )
        ed_want_next = st.checkbox(
            "Did you want to see her again?",
            value=_user_wanted_next_from_stored(row_d.get("user_wanted_next_date")) == 1,
            key=f"{key_prefix}_ed_want_{did}",
        )
        ed_notes = st.text_area(
            "Notes",
            value=str(row_d.get("notes") or ""),
            height=60,
            key=f"{key_prefix}_ed_notes_{did}",
        )
        _ed_ip = row_d.get("is_planned")
        ed_planned = st.checkbox(
            "Planned / not happened yet",
            value=(
                _ed_ip is not None
                and not (isinstance(_ed_ip, float) and pd.isna(_ed_ip))
                and int(_ed_ip) == 1
            ),
            key=f"{key_prefix}_ed_plan_{did}",
            help="Planned rows are excluded from KPIs and analytics until completed.",
        )
        _ed_sched0 = _parse_stored_datetime(row_d.get("scheduled_date"))
        if _ed_sched0 is None:
            _ed_sched0 = datetime.combine(ed_when, time(18, 0))
        ed_sched_d = st.date_input(
            "Scheduled date",
            value=_ed_sched0.date(),
            key=f"{key_prefix}_ed_sched_d_{did}",
        )
        ed_sched_t = st.time_input(
            "Scheduled time",
            value=_ed_sched0.time(),
            key=f"{key_prefix}_ed_sched_t_{did}",
        )
        ed_sched = datetime.combine(ed_sched_d, ed_sched_t)
        save_d = st.form_submit_button("Save outing")
        del_d = st.form_submit_button("Delete this outing")
        if save_d:
            if lock_person_id is not None:
                save_pid = int(lock_person_id)
            else:
                save_pid = all_ids[new_pi]
            er = None if ed_rate == "—" else int(ed_rate)
            sb.update_date_event(
                client,
                user_id,
                did,
                person_id=save_pid,
                occurred_on=ed_when,
                activity=ed_act,
                notes=ed_notes,
                rating=er,
                physical_escalation=ed_phys,
                outing_type=ed_ot,
                company_type=ed_co,
                thank_you=_thank_you_to_db(ed_ty),
                cost=float(ed_cost),
                is_planned=ed_planned,
                scheduled_at=ed_sched if ed_planned else None,
                initiator=ed_init,
                duration_hours=None if ed_dur is None else float(ed_dur),
                user_wanted_next_date=1 if ed_want_next else 0,
            )
            st.success("Outing updated.")
            st.rerun()
        if del_d:
            sb.delete_date_event(client, user_id, did)
            st.success("Outing deleted.")
            st.rerun()


def render_history_profile_readonly(
    client: Client, user_id: str,
    people_df: pd.DataFrame,
    pid: int,
) -> None:
    row = people_df[people_df["id"].astype(int) == int(pid)]
    if row.empty:
        st.warning("Person not found.")
        return
    r = row.iloc[0]
    stats = sb.load_person_pipeline_stats(client, user_id)
    mrow = merge_pipeline_stats(pd.DataFrame([r]), stats).iloc[0]
    nd = int(mrow.get("n_romantic_dates") or 0)
    ts = float(mrow.get("total_spent") or 0)
    last = mrow.get("last_occurred")
    n_all = sb.count_completed_outings_for_person(client, user_id, int(pid))
    c_img, c_txt = st.columns([0.35, 1])
    with c_img:
        render_profile_avatar(r.get("profile_image"), width=96)
    with c_txt:
        st.markdown(f"### {r['name']}")
        st.caption(str(r["status"]))
    if r.get("initial_met_via") and str(r["initial_met_via"]).strip():
        st.caption(f"Initially met via: **{r['initial_met_via']}**")
    st.caption(
        f"Dates (intentional outings): **{nd}** · "
        f"All completed outings: **{n_all}** · Total spent: **${ts:,.2f}**"
    )
    if pd.notna(last):
        st.caption(f"Last outing: **{pd.Timestamp(last).strftime('%Y-%m-%d')}**")


def render_history_person_edit_expander(
    client: Client, user_id: str,
    people_df: pd.DataFrame,
    pid: int,
    *,
    key_prefix: str = "hist",
    clear_session_pid_key: str | None = None,
) -> None:
    """Name, status, photo, delete — tucked away for a minimal History view."""
    sub = people_df[people_df["id"].astype(int) == int(pid)]
    if sub.empty:
        return
    sel = sub.iloc[0]
    pid = int(sel["id"])
    with st.expander("Edit person details or delete", expanded=False):
        _cur_prof = sel.get("profile_image")
        _cur_prof_s = (
            ""
            if _cur_prof is None or (isinstance(_cur_prof, float) and pd.isna(_cur_prof))
            else str(_cur_prof).strip()
        )
        with st.form(f"{key_prefix}_person_{pid}"):
            ed_name = st.text_input("Name", value=str(sel["name"]), key=f"{key_prefix}_pn_{pid}")
            si = STATUSES.index(sel["status"]) if sel["status"] in STATUSES else 0
            ed_stat = st.selectbox("Status", STATUSES, index=si, key=f"{key_prefix}_st_{pid}")
            ed_met = st.selectbox(
                "Initially met via",
                MET_VIA_OPTIONS,
                index=_met_via_select_index(sel["initial_met_via"]),
                key=f"{key_prefix}_mv_{pid}",
            )
            ed_purl = st.text_input(
                "Profile image URL (optional)",
                value=_cur_prof_s if profile_ref_is_url(_cur_prof_s) else "",
                key=f"{key_prefix}_ed_purl_{pid}",
            )
            ed_pfile = st.file_uploader(
                "Or upload a new profile photo",
                type=["jpg", "jpeg", "png", "webp", "gif", "heic"],
                key=f"{key_prefix}_ed_pfile_{pid}",
            )
            ed_rm_prof = st.checkbox(
                "Remove profile image",
                key=f"{key_prefix}_ed_rmprof_{pid}",
            )
            if st.form_submit_button("Save person"):
                nn = (ed_name or "").strip()
                if not nn:
                    st.error("Name is required.")
                else:
                    new_prof = _cur_prof_s
                    if ed_rm_prof:
                        unlink_profile_image_file(_cur_prof_s)
                        new_prof = ""
                    elif ed_pfile is not None:
                        unlink_profile_image_file(_cur_prof_s)
                        new_prof = save_profile_photo_upload(pid, ed_pfile)
                    elif (ed_purl or "").strip():
                        u = (ed_purl or "").strip()
                        if not profile_ref_is_url(u):
                            st.error("Image URL must start with http:// or https://")
                            st.stop()
                        if not profile_ref_is_url(_cur_prof_s):
                            unlink_profile_image_file(_cur_prof_s)
                        new_prof = u
                    sb.update_person(
                        client,
                        user_id,
                        pid,
                        name=nn,
                        status=ed_stat,
                        initial_met_via=ed_met,
                        profile_image=new_prof or None,
                    )
                    st.success("Person updated.")
                    st.rerun()

        st.markdown("**Delete person** (removes all outings for her)")
        del_ok = st.checkbox(
            "I understand this cannot be undone",
            key=f"{key_prefix}_confirm_del_person_{pid}",
        )
        if st.button("Delete this person", disabled=not del_ok, key=f"{key_prefix}_btn_del_p_{pid}"):
            delete_person_with_files(client, user_id, pid)
            if clear_session_pid_key and st.session_state.get(clear_session_pid_key) == int(
                pid
            ):
                st.session_state[clear_session_pid_key] = None
            st.success("Person deleted.")
            st.rerun()


def render_history_outings_expanders(
    client: Client, user_id: str,
    people_df: pd.DataFrame,
    pid: int,
    *,
    key_prefix: str = "hist",
) -> None:
    dates_sub = sb.load_dates_for_person(client, user_id, pid)
    if dates_sub.empty:
        st.caption("No outings logged yet — use **Log a new outing** on this tab when it’s shown.")
        return
    st.markdown("#### Outings")
    st.caption(
        "Open a row to edit or delete that outing. Log additional outings from **Log a new outing** below."
    )
    preview = dates_sub.copy()
    st.dataframe(
        _outings_dataframe_for_display(preview),
        hide_index=True,
        use_container_width=True,
    )
    for _, row_d in dates_sub.iterrows():
        with st.expander(_history_outing_expander_label(row_d), expanded=False):
            render_date_edit_form(
                client,
                user_id,
                people_df,
                row_d,
                key_prefix=key_prefix,
                lock_person_id=pid,
            )


def _plotly_calibration_curve(
    mean_pred: list[float],
    prob_true: list[float],
    *,
    n_bins: int,
    chart_title: str,
    series_name: str,
    accent: str,
    caption: str,
) -> None:
    """Shared reliability diagram (perfect diagonal + binned curve)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(color="#6b7280", width=2, dash="dash"),
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mean_pred,
            y=prob_true,
            mode="lines+markers",
            name=series_name,
            line=dict(color=accent, width=3),
            marker=dict(size=12, color=accent, line=dict(width=0)),
            hovertemplate=(
                "Mean predicted p=%{x:.3f}<br>Actual success rate=%{y:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=chart_title, font=dict(size=16)),
        xaxis_title="Predicted probability",
        yaxis_title="Actual success rate",
        xaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.08)"),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=48, r=24, t=56, b=48),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(caption)


def render_ml_calibration_chart(cfg: dict) -> None:
    """Reliability diagram from saved `calibration` block (Plotly, plotly_dark)."""
    cal = cfg.get("calibration")
    if not isinstance(cal, dict):
        st.caption(
            "No calibration curve stored yet — use **Analytics → Retrain ML engine** to refresh."
        )
        return
    mp = cal.get("mean_predicted_value")
    pt = cal.get("prob_true")
    if not isinstance(mp, list) or not isinstance(pt, list) or len(mp) != len(pt) or not mp:
        st.caption(
            "No calibration data — use **Analytics → Retrain ML engine** to refresh."
        )
        return
    n_bins = int(cal.get("n_bins", CALIBRATION_N_BINS))
    strat = str(cal.get("strategy", CALIBRATION_STRATEGY))
    _plotly_calibration_curve(
        [float(x) for x in mp],
        [float(x) for x in pt],
        n_bins=n_bins,
        chart_title=f"Calibration curve: ML model vs reality ({n_bins} bins, {strat})",
        series_name="ML model (LogReg)",
        accent=COLOR_ML,
        caption=(
            "**ML** trace matches the **ML** color on the Dashboard. **Gray dashed** = perfect calibration. "
            f"Bins: **{n_bins}** · **{strat}** on [0, 1] — same binning as Heuristic Diagnostics."
        ),
    )


def render_heuristic_calibration_chart(cfg: dict) -> None:
    """Reliability diagram for potential score ÷ 100 from `heuristic_calibration`."""
    cal = cfg.get("heuristic_calibration")
    if not isinstance(cal, dict):
        st.caption(
            "No heuristic calibration stored yet — use **Analytics → Retrain ML engine** to refresh."
        )
        return
    mp = cal.get("mean_predicted_value")
    pt = cal.get("prob_true")
    if not isinstance(mp, list) or not isinstance(pt, list) or len(mp) != len(pt) or not mp:
        st.caption(
            "No heuristic calibration data — use **Analytics → Retrain ML engine** to refresh."
        )
        return
    n_bins = int(cal.get("n_bins", CALIBRATION_N_BINS))
    strat = str(cal.get("strategy", CALIBRATION_STRATEGY))
    _plotly_calibration_curve(
        [float(x) for x in mp],
        [float(x) for x in pt],
        n_bins=n_bins,
        chart_title=f"Calibration curve: Heuristic (potential) vs reality ({n_bins} bins, {strat})",
        series_name="Heuristic (potential ÷ 100)",
        accent=COLOR_HEURISTIC,
        caption=(
            "**Heuristic** trace matches the **Heuristic** color on the Dashboard. **Gray dashed** = perfect calibration. "
            f"Bins: **{n_bins}** · **{strat}** on [0, 1] — same binning as Machine Learning Diagnostics."
        ),
    )


def _load_ml_config_dict(client: Client, user_id: str) -> dict | None:
    """Load saved ML diagnostics for the signed-in user from Supabase."""
    try:
        return sb.fetch_ml_config_dict(client, user_id)
    except Exception:
        return None


def _render_ml_saved_metrics_row(cfg: dict) -> None:
    """Accuracy, precision, AUC for the saved logistic regression (in-sample, training rows)."""
    m = cfg.get("metrics") or {}
    ma, mp, muc = m.get("accuracy"), m.get("precision"), m.get("roc_auc")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric(
            "Model Accuracy",
            f"{100 * float(ma):.0f}%" if ma is not None else "—",
            help="Logistic regression: correct class at 0.5 probability (in-sample).",
        )
    with mc2:
        st.metric(
            "Precision",
            f"{100 * float(mp):.0f}%" if mp is not None else "—",
            help="ML: when it predicts another date, how often did another date happen?",
        )
    with mc3:
        auc_txt = f"{float(muc):.2f}" if muc is not None else "—"
        st.metric(
            "Sorting Power (AUC)",
            auc_txt,
            help="ML: ranking quality on the training rows (predicted probability vs outcome).",
        )


def _render_heuristic_saved_metrics_row(cfg: dict) -> None:
    """Accuracy, precision, AUC for the Dashboard potential formula on the same training rows."""
    m = cfg.get("heuristic_metrics") or {}
    ma, mp, muc = m.get("accuracy"), m.get("precision"), m.get("roc_auc")
    if ma is None and mp is None and muc is None:
        st.caption(
            "No potential-formula metrics stored yet — **Retrain ML engine** refreshes them too."
        )
        return
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric(
            "Model Accuracy",
            f"{100 * float(ma):.0f}%" if ma is not None else "—",
            help="Potential score (0–100): predicted “success” if score ≥ 50 (in-sample).",
        )
    with mc2:
        st.metric(
            "Precision",
            f"{100 * float(mp):.0f}%" if mp is not None else "—",
            help="When potential ≥ 50, how often did another date actually happen?",
        )
    with mc3:
        auc_txt = f"{float(muc):.2f}" if muc is not None else "—"
        st.metric(
            "Sorting Power (AUC)",
            auc_txt,
            help="Ranking quality using potential ÷ 100 as a score vs outcome.",
        )


def render_ml_diagnostics_expander(cfg: dict | None) -> None:
    """ML metrics, feature weights, and logistic-regression calibration curve."""
    with st.expander("Machine Learning Diagnostics"):
        if cfg is None:
            st.caption(
                "Retrain the ML engine below to save your model to **Supabase** (balanced logistic regression, 10 features)."
            )
            return
        st.caption(
            "**Logistic regression (Engine 2)** — same labels as retrain: another date happened vs not, "
            "only rows where you wanted another date."
        )
        _render_ml_saved_metrics_row(cfg)
        names = list(cfg.get("feature_names") or ML_FEATURE_NAMES)
        coefs = cfg.get("coef") or []
        if len(names) != len(coefs):
            st.caption("Feature list length mismatch in saved config.")
        else:
            wf = pd.DataFrame({"feature": names, "weight": [float(c) for c in coefs]})
            wf = wf.iloc[wf["weight"].abs().argsort()[::-1]].reset_index(drop=True)
            st.markdown("**Feature weights**")
            st.dataframe(wf, hide_index=True, use_container_width=True)
        st.markdown("#### Calibration (ML)")
        st.caption(
            f"Each point is one bin on predicted probability: mean predicted p vs fraction of rows with another date. "
            f"**{CALIBRATION_N_BINS} {CALIBRATION_STRATEGY} bins** on [0, 1] — identical binning to **Heuristic Diagnostics**."
        )
        render_ml_calibration_chart(cfg)


def render_heuristic_diagnostics_expander(cfg: dict | None) -> None:
    """Potential / heuristic formula metrics (Engine 1), not the ML model."""
    with st.expander("Heuristic Diagnostics"):
        if cfg is None:
            st.info(
                "No saved config yet — use **Retrain ML engine with latest data** above."
            )
            return
        st.caption(
            "**Potential predictor (Engine 1)** — the same 🔥 heuristic as the Dashboard (ratings, physical gap, "
            "trajectory, duration, thank-you, momentum). These metrics use the **same training rows** as ML when you "
            "retrain: success = another date happened. **Decision rule:** score ≥ **50** on 0–100 counts as predicting success."
        )
        _render_heuristic_saved_metrics_row(cfg)
        st.markdown("#### Calibration (Heuristic)")
        st.caption(
            f"Bins use **potential ÷ 100** as predicted probability vs whether another date happened. "
            f"**{CALIBRATION_N_BINS} {CALIBRATION_STRATEGY} bins** on [0, 1] — same procedure as **Machine Learning Diagnostics**."
        )
        render_heuristic_calibration_chart(cfg)


def render_export_csv_expander(
    client: Client, user_id: str, *, use_expander: bool = True
) -> None:
    def _export_body() -> None:
        st.caption(
            "Download all outings joined with each person’s **status** and **Initially met via** "
            "(includes **initiator**, **Duration (hours)**, **user_wanted_next_date**, and **snapshot_potential** "
            "for offline analysis only — not all columns appear in app tables)."
        )
        _exp_df = augment_export_df_with_snapshot_potential(sb.load_dates_export_df(client, user_id))
        if _exp_df.empty:
            st.caption("No rows to export yet.")
        else:
            _csv_bytes = _exp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download dating_tracker_export.csv",
                data=_csv_bytes,
                file_name="dating_tracker_export.csv",
                mime="text/csv",
                key="export_dates_csv",
            )

    if use_expander:
        with st.expander("Advanced: Export data to CSV"):
            _export_body()
    else:
        st.markdown("#### Export data")
        _export_body()


def _csv_col(df: pd.DataFrame, *candidates: str) -> str | None:
    colmap = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        k = cand.strip().lower()
        if k in colmap:
            return colmap[k]
    return None


def run_csv_import(client: Client, user_id: str, raw: bytes) -> tuple[int, int, list[str]]:
    """Import rows from an export-style CSV (same columns as dating_tracker_export.csv)."""
    warnings: list[str] = []
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        return 0, 0, [f"Could not read CSV: {e}"]
    if df.empty:
        return 0, 0, ["File is empty."]
    pcol = _csv_col(df, "person_name", "Person Name")
    ocol = _csv_col(df, "occurred_on", "occurred on")
    if not pcol or not ocol:
        return 0, 0, ["Need at least **person_name** and **occurred_on** columns (use app export format)."]
    status_col = _csv_col(df, "person_status", "status")
    met_col = _csv_col(df, "initially_met_via", "initial_met_via")
    prof_col = _csv_col(df, "person_profile_image", "profile_image")
    ot_col = _csv_col(df, "outing_type")
    co_col = _csv_col(df, "company_type")
    phys_col = _csv_col(df, "physical_escalation")
    notes_col = _csv_col(df, "notes")
    act_col = _csv_col(df, "activity")
    ty_col = _csv_col(df, "thank_you")
    cost_col = _csv_col(df, "cost")
    plan_col = _csv_col(df, "is_planned")
    sched_col = _csv_col(df, "scheduled_date")
    init_col = _csv_col(df, "initiator")
    dur_col = _csv_col(df, "duration (hours)", "Duration (hours)", "duration_hours")
    want_col = _csv_col(df, "user_wanted_next_date")
    rate_col = _csv_col(df, "rating")

    existing = sb.load_people_df(client, user_id)
    name_to_id: dict[str, int] = {
        str(r["name"]).strip().lower(): int(r["id"]) for _, r in existing.iterrows()
    }

    seen_new: list[str] = []
    for _, row in df.iterrows():
        nm = str(row[pcol]).strip()
        if nm and nm.lower() not in name_to_id and nm.lower() not in {x.lower() for x in seen_new}:
            seen_new.append(nm)

    people_created = 0
    for nm in seen_new:
        k = nm.lower()
        sub = df[df[pcol].astype(str).str.strip().str.lower() == k]
        pr = sub.iloc[0]
        st_val = (
            str(pr[status_col]).strip()
            if status_col and pd.notna(pr.get(status_col)) and str(pr.get(status_col)).strip()
            else "Leads"
        )
        if st_val not in STATUSES:
            warnings.append(f"Unknown status {st_val!r} for {nm!r} — using Leads.")
            st_val = "Leads"
        imv = (
            normalize_initial_met_via(str(pr[met_col]).strip())
            if met_col and pd.notna(pr.get(met_col)) and str(pr.get(met_col)).strip()
            else None
        )
        prof = None
        if prof_col and pd.notna(pr.get(prof_col)) and str(pr.get(prof_col)).strip():
            prof = str(pr[prof_col]).strip()
        pid = sb.add_person(
            client,
            user_id,
            name=nm,
            status=st_val,
            initial_met_via=imv,
            profile_image=prof,
        )
        name_to_id[k] = pid
        people_created += 1

    dates_created = 0
    sort_df = df.copy()
    sort_df["_od"] = pd.to_datetime(sort_df[ocol], errors="coerce")
    sort_df = sort_df.sort_values("_od", kind="mergesort")
    for _, row in sort_df.iterrows():
        pname = str(row[pcol]).strip()
        if not pname:
            warnings.append("Skipped row with empty person name.")
            continue
        pid = name_to_id.get(pname.lower())
        if pid is None:
            warnings.append(f"No person match for {pname!r} — skipped one outing.")
            continue
        od = pd.to_datetime(row[ocol], errors="coerce")
        if pd.isna(od):
            warnings.append(f"Bad date for {pname!r} — skipped.")
            continue
        occurred_on = od.date()
        ot_raw = str(row[ot_col]).strip() if ot_col and pd.notna(row.get(ot_col)) else None
        outing_type = _normalize_outing_type(ot_raw)
        co_raw = str(row[co_col]).strip() if co_col and pd.notna(row.get(co_col)) else None
        company_type = _company_type_to_sql(_normalize_company_type(co_raw))
        phys = (
            normalize_physical_milestone(str(row[phys_col]).strip())
            if phys_col and pd.notna(row.get(phys_col)) and str(row.get(phys_col)).strip()
            else PHYSICAL_NONE
        )
        activity = (
            str(row[act_col]).strip()
            if act_col and pd.notna(row.get(act_col)) and str(row.get(act_col)).strip()
            else None
        )
        notes = (
            str(row[notes_col]).strip()
            if notes_col and pd.notna(row.get(notes_col)) and str(row.get(notes_col)).strip()
            else None
        )
        rating = None
        if rate_col and pd.notna(row.get(rate_col)):
            try:
                rating = int(row[rate_col])
                if rating < 1 or rating > 10:
                    rating = None
            except (TypeError, ValueError):
                rating = None
        thank_you: int | None = None
        if ty_col and pd.notna(row.get(ty_col)):
            try:
                thank_you = int(row[ty_col])
                if thank_you not in (0, 1):
                    thank_you = None
            except (TypeError, ValueError):
                thank_you = None
        cost = None
        if cost_col and pd.notna(row.get(cost_col)):
            try:
                cost = float(row[cost_col])
            except (TypeError, ValueError):
                cost = None
        is_planned = 0
        if plan_col and pd.notna(row.get(plan_col)):
            try:
                is_planned = 1 if int(row[plan_col]) else 0
            except (TypeError, ValueError):
                is_planned = 0
        scheduled_at: datetime | None = None
        if is_planned and sched_col and pd.notna(row.get(sched_col)):
            scheduled_at = pd.to_datetime(row[sched_col], errors="coerce")
            if pd.isna(scheduled_at):
                scheduled_at = datetime.combine(occurred_on, time(18, 0))
            else:
                scheduled_at = scheduled_at.to_pydatetime()
        initiator = (
            normalize_initiator(str(row[init_col]).strip())
            if init_col and pd.notna(row.get(init_col)) and str(row.get(init_col)).strip()
            else None
        )
        duration_hours: float | None = None
        if dur_col and pd.notna(row.get(dur_col)):
            try:
                duration_hours = float(row[dur_col])
            except (TypeError, ValueError):
                duration_hours = None
        user_wanted_next_date = 1
        if want_col and pd.notna(row.get(want_col)):
            try:
                user_wanted_next_date = 1 if int(row[want_col]) else 0
            except (TypeError, ValueError):
                user_wanted_next_date = 1
        try:
            sb.add_date_event(
                client,
                user_id,
                person_id=pid,
                occurred_on=occurred_on,
                activity=activity,
                notes=notes,
                rating=rating,
                physical_escalation=phys,
                outing_type=outing_type,
                company_type=company_type,
                thank_you=thank_you,
                cost=cost,
                is_planned=is_planned,
                scheduled_at=scheduled_at,
                initiator=initiator,
                duration_hours=duration_hours,
                user_wanted_next_date=user_wanted_next_date,
            )
            dates_created += 1
        except Exception as e:
            warnings.append(f"Row for {pname} on {occurred_on}: {e}")
    return people_created, dates_created, warnings


def render_import_csv_section(client: Client, user_id: str) -> None:
    st.markdown("#### Import data")
    st.caption(
        "Upload a CSV in the same format as **dating_tracker_export.csv** (export from above, or match its columns). "
        "People are matched or created by **person_name**; existing names get new outing rows only."
    )
    up = st.file_uploader("CSV file", type=["csv"], key="import_csv_upload")
    if st.button("Import CSV", key="import_csv_run"):
        if up is None:
            st.warning("Choose a CSV file first.")
        else:
            pc, dc, warns = run_csv_import(client, user_id, up.getvalue())
            if pc == 0 and dc == 0 and warns:
                for w in warns[:12]:
                    st.error(w)
                if len(warns) > 12:
                    st.caption(f"… and {len(warns) - 12} more messages.")
            else:
                st.success(f"Created **{pc}** new people and **{dc}** outings.")
                if warns:
                    with st.expander("Import notices", expanded=False):
                        for w in warns[:50]:
                            st.caption(w)
                        if len(warns) > 50:
                            st.caption(f"… {len(warns) - 50} more.")


def _metric_is_categorical(m: str) -> bool:
    return m in ("Company type", "Outing type", "Met via")


def render_metric_comparison_chart(df: pd.DataFrame, m1: str, m2: str) -> None:
    if m1 == m2:
        st.warning("Pick two different metrics.")
        return
    d = df.copy()
    d["occurred_on"] = pd.to_datetime(d["occurred_on"], errors="coerce")
    d = d.dropna(subset=["occurred_on"])
    pe = d["physical_escalation"].map(normalize_physical_milestone)
    d["kissed"] = (pe == PHYSICAL_KISSED).astype(int)
    d["co_lbl"] = d["company_type"].map(_company_display_label)
    d["ot_lbl"] = d["outing_type"].map(_outing_display_label)
    d["mv_lbl"] = d["met_via"].map(
        lambda x: normalize_initial_met_via(x)
        if x is not None and str(x).strip()
        else MET_VIA_ORGANIC
    )

    def series_for(metric: str) -> tuple[str, pd.Series]:
        if metric == "Cost ($)":
            return "x", pd.to_numeric(d["cost"], errors="coerce").fillna(0)
        if metric == "Rating (1-10)":
            return "y", pd.to_numeric(d["rating"], errors="coerce")
        if metric == "Kiss % (outing)":
            return "k", d["kissed"] * 100.0
        if metric == "Company type":
            return "c", d["co_lbl"]
        if metric == "Outing type":
            return "o", d["ot_lbl"]
        if metric == "Met via":
            return "m", d["mv_lbl"]
        return "", pd.Series(dtype=float)

    _n1, s1 = series_for(m1)
    _n2, s2 = series_for(m2)
    if len(d) == 0:
        st.caption("Not enough data.")
        return

    if _metric_is_categorical(m1) and _metric_is_categorical(m2):
        st.info("Pick one categorical metric (Company, Outing type, Met via) and one numeric metric.")
        return

    if _metric_is_categorical(m1) and not _metric_is_categorical(m2):
        tmp = pd.DataFrame({"cat": s1, "y": s2})
        plot = tmp.groupby("cat", observed=False)["y"].mean().reset_index()
        plot.columns = ["cat", "val"]
        fig = px.bar(
            plot,
            x="cat",
            y="val",
            template="plotly_dark",
            labels={"cat": m1, "val": m2},
        )
        fig.update_traces(marker_color=COLOR_HEURISTIC)
        fig.update_layout(
            height=320,
            xaxis_title=m1,
            yaxis_title=m2,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    if not _metric_is_categorical(m1) and _metric_is_categorical(m2):
        tmp = pd.DataFrame({"cat": s2, "x": s1})
        plot = tmp.groupby("cat", observed=False)["x"].mean().reset_index()
        plot.columns = ["cat", "val"]
        fig = px.bar(
            plot,
            x="cat",
            y="val",
            template="plotly_dark",
            labels={"cat": m2, "val": m1},
        )
        fig.update_traces(marker_color=COLOR_ML)
        fig.update_layout(
            height=320,
            xaxis_title=m2,
            yaxis_title=m1,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    plot_df = pd.DataFrame({"x": s1.astype(float), "y": s2.astype(float)}).dropna()
    if plot_df.empty:
        st.caption("Not enough overlapping numeric points.")
        return
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        template="plotly_dark",
        labels={"x": m1, "y": m2},
    )
    fig.update_traces(marker=dict(size=11, color=COLOR_HEURISTIC, opacity=0.85))
    fig.update_layout(
        height=340,
        xaxis_title=m1,
        yaxis_title=m2,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_tab_analytics(client: Client, user_id: str) -> None:
    st.subheader("Analytics")
    c_exp, c_ml = st.columns(2)
    with c_exp:
        render_export_csv_expander(client, user_id, use_expander=False)
        render_import_csv_section(client, user_id)
    with c_ml:
        st.markdown("#### Retrain ML engine")
        st.caption(
            "Uses only outings where you **wanted to see her again**. Target = another date happened vs you were still interested but it ended."
        )
        if st.button("Retrain ML Engine with Latest Data", key="ml_retrain_btn"):
            ok, msg = retrain_ml_model(client, user_id, n_bins=CALIBRATION_N_BINS, strategy=CALIBRATION_STRATEGY)
            if ok:
                st.toast("ML engine retrained.", icon="🤖")
                st.success(msg)
            else:
                st.warning(msg)

    _ml_cfg = _load_ml_config_dict(client, user_id)
    render_ml_diagnostics_expander(_ml_cfg)
    render_heuristic_diagnostics_expander(_ml_cfg)

    df = sb.load_completed_analytics_df(client, user_id)
    if df.empty:
        st.info("Log **completed** outings to see charts below.")
        return
    d = df.copy()
    d["occurred_on"] = pd.to_datetime(d["occurred_on"], errors="coerce")
    d = d.dropna(subset=["occurred_on"])
    d_rom = d[d["outing_type"].map(_normalize_outing_type) == OUTING_TYPE_DATE].copy()

    gran = st.radio(
        "Time grouping",
        ["Weekly", "Monthly", "Yearly"],
        horizontal=True,
        key="analytics_time_group",
    )
    freq_map = {"Weekly": "W-MON", "Monthly": "MS", "Yearly": "YS"}
    fk = freq_map[gran]
    g = d_rom.groupby(pd.Grouper(key="occurred_on", freq=fk))
    cnt = g.size().reset_index(name="count")
    cnt = cnt.rename(columns={"occurred_on": "period"})
    st.markdown("#### Dates over time")
    st.caption("Only intentional **Date** outings are counted; casual hangouts are excluded (same as Dashboard KPIs).")
    if cnt.empty or cnt["count"].sum() == 0:
        st.caption("No date outings in this range.")
    else:
        fig = px.line(
            cnt,
            x="period",
            y="count",
            markers=True,
            template="plotly_dark",
            labels={"period": gran, "count": "Dates"},
        )
        fig.update_traces(
            line=dict(color=COLOR_HEURISTIC, width=3), marker=dict(size=8)
        )
        fig.update_layout(
            height=340,
            xaxis_title=gran,
            yaxis_title="Dates",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Compare two metrics")
    opts = [
        "Cost ($)",
        "Rating (1-10)",
        "Kiss % (outing)",
        "Company type",
        "Outing type",
        "Met via",
    ]
    c1, c2 = st.columns(2)
    with c1:
        m1 = st.selectbox("First metric", opts, index=0, key="cmp_m1")
    with c2:
        m2 = st.selectbox("Second metric", opts, index=1, key="cmp_m2")
    render_metric_comparison_chart(df, m1, m2)


def render_dashboard_pipeline_log_form(
    client: Client, user_id: str,
    person_id: int,
    people_df: pd.DataFrame,
    *,
    key_prefix: str,
) -> None:
    """Streamlined log form for the pipeline action center (intentional dates only)."""
    row = people_df[people_df["id"].astype(int) == int(person_id)]
    if row.empty:
        st.warning("Person not found.")
        return
    name = str(row.iloc[0]["name"])
    dates_df = sb.load_dates_for_person(client, user_id, int(person_id))
    n_for_ty = len(filter_dates_for_potential(dates_df))
    lo = st.session_state.log_outing_seq
    st.markdown("#### Log a new date")
    with st.form(f"{key_prefix}_dashlog_{person_id}_{lo}"):
        st.caption(f"With **{name}**")
        when = st.date_input(
            "Date of outing",
            value=date.today(),
            key=f"{key_prefix}_{lo}_when",
        )
        activity = st.text_input(
            "Activity",
            placeholder="Dinner, hike, museum…",
            key=f"{key_prefix}_{lo}_act",
        )
        log_dur = st.number_input(
            "Duration (hours)",
            min_value=0.0,
            max_value=336.0,
            value=None,
            step=0.25,
            key=f"{key_prefix}_{lo}_dur",
        )
        log_cost = st.number_input(
            "Cost ($)",
            min_value=0.0,
            value=0.0,
            step=5.0,
            format="%.2f",
            key=f"{key_prefix}_{lo}_cost",
        )
        log_phys = st.selectbox(
            "Highest physical milestone",
            PHYSICAL_MILESTONE_OPTIONS,
            key=f"{key_prefix}_{lo}_phys",
        )
        log_init = st.selectbox(
            "Initiator",
            INITIATOR_OPTIONS,
            key=f"{key_prefix}_{lo}_init",
        )
        log_co = st.radio(
            "Company type",
            COMPANY_TYPES,
            format_func=lambda x: COMPANY_LABELS[x],
            horizontal=True,
            key=f"{key_prefix}_{lo}_co",
        )
        log_ty_sel: str | None = None
        if n_for_ty < 4:
            log_ty_sel = st.selectbox(
                "Did she send a thank-you text?",
                THANK_YOU_CHOICES,
                key=f"{key_prefix}_{lo}_ty",
            )
        dash_want_next = st.checkbox(
            "Did you want to see her again?",
            value=True,
            key=f"{key_prefix}_{lo}_want_next",
        )
        if st.form_submit_button("Save date"):
            ty_db = _thank_you_to_db(log_ty_sel) if log_ty_sel is not None else None
            sb.add_date_event(
                client,
                user_id,
                person_id=int(person_id),
                occurred_on=when,
                activity=activity,
                notes=None,
                rating=None,
                physical_escalation=log_phys,
                outing_type=OUTING_TYPE_DATE,
                company_type=log_co,
                thank_you=ty_db,
                cost=float(log_cost),
                initiator=log_init,
                duration_hours=None if log_dur is None else float(log_dur),
                user_wanted_next_date=1 if dash_want_next else 0,
            )
            st.session_state.log_outing_seq += 1
            st.success("Date saved.")
            st.rerun()


def render_pipeline_action_center(
    client: Client, user_id: str,
    people_df: pd.DataFrame,
    pid: int,
) -> None:
    row = people_df[people_df["id"].astype(int) == int(pid)]
    if row.empty:
        st.error("Person not found.")
        return
    r = row.iloc[0]
    dates_df = sb.load_dates_for_person(client, user_id, int(pid))
    bd = potential_score_breakdown(dates_df)
    c_img, c_txt = st.columns([0.32, 1])
    with c_img:
        render_profile_avatar(r.get("profile_image"), width=76)
    with c_txt:
        st.markdown(f"### {r['name']}")
        st.caption(str(r["status"]))
    ml_p = calculate_ml_probability(client, user_id, dates_df)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total dates", int(bd["n_dates"]))
    with m2:
        st.metric("Base score", f"{round(float(bd['base_score']))}")
    with m3:
        st.metric("Heuristic", f"{int(bd['score'])}%")
    with m4:
        st.metric(
            "ML probability",
            f"{ml_p}%" if ml_p is not None else "—",
        )
    st.caption(
        "**Heuristic** = tuned rules (half-life ratings, physical gap, trajectory, duration, thank-you), "
        f"then **momentum** (× decay by days since last date, **{MOMENTUM_HALF_LIFE_DAYS:.0f}**-day half-life). "
        "**ML probability** = logistic model on your history (retrain under **Analytics**)."
    )
    if int(bd["n_dates"]) > 0:
        st.caption(
            f"Pre-momentum peak: **{round(float(bd['max_potential']))}** · "
            f"Momentum **×{float(bd['momentum_factor']):.2f}**"
        )
    render_dashboard_pipeline_log_form(
        client,
        user_id, int(pid), people_df, key_prefix=f"pac_{pid}"
    )


@st.dialog("Action center — pipeline")
def pipeline_action_dialog(client: Client, user_id: str) -> None:
    pid = st.session_state.get("pipeline_ac_pid")
    if pid is None:
        return
    pdf = sb.load_people_df(client, user_id)
    render_pipeline_action_center(client, user_id, pdf, int(pid))
    if st.button("Close", key="pipeline_ac_close_btn"):
        st.session_state.pop("pipeline_ac_pid", None)
        st.rerun()


def render_roster_profile_panel(
    client: Client, user_id: str,
    people_df: pd.DataFrame,
    pid: int,
) -> None:
    """Profile details + past outings + log form (Dashboard roster). Uses open DB connection."""
    row = people_df[people_df["id"].astype(int) == int(pid)]
    if row.empty:
        st.error("Person not found.")
        if st.button("Dismiss", key="dash_prof_dismiss_missing"):
            st.session_state.pop("dash_roster_open_pid", None)
            st.rerun()
        return
    r = row.iloc[0]
    stats = sb.load_person_pipeline_stats(client, user_id)
    mrow = merge_pipeline_stats(pd.DataFrame([r]), stats).iloc[0]
    nd = int(mrow.get("n_romantic_dates") or 0)
    ts = float(mrow.get("total_spent") or 0)
    last = mrow.get("last_occurred")
    if nd >= VIP_MIN_ROMANTIC_DATES:
        st.markdown(
            '<span style="display:inline-block;background:linear-gradient(90deg,#5c4a1a,#8a7320);'
            "color:#fff7d6;padding:0.15rem 0.5rem;border-radius:6px;font-size:0.78rem;font-weight:600;"
            f'letter-spacing:0.03em;">VIP · {VIP_MIN_ROMANTIC_DATES}+ dates</span>',
            unsafe_allow_html=True,
        )
    c_img, c_txt = st.columns([0.35, 1])
    with c_img:
        render_profile_avatar(r.get("profile_image"), width=88)
    with c_txt:
        st.markdown(f"### {r['name']}")
        st.caption(str(r["status"]))
    if r.get("initial_met_via") and str(r["initial_met_via"]).strip():
        st.caption(f"Initially met via: **{r['initial_met_via']}**")
    st.caption(f"Dates logged: **{nd}** · Total spent: **${ts:,.2f}**")
    if pd.notna(last):
        st.caption(f"Last outing: **{pd.Timestamp(last).strftime('%Y-%m-%d')}**")

    dates_sub = sb.load_dates_for_person(client, user_id, int(pid))
    if "is_planned" in dates_sub.columns:
        completed = dates_sub[dates_sub["is_planned"].fillna(0).astype(int) == 0]
    else:
        completed = dates_sub
    st.markdown("#### Past outings")
    if completed.empty:
        st.caption("No completed outings yet.")
    else:
        st.dataframe(
            _outings_dataframe_for_display(completed),
            hide_index=True,
            use_container_width=True,
        )

    render_log_outing_form_fixed_person(
        conn, int(pid), people_df, key_prefix=f"dash_log_{pid}"
    )


def render_tab_dashboard(client: Client, user_id: str, people_df: pd.DataFrame) -> None:
    st.subheader("Dashboard")
    t, mo, yr = sb.kpi_date_counts(client, user_id)
    k1, k2, k3 = st.columns(3)
    _kpi_help = (
        "Counts outings logged as **Date** (intentional dates), not casual hangouts. "
        "Planned-but-not-happened rows are excluded."
    )
    with k1:
        st.metric("Total lifetime dates", t, help=_kpi_help)
    with k2:
        st.metric("Dates this month", mo, help=_kpi_help)
    with k3:
        st.metric("Dates this year", yr, help=_kpi_help)

    active_td = people_df[people_df["status"].isin(ACTIVE_PIPELINE_TD)]
    scores_by_id: dict[int, int] = {}
    ml_by_id: dict[int, int | None] = {}
    for _, pr in active_td.iterrows():
        pid_i = int(pr["id"])
        df_d = sb.load_dates_for_person(client, user_id, pid_i)
        scores_by_id[pid_i] = calculate_potential(pr, df_d)
        ml_by_id[pid_i] = calculate_ml_probability(client, user_id, df_d)

    ensure_default_pipeline_slots(client, user_id, people_df, scores_by_id)
    people_df = sb.load_people_df(client, user_id)

    st.markdown("#### The Pipeline (Top 5)")
    st.caption(
        "Order follows saved **roster slots**, then **potential** for new fills. "
        "Use **↑** / **↓** on a card to swap with the neighbor above or below (saved for **Talking** / **Dating**)."
    )
    slot_pids = get_pipeline_top5_slot_person_ids(people_df, scores_by_id)

    for i, pid_slot in enumerate(slot_pids):
        rank = i + 1
        with st.container(border=True):
            if pid_slot is None:
                st.markdown(
                    f'<p style="margin:0.15rem 0 0.35rem 0;color:#9ca3af;font-size:0.85rem;">Slot <b>#{rank}</b></p>',
                    unsafe_allow_html=True,
                )
                st.caption(PIPELINE_EMPTY_SLOT_LABEL)
                continue

            pr = people_df[people_df["id"].astype(int) == int(pid_slot)]
            if pr.empty:
                st.caption(f"#{rank} — Person not found.")
                continue
            row_p = pr.iloc[0]
            nm = str(row_p["name"])
            st_p = str(row_p["status"])
            h_sc = scores_by_id.get(int(pid_slot), 0)
            ml_sc = ml_by_id.get(int(pid_slot))
            warn = (
                " ⚠️"
                if ml_sc is not None and abs(int(h_sc) - int(ml_sc)) > 20
                else ""
            )
            head = st.columns([0.14, 1])
            with head[0]:
                st.markdown(f"**#{rank}**")
                render_profile_avatar(row_p.get("profile_image"), width=52)
            body = st.columns([1.15, 1])
            with body[0]:
                st.markdown(f"### {nm}{warn}")
                st.caption(st_p)
                if ml_sc is None:
                    st.caption("🤖 ML — retrain under **Analytics**")
            with body[1]:
                sc_cols = st.columns(2)
                with sc_cols[0]:
                    st.markdown(
                        f'<p style="margin:0;font-size:0.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:0.06em;">Heuristic</p>'
                        f'<p style="margin:0.1rem 0 0 0;font-size:2.1rem;font-weight:700;color:{COLOR_HEURISTIC};line-height:1.1;">{int(h_sc)}<span style="font-size:1rem;opacity:0.85;">%</span></p>',
                        unsafe_allow_html=True,
                    )
                with sc_cols[1]:
                    ml_disp = f"{int(ml_sc)}" if ml_sc is not None else "—"
                    ml_suffix = (
                        '<span style="font-size:1rem;opacity:0.85;">%</span>'
                        if ml_sc is not None
                        else ""
                    )
                    st.markdown(
                        f'<p style="margin:0;font-size:0.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:0.06em;">ML</p>'
                        f'<p style="margin:0.1rem 0 0 0;font-size:2.1rem;font-weight:700;color:{COLOR_ML};line-height:1.1;">{ml_disp}{ml_suffix}</p>',
                        unsafe_allow_html=True,
                    )
            row_btn = st.columns([0.12, 0.12, 1])
            with row_btn[0]:
                if st.button(
                    "↑",
                    key=f"pipe_up_{rank}_{pid_slot}",
                    disabled=rank <= 1,
                    help="Swap with slot above",
                ):
                    swap_pipeline_ranks(
                        client,
                        user_id, people_df, scores_by_id, rank, rank - 1
                    )
                    st.rerun()
            with row_btn[1]:
                if st.button(
                    "↓",
                    key=f"pipe_down_{rank}_{pid_slot}",
                    disabled=rank >= 5,
                    help="Swap with slot below",
                ):
                    swap_pipeline_ranks(
                        client,
                        user_id, people_df, scores_by_id, rank, rank + 1
                    )
                    st.rerun()
            with row_btn[2]:
                if st.button(
                    "Open details",
                    key=f"pipe_ac_{i}_{pid_slot}",
                    use_container_width=True,
                ):
                    st.session_state.pipeline_ac_pid = int(pid_slot)
                    st.rerun()

    st.divider()
    st.markdown("#### Prospects & Leads")
    st.caption("Promote a lead to **Talking** to pull her into the pipeline above.")
    leads_df = people_df[people_df["status"] == STATUS_LEADS].copy()
    if leads_df.empty:
        st.info("No **Leads** right now.")
    else:
        leads_df = leads_df.copy()
        leads_df["_sn"] = leads_df["name"].astype(str).str.lower()
        leads_df = leads_df.sort_values("_sn").drop(columns=["_sn"])
        n = len(leads_df)
        ncols = min(3, max(1, n))
        for r0 in range(0, n, ncols):
            cols = st.columns(ncols)
            for j in range(ncols):
                ix = r0 + j
                if ix >= n:
                    break
                lr = leads_df.iloc[ix]
                lid = int(lr["id"])
                with cols[j]:
                    with st.container(border=True):
                        c0, c1 = st.columns([0.22, 1])
                        with c0:
                            render_profile_avatar(lr.get("profile_image"), width=52)
                        with c1:
                            st.markdown(f"**{lr['name']}**")
                            if lr.get("initial_met_via"):
                                st.caption(f"Met via: {lr['initial_met_via']}")
                            if st.button(
                                "Promote to Active",
                                key=f"promote_lead_{lid}",
                            ):
                                sb.update_person_status(client, user_id, lid, "Talking")
                                st.rerun()


def render_tab_people(client: Client, user_id: str, people_df: pd.DataFrame) -> None:
    """Searchable roster + detail: profile, edit/delete person, outings, log new outing (any status)."""
    st.subheader("People")
    st.caption(
        "Search everyone in your database. Open someone to edit her record, review outings, or "
        "**log a new outing** — including new leads who are not on the Dashboard pipeline yet."
    )

    with st.container(border=True):
        st.subheader("Add someone")
        ap_seq = st.session_state.add_person_seq
        with st.form(f"add_person_{ap_seq}"):
            c1, c2, c3 = st.columns([2, 1, 2])
            with c1:
                new_name = st.text_input(
                    "Name",
                    placeholder="Alex",
                    key=f"ap_{ap_seq}_name",
                )
            with c2:
                new_status = st.selectbox(
                    "Status",
                    list(STATUSES),
                    key=f"ap_{ap_seq}_status",
                    help="**Leads** = met, not asked out yet. **Friend** = platonic. **Archived** = past only.",
                )
            with c3:
                new_met = st.selectbox(
                    "Initially met via",
                    MET_VIA_OPTIONS,
                    key=f"ap_{ap_seq}_met",
                )
            st.caption(
                "**Leads** = you’ve met her but haven’t asked her out / no outings logged yet."
            )
            ap_prof_url = st.text_input(
                "Profile image URL (optional)",
                placeholder="https://…",
                key=f"ap_{ap_seq}_purl",
            )
            ap_prof_file = st.file_uploader(
                "Or upload a profile photo",
                type=["jpg", "jpeg", "png", "webp", "gif", "heic"],
                key=f"ap_{ap_seq}_pfile",
            )
            if st.form_submit_button("Add to database"):
                nn = (new_name or "").strip()
                if not nn:
                    st.error("Name is required.")
                else:
                    pid_add = sb.add_person(
                        client,
                        user_id,
                        name=nn,
                        status=new_status,
                        initial_met_via=new_met,
                        profile_image=None,
                    )
                    prof: str | None = None
                    if ap_prof_file is not None:
                        prof = save_profile_photo_upload(pid_add, ap_prof_file)
                    elif (ap_prof_url or "").strip():
                        u = (ap_prof_url or "").strip()
                        if u.lower().startswith(("http://", "https://")):
                            prof = u
                        else:
                            st.warning(
                                "Image URL should start with http:// or https://"
                            )
                    if prof:
                        sb.update_person(
                            client,
                            user_id,
                            pid_add,
                            name=nn,
                            status=new_status,
                            initial_met_via=new_met,
                            profile_image=prof,
                        )
                    st.session_state.add_person_seq += 1
                    st.session_state[PEOPLE_TAB_SELECTED_PID_KEY] = int(pid_add)
                    st.session_state.pop("pipeline_ac_pid", None)
                    st.success("Added.")
                    st.rerun()

    st.divider()

    if people_df.empty:
        st.info("No people in your database yet — use **Add someone** above.")
        return

    c_list, c_detail = st.columns([1, 2], gap="large")

    with c_list:
        st.markdown("#### Directory")
        q = st.text_input(
            "Search",
            "",
            placeholder="Name or status…",
            key="people_dir_search",
        )
        roster = people_df.copy()
        roster["_sn"] = roster["name"].astype(str).str.lower()
        qstrip = (q or "").strip()
        if qstrip:
            qlow = qstrip.lower()
            esc = re.escape(qlow)
            m_name = roster["name"].astype(str).str.lower().str.contains(esc, na=False, regex=True)
            m_stat = roster["status"].astype(str).str.lower().str.contains(esc, na=False, regex=True)
            roster = roster[m_name | m_stat]
        roster = roster.sort_values("_sn").drop(columns=["_sn"])

        st.caption(f"{len(roster)} shown · click a name to open the profile panel →")

        cur_sel = st.session_state.get(PEOPLE_TAB_SELECTED_PID_KEY)

        if cur_sel is not None:
            if st.button("← Clear selection", key="people_clear_selection"):
                st.session_state[PEOPLE_TAB_SELECTED_PID_KEY] = None
                st.session_state.pop("pipeline_ac_pid", None)
                st.rerun()

        if roster.empty:
            st.info("No matches — try a shorter search.")
        else:
            for _, row in roster.iterrows():
                pid = int(row["id"])
                sel = cur_sel == pid
                btn_label = f"{row['name']} · {row['status']}"
                if sel:
                    btn_label = f"◆ {btn_label}"
                if st.button(
                    btn_label,
                    key=f"people_pick_{pid}",
                    use_container_width=True,
                    type="primary" if sel else "secondary",
                ):
                    st.session_state[PEOPLE_TAB_SELECTED_PID_KEY] = pid
                    st.session_state.pop("pipeline_ac_pid", None)
                    st.rerun()

    with c_detail:
        pid = st.session_state.get(PEOPLE_TAB_SELECTED_PID_KEY)
        if pid is None:
            st.info(
                "Select someone in the directory to view her profile, edit details, and log or edit outings."
            )
            return
        pipe_pid = st.session_state.get("pipeline_ac_pid")
        if pipe_pid is not None and int(pipe_pid) != int(pid):
            st.session_state.pop("pipeline_ac_pid", None)
            st.rerun()
        sub = people_df[people_df["id"].astype(int) == int(pid)]
        if sub.empty:
            st.session_state[PEOPLE_TAB_SELECTED_PID_KEY] = None
            st.warning("That person is no longer in the database.")
            st.rerun()

        st.markdown("#### Profile & outings")
        render_history_profile_readonly(client, user_id, people_df, int(pid))
        render_history_person_edit_expander(
            client,
            user_id,
            people_df,
            int(pid),
            key_prefix="people",
            clear_session_pid_key=PEOPLE_TAB_SELECTED_PID_KEY,
        )
        render_history_outings_expanders(client, user_id, people_df, int(pid), key_prefix="people")
        st.divider()
        render_log_outing_form_fixed_person(
            client,
            user_id,
            int(pid),
            people_df,
            key_prefix=f"people_log_{pid}",
        )


def main() -> None:
    st.set_page_config(page_title="Dating tracker", page_icon="📊", layout="wide")
    st.markdown(
        """
        <style>
            h1 { letter-spacing: -0.02em; margin-bottom: 0.25rem; }
            [data-testid="stTabs"] { margin-top: 0.5rem; }
            [data-testid="stRadio"] > div { flex-wrap: wrap; gap: 0.35rem; margin-top: 0.35rem; }
            div[data-testid="stVerticalBlockBorderWrapper"] > div {
                border-radius: 12px;
            }
            #MainMenu { visibility: hidden !important; }
            footer { visibility: hidden !important; }
            [data-testid="stToolbar"] { visibility: hidden !important; }
            [data-testid="stDecoration"] { display: none !important; }
            div[data-testid="stToolbarActions"] { display: none !important; }
            .stButton > button {
                border-radius: 10px !important;
                transition: transform 0.15s ease, box-shadow 0.2s ease, filter 0.15s ease !important;
            }
            .stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 20px rgba(0, 232, 168, 0.22) !important;
                filter: brightness(1.06);
            }
            .stDownloadButton > button {
                border-radius: 10px !important;
                transition: transform 0.15s ease, box-shadow 0.2s ease !important;
            }
            .stDownloadButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 18px rgba(56, 189, 248, 0.2) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not _supabase_secrets_ok():
        _render_supabase_missing_help()
        return

    client = _supabase_client()
    if not st.session_state.get(SESSION_ACCESS_TOKEN_KEY):
        render_auth_screen()
        return

    user_id = _require_authenticated_user(client)

    with st.sidebar:
        st.caption("Account")
        try:
            gu = client.auth.get_user()
            em = getattr(gu.user, "email", None) if gu and gu.user else None
        except Exception:
            em = None
        st.caption(em or f"User `{user_id[:8]}…`")
        if st.button("Log out", key="sidebar_logout"):
            try:
                client.auth.sign_out()
            except Exception:
                pass
            for k in (
                SESSION_USER_ID_KEY,
                SESSION_ACCESS_TOKEN_KEY,
                SESSION_REFRESH_TOKEN_KEY,
            ):
                st.session_state.pop(k, None)
            st.rerun()

    st.title("Dating tracker")

    if "log_outing_seq" not in st.session_state:
        st.session_state.log_outing_seq = 0
    if "add_person_seq" not in st.session_state:
        st.session_state.add_person_seq = 0
    if PEOPLE_TAB_SELECTED_PID_KEY not in st.session_state:
        st.session_state[PEOPLE_TAB_SELECTED_PID_KEY] = None

    prev_nav = st.session_state.get(APP_NAV_PREV_KEY)
    st.radio(
        "View",
        list(APP_NAV_SECTIONS),
        horizontal=True,
        key=APP_NAV_SECTION_KEY,
    )
    section = str(st.session_state[APP_NAV_SECTION_KEY])
    if prev_nav is not None:
        if prev_nav == "People" and section != "People":
            st.session_state.pop(PEOPLE_TAB_SELECTED_PID_KEY, None)
            st.session_state.pop("pipeline_ac_pid", None)
        elif prev_nav == "Dashboard" and section != "Dashboard":
            st.session_state.pop("pipeline_ac_pid", None)
    st.session_state[APP_NAV_PREV_KEY] = section

    people_df = sb.load_people_df(client, user_id)
    sb.ensure_default_roster_slots(client, user_id)
    people_df = sb.load_people_df(client, user_id)

    if section == "Dashboard":
        render_tab_dashboard(client, user_id, people_df)
    elif section == "People":
        render_tab_people(client, user_id, people_df)
    else:
        render_tab_analytics(client, user_id)

    if st.session_state.get("pipeline_ac_pid") is not None:
        pipeline_action_dialog(client, user_id)

    st.caption("Data stored in **Supabase** (per-user isolation).")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# GA4 Streamlit Dashboard — single file, secrets-only
# Requires Streamlit Secrets:
#   GA4_PROPERTY_ID="..."
#   [gcp_service_account] ... (service account fields)
# Optional:
#   APP_PASSWORD="..."          (simple password gate)
#   DASH_LOGO="assets/logo.svg"
#   SIDEBAR_LOGO="assets/internews.svg"

import os
import io
from pathlib import Path
from datetime import date, timedelta
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import numpy as np
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest, Dimension, Metric, Filter, FilterExpression,
    FilterExpressionList, OrderBy
)

# ──────────────────────────────────────────────────────────────────────────────
# Config / Styling
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Analytics Console", layout="wide")

# premium-ish UI like your first version
st.markdown("""
<style>
.main { background-color: #f8fafc; }
.stButton>button {
  width: 100%;
  border-radius: 12px;
  font-weight: 700;
  background-color: #0f172a;
  color: white;
  border: none;
  padding: 0.6rem;
  transition: all 0.2s;
}
.stButton>button:hover {
  background-color: #000000;
  color: white;
  transform: translateY(-1px);
}
div[data-testid="stExpander"] {
  border-radius: 15px;
  border: 1px solid #e2e8f0;
  background-color: white;
}
</style>
""", unsafe_allow_html=True)

DASH_LOGO = st.secrets.get("DASH_LOGO", os.getenv("DASH_LOGO", "assets/logo.svg"))
SIDEBAR_LOGO = st.secrets.get("SIDEBAR_LOGO", os.getenv("SIDEBAR_LOGO", "assets/internews.svg"))

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def fail_ui(msg: str):
    st.error(msg)
    st.stop()

def password_gate():
    app_pwd = str(st.secrets.get("APP_PASSWORD", "")).strip()
    if not app_pwd:
        return
    if st.session_state.get("authed"):
        return
    st.title("Вход")
    pwd = st.text_input("Пароль", type="password")
    if pwd and pwd == app_pwd:
        st.session_state["authed"] = True
        st.rerun()
    st.stop()

def render_logo(path: str, width: int | None = None):
    p = Path(path)
    if not p.exists():
        return
    ext = p.suffix.lower()
    if ext == ".svg":
        try:
            from urllib.parse import quote
            svg_txt = p.read_text(encoding="utf-8")
            data_uri = f"data:image/svg+xml;utf8,{quote(svg_txt)}"
            w_attr = f' style="width:{width}px;"' if width else ""
            st.markdown(f'<img src="{data_uri}"{w_attr}>', unsafe_allow_html=True)
        except Exception:
            pass
    else:
        st.image(str(p), use_column_width=(width is None), width=width)

INVISIBLE = ("\ufeff", "\u200b", "\u2060", "\u00a0")

def clean_line(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    for ch in INVISIBLE:
        s = s.replace(ch, "")
    return s.strip()

def strip_utm_and_fragment(raw_url: str) -> str:
    # remove utm_* query params and fragments; keep other query params
    p = urlparse(raw_url)
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q, doseq=True), ""))

def looks_like_domain_no_scheme(s: str) -> bool:
    # "www.domain.tld/..." or "domain.tld/..." without scheme
    s = s.strip()
    if not s or s.startswith("/"):
        return False
    # cheap heuristics: has dot in first token before slash and no spaces
    head = s.split("/")[0]
    return (" " not in s) and ("." in head) and (":" not in head)

def normalize_any_input_to_path_and_host(raw: str) -> tuple[str, str | None]:
    """
    Accepts:
      - https://domain/path...
      - http://domain/path...
      - www.domain/path...
      - domain/path...
      - /path...
      - path...
    Returns:
      path (always starting with "/"), host (if detected), both cleaned.
    """
    s = clean_line(raw)
    if not s:
        return "", None

    # If no scheme but looks like domain -> add https:// for parsing
    if looks_like_domain_no_scheme(s):
        s = "https://" + s

    if s.lower().startswith(("http://", "https://")):
        s2 = strip_utm_and_fragment(s)
        p = urlparse(s2)
        path = p.path or "/"
        host = p.hostname or None
        if not path.startswith("/"):
            path = "/" + path
        return path, host

    # Not a URL; treat as path
    if not s.startswith("/"):
        s = "/" + s
    return s, None

def collect_paths_hosts(raw_list: list[str]) -> tuple[list[str], list[str], list[str]]:
    """
    Returns:
      unique_paths: unique paths (order of first appearance)
      hosts: sorted unique hosts (if detected)
      order_list: paths in original order (duplicates kept)
    """
    seen = set()
    unique_paths = []
    hosts = set()
    order_list = []
    for raw in raw_list:
        path, host = normalize_any_input_to_path_and_host(raw)
        if not path:
            continue
        order_list.append(path)
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
        if host:
            hosts.add(host)
    return unique_paths, sorted(hosts), order_list

def read_uploaded_lines(uploaded) -> list[str]:
    if uploaded is None:
        return []
    try:
        if uploaded.name.lower().endswith(".txt"):
            # bytes -> lines
            return [clean_line(b.decode("utf-8", errors="ignore")) for b in uploaded.readlines()]
        # csv
        dfu = pd.read_csv(uploaded, header=None)
        return [clean_line(x) for x in dfu.iloc[:, 0].tolist()]
    except Exception:
        return []

# ──────────────────────────────────────────────────────────────────────────────
# GA4 client
# ──────────────────────────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/analytics.readonly"]

@st.cache_resource
def ga_client() -> BetaAnalyticsDataClient:
    sa = st.secrets.get("gcp_service_account")
    if not sa:
        fail_ui("Missing Streamlit Secret: gcp_service_account")
    creds = service_account.Credentials.from_service_account_info(dict(sa), scopes=SCOPES)
    return BetaAnalyticsDataClient(credentials=creds)

def default_property_id() -> str:
    pid = str(st.secrets.get("GA4_PROPERTY_ID", "")).strip()
    if not pid:
        fail_ui("Missing Streamlit Secret: GA4_PROPERTY_ID")
    return pid

# ──────────────────────────────────────────────────────────────────────────────
# GA4 queries
# ──────────────────────────────────────────────────────────────────────────────
# We keep only safe/compatible metrics for pagePath report
METRICS_PAGE = ["screenPageViews", "activeUsers", "userEngagementDuration"]

def make_path_filter(paths_batch: list[str]) -> FilterExpression:
    # Default + reliable: BEGINS_WITH for each provided path
    exprs = [
        FilterExpression(
            filter=Filter(
                field_name="pagePath",
                string_filter=Filter.StringFilter(
                    value=pth,
                    match_type=Filter.StringFilter.MatchType.BEGINS_WITH,
                    case_sensitive=False,
                )
            )
        )
        for pth in paths_batch
    ]
    return FilterExpression(or_group=FilterExpressionList(expressions=exprs))

@st.cache_data(ttl=300, show_spinner=False)
def fetch_by_paths_cached(property_id: str, paths: tuple, hosts: tuple, start_date: str, end_date: str, order_keys: tuple) -> pd.DataFrame:
    return fetch_ga4_by_paths(property_id, list(paths), list(hosts), start_date, end_date, list(order_keys))

def fetch_ga4_by_paths(property_id: str, paths_in: list[str], hosts_in: list[str], start_date: str, end_date: str, order_keys: list[str]) -> pd.DataFrame:
    if not property_id:
        property_id = default_property_id()

    if not paths_in:
        return pd.DataFrame(columns=["pagePath", "pageTitle", "hostName"] + METRICS_PAGE)

    client = ga_client()
    rows, BATCH = [], 25

    for i in range(0, len(paths_in), BATCH):
        batch = paths_in[i:i+BATCH]
        base = make_path_filter(batch)
        dim_filter = base

        dims = [Dimension(name="pagePath"), Dimension(name="pageTitle")]

        # hostName filter is optional: only if we detected hosts from input
        if hosts_in:
            host_expr = FilterExpression(
                filter=Filter(
                    field_name="hostName",
                    in_list_filter=Filter.InListFilter(values=hosts_in[:50])
                )
            )
            dim_filter = FilterExpression(and_group=FilterExpressionList(expressions=[base, host_expr]))
            dims.append(Dimension(name="hostName"))

        req = RunReportRequest(
            property=f"properties/{property_id}",
            dimensions=dims,
            metrics=[Metric(name=m) for m in METRICS_PAGE],
            date_ranges=[{"start_date": start_date, "end_date": end_date}],
            dimension_filter=dim_filter,
            limit=100000,
        )

        resp = client.run_report(req)
        for r in resp.rows:
            rec = {}
            idx = 0
            rec["pagePath"] = r.dimension_values[idx].value; idx += 1
            rec["pageTitle"] = r.dimension_values[idx].value; idx += 1
            if hosts_in:
                rec["hostName"] = r.dimension_values[idx].value
            for j, m in enumerate(METRICS_PAGE):
                rec[m] = r.metric_values[j].value
            rows.append(rec)

    df = pd.DataFrame(rows)

    for m in METRICS_PAGE:
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)

    # aggregate by pagePath (same as your working code)
    agg = {m: "sum" for m in METRICS_PAGE}
    agg["pageTitle"] = "first"
    if "hostName" in df.columns:
        agg["hostName"] = "first"

    if not df.empty:
        df = df.groupby(["pagePath"], as_index=False).agg(agg)

    # fill missing with zeros (preserve order)
    present = set(df["pagePath"].tolist()) if not df.empty else set()
    missing_unique = [p for p in paths_in if p not in present]
    if missing_unique:
        base_zero = {
            "pagePath": None,
            "pageTitle": "",
            "screenPageViews": 0,
            "activeUsers": 0,
            "userEngagementDuration": 0,
        }
        zeros = pd.DataFrame([dict(base_zero, **{"pagePath": p}) for p in missing_unique])
        if "hostName" in df.columns:
            zeros["hostName"] = hosts_in[0] if hosts_in else ""
        df = pd.concat([df, zeros], ignore_index=True)

    df = df.set_index("pagePath").reindex(order_keys).reset_index()

    # derived metrics: Views / ActiveUser, Avg Engagement Time (s)
    den = pd.to_numeric(df["activeUsers"], errors="coerce").replace(0, np.nan).astype(float)
    df["viewsPerActiveUser"] = (pd.to_numeric(df["screenPageViews"], errors="coerce").astype(float) / den).fillna(0).round(2)
    df["avgEngagementTime_sec"] = (pd.to_numeric(df["userEngagementDuration"], errors="coerce").astype(float) / den).fillna(0).round(1)

    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_top_materials_cached(property_id: str, start_date: str, end_date: str, limit: int) -> pd.DataFrame:
    # keep it compatible: no averageEngagementTime metric here to avoid InvalidArgument
    client = ga_client()
    req = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="pagePath"), Dimension(name="pageTitle")],
        metrics=[Metric(name="screenPageViews"), Metric(name="activeUsers"), Metric(name="userEngagementDuration")],
        date_ranges=[{"start_date": start_date, "end_date": end_date}],
        order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"), desc=True)],
        limit=int(limit),
    )
    resp = client.run_report(req)
    rows = []
    for r in resp.rows:
        views = int(float(r.metric_values[0].value or 0))
        users = int(float(r.metric_values[1].value or 0))
        eng = float(r.metric_values[2].value or 0)
        rows.append({
            "Path": r.dimension_values[0].value,
            "Title": r.dimension_values[1].value,
            "Views": views,
            "Unique Users": users,
            "Avg Engagement Time (s)": round(eng / max(users, 1), 1),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_site_totals_cached(property_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    client = ga_client()
    req = RunReportRequest(
        property=f"properties/{property_id}",
        metrics=[Metric(name="sessions"), Metric(name="totalUsers"), Metric(name="screenPageViews")],
        date_ranges=[{"start_date": start_date, "end_date": end_date}],
        limit=1,
    )
    resp = client.run_report(req)
    row = resp.rows[0].metric_values if resp.rows else []
    return pd.DataFrame([{
        "sessions": int(row[0].value) if row else 0,
        "totalUsers": int(row[1].value) if row else 0,
        "screenPageViews": int(row[2].value) if row else 0,
    }])

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
password_gate()

# Header
head_col1, head_col2 = st.columns([4, 1])
with head_col1:
    st.title("Analytics Console")
    st.markdown("Professional content performance and user engagement reporting.")
with head_col2:
    # prefer local logo if present; else keep GA icon
    if Path(DASH_LOGO).exists():
        render_logo(DASH_LOGO, width=90)
    else:
        st.image("https://www.gstatic.com/analytics-suite/header/suite/v2/ic_analytics.svg", width=80)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### Reporting Period")
    today = date.today()
    date_from = st.date_input("Date From", value=today - timedelta(days=30))
    date_to = st.date_input("Date To", value=today)

    st.divider()
    pid_default = default_property_id()
    property_id = st.text_input("GA4 Property ID", value=pid_default)

    st.divider()
    st.markdown("### Developed by")
    st.markdown("**Alexey Terekhov**")
    st.markdown("[terekhov.digital@gmail.com](mailto:terekhov.digital@gmail.com)")
    if Path(SIDEBAR_LOGO).exists():
        st.markdown("<br>", unsafe_allow_html=True)
        render_logo(SIDEBAR_LOGO, width=160)

tab1, tab2, tab3 = st.tabs(["URL Analytics", "Top Materials", "Global Performance"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — URL Analytics (auto: any input -> path)
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("URL Analytics")

    cA, cB = st.columns([3, 2])
    with cA:
        uinput = st.text_area(
            "Вставьте URL или пути (по одному в строке)",
            height=200,
            placeholder=(
                "https://www.site.com/path\n"
                "www.site.com/path\n"
                "site.com/path\n"
                "/path\n"
                "path"
            ),
        )
    with cB:
        uploaded = st.file_uploader("Или загрузите .txt/.csv (1 в строке)", type=["txt", "csv"])

    lines = []
    if uinput:
        lines.extend([clean_line(x) for x in uinput.splitlines() if clean_line(x)])
    up_lines = read_uploaded_lines(uploaded)
    lines.extend([x for x in up_lines if x])

    # IMPORTANT: we treat everything as "path analytics" reliably
    # (URL lookup by full URL is unstable across GA4 properties; pagePath is reliable.)
    # But we still "understand" any url format by extracting path + host.
    unique_paths, hostnames, order_paths = collect_paths_hosts(lines)

    # For UX counters: count url-like vs path-like
    url_like = sum(1 for x in lines if looks_like_domain_no_scheme(x) or x.lower().startswith(("http://", "https://")))
    path_like = len(lines) - url_like

    host_txt = f" | Hosts: {', '.join(hostnames)}" if hostnames else ""
    st.caption(f"Lines: {len(lines)} | URLs: {url_like} | Paths: {path_like}{host_txt}")

    if st.button("Analyze"):
        if date_from > date_to:
            fail_ui("Date From must be <= Date To.")
        pid = property_id.strip()
        if not pid:
            fail_ui("GA4 Property ID is empty.")
        if not lines:
            fail_ui("Добавьте хотя бы один URL/путь.")

        with st.spinner("Fetching GA4 by paths..."):
            df_p = fetch_by_paths_cached(
                pid,
                tuple(unique_paths),
                tuple(hostnames),
                str(date_from),
                str(date_to),
                tuple(order_paths),
            )

        if df_p.empty:
            st.info("No data returned for these identifiers.")
        else:
            show = df_p.reindex(columns=[
                "pagePath",
                "pageTitle",
                "screenPageViews",
                "activeUsers",
                "viewsPerActiveUser",
                "avgEngagementTime_sec",
            ]).rename(columns={
                "pagePath": "Path",
                "pageTitle": "Title",
                "screenPageViews": "Views",
                "activeUsers": "Unique Users",
                "viewsPerActiveUser": "Views / Unique User",
                "avgEngagementTime_sec": "Avg Engagement Time (s)",
            })

            st.success(f"Found {len(show)} rows.")
            st.dataframe(show, use_container_width=True, hide_index=True)

            tot_views = int(pd.to_numeric(df_p["screenPageViews"], errors="coerce").sum())
            tot_users = int(pd.to_numeric(df_p["activeUsers"], errors="coerce").sum())
            ratio = tot_views / max(tot_users, 1)
            avg_eng = float(pd.to_numeric(df_p["userEngagementDuration"], errors="coerce").sum() / max(tot_users, 1))

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Views", f"{tot_views:,}")
            k2.metric("Unique Users", f"{tot_users:,}")
            k3.metric("Views / Unique User", f"{ratio:.2f}")
            k4.metric("Avg Engagement Time (s)", f"{avg_eng:.1f}")

            st.download_button(
                "Export (CSV)",
                show.to_csv(index=False).encode("utf-8"),
                "ga4_url_analytics.csv",
                "text/csv",
            )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Top Materials
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("High-Performance Content")
    c1, c2 = st.columns([1, 2])
    with c1:
        limit = st.number_input("Limit", min_value=1, max_value=500, value=10)

    if st.button("Extract Top Content"):
        if date_from > date_to:
            fail_ui("Date From must be <= Date To.")
        pid = property_id.strip()
        if not pid:
            fail_ui("GA4 Property ID is empty.")

        with st.spinner(f"Extracting top {int(limit)} materials..."):
            df_top = fetch_top_materials_cached(pid, str(date_from), str(date_to), int(limit))

        if df_top.empty:
            st.info("No data returned for this period.")
        else:
            st.dataframe(df_top, use_container_width=True, hide_index=True)
            st.download_button(
                "Export Ranking (CSV)",
                df_top.to_csv(index=False).encode("utf-8"),
                "ga4_top.csv",
                "text/csv",
            )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Global Performance
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Global Site Summary")

    if st.button("Refresh Site Totals"):
        if date_from > date_to:
            fail_ui("Date From must be <= Date To.")
        pid = property_id.strip()
        if not pid:
            fail_ui("GA4 Property ID is empty.")

        with st.spinner("Aggregating..."):
            totals = fetch_site_totals_cached(pid, str(date_from), str(date_to))
            s = int(totals.loc[0, "sessions"])
            u = int(totals.loc[0, "totalUsers"])
            v = int(totals.loc[0, "screenPageViews"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Sessions", f"{s:,}")
        c2.metric("Unique Users", f"{u:,}")
        c3.metric("Page Views", f"{v:,}")

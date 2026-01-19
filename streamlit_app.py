# -*- coding: utf-8 -*-
import os
import io
from pathlib import Path
from datetime import date, timedelta
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import numpy as np
import pandas as pd
import streamlit as st

from google.oauth2 import service_account
from google.api_core.exceptions import InvalidArgument
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest, Dimension, Metric,
    Filter, FilterExpression, FilterExpressionList, OrderBy
)

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="GA4 Professional Dashboard", layout="wide")

st.markdown(
    """
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
    .metric-container {
        background-color: white;
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stExpander"] {
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# Конфиг / Secrets
# ─────────────────────────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/analytics.readonly"]
DASH_LOGO = st.secrets.get("DASH_LOGO", os.getenv("DASH_LOGO", ""))
SIDEBAR_LOGO = st.secrets.get("SIDEBAR_LOGO", os.getenv("SIDEBAR_LOGO", ""))
INVISIBLE = ("\ufeff", "\u200b", "\u2060", "\u00a0")

DROP_QUERY_KEYS = {
    "gclid", "fbclid", "yclid", "msclkid", "gbraid", "wbraid",
    "mc_cid", "mc_eid", "igshid", "ref", "ref_src",
}
# utm_* режем всегда

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

@st.cache_resource
def ga_client() -> BetaAnalyticsDataClient:
    sa = st.secrets.get("gcp_service_account")
    if not sa:
        fail_ui("Не найден секрет **gcp_service_account** в Streamlit Secrets.")
    creds = service_account.Credentials.from_service_account_info(dict(sa), scopes=SCOPES)
    return BetaAnalyticsDataClient(credentials=creds)

def default_property_id() -> str:
    pid = str(st.secrets.get("GA4_PROPERTY_ID", "")).strip()
    if not pid:
        fail_ui("Не задан секрет **GA4_PROPERTY_ID**.")
    return pid

def render_logo(path: str, width: int | None = None):
    if not path:
        return
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
            return
    else:
        st.image(str(p), use_container_width=(width is None), width=width)

# ─────────────────────────────────────────────────────────────────────────────
# Ввод / парсинг
# ─────────────────────────────────────────────────────────────────────────────
def clean_line(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    for ch in INVISIBLE:
        s = s.replace(ch, "")
    return s.strip()

def normalize_url(raw_url: str) -> str:
    p = urlparse(raw_url)
    q = []
    for k, v in parse_qsl(p.query, keep_blank_values=True):
        kl = k.lower()
        if kl.startswith("utm_"):
            continue
        if kl in DROP_QUERY_KEYS:
            continue
        q.append((k, v))
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q, doseq=True), ""))

def url_to_path_host(u: str) -> tuple[str, str | None]:
    s = clean_line(u)
    if not s:
        return "", None
    if s.lower().startswith(("http://", "https://")):
        s2 = normalize_url(s)
        p = urlparse(s2)
        return (p.path or "/"), (p.hostname or None)
    if not s.startswith("/"):
        s = "/" + s
    return s, None

def collect_paths_hosts(raw_list: list[str]) -> tuple[list[str], list[str], list[str]]:
    seen = set()
    unique_paths: list[str] = []
    hosts = set()
    order_list: list[str] = []
    for raw in raw_list:
        path, host = url_to_path_host(raw)
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
    name = (uploaded.name or "").lower()
    if name.endswith(".txt"):
        txt = uploaded.getvalue().decode("utf-8", errors="ignore")
        return [clean_line(x) for x in txt.splitlines() if clean_line(x)]
    if name.endswith(".csv"):
        dfu = pd.read_csv(uploaded, header=None)
        col = dfu.iloc[:, 0].astype(str).tolist()
        return [clean_line(x) for x in col if clean_line(x)]
    return []

# ─────────────────────────────────────────────────────────────────────────────
# GA4 helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_path_filter(paths_batch: list[str], match_type: str) -> FilterExpression:
    mt = Filter.StringFilter.MatchType.BEGINS_WITH
    if match_type == "exact":
        mt = Filter.StringFilter.MatchType.EXACT
    elif match_type == "contains":
        mt = Filter.StringFilter.MatchType.CONTAINS

    exprs = [
        FilterExpression(
            filter=Filter(
                field_name="pagePath",
                string_filter=Filter.StringFilter(
                    value=pth,
                    match_type=mt,
                    case_sensitive=False,
                )
            )
        )
        for pth in paths_batch
    ]
    return FilterExpression(or_group=FilterExpressionList(expressions=exprs))

def _build_batch_request(
    property_id: str,
    mode: str,
    batch: list[str],
    host_list: list[str],
    start_date: str,
    end_date: str,
    path_match: str,
    include_aet: bool,
    include_host: bool
) -> RunReportRequest:
    metrics = [Metric(name="screenPageViews"), Metric(name="activeUsers")]
    if include_aet:
        metrics.append(Metric(name="averageEngagementTime"))

    if mode == "path":
        base = make_path_filter(batch, path_match)
        dims = [Dimension(name="pagePath"), Dimension(name="pageTitle")]

        dim_filter = base
        if include_host and host_list:
            host_expr = FilterExpression(
                filter=Filter(
                    field_name="hostName",
                    in_list_filter=Filter.InListFilter(values=host_list[:50])
                )
            )
            dim_filter = FilterExpression(and_group=FilterExpressionList(expressions=[base, host_expr]))
            dims.append(Dimension(name="hostName"))

        return RunReportRequest(
            property=f"properties/{property_id}",
            dimensions=dims,
            metrics=metrics,
            date_ranges=[{"start_date": start_date, "end_date": end_date}],
            dimension_filter=dim_filter,
            limit=100000,
        )

    # mode == "url"
    dims = [Dimension(name="pageLocation"), Dimension(name="pageTitle")]
    return RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=dims,
        metrics=metrics,
        date_ranges=[{"start_date": start_date, "end_date": end_date}],
        dimension_filter=FilterExpression(
            filter=Filter(
                field_name="pageLocation",
                in_list_filter=Filter.InListFilter(values=batch)
            )
        ),
        limit=100000,
    )

def run_report_with_fallback(
    client: BetaAnalyticsDataClient,
    property_id: str,
    mode: str,
    batch: list[str],
    host_list: list[str],
    start_date: str,
    end_date: str,
    path_match: str,
) -> tuple[object, list[str], bool, bool]:
    """
    Возвращает: (resp, metric_names, used_aet, used_host)
    """
    # 1) пробуем: AET + host
    for include_aet, include_host in [
        (True, True),
        (False, True),
        (True, False),
        (False, False),
    ]:
        req = _build_batch_request(
            property_id=property_id,
            mode=mode,
            batch=batch,
            host_list=host_list,
            start_date=start_date,
            end_date=end_date,
            path_match=path_match,
            include_aet=include_aet,
            include_host=include_host,
        )
        try:
            resp = client.run_report(req)
            metric_names = [m.name for m in req.metrics]
            return resp, metric_names, include_aet, include_host
        except InvalidArgument:
            continue

    raise InvalidArgument("GA4 запрос невалиден для этого property (dimensions/metrics несовместимы).")

# ─────────────────────────────────────────────────────────────────────────────
# GA4 queries (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_batch(
    property_id: str,
    identifiers: tuple[str, ...],
    hosts: tuple[str, ...],
    start_date: str,
    end_date: str,
    order_keys: tuple[str, ...],
    mode: str,
    path_match: str,
) -> pd.DataFrame:
    client = ga_client()
    rows: list[dict] = []
    BATCH = 25

    id_list = list(identifiers)
    host_list = list(hosts)
    key_col = "pagePath" if mode == "path" else "pageLocation"

    for i in range(0, len(id_list), BATCH):
        batch = id_list[i:i + BATCH]

        resp, metric_names, used_aet, used_host = run_report_with_fallback(
            client=client,
            property_id=property_id,
            mode=mode,
            batch=batch,
            host_list=host_list,
            start_date=start_date,
            end_date=end_date,
            path_match=path_match,
        )

        for r in resp.rows:
            rec: dict = {}
            dvals = r.dimension_values
            mvals = r.metric_values

            rec[key_col] = dvals[0].value
            rec["pageTitle"] = dvals[1].value if len(dvals) > 1 else ""
            if mode == "path" and used_host and host_list and len(dvals) > 2:
                rec["hostName"] = dvals[2].value

            def mv(name: str) -> float:
                if name not in metric_names:
                    return 0.0
                idx = metric_names.index(name)
                return float(mvals[idx].value or 0)

            rec["screenPageViews"] = mv("screenPageViews")
            rec["activeUsers"] = mv("activeUsers")
            rec["averageEngagementTime"] = mv("averageEngagementTime")
            rows.append(rec)

    df = pd.DataFrame(rows)

    # Пустой ответ
    if df.empty:
        base_cols = [key_col, "pageTitle", "screenPageViews", "activeUsers", "averageEngagementTime"]
        return pd.DataFrame(columns=base_cols)

    for c in ["screenPageViews", "activeUsers", "averageEngagementTime"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Аггрегация по ключу
    agg = {
        "screenPageViews": "sum",
        "activeUsers": "sum",
        "pageTitle": "first",
    }
    # averageEngagementTime — среднее по ключу (если GA4 отдаёт на этом уровне)
    agg["averageEngagementTime"] = "mean"

    if "hostName" in df.columns:
        agg["hostName"] = "first"

    df = df.groupby([key_col], as_index=False).agg(agg)

    # Добавляем отсутствующие (нули), чтобы совпадало с input
    present = set(df[key_col].tolist())
    missing = [p for p in id_list if p not in present]
    if missing:
        zeros = pd.DataFrame({
            key_col: missing,
            "pageTitle": [""] * len(missing),
            "screenPageViews": [0] * len(missing),
            "activeUsers": [0] * len(missing),
            "averageEngagementTime": [0] * len(missing),
        })
        if "hostName" in df.columns:
            zeros["hostName"] = host_list[0] if host_list else ""
        df = pd.concat([df, zeros], ignore_index=True)

    # Восстановить порядок строк как во входе
    df = df.set_index(key_col).reindex(list(order_keys)).reset_index()

    # Типы/производные
    df["pageTitle"] = df["pageTitle"].fillna("")
    df["screenPageViews"] = pd.to_numeric(df["screenPageViews"], errors="coerce").fillna(0).astype(int)
    df["activeUsers"] = pd.to_numeric(df["activeUsers"], errors="coerce").fillna(0).astype(int)
    df["averageEngagementTime"] = pd.to_numeric(df["averageEngagementTime"], errors="coerce").fillna(0).round(1)

    den = pd.to_numeric(df["activeUsers"], errors="coerce").replace(0, np.nan).astype(float)
    df["viewsPerActiveUser"] = (df["screenPageViews"].astype(float).div(den)).fillna(0).round(2)

    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_top_materials(property_id: str, start_date: str, end_date: str, limit: int) -> pd.DataFrame:
    # Важно: averageEngagementTime часто несовместима с pagePath/pageTitle в топ-рейтинге.
    client = ga_client()
    req = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="pagePath"), Dimension(name="pageTitle")],
        metrics=[Metric(name="screenPageViews"), Metric(name="activeUsers")],
        date_ranges=[{"start_date": start_date, "end_date": end_date}],
        order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"), desc=True)],
        limit=int(limit),
    )
    resp = client.run_report(req)
    rows = []
    for r in resp.rows:
        rows.append({
            "Path": r.dimension_values[0].value,
            "Title": r.dimension_values[1].value,
            "Views": int(float(r.metric_values[0].value or 0)),
            "Active Users": int(float(r.metric_values[1].value or 0)),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_site_totals(property_id: str, start_date: str, end_date: str) -> tuple[int, int, int]:
    client = ga_client()
    req = RunReportRequest(
        property=f"properties/{property_id}",
        metrics=[Metric(name="sessions"), Metric(name="totalUsers"), Metric(name="screenPageViews")],
        date_ranges=[{"start_date": start_date, "end_date": end_date}],
        limit=1,
    )
    resp = client.run_report(req)
    if not resp.rows:
        return 0, 0, 0
    v = resp.rows[0].metric_values
    return int(float(v[0].value or 0)), int(float(v[1].value or 0)), int(float(v[2].value or 0))

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
password_gate()

prop_id_default = default_property_id()

with st.sidebar:
    st.markdown("### Reporting Period")
    today = date.today()
    date_from = st.date_input("Date From", value=today - timedelta(days=30))
    date_to = st.date_input("Date To", value=today)

    st.divider()
    st.markdown("### Property")
    property_id = st.text_input("GA4 Property ID", value=prop_id_default)

    st.divider()
    st.markdown("### Developed by")
    st.markdown("**Alexey Terekhov**")
    st.markdown("[terekhov.digital@gmail.com](mailto:terekhov.digital@gmail.com)")

    if SIDEBAR_LOGO:
        st.markdown("<br>", unsafe_allow_html=True)
        render_logo(SIDEBAR_LOGO, width=160)

head_col1, head_col2 = st.columns([4, 1])
with head_col1:
    st.title("Analytics Console")
    st.markdown("Professional content performance and user engagement reporting.")
with head_col2:
    if DASH_LOGO:
        render_logo(DASH_LOGO, width=72)
    else:
        st.image("https://www.gstatic.com/analytics-suite/header/suite/v2/ic_analytics.svg", width=80)

st.divider()

tab1, tab2, tab3 = st.tabs(["Batch URL Analysis", "Top Materials", "Global Performance"])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Batch Processing")

    smode = st.radio("Extraction Logic", ["By Path", "By URL"], horizontal=True)
    mode_key = "path" if smode == "By Path" else "url"

    path_match = "begins_with"
    if mode_key == "path":
        pm_ui = st.radio("Path matching", ["Begins with", "Exact", "Contains"], horizontal=True)
        path_match = {"Begins with": "begins_with", "Exact": "exact", "Contains": "contains"}[pm_ui]

    cA, cB = st.columns([3, 2])
    with cA:
        uinput = st.text_area(
            "Identifiers (one per line)",
            height=200,
            placeholder="/content/news-item\nhttps://domain.com/page"
        )
    with cB:
        uploaded = st.file_uploader("Or upload .txt/.csv (1 per line)", type=["txt", "csv"])

    lines = []
    if uinput:
        lines.extend([clean_line(x) for x in uinput.splitlines() if clean_line(x)])
    lines.extend(read_uploaded_lines(uploaded))

    if mode_key == "path":
        page_paths, hostnames, order_keys = collect_paths_hosts(lines)
        st.caption(
            f"Lines: {len(lines)} | Unique paths: {len(page_paths)} | "
            f"Hosts: {', '.join(hostnames) if hostnames else '—'}"
        )
        identifiers = page_paths
        hosts = hostnames
    else:
        urls_clean = []
        seen = set()
        for u in lines:
            if not u.lower().startswith(("http://", "https://")):
                continue
            u2 = normalize_url(u)
            if u2 not in seen:
                seen.add(u2)
                urls_clean.append(u2)
        st.caption(f"URLs: {len(urls_clean)}")
        identifiers = urls_clean
        hosts = []
        order_keys = urls_clean

    if st.button("Execute Analysis"):
        if date_from > date_to:
            fail_ui("Date From must be <= Date To.")
        if not property_id.strip():
            fail_ui("GA4 Property ID is empty.")
        if not identifiers:
            fail_ui("Input required: add at least one identifier.")

        try:
            with st.spinner("Fetching data..."):
                df_batch = fetch_batch(
                    property_id=property_id.strip(),
                    identifiers=tuple(identifiers),
                    hosts=tuple(hosts),
                    start_date=str(date_from),
                    end_date=str(date_to),
                    order_keys=tuple(order_keys),
                    mode=mode_key,
                    path_match=path_match,
                )
        except InvalidArgument:
            fail_ui("GA4 отклонил запрос (несовместимые dimensions/metrics для этого property).")

        if df_batch.empty:
            st.info("No data returned for these identifiers.")
        else:
            st.success(f"Processed {len(df_batch)} rows.")

            key_col = "pagePath" if mode_key == "path" else "pageLocation"
            show_cols = [
                key_col, "pageTitle",
                "screenPageViews", "activeUsers",
                "viewsPerActiveUser",
                "averageEngagementTime",
            ]
            show = df_batch.reindex(columns=[c for c in show_cols if c in df_batch.columns]).rename(columns={
                key_col: "Identifier",
                "pageTitle": "Title",
                "screenPageViews": "Views",
                "activeUsers": "Active Users",
                "viewsPerActiveUser": "Views / Active User",
                "averageEngagementTime": "Avg Engagement Time (s)",
            })

            st.dataframe(show, use_container_width=True, hide_index=True)

            tot_views = int(pd.to_numeric(df_batch["screenPageViews"], errors="coerce").sum())
            tot_users = int(pd.to_numeric(df_batch["activeUsers"], errors="coerce").sum())
            ratio = (tot_views / max(tot_users, 1))

            # mean по строкам — если метрика недоступна и упала в 0, будет 0.0 (и это лучше, чем падение)
            avg_eng = float(pd.to_numeric(df_batch["averageEngagementTime"], errors="coerce").fillna(0).mean())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Views", f"{tot_views:,}")
            k2.metric("Active Users", f"{tot_users:,}")
            k3.metric("Views / Active User", f"{ratio:.2f}")
            k4.metric("Avg Engagement Time (s)", f"{avg_eng:.1f}")

            st.download_button(
                "Export Results (CSV)",
                show.to_csv(index=False).encode("utf-8"),
                "ga4_batch_report.csv",
                "text/csv"
            )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("High-Performance Content")
    lim_col, _ = st.columns([1, 2])
    with lim_col:
        limit = st.number_input("Limit", min_value=1, max_value=500, value=10)

    if st.button("Extract Top Content"):
        if date_from > date_to:
            fail_ui("Date From must be <= Date To.")
        if not property_id.strip():
            fail_ui("GA4 Property ID is empty.")

        try:
            with st.spinner(f"Extracting top {int(limit)} materials..."):
                df_top = fetch_top_materials(property_id.strip(), str(date_from), str(date_to), int(limit))
        except InvalidArgument:
            fail_ui("GA4 отклонил запрос Top Materials для этого property.")

        if df_top.empty:
            st.info("No data returned for this period.")
        else:
            st.dataframe(df_top, use_container_width=True, hide_index=True)
            st.download_button(
                "Export Ranking (CSV)",
                df_top.to_csv(index=False).encode("utf-8"),
                "ga4_top.csv",
                "text/csv"
            )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Global Site Summary")
    if st.button("Refresh Site Totals"):
        if date_from > date_to:
            fail_ui("Date From must be <= Date To.")
        if not property_id.strip():
            fail_ui("GA4 Property ID is empty.")

        try:
            with st.spinner("Aggregating..."):
                s, u, v = fetch_site_totals(property_id.strip(), str(date_from), str(date_to))
        except InvalidArgument:
            fail_ui("GA4 отклонил запрос totals для этого property.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Sessions", f"{s:,}")
        c2.metric("Total Users", f"{u:,}")
        c3.metric("Page Views", f"{v:,}")

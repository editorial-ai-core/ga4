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
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest, Dimension, Metric, Filter, FilterExpression, FilterExpressionList, OrderBy
)

# ─────────────────────────────────────────────────────────────────────────────
# UI / Style
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
# Secrets / Config
# ─────────────────────────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/analytics.readonly"]

DASH_LOGO = st.secrets.get("DASH_LOGO", os.getenv("DASH_LOGO", "assets/logo.svg"))
SIDEBAR_LOGO = st.secrets.get("SIDEBAR_LOGO", os.getenv("SIDEBAR_LOGO", "assets/internews.svg"))

INVISIBLE = ("\ufeff", "\u200b", "\u2060", "\u00a0")

def fail_ui(msg: str):
    st.error(msg)
    st.stop()

def password_gate():
    app_pwd = st.secrets.get("APP_PASSWORD", "").strip()
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
        fail_ui("Не найден секрет **gcp_service_account**. Добавь его в Streamlit Secrets.")
    creds = service_account.Credentials.from_service_account_info(dict(sa), scopes=SCOPES)
    return BetaAnalyticsDataClient(credentials=creds)

def default_property_id() -> str:
    pid = str(st.secrets.get("GA4_PROPERTY_ID", "")).strip()
    if not pid:
        fail_ui("Не задан секрет **GA4_PROPERTY_ID**.")
    return pid

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
            return
    else:
        st.image(str(p), use_container_width=(width is None), width=width)

# ─────────────────────────────────────────────────────────────────────────────
# Input helpers
# ─────────────────────────────────────────────────────────────────────────────
def clean_line(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    for ch in INVISIBLE:
        s = s.replace(ch, "")
    return s.strip()

def strip_utm_and_fragment(raw_url: str) -> str:
    p = urlparse(raw_url)
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
         if not k.lower().startswith("utm_")]
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q, doseq=True), ""))

def normalize_path(p: str) -> str:
    p = clean_line(p)
    if not p:
        return ""
    if p.lower().startswith(("http://", "https://")):
        p = urlparse(strip_utm_and_fragment(p)).path or "/"
    if not p.startswith("/"):
        p = "/" + p
    return p

def path_variants(p: str) -> list[str]:
    """
    GA4 часто хранит и /path и /path/ (или только один вариант).
    Берём оба, чтобы не ловить "0".
    """
    p = normalize_path(p)
    if not p:
        return []
    if p == "/":
        return ["/"]
    if p.endswith("/"):
        a = p.rstrip("/")
        return [a, a + "/"]
    return [p, p + "/"]

def url_variants(raw_url: str) -> list[str]:
    """
    Для pageLocation exact-match:
    - режем utm
    - делаем варианты с / без / на конце
    - делаем варианты www/без www (очень частая причина нулей)
    """
    u = clean_line(raw_url)
    if not u.lower().startswith(("http://", "https://")):
        return []
    u = strip_utm_and_fragment(u)
    p = urlparse(u)

    scheme = p.scheme
    host = p.netloc
    path = p.path or "/"

    # base без query (utm уже сняли, оставшийся query оставим как есть)
    query = p.query
    def build(netloc: str, path_: str) -> str:
        return urlunparse((scheme, netloc, path_, "", query, ""))

    # слеш-варианты
    if path != "/" and path.endswith("/"):
        path_a = path.rstrip("/")
        path_b = path
    else:
        path_a = path
        path_b = path if path.endswith("/") or path == "/" else path + "/"

    hosts = [host]
    host_l = host.lower()
    if host_l.startswith("www."):
        hosts.append(host[4:])
    else:
        hosts.append("www." + host)

    out = []
    for h in hosts:
        for pa in [path_a, path_b]:
            v = build(h, pa)
            if v and v not in out:
                out.append(v)
    return out

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
# GA4 queries
# ─────────────────────────────────────────────────────────────────────────────
# ВАЖНО: averageEngagementTime (сек) вместо userEngagementDuration
METRICS_URLS = ["screenPageViews", "activeUsers", "averageEngagementTime"]

def make_path_filter(paths_batch: list[str], match_type: Filter.StringFilter.MatchType) -> FilterExpression:
    exprs = [
        FilterExpression(
            filter=Filter(
                field_name="pagePath",
                string_filter=Filter.StringFilter(
                    value=pth,
                    match_type=match_type,
                    case_sensitive=False,
                )
            )
        )
        for pth in paths_batch
    ]
    return FilterExpression(or_group=FilterExpressionList(expressions=exprs))

def fetch_ga4_by_identifiers(
    property_id: str,
    identifiers: list[str],
    start_date: str,
    end_date: str,
    mode: str,
    match: str = "begins_with",
) -> pd.DataFrame:
    """
    mode:
      - "path": identifiers -> pagePath; match: begins_with (default) / exact / contains
      - "url" : identifiers -> pageLocation; match всегда exact через in_list_filter (как в рабочей схеме)
    """
    if not identifiers:
        key_col = "pagePath" if mode == "path" else "pageLocation"
        cols = [key_col, "pageTitle"] + METRICS_URLS
        return pd.DataFrame(columns=cols)

    client = ga_client()
    rows, BATCH = [], 25

    if mode == "path":
        match_map = {
            "begins_with": Filter.StringFilter.MatchType.BEGINS_WITH,
            "exact": Filter.StringFilter.MatchType.EXACT,
            "contains": Filter.StringFilter.MatchType.CONTAINS,
        }
        mt = match_map.get(match, Filter.StringFilter.MatchType.BEGINS_WITH)

    for i in range(0, len(identifiers), BATCH):
        batch = identifiers[i:i + BATCH]

        if mode == "path":
            req = RunReportRequest(
                property=f"properties/{property_id}",
                dimensions=[Dimension(name="pagePath"), Dimension(name="pageTitle")],
                metrics=[Metric(name=m) for m in METRICS_URLS],
                date_ranges=[{"start_date": start_date, "end_date": end_date}],
                dimension_filter=make_path_filter(batch, mt),
                limit=100000,
            )
            key_name = "pagePath"
        else:
            req = RunReportRequest(
                property=f"properties/{property_id}",
                dimensions=[Dimension(name="pageLocation"), Dimension(name="pageTitle")],
                metrics=[Metric(name=m) for m in METRICS_URLS],
                date_ranges=[{"start_date": start_date, "end_date": end_date}],
                dimension_filter=FilterExpression(
                    filter=Filter(
                        field_name="pageLocation",
                        in_list_filter=Filter.InListFilter(values=batch)
                    )
                ),
                limit=100000,
            )
            key_name = "pageLocation"

        resp = client.run_report(req)
        for r in resp.rows:
            rec = {
                key_name: r.dimension_values[0].value,
                "pageTitle": r.dimension_values[1].value,
            }
            for j, m in enumerate(METRICS_URLS):
                rec[m] = r.metric_values[j].value
            rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for m in METRICS_URLS:
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)

    agg = {m: "sum" for m in ["screenPageViews", "activeUsers"]}
    # averageEngagementTime в GA4 — “average”, логичнее агрегировать как mean.
    # Но если у тебя несколько строк на один URL, чаще всего это разная title/параметры,
    # а метрика уже средняя. Берём mean.
    agg["averageEngagementTime"] = "mean"
    agg["pageTitle"] = "first"

    df = df.groupby([key_name], as_index=False).agg(agg)

    den = pd.to_numeric(df["activeUsers"], errors="coerce").replace(0, np.nan).astype(float)
    df["viewsPerActiveUser"] = (df["screenPageViews"].astype(float).div(den)).fillna(0).round(2)

    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_url_mode(property_id: str, urls_in: tuple[str, ...], start_date: str, end_date: str) -> pd.DataFrame:
    return fetch_ga4_by_identifiers(property_id, list(urls_in), start_date, end_date, mode="url")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_path_mode(property_id: str, paths_in: tuple[str, ...], start_date: str, end_date: str) -> pd.DataFrame:
    return fetch_ga4_by_identifiers(property_id, list(paths_in), start_date, end_date, mode="path", match="begins_with")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_top_materials(property_id: str, start_date: str, end_date: str, limit: int) -> pd.DataFrame:
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
            "Unique Users": int(float(r.metric_values[1].value or 0)),
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
# App layout
# ─────────────────────────────────────────────────────────────────────────────
password_gate()

with st.sidebar:
    st.markdown("### Reporting Period")
    today = date.today()
    date_from = st.date_input("Date From", value=today - timedelta(days=30))
    date_to = st.date_input("Date To", value=today)

    st.divider()
    st.markdown("### Property")
    property_id = st.text_input("GA4 Property ID", value=default_property_id())

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
    if DASH_LOGO and Path(DASH_LOGO).exists():
        render_logo(DASH_LOGO, width=80)
    else:
        st.image("https://www.gstatic.com/analytics-suite/header/suite/v2/ic_analytics.svg", width=80)

st.divider()

tab1, tab2, tab3 = st.tabs(["URL Analytics", "Top Materials", "Global Performance"])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — URL Analytics (no toggles)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("URL Analytics")

    c1, c2 = st.columns([3, 2])
    with c1:
        text = st.text_area(
            "Вставьте URL или пути (по одному в строке)",
            height=170,
            placeholder="https://example.com/path\n/just/path\njust/path",
        )
    with c2:
        uploaded = st.file_uploader("Или загрузите .txt/.csv (1 в строке)", type=["txt", "csv"])

    lines = []
    if text:
        lines.extend([clean_line(x) for x in text.splitlines() if clean_line(x)])
    lines.extend(read_uploaded_lines(uploaded))

    url_lines = [x for x in lines if x.lower().startswith(("http://", "https://"))]
    path_lines = [x for x in lines if not x.lower().startswith(("http://", "https://"))]

    # URL candidates (pageLocation)
    url_candidates = []
    seen_u = set()
    for u in url_lines:
        for v in url_variants(u):
            if v not in seen_u:
                seen_u.add(v)
                url_candidates.append(v)

    # Path candidates (pagePath begins_with)
    # Важно: из URL тоже извлекаем путь, потому что иногда pageLocation не совпадает,
    # а pagePath совпадает (и наоборот). Это и есть "проверяем оба варианта".
    all_for_path = path_lines + url_lines
    path_candidates = []
    seen_p = set()
    order_norm = []
    for raw in all_for_path:
        p = normalize_path(raw)
        pn = p.rstrip("/") if (p != "/" and p.endswith("/")) else p
        if pn:
            order_norm.append(pn)
        for pv in path_variants(p):
            if pv not in seen_p:
                seen_p.add(pv)
                path_candidates.append(pv)

    st.caption(f"Lines: {len(lines)} | URLs: {len(url_lines)} | Paths: {len(set(order_norm))}")

    if st.button("Analyze"):
        if not lines:
            fail_ui("Добавьте хотя бы один URL/путь.")
        if date_from > date_to:
            fail_ui("Date From must be <= Date To.")
        pid = property_id.strip()
        if not pid:
            fail_ui("GA4 Property ID is empty.")

        frames = []

        # 1) URL exact (pageLocation) — как в твоей рабочей схеме
        if url_candidates:
            with st.spinner("Fetching GA4 by URL (pageLocation)..."):
                df_u = fetch_url_mode(pid, tuple(url_candidates), str(date_from), str(date_to))
            if not df_u.empty:
                df_u = df_u.rename(columns={"pageLocation": "Identifier"})
                df_u["Source"] = "URL"
                frames.append(df_u)

        # 2) Path begins_with (pagePath) — тоже делаем всегда (проверка “оба варианта”)
        if path_candidates:
            with st.spinner("Fetching GA4 by path (pagePath)..."):
                df_p = fetch_path_mode(pid, tuple(path_candidates), str(date_from), str(date_to))
            if not df_p.empty:
                df_p = df_p.rename(columns={"pagePath": "Identifier"})
                df_p["Source"] = "Path"
                frames.append(df_p)

        if not frames:
            st.info("No data returned for these identifiers.")
        else:
            df = pd.concat(frames, ignore_index=True)

            # Нормализуем идентификатор для склейки:
            # - для URL: убираем trailing /
            # - для Path: тоже убираем trailing /
            def _norm_id(x: str) -> str:
                x = str(x or "")
                if x.startswith("http"):
                    return x.rstrip("/")
                return x.rstrip("/") if x != "/" else "/"

            df["Identifier_norm"] = df["Identifier"].apply(_norm_id)

            # Склеиваем результаты URL+Path по нормализованному ключу (если оба нашли одно и то же)
            # Приоритет: если есть URL-строка — берём её как "Identifier", иначе path.
            # Views/Users суммируем, AvgEngagementTime усредняем.
            df["screenPageViews"] = pd.to_numeric(df["screenPageViews"], errors="coerce").fillna(0)
            df["activeUsers"] = pd.to_numeric(df["activeUsers"], errors="coerce").fillna(0)
            df["averageEngagementTime"] = pd.to_numeric(df["averageEngagementTime"], errors="coerce").fillna(0)
            df["viewsPerActiveUser"] = pd.to_numeric(df.get("viewsPerActiveUser", 0), errors="coerce").fillna(0)

            grouped = (
                df.sort_values(by=["Source"])  # URL/Path порядок не критичен
                  .groupby("Identifier_norm", as_index=False)
                  .agg({
                      "Identifier": "first",
                      "pageTitle": "first",
                      "screenPageViews": "sum",
                      "activeUsers": "sum",
                      "averageEngagementTime": "mean",
                  })
            )

            den = pd.to_numeric(grouped["activeUsers"], errors="coerce").replace(0, np.nan).astype(float)
            grouped["viewsPerActiveUser"] = (grouped["screenPageViews"].astype(float).div(den)).fillna(0).round(2)

            # Порядок вывода: по порядку ввода путей (если юзер вставил URL — порядок всё равно сохраняем через order_norm)
            # Если order_norm пуст (например, только URL без path?) — выводим как есть.
            if order_norm:
                base = pd.DataFrame({"Identifier_norm": [p for p in order_norm if p]})
                out = base.merge(grouped, on="Identifier_norm", how="left")
                # если строка по URL не сматчилась по path_norm — добавим её в конец
                present = set(out["Identifier_norm"].dropna().tolist())
                extra = grouped[~grouped["Identifier_norm"].isin(present)]
                out = pd.concat([out, extra], ignore_index=True)
            else:
                out = grouped

            for c in ["screenPageViews", "activeUsers", "averageEngagementTime", "viewsPerActiveUser"]:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

            out["screenPageViews"] = out["screenPageViews"].astype(int)
            out["activeUsers"] = out["activeUsers"].astype(int)
            out["averageEngagementTime"] = out["averageEngagementTime"].round(1)
            out["pageTitle"] = out["pageTitle"].fillna("")
            out["Identifier"] = out["Identifier"].fillna(out["Identifier_norm"]).fillna("")

            show = out.rename(columns={
                "Identifier": "URL/Path",
                "pageTitle": "Title",
                "screenPageViews": "Views",
                "activeUsers": "Unique Users",
                "viewsPerActiveUser": "Views / Unique User",
                "averageEngagementTime": "Avg Engagement Time (s)",
            })[["URL/Path", "Title", "Views", "Unique Users", "Views / Unique User", "Avg Engagement Time (s)"]]

            if (show["Views"].sum() == 0) and (show["Unique Users"].sum() == 0):
                st.info("No data returned for these identifiers.")
            else:
                st.success("Done.")
                st.dataframe(show, use_container_width=True, hide_index=True)

                tot_views = int(show["Views"].sum())
                tot_users = int(show["Unique Users"].sum())
                ratio = tot_views / max(tot_users, 1)
                avg_eng = float(pd.to_numeric(show["Avg Engagement Time (s)"], errors="coerce").fillna(0).mean())

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Views", f"{tot_views:,}")
                k2.metric("Unique Users", f"{tot_users:,}")
                k3.metric("Views / Unique User", f"{ratio:.2f}")
                k4.metric("Avg Engagement Time (s)", f"{avg_eng:.1f}")

                st.download_button(
                    "Export (CSV)",
                    show.to_csv(index=False).encode("utf-8"),
                    "ga4_url_analytics.csv",
                    "text/csv"
                )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Top Materials
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("High-Performance Content")
    lim_col, _ = st.columns([1, 2])
    with lim_col:
        limit = st.number_input("Limit", min_value=1, max_value=500, value=10)

    if st.button("Extract Top Content"):
        if date_from > date_to:
            fail_ui("Date From must be <= Date To.")
        pid = property_id.strip()
        if not pid:
            fail_ui("GA4 Property ID is empty.")

        with st.spinner(f"Extracting top {int(limit)} materials..."):
            df_top = fetch_top_materials(pid, str(date_from), str(date_to), int(limit))

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
# Tab 3 — Global Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Global Site Summary")

    if st.button("Refresh Site Totals"):
        if date_from > date_to:
            fail_ui("Date From must be <= Date To.")
        pid = property_id.strip()
        if not pid:
            fail_ui("GA4 Property ID is empty.")

        with st.spinner("Aggregating..."):
            s, u, v = fetch_site_totals(pid, str(date_from), str(date_to))

        c1, c2, c3 = st.columns(3)
        c1.metric("Sessions", f"{s:,}")
        c2.metric("Unique Users", f"{u:,}")
        c3.metric("Page Views", f"{v:,}")

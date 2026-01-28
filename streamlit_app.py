import os
from pathlib import Path
from datetime import date, timedelta
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import numpy as np
import pandas as pd
import streamlit as st

from google.oauth2 import service_account
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest,
    Dimension,
    Metric,
    Filter,
    FilterExpression,
    FilterExpressionList,
    OrderBy,
)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analytics Console",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# STYLES
# ──────────────────────────────────────────────────────────────────────────────
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
}
.stButton>button:hover {
  background-color: black;
}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/analytics.readonly"]
METRICS_PAGE = ["screenPageViews", "activeUsers", "userEngagementDuration"]

# ──────────────────────────────────────────────────────────────────────────────
# AUTH / CLIENT
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def ga_client() -> BetaAnalyticsDataClient:
    sa = st.secrets.get("gcp_service_account")
    if not sa:
        st.error("Missing gcp_service_account secret")
        st.stop()
    creds = service_account.Credentials.from_service_account_info(
        dict(sa), scopes=SCOPES
    )
    return BetaAnalyticsDataClient(credentials=creds)


def default_property_id() -> str:
    pid = str(st.secrets.get("GA4_PROPERTY_ID", "")).strip()
    if not pid:
        st.error("Missing GA4_PROPERTY_ID secret")
        st.stop()
    return pid


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def clean_line(s: str) -> str:
    if not s:
        return ""
    return str(s).strip()


def strip_utm(url: str) -> str:
    p = urlparse(url)
    q = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm_")]
    return urlunparse((p.scheme, p.netloc, p.path, "", urlencode(q), ""))


def normalize_to_path(raw: str) -> str:
    raw = clean_line(raw)
    if not raw:
        return ""
    if raw.startswith("http"):
        raw = strip_utm(raw)
        return urlparse(raw).path or "/"
    if not raw.startswith("/"):
        raw = "/" + raw
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# GA4 QUERIES
# ──────────────────────────────────────────────────────────────────────────────
def make_path_filter(paths):
    return FilterExpression(
        or_group=FilterExpressionList(
            expressions=[
                FilterExpression(
                    filter=Filter(
                        field_name="pagePath",
                        string_filter=Filter.StringFilter(
                            value=p,
                            match_type=Filter.StringFilter.MatchType.BEGINS_WITH,
                            case_sensitive=False,
                        ),
                    )
                )
                for p in paths
            ]
        )
    )


@st.cache_data(ttl=300)
def fetch_url_analytics(property_id, paths, start_date, end_date):
    client = ga_client()
    req = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="pagePath"), Dimension(name="pageTitle")],
        metrics=[Metric(name=m) for m in METRICS_PAGE],
        dimension_filter=make_path_filter(paths),
        date_ranges=[{"start_date": start_date, "end_date": end_date}],
        limit=100000,
    )
    resp = client.run_report(req)

    rows = []
    for r in resp.rows:
        users = int(float(r.metric_values[1].value or 0))
        rows.append(
            {
                "Path": r.dimension_values[0].value,
                "Title": r.dimension_values[1].value,
                "Views": int(float(r.metric_values[0].value or 0)),
                "Unique Users": users,
                "Avg Engagement Time (s)": round(
                    float(r.metric_values[2].value or 0) / max(users, 1), 1
                ),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def fetch_top_materials(property_id, start_date, end_date, limit):
    client = ga_client()
    req = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="pagePath"), Dimension(name="pageTitle")],
        metrics=[Metric(name="screenPageViews"), Metric(name="activeUsers")],
        date_ranges=[{"start_date": start_date, "end_date": end_date}],
        order_bys=[
            OrderBy(
                metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"), desc=True
            )
        ],
        limit=limit,
    )
    resp = client.run_report(req)

    return pd.DataFrame(
        [
            {
                "Path": r.dimension_values[0].value,
                "Title": r.dimension_values[1].value,
                "Views": int(float(r.metric_values[0].value)),
                "Users": int(float(r.metric_values[1].value)),
            }
            for r in resp.rows
        ]
    )


@st.cache_data(ttl=300)
def fetch_site_totals(property_id, start_date, end_date):
    client = ga_client()
    req = RunReportRequest(
        property=f"properties/{property_id}",
        metrics=[
            Metric(name="sessions"),
            Metric(name="totalUsers"),
            Metric(name="screenPageViews"),
        ],
        date_ranges=[{"start_date": start_date, "end_date": end_date}],
    )
    resp = client.run_report(req)
    r = resp.rows[0].metric_values
    return {
        "Sessions": int(r[0].value),
        "Users": int(r[1].value),
        "Views": int(r[2].value),
    }


@st.cache_data(ttl=300)
def fetch_demographics(property_id, start_date, end_date):
    client = ga_client()
    req = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="userGender")],
        metrics=[
            Metric(name="totalUsers"),
            Metric(name="screenPageViews"),
            Metric(name="userEngagementDuration"),
        ],
        date_ranges=[{"start_date": start_date, "end_date": end_date}],
    )
    resp = client.run_report(req)

    rows = []
    for r in resp.rows:
        users = int(float(r.metric_values[0].value or 0))
        rows.append(
            {
                "Gender": r.dimension_values[0].value or "Unknown",
                "Users": users,
                "Views": int(float(r.metric_values[1].value or 0)),
                "Avg Engagement Time (s)": round(
                    float(r.metric_values[2].value or 0) / max(users, 1), 1
                ),
            }
        )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("Analytics Console")

today = date.today()
date_from = st.sidebar.date_input("Date From", today - timedelta(days=30))
date_to = st.sidebar.date_input("Date To", today)
property_id = st.sidebar.text_input(
    "GA4 Property ID", value=default_property_id()
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["URL Analytics", "Top Materials", "Global Performance", "Demographics"]
)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    urls = st.text_area("Paste URLs or paths (one per line)", height=200)
    if st.button("Collect"):
        paths = [
            normalize_to_path(x)
            for x in urls.splitlines()
            if normalize_to_path(x)
        ]
        df = fetch_url_analytics(
            property_id, paths, str(date_from), str(date_to)
        )
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Export CSV",
            df.to_csv(index=False).encode(),
            "url_analytics.csv",
        )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    limit = st.number_input("Limit", 1, 500, 10)
    if st.button("Load Top Materials"):
        df = fetch_top_materials(
            property_id, str(date_from), str(date_to), int(limit)
        )
        st.dataframe(df, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    if st.button("Refresh Totals"):
        t = fetch_site_totals(property_id, str(date_from), str(date_to))
        c1, c2, c3 = st.columns(3)
        c1.metric("Sessions", f"{t['Sessions']:,}")
        c2.metric("Users", f"{t['Users']:,}")
        c3.metric("Views", f"{t['Views']:,}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — DEMOGRAPHICS
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Demographics — Gender")

    if st.button("Load Demographics"):
        df = fetch_demographics(property_id, str(date_from), str(date_to))

        if df.empty:
            st.warning(
                "No gender data returned.\n\n"
                "Possible reasons:\n"
                "• Google Signals is disabled in GA4\n"
                "• Not enough users for this period\n"
                "• Date range is before Signals activation"
            )
        else:
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "Export Demographics CSV",
                df.to_csv(index=False).encode("utf-8"),
                "demographics_gender.csv",
            )


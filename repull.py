#!/usr/bin/env python3
"""
Re-pull incomplete slices with aggressive retry/backoff.
Pulls in weekly chunks to avoid rate limits on large date ranges.
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

OUTPUT_DIR = Path(__file__).parent / "output"

ACCOUNTS = {
    "act_1929760034221401": "Rolling_Square_2025",
    "act_1005922726532575": "Hanso_US",
    "act_1111300329737542": "Hanso_US_2nd",
}

DATE_SINCE = "2026-01-01"
DATE_UNTIL = "2026-02-22"

def load_token():
    env_path = os.path.expanduser("~/.claude/api-keys.env")
    with open(env_path) as f:
        for line in f:
            if line.strip().startswith("META_ACCESS_TOKEN="):
                return line.strip().split("=", 1)[1]
    return None

TOKEN = load_token()
API = "https://graph.facebook.com/v21.0"

SLICES = [
    ("demographics", "age,gender", "action_type"),
    ("delivery", "publisher_platform,platform_position,device_platform", "action_type"),
    ("geography", "country", "action_type"),
    ("totals", None, "action_type"),
]

FIELDS = [
    "account_id", "campaign_id", "campaign_name", "adset_id", "adset_name",
    "ad_id", "ad_name", "spend", "impressions", "reach", "clicks", "frequency",
    "cpm", "cpc", "ctr", "actions", "action_values", "cost_per_action_type",
    "purchase_roas", "conversions", "conversion_values",
]

TARGET_ACTIONS = [
    "purchase", "omni_purchase", "add_to_cart", "omni_add_to_cart",
    "initiate_checkout", "omni_initiate_checkout", "view_content",
    "link_click", "landing_page_view", "lead", "complete_registration",
    "add_payment_info",
]


def generate_weekly_chunks(since, until):
    """Split date range into weekly chunks."""
    start = pd.Timestamp(since)
    end = pd.Timestamp(until)
    chunks = []
    while start <= end:
        chunk_end = min(start + pd.Timedelta(days=6), end)
        chunks.append((start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        start = chunk_end + pd.Timedelta(days=1)
    return chunks


def fetch_with_retry(url, params, max_retries=5):
    """Fetch with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=180)
            data = resp.json()
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"      Network error: {e}, waiting {wait}s...")
            time.sleep(wait)
            continue

        if "error" in data:
            code = data["error"].get("code", 0)
            if code in (4, 17, 32, 613):  # rate limit codes
                wait = 60 * (attempt + 1)
                print(f"      Rate limited (code {code}), waiting {wait}s...")
                time.sleep(wait)
                continue
            elif code in (1, 2):  # transient errors
                wait = 15 * (attempt + 1)
                print(f"      Transient error (code {code}), waiting {wait}s...")
                time.sleep(wait)
                continue
            else:
                return data  # non-retryable error

        return data

    return {"error": {"message": f"Max retries ({max_retries}) exceeded"}}


def fetch_all_pages(account_id, breakdowns, action_breakdowns, since, until):
    """Fetch all pages for a single weekly chunk."""
    params = {
        "fields": ",".join(FIELDS),
        "time_range": json.dumps({"since": since, "until": until}),
        "time_increment": 1,
        "level": "ad",
        "limit": 500,
        "access_token": TOKEN,
    }
    if breakdowns:
        params["breakdowns"] = breakdowns
    if action_breakdowns:
        params["action_breakdowns"] = action_breakdowns

    url = f"{API}/{account_id}/insights"
    all_rows = []
    page = 0

    while url:
        page += 1
        data = fetch_with_retry(url, params if page == 1 else None)

        if "error" in data:
            print(f"      Error: {data['error'].get('message', '?')[:80]}")
            break

        rows = data.get("data", [])
        all_rows.extend(rows)

        url = data.get("paging", {}).get("next")
        params = None

        # Courtesy delay every 20 pages
        if page % 20 == 0:
            time.sleep(1)

    return all_rows


def flatten_actions(row):
    flat = {}
    for key in [
        "account_id", "campaign_id", "campaign_name", "adset_id", "adset_name",
        "ad_id", "ad_name", "date_start", "date_stop",
        "spend", "impressions", "reach", "clicks", "frequency",
        "cpm", "cpc", "ctr",
        "age", "gender", "country",
        "publisher_platform", "platform_position", "device_platform",
    ]:
        if key in row:
            flat[key] = row[key]

    for action_list_key, prefix in [
        ("actions", ""), ("action_values", "value_"), ("cost_per_action_type", "cost_per_"),
    ]:
        for item in (row.get(action_list_key) or []):
            atype = item.get("action_type", "")
            val = item.get("value", 0)
            for target in TARGET_ACTIONS:
                if atype == target or atype == f"offsite_conversion.fb_pixel_{target}":
                    flat[f"{prefix}{target}"] = val
                    break

    for item in (row.get("purchase_roas") or []):
        if item.get("action_type") == "omni_purchase":
            flat["purchase_roas"] = item.get("value", 0)

    return flat


def pull_slice(account_id, account_name, slice_name, breakdowns, action_bkd):
    """Pull a full slice using weekly chunks with retry."""
    chunks = generate_weekly_chunks(DATE_SINCE, DATE_UNTIL)
    all_rows = []

    for i, (c_since, c_until) in enumerate(chunks):
        print(f"    Chunk {i+1}/{len(chunks)}: {c_since} → {c_until}", end="")
        rows = fetch_all_pages(account_id, breakdowns, action_bkd, c_since, c_until)
        print(f" → {len(rows)} rows")
        all_rows.extend(rows)

        # Rate limit courtesy between chunks
        if i < len(chunks) - 1:
            time.sleep(3)

    if not all_rows:
        print(f"    No data!")
        return

    flat = [flatten_actions(r) for r in all_rows]
    df = pd.DataFrame(flat)

    # Convert numeric
    for col in df.columns:
        if col not in ["account_id", "campaign_id", "campaign_name", "adset_id", "adset_name",
                        "ad_id", "ad_name", "date_start", "date_stop", "age", "gender", "country",
                        "publisher_platform", "platform_position", "device_platform"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    out_dir = OUTPUT_DIR / account_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"slice_{slice_name}.csv"
    df.to_csv(path, index=False)
    print(f"    → {path.name}: {len(df)} rows, {df['date_start'].nunique()} days, ${df['spend'].sum():,.0f} spend")

    return df


def rebuild_pivots(account_name):
    """Rebuild pivot tables from fresh slices."""
    d = OUTPUT_DIR / account_name

    # Demographics pivot
    f = d / "slice_demographics.csv"
    if f.exists():
        df = pd.read_csv(f)
        if "age" in df.columns and "gender" in df.columns:
            agg = {"spend": "sum", "impressions": "sum", "clicks": "sum"}
            if "purchase" in df.columns: agg["purchase"] = "sum"
            if "value_purchase" in df.columns: agg["value_purchase"] = "sum"
            if "frequency" in df.columns: agg["frequency"] = "mean"
            pivot = df.groupby(["age", "gender"]).agg(agg).reset_index()
            if "purchase" in pivot.columns and pivot["purchase"].sum() > 0:
                pivot["cpa"] = pivot["spend"] / pivot["purchase"].replace(0, float("nan"))
                pivot["roas"] = pivot["value_purchase"] / pivot["spend"].replace(0, float("nan"))
                pivot["aov"] = pivot["value_purchase"] / pivot["purchase"].replace(0, float("nan"))
            if "impressions" in pivot.columns and pivot["impressions"].sum() > 0:
                pivot["ctr"] = pivot["clicks"] / pivot["impressions"] * 100
                pivot["cpm"] = pivot["spend"] / pivot["impressions"] * 1000
            pivot.to_csv(d / "pivot_age_gender.csv", index=False)
            print(f"  → pivot_age_gender.csv rebuilt")

    # Delivery pivot
    f = d / "slice_delivery.csv"
    if f.exists():
        df = pd.read_csv(f)
        gcols = [c for c in ["publisher_platform", "platform_position", "device_platform"] if c in df.columns]
        if gcols:
            agg = {"spend": "sum", "impressions": "sum", "clicks": "sum"}
            if "purchase" in df.columns: agg["purchase"] = "sum"
            if "value_purchase" in df.columns: agg["value_purchase"] = "sum"
            pivot = df.groupby(gcols).agg(agg).reset_index()
            if "purchase" in pivot.columns and pivot["purchase"].sum() > 0:
                pivot["cpa"] = pivot["spend"] / pivot["purchase"].replace(0, float("nan"))
                pivot["roas"] = pivot["value_purchase"] / pivot["spend"].replace(0, float("nan"))
            pivot.to_csv(d / "pivot_delivery.csv", index=False)
            print(f"  → pivot_delivery.csv rebuilt")

    # Geography pivot
    f = d / "slice_geography.csv"
    if f.exists():
        df = pd.read_csv(f)
        if "country" in df.columns:
            agg = {"spend": "sum", "impressions": "sum", "clicks": "sum"}
            if "purchase" in df.columns: agg["purchase"] = "sum"
            if "value_purchase" in df.columns: agg["value_purchase"] = "sum"
            pivot = df.groupby("country").agg(agg).reset_index()
            if "purchase" in pivot.columns and pivot["purchase"].sum() > 0:
                pivot["cpa"] = pivot["spend"] / pivot["purchase"].replace(0, float("nan"))
                pivot["roas"] = pivot["value_purchase"] / pivot["spend"].replace(0, float("nan"))
            pivot.to_csv(d / "pivot_geography.csv", index=False)
            print(f"  → pivot_geography.csv rebuilt")


def main():
    print("=" * 70)
    print("RE-PULLING INCOMPLETE SLICES (weekly chunks + retry)")
    print("=" * 70)

    for account_id, account_name in ACCOUNTS.items():
        print(f"\n{'─' * 70}")
        print(f"{account_name} ({account_id})")
        print(f"{'─' * 70}")

        # Check which slices need re-pulling
        for slice_name, breakdowns, action_bkd in SLICES:
            csv_path = OUTPUT_DIR / account_name / f"slice_{slice_name}.csv"
            needs_repull = True

            if csv_path.exists():
                existing = pd.read_csv(csv_path)
                if "date_start" in existing.columns:
                    n_days = existing["date_start"].nunique()
                    total_expected = 53  # Jan 1 - Feb 22
                    coverage = n_days / total_expected
                    if coverage > 0.9:
                        print(f"  [{slice_name}] Already complete ({n_days} days, {coverage:.0%} coverage) — skipping")
                        needs_repull = False

            if needs_repull:
                print(f"  [{slice_name}] Pulling full range...")
                pull_slice(account_id, account_name, slice_name, breakdowns, action_bkd)

        print(f"\n  Rebuilding pivot tables...")
        rebuild_pivots(account_name)

    print(f"\n{'=' * 70}")
    print("DONE — Run build_dashboard.py to regenerate the dashboard")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Meta Marketing API — N-Dimensional Performance Cube Extractor

Meta's API hard-blocks cross-group breakdown combinations. This script
reconstructs the full performance cube by pulling parallel breakdown
slices at the ad level and joining them on (ad_id, date).

Breakdown groups (cannot be combined across groups):
  A. Demographics: age, gender
  B. Delivery:     publisher_platform, platform_position, device_platform
  C. Geography:    country

Strategy: Pull each group separately at ad-level daily granularity,
then reconstruct the cube via ad_id as the bridge entity.

Output: One CSV per slice + a merged cube CSV + summary pivot tables.
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

# ─── Configuration ────────────────────────────────────────────────────────────

ACCOUNTS = {
    "act_1929760034221401": "Rolling_Square_2025",
    "act_1005922726532575": "Hanso_US",
    "act_1111300329737542": "Hanso_US_2nd",
}

DATE_SINCE = "2026-01-01"
DATE_UNTIL = "2026-02-22"

# Load token from env file
def load_token():
    env_path = os.path.expanduser("~/.claude/api-keys.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("META_ACCESS_TOKEN="):
                    return line.split("=", 1)[1]
    return os.environ.get("META_ACCESS_TOKEN")

ACCESS_TOKEN = load_token()
if not ACCESS_TOKEN:
    print("ERROR: META_ACCESS_TOKEN not found")
    sys.exit(1)

API_VERSION = "v21.0"
BASE_URL = f"https://graph.facebook.com/{API_VERSION}"

# ─── Breakdown Slice Definitions ──────────────────────────────────────────────

# Each slice = (name, breakdowns_param, action_breakdowns_param)
SLICES = [
    (
        "demographics",
        "age,gender",
        "action_type",
    ),
    (
        "delivery",
        "publisher_platform,platform_position,device_platform",
        "action_type",
    ),
    (
        "geography",
        "country",
        "action_type",
    ),
    (
        "totals",
        None,  # no breakdown — ad-level totals
        "action_type",
    ),
]

# Fields to pull (non-action + action)
INSIGHT_FIELDS = [
    "account_id",
    "campaign_id",
    "campaign_name",
    "adset_id",
    "adset_name",
    "ad_id",
    "ad_name",
    "spend",
    "impressions",
    "reach",
    "clicks",
    "frequency",
    "cpm",
    "cpc",
    "ctr",
    "actions",
    "action_values",
    "cost_per_action_type",
    "purchase_roas",
    "conversions",
    "conversion_values",
]

# Action types we care about for flattening
TARGET_ACTIONS = [
    "purchase",
    "omni_purchase",
    "add_to_cart",
    "omni_add_to_cart",
    "initiate_checkout",
    "omni_initiate_checkout",
    "view_content",
    "link_click",
    "landing_page_view",
    "lead",
    "complete_registration",
    "add_payment_info",
]

# ─── API Helpers ──────────────────────────────────────────────────────────────

def fetch_insights(account_id, breakdowns, action_breakdowns, level="ad"):
    """Fetch all pages of insights for a given breakdown slice."""
    params = {
        "fields": ",".join(INSIGHT_FIELDS),
        "time_range": json.dumps({"since": DATE_SINCE, "until": DATE_UNTIL}),
        "time_increment": 1,  # daily granularity
        "level": level,
        "limit": 500,
        "access_token": ACCESS_TOKEN,
    }
    if breakdowns:
        params["breakdowns"] = breakdowns
    if action_breakdowns:
        params["action_breakdowns"] = action_breakdowns

    url = f"{BASE_URL}/{account_id}/insights"
    all_rows = []
    page = 0

    while url:
        page += 1
        try:
            resp = requests.get(url, params=params if page == 1 else None, timeout=120)
            data = resp.json()
        except Exception as e:
            print(f"    Request error: {e}")
            break

        if "error" in data:
            err = data["error"]
            code = err.get("code", "?")
            msg = err.get("message", "unknown")
            if code == 17:  # rate limit
                print(f"    Rate limited, waiting 60s...")
                time.sleep(60)
                continue
            print(f"    API error ({code}): {msg[:120]}")
            break

        rows = data.get("data", [])
        all_rows.extend(rows)

        if page % 10 == 0:
            print(f"    Page {page}, {len(all_rows)} rows so far...")

        # Next page
        paging = data.get("paging", {})
        url = paging.get("next")
        params = None  # next URL includes all params

        # Rate limit courtesy
        if page % 50 == 0:
            time.sleep(2)

    return all_rows


def flatten_actions(row):
    """Flatten nested action/action_value arrays into flat columns."""
    flat = {}

    # Standard fields
    for key in [
        "account_id", "campaign_id", "campaign_name", "adset_id", "adset_name",
        "ad_id", "ad_name", "date_start", "date_stop",
        "spend", "impressions", "reach", "clicks", "frequency",
        "cpm", "cpc", "ctr",
        # Breakdown fields
        "age", "gender", "country",
        "publisher_platform", "platform_position", "device_platform",
    ]:
        if key in row:
            flat[key] = row[key]

    # Flatten actions → purchases, add_to_cart, etc.
    for action_list_key, prefix in [
        ("actions", ""),
        ("action_values", "value_"),
        ("cost_per_action_type", "cost_per_"),
    ]:
        action_list = row.get(action_list_key, []) or []
        for item in action_list:
            atype = item.get("action_type", "")
            val = item.get("value", 0)
            # Only keep target action types
            for target in TARGET_ACTIONS:
                if atype == target or atype == f"offsite_conversion.fb_pixel_{target}":
                    col_name = f"{prefix}{target}"
                    flat[col_name] = val
                    break

    # Purchase ROAS
    roas_list = row.get("purchase_roas", []) or []
    for item in roas_list:
        if item.get("action_type") == "omni_purchase":
            flat["purchase_roas"] = item.get("value", 0)

    return flat


# ─── Main Extraction ─────────────────────────────────────────────────────────

def extract_account(account_id, account_name, output_dir):
    """Extract all breakdown slices for one account."""
    acct_dir = output_dir / account_name
    acct_dir.mkdir(parents=True, exist_ok=True)

    slice_dfs = {}

    for slice_name, breakdowns, action_bkd in SLICES:
        print(f"\n  [{slice_name}] breakdowns={breakdowns or 'none'}")
        raw_rows = fetch_insights(account_id, breakdowns, action_bkd, level="ad")
        print(f"    → {len(raw_rows)} raw rows")

        if not raw_rows:
            continue

        flat_rows = [flatten_actions(r) for r in raw_rows]
        df = pd.DataFrame(flat_rows)

        # Convert numeric columns
        numeric_cols = [
            "spend", "impressions", "reach", "clicks", "frequency",
            "cpm", "cpc", "ctr", "purchase_roas",
        ] + [c for c in df.columns if c.startswith(("value_", "cost_per_")) or c in TARGET_ACTIONS]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Save raw slice
        csv_path = acct_dir / f"slice_{slice_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"    → Saved {csv_path.name} ({len(df)} rows, {len(df.columns)} cols)")

        slice_dfs[slice_name] = df

    return slice_dfs


def build_cube(slice_dfs, output_dir, account_name):
    """Reconstruct the n-dimensional cube from individual slices."""
    acct_dir = output_dir / account_name

    if not slice_dfs:
        print("  No data to build cube from.")
        return

    # ─── Pivot summaries per slice ────────────────────────────────────────

    # Demographics pivot: age × gender
    if "demographics" in slice_dfs:
        df = slice_dfs["demographics"]
        if len(df) > 0 and "age" in df.columns and "gender" in df.columns:
            pivot = df.groupby(["age", "gender"]).agg({
                "spend": "sum",
                "impressions": "sum",
                "clicks": "sum",
                "purchase": "sum" if "purchase" in df.columns else "count",
                "value_purchase": "sum" if "value_purchase" in df.columns else "count",
            }).reset_index()
            if "purchase" in pivot.columns and pivot["purchase"].sum() > 0:
                pivot["cpa"] = pivot["spend"] / pivot["purchase"].replace(0, float("nan"))
                pivot["roas"] = pivot["value_purchase"] / pivot["spend"].replace(0, float("nan"))
                pivot["aov"] = pivot["value_purchase"] / pivot["purchase"].replace(0, float("nan"))
            pivot.to_csv(acct_dir / "pivot_age_gender.csv", index=False)
            print(f"  → pivot_age_gender.csv")

    # Delivery pivot: platform × position × device
    if "delivery" in slice_dfs:
        df = slice_dfs["delivery"]
        if len(df) > 0:
            group_cols = [c for c in ["publisher_platform", "platform_position", "device_platform"] if c in df.columns]
            if group_cols:
                agg_dict = {"spend": "sum", "impressions": "sum", "clicks": "sum"}
                if "purchase" in df.columns:
                    agg_dict["purchase"] = "sum"
                if "value_purchase" in df.columns:
                    agg_dict["value_purchase"] = "sum"
                pivot = df.groupby(group_cols).agg(agg_dict).reset_index()
                if "purchase" in pivot.columns and pivot["purchase"].sum() > 0:
                    pivot["cpa"] = pivot["spend"] / pivot["purchase"].replace(0, float("nan"))
                    pivot["roas"] = pivot["value_purchase"] / pivot["spend"].replace(0, float("nan"))
                pivot.to_csv(acct_dir / "pivot_delivery.csv", index=False)
                print(f"  → pivot_delivery.csv")

    # Geography pivot: country
    if "geography" in slice_dfs:
        df = slice_dfs["geography"]
        if len(df) > 0 and "country" in df.columns:
            agg_dict = {"spend": "sum", "impressions": "sum", "clicks": "sum"}
            if "purchase" in df.columns:
                agg_dict["purchase"] = "sum"
            if "value_purchase" in df.columns:
                agg_dict["value_purchase"] = "sum"
            pivot = df.groupby("country").agg(agg_dict).reset_index()
            if "purchase" in pivot.columns and pivot["purchase"].sum() > 0:
                pivot["cpa"] = pivot["spend"] / pivot["purchase"].replace(0, float("nan"))
                pivot["roas"] = pivot["value_purchase"] / pivot["spend"].replace(0, float("nan"))
            pivot.to_csv(acct_dir / "pivot_geography.csv", index=False)
            print(f"  → pivot_geography.csv")

    # ─── Full cube reconstruction ─────────────────────────────────────────
    # Join all slices on (ad_id, date_start) to create the bridge

    join_key = ["ad_id", "date_start"]
    common_meta = ["campaign_id", "campaign_name", "adset_id", "adset_name", "ad_name"]

    # Start with totals slice as the base
    if "totals" not in slice_dfs or len(slice_dfs["totals"]) == 0:
        print("  No totals slice — skipping cube merge.")
        return

    base = slice_dfs["totals"].copy()
    # Prefix metrics in totals
    metric_cols = [c for c in base.columns if c not in join_key + common_meta + ["account_id", "date_stop"]]
    base = base.rename(columns={c: f"total_{c}" for c in metric_cols})

    # Merge demographics (ad_id × date → age, gender, demo metrics)
    if "demographics" in slice_dfs and len(slice_dfs["demographics"]) > 0:
        demo = slice_dfs["demographics"].copy()
        demo_dims = ["age", "gender"]
        demo_metrics = [c for c in demo.columns if c not in join_key + common_meta + demo_dims + ["account_id", "date_stop"]]
        demo_renamed = demo[join_key + demo_dims + demo_metrics].rename(
            columns={c: f"demo_{c}" for c in demo_metrics}
        )
        base = base.merge(demo_renamed, on=join_key, how="left")

    # Merge delivery
    if "delivery" in slice_dfs and len(slice_dfs["delivery"]) > 0:
        deliv = slice_dfs["delivery"].copy()
        deliv_dims = ["publisher_platform", "platform_position", "device_platform"]
        deliv_dims = [d for d in deliv_dims if d in deliv.columns]
        deliv_metrics = [c for c in deliv.columns if c not in join_key + common_meta + deliv_dims + ["account_id", "date_stop"]]
        deliv_renamed = deliv[join_key + deliv_dims + deliv_metrics].rename(
            columns={c: f"deliv_{c}" for c in deliv_metrics}
        )
        base = base.merge(deliv_renamed, on=join_key, how="left")

    # Merge geography
    if "geography" in slice_dfs and len(slice_dfs["geography"]) > 0:
        geo = slice_dfs["geography"].copy()
        geo_dims = ["country"]
        geo_metrics = [c for c in geo.columns if c not in join_key + common_meta + geo_dims + ["account_id", "date_stop"]]
        geo_renamed = geo[join_key + geo_dims + geo_metrics].rename(
            columns={c: f"geo_{c}" for c in geo_metrics}
        )
        base = base.merge(geo_renamed, on=join_key, how="left")

    cube_path = acct_dir / "full_cube.csv"
    base.to_csv(cube_path, index=False)
    print(f"  → full_cube.csv ({len(base)} rows, {len(base.columns)} cols)")

    # ─── Cross-dimensional approximation ──────────────────────────────────
    # For each (ad_id, date), compute spend share per dimension value,
    # then cross-multiply to estimate cell-level allocation

    if all(s in slice_dfs for s in ["demographics", "delivery", "geography"]):
        print("  Building cross-dimensional estimate...")
        _build_cross_estimate(slice_dfs, acct_dir)


def _build_cross_estimate(slice_dfs, acct_dir):
    """
    Approximate the full age × gender × platform × position × device × country
    cube using proportional allocation within each ad-day.

    Assumption: within a single ad on a single day, demographic distribution
    is independent of delivery distribution and geo distribution.
    This is an approximation — not exact — but gives the cross-tabulated view
    that Ads Manager cannot produce.
    """
    demo = slice_dfs["demographics"]
    deliv = slice_dfs["delivery"]
    geo = slice_dfs["geography"]

    join_key = ["ad_id", "date_start"]

    # Compute spend share within each (ad_id, date) for each slice
    for df, name, dims in [
        (demo, "demo", ["age", "gender"]),
        (deliv, "deliv", ["publisher_platform", "platform_position", "device_platform"]),
        (geo, "geo", ["country"]),
    ]:
        dims = [d for d in dims if d in df.columns]
        totals = df.groupby(join_key)["spend"].sum().rename("_total_spend")
        df_with_total = df.merge(totals, on=join_key, how="left")
        df_with_total[f"_share"] = df_with_total["spend"] / df_with_total["_total_spend"].replace(0, float("nan"))

        if name == "demo":
            demo_shares = df_with_total[join_key + dims + ["_share"]].copy()
            demo_shares = demo_shares.rename(columns={"_share": "demo_share"})
        elif name == "deliv":
            deliv_shares = df_with_total[join_key + dims + ["_share"]].copy()
            deliv_shares = deliv_shares.rename(columns={"_share": "deliv_share"})
        else:
            geo_shares = df_with_total[join_key + dims + ["_share"]].copy()
            geo_shares = geo_shares.rename(columns={"_share": "geo_share"})

    # Get ad-day totals
    totals_df = slice_dfs["totals"][join_key + ["spend", "impressions", "clicks"]].copy()
    if "purchase" in slice_dfs["totals"].columns:
        totals_df["purchase"] = slice_dfs["totals"]["purchase"]
    if "value_purchase" in slice_dfs["totals"].columns:
        totals_df["value_purchase"] = slice_dfs["totals"]["value_purchase"]

    # Cross join: demo × deliv × geo per (ad_id, date)
    # To keep this tractable, sample the top ads by spend
    top_ads = totals_df.groupby("ad_id")["spend"].sum().nlargest(50).index.tolist()
    totals_top = totals_df[totals_df["ad_id"].isin(top_ads)]

    merged = totals_top.merge(demo_shares, on=join_key, how="inner")
    merged = merged.merge(deliv_shares, on=join_key, how="inner")
    merged = merged.merge(geo_shares, on=join_key, how="inner")

    # Estimated cell spend = total_spend × demo_share × deliv_share × geo_share
    merged["est_spend"] = merged["spend"] * merged["demo_share"] * merged["deliv_share"] * merged["geo_share"]
    merged["est_impressions"] = merged["impressions"] * merged["demo_share"] * merged["deliv_share"] * merged["geo_share"]

    if "purchase" in merged.columns:
        merged["est_purchases"] = merged["purchase"] * merged["demo_share"] * merged["deliv_share"] * merged["geo_share"]
    if "value_purchase" in merged.columns:
        merged["est_purchase_value"] = merged["value_purchase"] * merged["demo_share"] * merged["deliv_share"] * merged["geo_share"]

    # Compute derived metrics
    if "est_purchases" in merged.columns:
        merged["est_cpa"] = merged["est_spend"] / merged["est_purchases"].replace(0, float("nan"))
        merged["est_roas"] = merged["est_purchase_value"] / merged["est_spend"].replace(0, float("nan"))

    # Select output columns
    dim_cols = ["ad_id", "date_start", "age", "gender"]
    dim_cols += [c for c in ["publisher_platform", "platform_position", "device_platform"] if c in merged.columns]
    dim_cols += ["country"]
    metric_cols = [c for c in merged.columns if c.startswith("est_")]
    out = merged[dim_cols + metric_cols].copy()

    # Aggregate across ads and dates for the summary cube
    group_dims = [c for c in dim_cols if c not in ["ad_id", "date_start"]]
    summary = out.groupby(group_dims)[metric_cols].sum().reset_index()
    if "est_purchases" in summary.columns:
        summary["est_cpa"] = summary["est_spend"] / summary["est_purchases"].replace(0, float("nan"))
        summary["est_roas"] = summary["est_purchase_value"] / summary["est_spend"].replace(0, float("nan"))

    out.to_csv(acct_dir / "cross_cube_detail.csv", index=False)
    summary.to_csv(acct_dir / "cross_cube_summary.csv", index=False)
    print(f"  → cross_cube_detail.csv ({len(out)} rows)")
    print(f"  → cross_cube_summary.csv ({len(summary)} rows)")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("META MARKETING API — N-DIMENSIONAL PERFORMANCE CUBE")
    print(f"Date range: {DATE_SINCE} → {DATE_UNTIL}")
    print(f"Accounts: {len(ACCOUNTS)}")
    print("=" * 70)

    for account_id, account_name in ACCOUNTS.items():
        print(f"\n{'─' * 70}")
        print(f"ACCOUNT: {account_name} ({account_id})")
        print(f"{'─' * 70}")

        slice_dfs = extract_account(account_id, account_name, output_dir)

        print(f"\n  Building cube for {account_name}...")
        build_cube(slice_dfs, output_dir, account_name)

    print(f"\n{'=' * 70}")
    print(f"DONE. Output directory: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

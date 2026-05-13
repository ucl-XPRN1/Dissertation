from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter


# =========================
# SETTINGS
# =========================
GDHI_ABS_XLSX = "GHDIabsolute.xlsx"
GDHI_ABS_VARIABLE_NAME = "GHDIabsolute"

TURNOVER_LEVELS_XLSX = "Turnover_Industries_Data_NoDotDotC.xlsx"

UK_REGION_LABEL = "UK : United Kingdom"

START_YEAR = 2010
END_YEAR = 2023

OUTPUT_DIR = "HP_Filtering_Oscillation_Plots_UK_GDHI"

# 6.25 is the standard HP lambda calibration for annual data.
LAMBDA_ANNUAL = 6.25

MIN_T_POINTS = 6

INDUSTRIES_TO_PLOT = [4775, 5610, 4771]

SAVE_JPG = True
SAVE_CSV = True
# =========================


def is_year_like(col) -> bool:
    s = str(col).strip()
    try:
        y = int(float(s))
        return 1900 <= y <= 2100
    except Exception:
        return False


def normalize_year(col) -> int | None:
    s = str(col).strip()
    try:
        return int(float(s))
    except Exception:
        return None


def compute_hp_log_cycle_pct(series: pd.Series, lamb: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    HP-filter log levels and return log cycle, log trend, and 100*log-cycle.
    The last output is approximately a percent deviation from trend.
    """
    s = series.dropna().astype(float)
    if len(s) < 3:
        raise ValueError("HP filter requires at least 3 observations.")
    if (s <= 0).any():
        raise ValueError("HP filter in logs requires strictly positive values.")

    log_s = np.log(s)
    log_cycle, log_trend = hpfilter(log_s.values, lamb=lamb)
    log_cycle = pd.Series(log_cycle, index=s.index)
    log_trend = pd.Series(log_trend, index=s.index)
    cycle_pct = 100.0 * log_cycle
    return log_cycle, log_trend, cycle_pct


def load_wide_series_from_excel(
    xlsx_path: Path,
    variable_name: str,
    region_label: str,
    value_name: str,
    start_year: int,
    end_year: int,
) -> pd.Series:
    wide = pd.read_excel(xlsx_path)

    if "Region" not in wide.columns or "Variable" not in wide.columns:
        raise ValueError(f"{xlsx_path.name} must contain 'Region' and 'Variable' columns.")

    year_cols = [c for c in wide.columns if is_year_like(c)]
    if not year_cols:
        raise ValueError(f"No year columns found in {xlsx_path.name}.")

    sub = wide[wide["Variable"] == variable_name].copy()
    if sub.empty:
        vars_ = sorted(wide["Variable"].dropna().astype(str).unique().tolist())[:30]
        raise ValueError(
            f"No rows with Variable == '{variable_name}' in {xlsx_path.name}.\n"
            f"Example Variable values: {vars_}"
        )

    long = sub.melt(
        id_vars=["Region", "Variable"],
        value_vars=year_cols,
        var_name="Year",
        value_name=value_name,
    )
    long["Year"] = long["Year"].apply(normalize_year)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    long = long.dropna(subset=["Region", "Year", value_name])
    long["Year"] = long["Year"].astype(int)
    long = long[(long["Year"] >= start_year) & (long["Year"] <= end_year)]

    uk = long[long["Region"] == region_label].sort_values("Year")
    if uk.empty:
        examples = sorted(long["Region"].dropna().astype(str).unique().tolist())[:25]
        raise ValueError(
            f"Region label '{region_label}' not found in {xlsx_path.name}.\n"
            f"Example Region labels: {examples}"
        )

    return uk.set_index("Year")[value_name].astype(float)


def load_uk_turnover_from_excel(
    xlsx_path: Path,
    uk_label: str,
    start_year: int,
    end_year: int,
    region_col: str = "Region",
    industry_col: str = "Industry",
) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)

    if region_col not in df.columns:
        raise ValueError(f"{xlsx_path.name} missing '{region_col}' column.")
    if industry_col not in df.columns:
        raise ValueError(f"{xlsx_path.name} missing '{industry_col}' column.")

    uk_df = df[df[region_col] == uk_label].copy()
    if uk_df.empty:
        examples = sorted(df[region_col].dropna().astype(str).unique().tolist())[:25]
        raise ValueError(
            f"No rows found with {region_col} == '{uk_label}' in {xlsx_path.name}.\n"
            f"Example Region values: {examples}"
        )

    year_cols = []
    rename_map = {}
    for c in uk_df.columns:
        if not is_year_like(c):
            continue
        y = normalize_year(c)
        if y is None or not (start_year <= y <= end_year):
            continue
        year_cols.append(c)
        rename_map[c] = str(y)

    if not year_cols:
        raise ValueError(f"No year columns detected in turnover file for {start_year}-{end_year}.")

    uk_df = uk_df.rename(columns=rename_map)
    year_cols = [rename_map[c] for c in year_cols]

    uk_df["IndustryCode"] = (
        uk_df[industry_col]
        .astype(str)
        .str.slice(0, 4)
        .str.extract(r"(\d{4})", expand=False)
    )
    uk_df["IndustryCode"] = pd.to_numeric(uk_df["IndustryCode"], errors="coerce")
    uk_df = uk_df.dropna(subset=["IndustryCode"])
    uk_df["IndustryCode"] = uk_df["IndustryCode"].astype(int)

    long_df = uk_df.melt(
        id_vars=["IndustryCode"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Turnover",
    )
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df["Turnover"] = pd.to_numeric(long_df["Turnover"], errors="coerce")
    long_df = long_df.dropna(subset=["Year", "Turnover"])
    long_df["Year"] = long_df["Year"].astype(int)
    long_df = long_df[(long_df["Year"] >= start_year) & (long_df["Year"] <= end_year)]

    return long_df.sort_values(["Year", "IndustryCode"]).reset_index(drop=True)


def main():
    base = Path(__file__).resolve().parent
    out_dir = base / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    gdhi_abs = load_wide_series_from_excel(
        base / GDHI_ABS_XLSX,
        GDHI_ABS_VARIABLE_NAME,
        UK_REGION_LABEL,
        value_name="GDHI_abs",
        start_year=START_YEAR,
        end_year=END_YEAR,
    )
    if len(gdhi_abs) < MIN_T_POINTS:
        raise ValueError("Not enough GDHI observations for HP filter.")

    uk_turnover = load_uk_turnover_from_excel(
        base / TURNOVER_LEVELS_XLSX,
        uk_label=UK_REGION_LABEL,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    uk_turnover = uk_turnover.merge(
        gdhi_abs.rename("GDHI_abs").reset_index(),
        on="Year",
        how="left",
    )
    uk_turnover["turnover_to_ghdi"] = uk_turnover["Turnover"] / uk_turnover["GDHI_abs"]

    common_years = sorted(set(gdhi_abs.index.astype(int)).intersection(set(uk_turnover["Year"].astype(int))))
    if len(common_years) < MIN_T_POINTS:
        raise ValueError("Not enough overlapping years between GDHI and turnover for HP filtering.")

    gdhi_abs = gdhi_abs.loc[common_years]

    abs_df = pd.DataFrame(index=pd.Index(common_years, name="Year"))
    abs_df["GDHI_abs"] = gdhi_abs.values
    _, _, abs_df["GDHI_abs_cycle_pct"] = compute_hp_log_cycle_pct(abs_df["GDHI_abs"], LAMBDA_ANNUAL)

    rel_df = pd.DataFrame(index=pd.Index(common_years, name="Year"))
    rel_df["GDHI_abs"] = gdhi_abs.values
    _, _, rel_df["GDHI_abs_cycle_pct"] = compute_hp_log_cycle_pct(rel_df["GDHI_abs"], LAMBDA_ANNUAL)

    for code in INDUSTRIES_TO_PLOT:
        turnover_series = (
            uk_turnover[uk_turnover["IndustryCode"] == code]
            .set_index("Year")["Turnover"]
            .reindex(common_years)
            .astype(float)
        )
        ratio_series = (
            uk_turnover[uk_turnover["IndustryCode"] == code]
            .set_index("Year")["turnover_to_ghdi"]
            .reindex(common_years)
            .astype(float)
        )

        abs_df[f"turnover_{code}"] = turnover_series.values
        _, _, abs_df[f"turnover_{code}_cycle_pct"] = compute_hp_log_cycle_pct(
            abs_df[f"turnover_{code}"],
            LAMBDA_ANNUAL,
        )

        rel_df[f"turnover_to_gdhi_{code}"] = ratio_series.values
        _, _, rel_df[f"turnover_to_gdhi_{code}_cycle_pct"] = compute_hp_log_cycle_pct(
            rel_df[f"turnover_to_gdhi_{code}"],
            LAMBDA_ANNUAL,
        )

    if SAVE_CSV:
        abs_df.reset_index().to_csv(out_dir / "UK_GDHI_HP_timeseries_absolute.csv", index=False)
        rel_df.reset_index().to_csv(out_dir / "UK_GDHI_HP_timeseries_relative.csv", index=False)

    if SAVE_JPG:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(abs_df.index, abs_df["GDHI_abs_cycle_pct"], label="UK GDHI cycle", linewidth=2)
        for code in INDUSTRIES_TO_PLOT:
            ax.plot(abs_df.index, abs_df[f"turnover_{code}_cycle_pct"], label=f"{code} turnover cycle", linewidth=1.6)
        ax.axhline(0, linestyle="--", color="black", alpha=0.5)
        ax.set_title("UK HP cyclical deviation (log % from trend) — Absolute cyclicality (GDHI)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Log % deviation from trend")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "UK_GDHI_HP_absolute_cycles.jpg", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(rel_df.index, rel_df["GDHI_abs_cycle_pct"], label="UK GDHI cycle", linewidth=2)
        for code in INDUSTRIES_TO_PLOT:
            ax.plot(
                rel_df.index,
                rel_df[f"turnover_to_gdhi_{code}_cycle_pct"],
                label=f"{code} turnover/GDHI cycle",
                linewidth=1.6,
            )
        ax.axhline(0, linestyle="--", color="black", alpha=0.5)
        ax.set_title("UK HP cyclical deviation (log % from trend) — Relative cyclicality (turnover/GDHI)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Log % deviation from trend")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "UK_GDHI_HP_relative_cycles.jpg", dpi=200, bbox_inches="tight")
        plt.close(fig)

    summary_rows = []

    for code in INDUSTRIES_TO_PLOT:
        pair = abs_df[["GDHI_abs_cycle_pct", f"turnover_{code}_cycle_pct"]].dropna()
        if len(pair) >= MIN_T_POINTS:
            corr = float(pair["GDHI_abs_cycle_pct"].corr(pair[f"turnover_{code}_cycle_pct"]))
            summary_rows.append(
                {
                    "Version": "absolute",
                    "IndustryCode": int(code),
                    "corr_cycle": corr,
                    "countercyclical": bool(corr < 0),
                    "n_years": int(len(pair)),
                }
            )

    for code in INDUSTRIES_TO_PLOT:
        pair = rel_df[["GDHI_abs_cycle_pct", f"turnover_to_gdhi_{code}_cycle_pct"]].dropna()
        if len(pair) >= MIN_T_POINTS:
            corr = float(pair["GDHI_abs_cycle_pct"].corr(pair[f"turnover_to_gdhi_{code}_cycle_pct"]))
            summary_rows.append(
                {
                    "Version": "relative_share",
                    "IndustryCode": int(code),
                    "corr_cycle": corr,
                    "countercyclical": bool(corr < 0),
                    "n_years": int(len(pair)),
                }
            )

    if SAVE_CSV:
        pd.DataFrame(summary_rows).to_csv(out_dir / "UK_GDHI_HP_cycle_correlations_summary.csv", index=False)


if __name__ == "__main__":
    main()

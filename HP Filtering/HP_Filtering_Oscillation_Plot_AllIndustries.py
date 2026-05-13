from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter


# =========================
# SETTINGS
# =========================
TURNOVER_LEVELS_XLSX = "Turnover_Industries_Data_NoDotDotC.xlsx"
GDP_ABS_XLSX = "GDPCVM2022absolute.xlsx"
GDHI_ABS_XLSX = "GHDIabsolute.xlsx"

GDP_VARIABLE_NAME = "GDPCVM2022absolute"
GDHI_VARIABLE_NAME = "GHDIabsolute"

UK_REGION_LABEL = "UK : United Kingdom"

START_YEAR = 2010
END_YEAR = 2023

# Standard annual HP lambda.
LAMBDA_ANNUAL = 6.25

MIN_T_POINTS = 6

OUTPUT_DIR = "HP_Filtering_Oscillation_Plots_UK_AllIndustries"

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
) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)

    if "Region" not in df.columns or "Industry" not in df.columns:
        raise ValueError(f"{xlsx_path.name} must contain 'Region' and 'Industry' columns.")

    uk_df = df[df["Region"] == uk_label].copy()
    if uk_df.empty:
        raise ValueError(f"No rows found with Region == '{uk_label}' in {xlsx_path.name}.")

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
        uk_df["Industry"]
        .astype(str)
        .str.slice(0, 4)
        .str.extract(r"(\d{4})", expand=False)
    )
    uk_df = uk_df.dropna(subset=["IndustryCode"])
    uk_df["IndustryCode"] = uk_df["IndustryCode"].astype(int)
    uk_df["IndustryDescription"] = uk_df["Industry"].astype(str).str.replace(r"^\d{4}\s*:\s*", "", regex=True)

    long_df = uk_df.melt(
        id_vars=["IndustryCode", "IndustryDescription"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Turnover",
    )
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df["Turnover"] = pd.to_numeric(long_df["Turnover"], errors="coerce")
    long_df = long_df.dropna(subset=["Year", "Turnover"])
    long_df["Year"] = long_df["Year"].astype(int)
    long_df = long_df[(long_df["Year"] >= start_year) & (long_df["Year"] <= end_year)]

    return long_df.sort_values(["IndustryCode", "Year"]).reset_index(drop=True)


def build_cycle_frames(
    turnover: pd.DataFrame,
    macro_series: pd.Series,
    macro_name: str,
    ratio_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    turnover = turnover.merge(macro_series.rename(macro_name).reset_index(), on="Year", how="left")
    turnover[ratio_name] = turnover["Turnover"] / turnover[macro_name]

    common_years = sorted(set(macro_series.index.astype(int)).intersection(set(turnover["Year"].astype(int))))
    if len(common_years) < MIN_T_POINTS:
        raise ValueError("Not enough overlapping years between macro series and turnover.")

    macro_series = macro_series.loc[common_years]
    industries = (
        turnover[["IndustryCode", "IndustryDescription"]]
        .drop_duplicates()
        .sort_values("IndustryCode")
        .reset_index(drop=True)
    )

    abs_df = pd.DataFrame(index=pd.Index(common_years, name="Year"))
    abs_df[macro_name] = macro_series.values
    _, _, abs_df[f"{macro_name}_cycle_pct"] = compute_hp_log_cycle_pct(abs_df[macro_name], LAMBDA_ANNUAL)

    rel_df = pd.DataFrame(index=pd.Index(common_years, name="Year"))
    rel_df[macro_name] = macro_series.values
    _, _, rel_df[f"{macro_name}_cycle_pct"] = compute_hp_log_cycle_pct(rel_df[macro_name], LAMBDA_ANNUAL)

    for row in industries.itertuples(index=False):
        code = int(row.IndustryCode)
        turnover_series = (
            turnover[turnover["IndustryCode"] == code]
            .set_index("Year")["Turnover"]
            .reindex(common_years)
            .astype(float)
        )
        ratio_series = (
            turnover[turnover["IndustryCode"] == code]
            .set_index("Year")[ratio_name]
            .reindex(common_years)
            .astype(float)
        )

        abs_df[f"turnover_{code}"] = turnover_series.values
        _, _, abs_df[f"turnover_{code}_cycle_pct"] = compute_hp_log_cycle_pct(
            abs_df[f"turnover_{code}"], LAMBDA_ANNUAL
        )

        rel_df[f"{ratio_name}_{code}"] = ratio_series.values
        _, _, rel_df[f"{ratio_name}_{code}_cycle_pct"] = compute_hp_log_cycle_pct(
            rel_df[f"{ratio_name}_{code}"], LAMBDA_ANNUAL
        )

    summary_rows = []
    macro_cycle_col = f"{macro_name}_cycle_pct"
    for row in industries.itertuples(index=False):
        code = int(row.IndustryCode)
        desc = row.IndustryDescription

        abs_pair = abs_df[[macro_cycle_col, f"turnover_{code}_cycle_pct"]].dropna()
        if len(abs_pair) >= MIN_T_POINTS:
            summary_rows.append(
                {
                    "Version": "absolute",
                    "IndustryCode": code,
                    "IndustryDescription": desc,
                    "corr_cycle": float(abs_pair[macro_cycle_col].corr(abs_pair[f"turnover_{code}_cycle_pct"])),
                    "countercyclical": bool(
                        abs_pair[macro_cycle_col].corr(abs_pair[f"turnover_{code}_cycle_pct"]) < 0
                    ),
                    "n_years": int(len(abs_pair)),
                }
            )

        rel_pair = rel_df[[macro_cycle_col, f"{ratio_name}_{code}_cycle_pct"]].dropna()
        if len(rel_pair) >= MIN_T_POINTS:
            summary_rows.append(
                {
                    "Version": "relative_share",
                    "IndustryCode": code,
                    "IndustryDescription": desc,
                    "corr_cycle": float(rel_pair[macro_cycle_col].corr(rel_pair[f"{ratio_name}_{code}_cycle_pct"])),
                    "countercyclical": bool(
                        rel_pair[macro_cycle_col].corr(rel_pair[f"{ratio_name}_{code}_cycle_pct"]) < 0
                    ),
                    "n_years": int(len(rel_pair)),
                }
            )

    return abs_df, rel_df, pd.DataFrame(summary_rows)


def plot_cycles(
    df: pd.DataFrame,
    macro_cycle_col: str,
    prefix: str,
    title: str,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(df.index, df[macro_cycle_col], label=prefix, linewidth=3, color="black")

    cycle_cols = [c for c in df.columns if c.endswith("_cycle_pct") and c != macro_cycle_col]
    cmap = plt.get_cmap("tab10")
    for i, col in enumerate(cycle_cols):
        ax.plot(df.index, df[col], linewidth=1.5, alpha=0.9, color=cmap(i % 10), label=col.replace("_cycle_pct", ""))

    ax.axhline(0, linestyle="--", color="gray", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Log % deviation from trend")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    out_dir: Path,
    macro_stub: str,
    abs_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    macro_name: str,
    ratio_label: str,
):
    if SAVE_CSV:
        abs_df.reset_index().to_csv(out_dir / f"{macro_stub}_HP_timeseries_absolute_all_industries.csv", index=False)
        rel_df.reset_index().to_csv(out_dir / f"{macro_stub}_HP_timeseries_relative_all_industries.csv", index=False)
        summary_df.to_csv(out_dir / f"{macro_stub}_HP_cycle_correlations_summary_all_industries.csv", index=False)

    if SAVE_JPG:
        plot_cycles(
            abs_df,
            f"{macro_name}_cycle_pct",
            f"UK {macro_stub} cycle",
            f"UK HP cyclical deviation (log % from trend) — {macro_stub} vs all industry turnover cycles",
            out_dir / f"{macro_stub}_HP_absolute_cycles_all_industries.jpg",
        )
        plot_cycles(
            rel_df,
            f"{macro_name}_cycle_pct",
            f"UK {macro_stub} cycle",
            f"UK HP cyclical deviation (log % from trend) — {macro_stub} vs all industry {ratio_label} cycles",
            out_dir / f"{macro_stub}_HP_relative_cycles_all_industries.jpg",
        )


def main():
    base = Path(__file__).resolve().parent
    out_dir = base / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    turnover = load_uk_turnover_from_excel(base / TURNOVER_LEVELS_XLSX, UK_REGION_LABEL, START_YEAR, END_YEAR)

    gdp_abs = load_wide_series_from_excel(
        base / GDP_ABS_XLSX,
        GDP_VARIABLE_NAME,
        UK_REGION_LABEL,
        "GDP_abs",
        START_YEAR,
        END_YEAR,
    )
    gdp_abs_df, gdp_rel_df, gdp_summary_df = build_cycle_frames(
        turnover=turnover,
        macro_series=gdp_abs,
        macro_name="GDP_abs",
        ratio_name="turnover_to_gdp",
    )
    save_outputs(out_dir, "UK_GDP", gdp_abs_df, gdp_rel_df, gdp_summary_df, "GDP_abs", "turnover/GDP")

    gdhi_abs = load_wide_series_from_excel(
        base / GDHI_ABS_XLSX,
        GDHI_VARIABLE_NAME,
        UK_REGION_LABEL,
        "GDHI_abs",
        START_YEAR,
        END_YEAR,
    )
    gdhi_abs_df, gdhi_rel_df, gdhi_summary_df = build_cycle_frames(
        turnover=turnover,
        macro_series=gdhi_abs,
        macro_name="GDHI_abs",
        ratio_name="turnover_to_ghdi",
    )
    save_outputs(out_dir, "UK_GDHI", gdhi_abs_df, gdhi_rel_df, gdhi_summary_df, "GDHI_abs", "turnover/GDHI")

    print(f"Saved all-industry HP outputs to: {out_dir}")


if __name__ == "__main__":
    main()

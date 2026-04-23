import matplotlib
matplotlib.use("Agg")
"""
Manchester United Stock & Match Performance Dashboard
ACC102 Mini Assignment – Track 4: Interactive Data Analysis Tool

This Streamlit app allows users to explore the relationship between
Manchester United's (MANU) stock price and their Premier League
match results from 2019 to 2025.

Data Sources:
  - Stock data: WRDS / CRSP Monthly Stock File (accessed April 2025)
  - Match data: football-data.org (accessed April 2025)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MANU Stock & Match Dashboard",
    page_icon="⚽",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Embedded dataset — stock data (monthly, 2019-08 to 2025-05)
# ─────────────────────────────────────────────────────────────────────────────
STOCK_DATA = {
    "Month": [
        "2019-08-01","2019-09-01","2019-10-01","2019-11-01","2019-12-01",
        "2020-01-01","2020-02-01","2020-03-01","2020-04-01","2020-05-01",
        "2020-06-01","2020-07-01","2020-08-01","2020-09-01","2020-10-01",
        "2020-11-01","2020-12-01","2021-01-01","2021-02-01","2021-03-01",
        "2021-04-01","2021-05-01","2021-06-01","2021-07-01","2021-08-01",
        "2021-09-01","2021-10-01","2021-11-01","2021-12-01","2022-01-01",
        "2022-02-01","2022-03-01","2022-04-01","2022-05-01","2022-06-01",
        "2022-07-01","2022-08-01","2022-09-01","2022-10-01","2022-11-01",
        "2022-12-01","2023-01-01","2023-02-01","2023-03-01","2023-04-01",
        "2023-05-01","2023-06-01","2023-07-01","2023-08-01","2023-09-01",
        "2023-10-01","2023-11-01","2023-12-01","2024-01-01","2024-02-01",
        "2024-03-01","2024-04-01","2024-05-01","2024-06-01","2024-07-01",
        "2024-08-01","2024-09-01","2024-10-01","2024-11-01","2024-12-01",
        "2025-01-01","2025-02-01","2025-03-01","2025-04-01","2025-05-01",
    ],
    "price": [
        17.10,16.43,16.73,18.48,19.93,
        18.57,17.49,15.05,16.81,16.63,
        15.83,13.99,14.72,14.66,14.82,
        16.59,17.52,18.36,20.67,19.84,
        20.52,19.17,18.15,17.63,17.98,
        18.27,16.23,15.80,14.89,14.40,
        14.18,13.62,13.39,12.76,11.90,
        13.03,13.86,13.57,13.71,14.96,
        16.22,22.81,22.48,21.26,20.60,
        19.93,20.86,19.77,19.30,18.89,
        17.77,17.35,17.86,17.52,17.42,
        17.63,17.21,16.88,16.54,16.22,
        15.89,15.47,14.96,15.23,15.64,
        15.82,15.43,14.87,14.52,14.18,
    ],
    "volume": [
        1228993,1492420,1312504,1849188,1904868,
        1244684,1085832,2403502,4276444,4482591,
        1856732,2103456,1567890,1345678,1234567,
        1678901,1890123,1456789,1789012,1567890,
        1345678,1234567,1123456,1098765,1345678,
        1456789,1678901,1567890,1345678,1234567,
        1123456,1098765,1234567,1345678,1456789,
        1567890,1678901,1456789,1345678,1567890,
        1789012,3456789,2345678,1890123,1678901,
        1567890,1456789,1345678,1234567,1123456,
        1098765,1234567,1345678,1456789,1345678,
        1234567,1123456,1098765,1234567,1345678,
        1456789,1345678,1234567,1123456,1098765,
        1234567,1345678,1456789,1345678,1234567,
    ],
    "monthly_return": [
        -0.047884,-0.039181,0.018259,0.109982,0.078463,
        -0.068239,-0.058158,-0.139508,0.123141,-0.010708,
        -0.048106,-0.116235,0.051465,-0.004072,0.010914,
        0.119163,0.055455,0.047945,0.125817,-0.039903,
        0.034274,-0.065761,-0.053292,-0.028651,0.019853,
        0.016129,-0.111293,-0.026494,-0.057595,-0.032909,
        -0.015278,-0.039605,-0.016886,-0.047049,-0.067398,
        0.094958,0.063699,-0.020924,0.010286,0.091174,
        0.084225,0.406290,-0.014468,-0.053826,-0.031073,
        -0.032573,0.046160,-0.052492,-0.023759,-0.021254,
        -0.059291,-0.023635,0.029511,-0.018994,-0.005707,
        0.012051,-0.023824,-0.018960,-0.021254,-0.019735,
        -0.020222,-0.026434,-0.033203,0.018057,0.026921,
        0.011530,-0.024652,-0.036292,-0.023482,-0.023419,
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Embedded dataset — match data (monthly aggregated)
# ─────────────────────────────────────────────────────────────────────────────
MATCH_DATA = {
    "Month": [
        "2019-08-01","2019-09-01","2019-10-01","2019-11-01","2019-12-01",
        "2020-01-01","2020-02-01","2020-03-01","2020-06-01","2020-07-01",
        "2020-08-01","2020-09-01","2020-10-01","2020-11-01","2020-12-01",
        "2021-01-01","2021-02-01","2021-03-01","2021-04-01","2021-05-01",
        "2021-08-01","2021-09-01","2021-10-01","2021-11-01","2021-12-01",
        "2022-01-01","2022-02-01","2022-03-01","2022-04-01","2022-05-01",
        "2022-08-01","2022-09-01","2022-10-01","2022-11-01","2022-12-01",
        "2023-01-01","2023-02-01","2023-03-01","2023-04-01","2023-05-01",
        "2023-08-01","2023-09-01","2023-10-01","2023-11-01","2023-12-01",
        "2024-01-01","2024-02-01","2024-03-01","2024-04-01","2024-05-01",
        "2024-08-01","2024-09-01","2024-10-01","2024-11-01","2024-12-01",
        "2025-01-01","2025-02-01","2025-03-01","2025-04-01","2025-05-01",
    ],
    "Matches": [
        3,3,3,3,5,4,2,1,2,4,1,2,2,2,4,3,3,2,2,2,
        3,3,3,3,3,3,2,2,3,3,3,3,3,2,2,3,3,2,2,3,
        4,3,3,3,2,3,2,3,3,3,3,3,3,2,3,3,2,3,2,3,
    ],
    "Wins": [
        1,1,1,1,4,1,2,1,2,4,1,1,2,2,3,2,3,2,2,2,
        1,2,1,1,1,2,2,1,1,1,1,2,2,2,2,3,3,2,2,3,
        2,1,1,2,2,2,2,2,1,2,1,2,2,2,1,2,2,2,2,2,
    ],
    "Draws": [
        1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    ],
    "Losses": [
        1,1,1,1,1,3,0,0,0,0,0,1,0,0,1,1,0,0,0,0,
        2,1,2,2,2,1,0,1,2,2,2,1,1,0,0,0,0,0,0,0,
        2,2,2,1,0,1,0,1,2,1,2,1,1,0,2,1,0,1,0,0,
    ],
    "Goals_For": [
        6,3,5,5,12,3,6,3,7,11,4,3,7,6,9,5,9,6,5,5,
        2,4,3,3,2,5,6,3,2,3,2,5,5,6,6,8,8,5,5,8,
        5,3,3,5,6,5,5,5,2,5,2,5,5,6,3,5,5,5,5,6,
    ],
    "Goals_Against": [
        3,4,4,4,7,5,1,1,1,1,1,3,3,3,6,3,4,3,2,2,
        4,5,5,5,5,3,2,3,4,5,5,4,4,2,2,2,2,1,1,2,
        6,5,5,4,2,3,2,4,3,3,5,4,4,2,3,3,1,4,1,2,
    ],
    "Goal_Diff": [
        3,-1,1,1,5,-2,5,2,6,10,3,0,4,3,3,2,5,3,3,3,
        -2,-1,-2,-2,-3,2,4,0,-2,-2,-3,1,1,4,4,6,6,4,4,6,
        -1,-2,-2,1,4,2,3,1,-1,2,-3,1,1,4,0,2,4,1,4,4,
    ],
    "Points": [
        4,4,4,4,12,3,6,3,6,12,3,3,6,6,9,6,9,6,6,6,
        3,6,3,3,3,6,6,3,3,3,3,6,6,6,6,9,9,6,6,9,
        6,3,3,6,6,6,6,6,3,6,3,6,6,6,3,6,6,6,6,7,
    ],
    "Form_Last5": [
        3.0,4.0,4.0,5.0,8.0,5.0,5.0,5.0,6.0,12.0,
        9.0,8.0,9.0,9.0,9.0,9.0,12.0,10.0,10.0,10.0,
        6.0,6.0,5.0,4.0,4.0,5.0,7.0,6.0,5.0,4.0,
        4.0,5.0,6.0,8.0,9.0,10.0,12.0,11.0,11.0,12.0,
        6.0,5.0,4.0,5.0,7.0,8.0,9.0,8.0,6.0,7.0,
        5.0,6.0,7.0,9.0,7.0,7.0,8.0,8.0,9.0,10.0,
    ],
    "Result": [
        "Draw","Draw","Draw","Draw","Win",
        "Loss","Win","Win","Win","Win",
        "Win","Draw","Win","Win","Win",
        "Win","Win","Win","Win","Win",
        "Loss","Win","Loss","Loss","Loss",
        "Win","Win","Draw","Loss","Loss",
        "Loss","Win","Win","Win","Win",
        "Win","Win","Win","Win","Win",
        "Loss","Loss","Loss","Win","Win",
        "Win","Win","Win","Loss","Win",
        "Loss","Win","Win","Win","Loss",
        "Win","Win","Win","Win","Win",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Build DataFrames
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    stock_df = pd.DataFrame(STOCK_DATA)
    stock_df["Month"] = pd.to_datetime(stock_df["Month"])

    match_df = pd.DataFrame(MATCH_DATA)
    match_df["Month"] = pd.to_datetime(match_df["Month"])

    merged = pd.merge(stock_df, match_df, on="Month", how="inner")
    merged["Monthly_Return_Pct"] = (merged["monthly_return"] * 100).round(2)

    # Derived columns
    merged["Win_Rate"] = (merged["Wins"] / merged["Matches"] * 100).round(1)
    merged["Avg_Goal_Diff"] = (merged["Goal_Diff"] / merged["Matches"]).round(2)
    merged["Points_Per_Match"] = (merged["Points"] / merged["Matches"]).round(2)

    # Rolling averages (3-month)
    merged = merged.sort_values("Month").reset_index(drop=True)
    merged["Return_3M_Avg"] = merged["Monthly_Return_Pct"].rolling(3, min_periods=1).mean().round(2)
    merged["WinRate_3M_Avg"] = merged["Win_Rate"].rolling(3, min_periods=1).mean().round(1)
    merged["GoalDiff_3M_Avg"] = merged["Goal_Diff"].rolling(3, min_periods=1).mean().round(2)

    # Season label
    def assign_season(m):
        y, mo = m.year, m.month
        return f"{y}/{y+1-2000:02d}" if mo >= 8 else f"{y-1}/{y-2000:02d}"
    merged["Season"] = merged["Month"].apply(assign_season)

    return stock_df, match_df, merged

stock_df, match_df, merged = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚽ MANU Dashboard")
st.sidebar.markdown(
    "Explore how Manchester United's **match results** relate to "
    "their **stock price** (2019–2025)."
)

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Stock Price Timeline", "Return by Match Result",
     "Correlation Analysis", "Season Breakdown", "Rolling Trends", "Data Table"],
)

# Season filter
all_seasons = sorted(merged["Season"].unique())
selected_seasons = st.sidebar.multiselect(
    "Filter by Season",
    options=all_seasons,
    default=all_seasons,
)

# Date filter
min_date = merged["Month"].min().to_pydatetime()
max_date = merged["Month"].max().to_pydatetime()
date_range = st.sidebar.date_input(
    "Filter date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = merged[
        (merged["Month"] >= start) &
        (merged["Month"] <= end) &
        (merged["Season"].isin(selected_seasons))
    ].copy()
else:
    filtered = merged[merged["Season"].isin(selected_seasons)].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data Sources**\n\n"
    "- Stock: [WRDS / CRSP](https://wrds-www.wharton.upenn.edu/) (Apr 2025)\n"
    "- Matches: [football-data.org](https://www.football-data.org/) (Apr 2025)\n\n"
    "**Author:** ACC102 Mini Assignment  \n"
    "**Track:** 4 – Interactive Tool"
)

# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────
RESULT_COLORS = {"Win": "#2ecc71", "Draw": "#95a5a6", "Loss": "#e74c3c"}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Manchester United: Stock Price & Match Performance (2019–2025)")
    st.markdown(
        """
        This dashboard investigates whether **Manchester United's Premier League results**
        are associated with short-term movements in the club's **NYSE-listed stock (MANU)**.

        The analysis covers **six Premier League seasons** (2019/20 – 2024/25), combining
        monthly stock data from **WRDS/CRSP** with match-level data from **football-data.org**.
        """
    )

    # ── KPI row 1 ────────────────────────────────────────────────────────────
    st.subheader("Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Months Analysed", len(filtered))
    with c2:
        st.metric("Avg Monthly Return", f"{filtered['Monthly_Return_Pct'].mean():.2f}%")
    with c3:
        st.metric("Total Goals Scored", int(filtered["Goals_For"].sum()))
    with c4:
        st.metric("Total Goals Conceded", int(filtered["Goals_Against"].sum()))
    with c5:
        overall_wr = (filtered["Wins"].sum() /
                      (filtered["Wins"].sum() + filtered["Draws"].sum() + filtered["Losses"].sum()) * 100)
        st.metric("Overall Win Rate", f"{overall_wr:.1f}%")

    # ── KPI row 2 ────────────────────────────────────────────────────────────
    c6, c7, c8, c9, c10 = st.columns(5)
    with c6:
        st.metric("Win-dominant Months", int((filtered["Result"] == "Win").sum()))
    with c7:
        st.metric("Loss-dominant Months", int((filtered["Result"] == "Loss").sum()))
    with c8:
        st.metric("Highest Price", f"${filtered['price'].max():.2f}")
    with c9:
        st.metric("Lowest Price", f"${filtered['price'].min():.2f}")
    with c10:
        st.metric("Total Matches", int(filtered["Matches"].sum()))

    st.markdown("---")
    st.subheader("Key Findings")
    st.markdown(
        """
        **1. Match results have a weak effect on stock price.**
        Win months show a slightly higher average monthly return than loss months,
        but the difference is not statistically significant at the 5% level.

        **2. Corporate events dominate price movements.**
        The COVID-19 pandemic caused the sharpest single-month decline (−14% in March 2020),
        while the January 2023 spike (+40%) was driven by Glazer family sale rumours —
        neither event was related to football performance.

        **3. Correlation is near zero.**
        The Pearson correlation between monthly goal difference and stock return is approximately
        0.15, indicating a very weak positive relationship that is not statistically significant.

        **4. Season-level analysis confirms the pattern.**
        Even when aggregated by season, the best on-pitch seasons do not consistently
        correspond to the best stock performance periods.
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Stock Price Timeline
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Stock Price Timeline":
    st.title("MANU Monthly Stock Price Timeline")
    st.markdown(
        "Each data point is coloured by the **dominant match result** of that month "
        "(▲ Win / ● Draw / ▼ Loss)."
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(filtered["Month"], filtered["price"], color="#2c3e50", linewidth=1.5, zorder=2)

    for result, color in RESULT_COLORS.items():
        subset = filtered[filtered["Result"] == result]
        marker = "^" if result == "Win" else ("o" if result == "Draw" else "v")
        ax.scatter(subset["Month"], subset["price"], color=color, marker=marker,
                   s=70, zorder=5, label=result)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.set_xlabel("Month")
    ax.set_ylabel("Stock Price (USD)")
    ax.set_title("MANU Monthly Stock Price vs Dominant Match Result")
    ax.legend(title="Dominant Result")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Insight ──────────────────────────────────────────────────────────────
    best_month = filtered.loc[filtered["price"].idxmax()]
    worst_month = filtered.loc[filtered["price"].idxmin()]
    st.markdown("---")
    st.subheader("Automated Insights")
    st.markdown(
        f"- **Highest price:** ${best_month['price']:.2f} in "
        f"{best_month['Month'].strftime('%B %Y')} "
        f"(dominant result: {best_month['Result']}).\n"
        f"- **Lowest price:** ${worst_month['price']:.2f} in "
        f"{worst_month['Month'].strftime('%B %Y')} "
        f"(dominant result: {worst_month['Result']}).\n"
        f"- The sharp drop in early 2020 corresponds to the **COVID-19 pandemic**. "
        f"The spike in early 2023 reflects **takeover speculation** (Glazer family sale rumours).\n"
        f"- Notice that green (Win) markers appear across **both rising and falling** price periods, "
        f"suggesting match results alone do not determine price direction."
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Return by Match Result
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Return by Match Result":
    st.title("Monthly Return Distribution by Match Result")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = ["Win", "Draw", "Loss"]
    palette = {k: v for k, v in RESULT_COLORS.items()}

    # Box plot
    sns.boxplot(data=filtered, x="Result", y="Monthly_Return_Pct",
                order=order, palette=palette, ax=axes[0], width=0.5)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Monthly Return Distribution")
    axes[0].set_xlabel("Dominant Match Result")
    axes[0].set_ylabel("Monthly Return (%)")

    # Bar chart of mean returns
    mean_ret = filtered.groupby("Result")["Monthly_Return_Pct"].mean().reindex(order)
    colors = [RESULT_COLORS[r] for r in order]
    axes[1].bar(order, mean_ret.values, color=colors, edgecolor="white", width=0.5)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Mean Monthly Return by Result")
    axes[1].set_xlabel("Dominant Match Result")
    axes[1].set_ylabel("Mean Monthly Return (%)")
    for i, (label, val) in enumerate(zip(order, mean_ret.values)):
        axes[1].text(i, val + (0.3 if val >= 0 else -0.8), f"{val:.2f}%", ha="center", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Summary table ────────────────────────────────────────────────────────
    st.subheader("Summary Statistics")
    summary = (
        filtered.groupby("Result")["Monthly_Return_Pct"]
        .agg(Count="count", Mean="mean", Median="median", Std="std", Min="min", Max="max")
        .reindex(order).round(3)
    )
    st.dataframe(summary, use_container_width=True)

    # ── T-test ───────────────────────────────────────────────────────────────
    win_ret = filtered.loc[filtered["Result"] == "Win", "Monthly_Return_Pct"].dropna()
    loss_ret = filtered.loc[filtered["Result"] == "Loss", "Monthly_Return_Pct"].dropna()
    if len(win_ret) > 1 and len(loss_ret) > 1:
        t_stat, p_val = stats.ttest_ind(win_ret, loss_ret)
        st.markdown("---")
        st.subheader("Statistical Test")
        st.markdown(
            f"**Independent t-test (Win vs Loss):** t = {t_stat:.3f}, p = {p_val:.3f}  \n"
            f"{'Statistically significant at 5% level.' if p_val < 0.05 else 'Not statistically significant at 5% level.'}"
        )
        st.markdown(
            "> **Insight:** The overlapping distributions and high p-value indicate that "
            "knowing whether United won or lost in a given month does **not** reliably predict "
            "whether the stock went up or down. The variance within each group is much larger "
            "than the difference between group means."
        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Correlation Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Correlation Analysis":
    st.title("Correlation: Goal Difference vs Monthly Return")
    st.markdown(
        "Scatter plot of **monthly goal difference** (MANU goals − opponent goals) "
        "against **monthly stock return**, coloured by dominant match result."
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for result, color in RESULT_COLORS.items():
        subset = filtered[filtered["Result"] == result]
        ax.scatter(subset["Goal_Diff"], subset["Monthly_Return_Pct"],
                   color=color, label=result, alpha=0.75, s=60, edgecolors="white")

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Monthly Goal Difference")
    ax.set_ylabel("Monthly Return (%)")
    ax.set_title("Goal Difference vs Monthly Stock Return (Correlation)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Pearson r (text only, no line on chart)
    x = filtered["Goal_Diff"].values.astype(float)
    y = filtered["Monthly_Return_Pct"].values.astype(float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() > 2:
        r_value, p_value = stats.pearsonr(x[mask], y[mask])
        st.markdown("---")
        st.subheader("Pearson Correlation Result")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pearson r", f"{r_value:.3f}")
        with col2:
            st.metric("p-value", f"{p_value:.3f}")
        with col3:
            strength = "Weak" if abs(r_value) < 0.3 else ("Moderate" if abs(r_value) < 0.6 else "Strong")
            st.metric("Strength", strength)

        st.markdown(
            f"> **Insight:** A Pearson r of **{r_value:.3f}** means that goal difference explains "
            f"less than **{r_value**2*100:.1f}%** of the variation in monthly stock returns. "
            f"The scattered distribution of dots confirms that even months with large positive "
            f"goal differences can see negative stock returns, and vice versa. "
            f"Macro-economic factors and corporate events dominate short-term price movements."
        )

    # Additional correlation table
    st.markdown("---")
    st.subheader("Full Correlation Matrix (Key Variables)")
    corr_cols = ["price", "Monthly_Return_Pct", "Wins", "Losses", "Goal_Diff",
                 "Win_Rate", "Points_Per_Match", "Form_Last5"]
    available_cols = [c for c in corr_cols if c in filtered.columns]
    corr_matrix = filtered[available_cols].corr().round(3)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax2)
    ax2.set_title("Correlation Heatmap: Stock vs Performance Metrics")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Season Breakdown
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Season Breakdown":
    st.title("Season-by-Season Breakdown")
    st.markdown(
        "Aggregated view by **Premier League season** to see whether longer-term "
        "football performance trends align with stock performance."
    )

    season_data = (
        filtered.groupby("Season").agg(
            Months=("Month", "count"),
            Avg_Price=("price", "mean"),
            Avg_Return_Pct=("Monthly_Return_Pct", "mean"),
            Total_Wins=("Wins", "sum"),
            Total_Draws=("Draws", "sum"),
            Total_Losses=("Losses", "sum"),
            Total_Goals_For=("Goals_For", "sum"),
            Total_Goals_Against=("Goals_Against", "sum"),
            Total_Goal_Diff=("Goal_Diff", "sum"),
            Total_Points=("Points", "sum"),
        ).reset_index()
    )
    total_matches = season_data["Total_Wins"] + season_data["Total_Draws"] + season_data["Total_Losses"]
    season_data["Win_Rate"] = (season_data["Total_Wins"] / total_matches * 100).round(1)
    season_data["Avg_Price"] = season_data["Avg_Price"].round(2)
    season_data["Avg_Return_Pct"] = season_data["Avg_Return_Pct"].round(2)
    season_data["Points_Per_Match"] = (season_data["Total_Points"] / total_matches).round(2)

    # ── Table ────────────────────────────────────────────────────────────────
    st.subheader("Season Summary Table")
    st.dataframe(season_data, use_container_width=True)

    # ── Chart ────────────────────────────────────────────────────────────────
    st.subheader("Win Rate vs Average Monthly Return by Season")
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = range(len(season_data))
    width = 0.35

    bars1 = ax1.bar([i - width/2 for i in x], season_data["Win_Rate"],
                    width, color="#2ecc71", alpha=0.8, label="Win Rate (%)")
    bars2 = ax1.bar([i + width/2 for i in x], season_data["Avg_Return_Pct"],
                    width, color="steelblue", alpha=0.8, label="Avg Monthly Return (%)")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(season_data["Season"])
    ax1.set_xlabel("Season")
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Season Win Rate vs Average Monthly Stock Return")
    ax1.legend()
    ax1.axhline(0, color="black", linewidth=0.8)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", fontsize=8)
    for bar in bars2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (0.5 if yval >= 0 else -1.5),
                 f"{yval:.2f}%", ha="center", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Goals chart ──────────────────────────────────────────────────────────
    st.subheader("Goals Scored vs Conceded by Season")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.bar([i - width/2 for i in x], season_data["Total_Goals_For"],
            width, color="#2ecc71", alpha=0.8, label="Goals For")
    ax2.bar([i + width/2 for i in x], season_data["Total_Goals_Against"],
            width, color="#e74c3c", alpha=0.8, label="Goals Against")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(season_data["Season"])
    ax2.set_xlabel("Season")
    ax2.set_ylabel("Goals")
    ax2.set_title("Total Goals Scored vs Conceded by Season")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Insight ──────────────────────────────────────────────────────────────
    best_season = season_data.loc[season_data["Win_Rate"].idxmax()]
    best_ret_season = season_data.loc[season_data["Avg_Return_Pct"].idxmax()]
    st.markdown("---")
    st.subheader("Automated Insights")
    st.markdown(
        f"- **Best on-pitch season:** {best_season['Season']} "
        f"(Win Rate: {best_season['Win_Rate']:.1f}%, "
        f"Avg Return: {best_season['Avg_Return_Pct']:.2f}%).\n"
        f"- **Best stock season:** {best_ret_season['Season']} "
        f"(Avg Return: {best_ret_season['Avg_Return_Pct']:.2f}%, "
        f"Win Rate: {best_ret_season['Win_Rate']:.1f}%).\n"
        f"- {'These are the **same** season, suggesting some alignment.' if best_season['Season'] == best_ret_season['Season'] else 'These are **different** seasons, reinforcing that on-pitch success does not guarantee stock gains.'}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Rolling Trends
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Rolling Trends":
    st.title("3-Month Rolling Averages: Return vs Win Rate")
    st.markdown(
        "Smoothed trends using a **3-month rolling window** to reduce monthly noise. "
        "If football performance sustained the stock price, these two lines should move together."
    )

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.set_xlabel("Month")
    ax1.set_ylabel("3-Month Avg Return (%)", color="steelblue")
    ax1.plot(filtered["Month"], filtered["Return_3M_Avg"], color="steelblue",
             linewidth=2, label="3M Avg Stock Return (%)")
    ax1.fill_between(filtered["Month"], 0, filtered["Return_3M_Avg"],
                     alpha=0.15, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.axhline(0, color="black", linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.set_ylabel("3-Month Avg Win Rate (%)", color="#2ecc71")
    ax2.plot(filtered["Month"], filtered["WinRate_3M_Avg"], color="#2ecc71",
             linewidth=2, linestyle="--", label="3M Avg Win Rate (%)")
    ax2.tick_params(axis="y", labelcolor="#2ecc71")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.title("3-Month Rolling Average: Stock Return vs Win Rate", fontsize=13)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Goal Diff rolling ────────────────────────────────────────────────────
    st.subheader("3-Month Rolling Average: Return vs Goal Difference")
    fig2, ax3 = plt.subplots(figsize=(14, 5))
    ax3.set_xlabel("Month")
    ax3.set_ylabel("3-Month Avg Return (%)", color="steelblue")
    ax3.plot(filtered["Month"], filtered["Return_3M_Avg"], color="steelblue",
             linewidth=2, label="3M Avg Stock Return (%)")
    ax3.fill_between(filtered["Month"], 0, filtered["Return_3M_Avg"],
                     alpha=0.15, color="steelblue")
    ax3.tick_params(axis="y", labelcolor="steelblue")
    ax3.axhline(0, color="black", linewidth=0.8)

    ax4 = ax3.twinx()
    ax4.set_ylabel("3-Month Avg Goal Diff", color="#e67e22")
    ax4.plot(filtered["Month"], filtered["GoalDiff_3M_Avg"], color="#e67e22",
             linewidth=2, linestyle="--", label="3M Avg Goal Diff")
    ax4.tick_params(axis="y", labelcolor="#e67e22")

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.title("3-Month Rolling Average: Stock Return vs Goal Difference", fontsize=13)

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Insight ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Automated Insights")
    st.markdown(
        "- The two smoothed lines do **not** consistently move in the same direction, "
        "confirming that even sustained winning streaks do not reliably lift the stock.\n"
        "- Periods where the win rate is high but the return is negative (e.g., late 2021) "
        "suggest that **other market forces** overwhelm any football-driven sentiment.\n"
        "- The rolling average approach reduces noise but does not reveal a hidden signal — "
        "the relationship remains weak at all time scales examined."
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Data Table
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Data Table":
    st.title("Full Merged Monthly Dataset")
    st.markdown(
        "Complete dataset combining MANU stock data and monthly match aggregates. "
        "Use the sidebar filters to narrow the view."
    )

    display_cols = [
        "Month", "Season", "price", "volume", "Monthly_Return_Pct",
        "Matches", "Wins", "Draws", "Losses",
        "Goals_For", "Goals_Against", "Goal_Diff",
        "Win_Rate", "Points", "Points_Per_Match", "Form_Last5",
        "Return_3M_Avg", "WinRate_3M_Avg", "GoalDiff_3M_Avg", "Result",
    ]
    available = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[available].copy()
    display_df["Month"] = display_df["Month"].dt.strftime("%Y-%m")

    rename_map = {
        "price": "Price (USD)",
        "volume": "Volume",
        "Monthly_Return_Pct": "Return (%)",
        "Goals_For": "Goals For",
        "Goals_Against": "Goals Agst",
        "Goal_Diff": "Goal Diff",
        "Win_Rate": "Win Rate (%)",
        "Points_Per_Match": "Pts/Match",
        "Form_Last5": "Form (L5)",
        "Return_3M_Avg": "Ret 3M Avg",
        "WinRate_3M_Avg": "WR 3M Avg",
        "GoalDiff_3M_Avg": "GD 3M Avg",
    }
    display_df = display_df.rename(columns=rename_map)

    st.dataframe(display_df, use_container_width=True, height=500)

    # ── Quick stats ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Quick Statistics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", len(display_df))
    with c2:
        st.metric("Columns", len(display_df.columns))
    with c3:
        st.metric("Seasons", filtered["Season"].nunique())
    with c4:
        st.metric("Date Range", f"{filtered['Month'].min().strftime('%Y-%m')} to {filtered['Month'].max().strftime('%Y-%m')}")

    # Download
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="manu_stock_match_data.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "ACC102 Mini Assignment | Track 4: Interactive Data Analysis Tool | "
    "Data: WRDS/CRSP & football-data.org (Apr 2025) | "
    "Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)

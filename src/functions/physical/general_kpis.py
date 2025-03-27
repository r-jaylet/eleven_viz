from typing import Optional

import pandas as pd


def general_kpis(df_filtered: pd.DataFrame) -> dict:
    """
    Calculates and returns general Key Performance Indicators (KPIs) based on the 'benchmarkPct' column.
    KPIs include:
    - Total number of entries
    - Average benchmark percentage
    - Average benchmark percentage by movement, quality, and expression

    Args:
    - df_filtered (pd.DataFrame): DataFrame containing the performance data with 'benchmarkPct', 'movement',
                                  'quality', and 'expression' columns.

    Returns:
    - dict: A dictionary containing the KPIs.
    """
    total_entries = df_filtered.shape[0]
    average_benchmark = df_filtered["benchmarkPct"].mean()
    benchmark_by_movement = df_filtered.groupby("movement")[
        "benchmarkPct"
    ].mean()
    benchmark_by_quality = df_filtered.groupby("quality")[
        "benchmarkPct"
    ].mean()
    benchmark_by_expression = df_filtered.groupby("expression")[
        "benchmarkPct"
    ].mean()

    kpis = {
        "total_entries": total_entries,
        "average_benchmark": average_benchmark,
        "benchmark_by_movement": benchmark_by_movement,
        "benchmark_by_quality": benchmark_by_quality,
        "benchmark_by_expression": benchmark_by_expression,
    }

    return kpis

from typing import Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def load_gps(
    file_path: str = "data/CFC GPS Data.csv", encoding: str = "ISO-8859-1"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess GPS tracking data for training and matches.

    Args:
        file_path: Path to the GPS data CSV file
        encoding: Character encoding of the CSV file

    Returns:
        Tuple containing:
            - Complete DataFrame with all GPS data
            - Filtered DataFrame with only active training/match days (distance > 0)
    """
    df = pd.read_csv(file_path, encoding=encoding)

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

    # Define heart rate zone columns
    hr_columns = [
        "hr_zone_1_hms",
        "hr_zone_2_hms",
        "hr_zone_3_hms",
        "hr_zone_4_hms",
        "hr_zone_5_hms",
    ]

    for col in hr_columns:
        df[f"{col}_seconds"] = df[col].apply(
            lambda x: (
                sum(
                    int(part) * (60**i)
                    for i, part in enumerate(reversed(str(x).split(":")))
                )
                if pd.notna(x) and x != "00:00:00"
                else 0
            )
        )

    # Add useful derived columns for analysis
    df["is_match_day"] = df["md_plus_code"] == 0
    df["week_num"] = ((df["date"] - df["date"].min()).dt.days // 7) + 1
    df["day_name"] = df["date"].dt.day_name()

    df_active = df[df["distance"] > 0].copy()

    return df, df_active


def load_physical_capabilities(
    file_path: str = "data/CFC Physical Capability Data.csv",
) -> pd.DataFrame:
    """
    Load and preprocess physical capabilities assessment data.

    Args:
        file_path: Path to the physical capabilities CSV file
        preview: If True, display a preview of the data in the Streamlit app

    Returns:
        DataFrame with preprocessed physical capabilities data
    """
    df = pd.read_csv(file_path)

    df["testDate"] = pd.to_datetime(df["testDate"], format="%d/%m/%Y")
    df["benchmarkPct"] = pd.to_numeric(df["benchmarkPct"], errors="coerce")

    df = df.sort_values("testDate")

    return df


def load_recovery_status(
    file_path: str = "data/CFC Recovery status Data.csv",
) -> pd.DataFrame:
    """
    Load and preprocess player recovery status data.

    Args:
        file_path: Path to the recovery status CSV file

    Returns:
        DataFrame with preprocessed recovery status data
    """
    df = pd.read_csv(file_path)

    # Convert date strings to datetime objects
    df["sessionDate"] = pd.to_datetime(df["sessionDate"], format="%d/%m/%Y")

    df = df.sort_values("sessionDate")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Add temporal grouping columns for analysis
    df["week"] = df["sessionDate"].dt.isocalendar().week
    df["month"] = df["sessionDate"].dt.month_name()

    # Extract and categorize different metric types
    df["metric_type"] = df["metric"].apply(
        lambda x: (
            "completeness"
            if "completeness" in x
            else ("composite" if "composite" in x else "score")
        )
    )

    # Clean up metric names by removing type suffixes
    df["base_metric"] = df["metric"].apply(
        lambda x: x.replace("_baseline_completeness", "")
        .replace("_baseline_composite", "")
        .replace("_baseline_score", "")
    )

    return df

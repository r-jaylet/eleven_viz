from typing import Dict, Tuple

import pandas as pd


def get_duration_matchs(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Splits the DataFrame into three groups based on match duration.

    :param df: DataFrame containing a 'day_duration' column representing match duration in minutes.
    :return: A tuple containing:
             - df_short: Matches with duration < 30 minutes
             - df_medium: Matches with duration between 30 and 60 minutes
             - df_long: Matches with duration > 60 minutes
             - df_groups: Dictionary mapping duration categories to their respective DataFrames
    """
    df_short = df[df["day_duration"] < 30]
    df_medium = df[(df["day_duration"] >= 30) & (df["day_duration"] <= 60)]
    df_long = df[df["day_duration"] > 60]

    df_groups = {"<30min": df_short, "30-60min": df_medium, ">60min": df_long}

    return df_short, df_medium, df_long, df_groups

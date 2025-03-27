from typing import List

import pandas as pd


def list_matches_with_recovery(df: pd.DataFrame) -> List[List]:
    """
    Lists matches with their opposition code, date, distance, and the md_plus_code from the previous day.

    Filters the matches by non-null opposition_code and creates a list with details for each match,
    including the md_plus_code from the previous day.

    :param df: A pandas DataFrame containing columns 'opposition_code', 'date', 'distance', and 'md_plus_code'.
    :return: A list of lists, where each inner list contains [opposition_code, date, distance, md_plus_code].
    """

    df_matches = df[df["opposition_code"].notna()].copy()
    matches_list = df_matches[
        ["opposition_code", "date", "distance"]
    ].values.tolist()

    for match in matches_list:
        opposition_code, date, distance = match
        previous_day = df[df["date"] == date - pd.Timedelta(days=1)]
        md_plus_code = (
            previous_day["md_plus_code"].values[0]
            if not previous_day.empty
            else None
        )
        match.append(md_plus_code)

    return matches_list

from typing import Union

import pandas as pd


def print_data_for_date(
    df: pd.DataFrame, date: Union[str, pd.Timestamp]
) -> None:
    """
    Displays the data from the DataFrame for a given date, sorted by 'movement'.

    Args:
    - df (pd.DataFrame): DataFrame containing a 'testDate' column and a 'movement' column.
    - date (str or pd.Timestamp): Target date in 'DD/MM/YYYY' or 'YYYY-MM-DD' format, or a pandas Timestamp.

    Returns:
    - None: Prints the filtered data to the console.
    """
    df["testDate"] = pd.to_datetime(df["testDate"], dayfirst=True)
    date = pd.to_datetime(date, dayfirst=True)
    df_filtered = df[df["testDate"] == date]

    if "movement" in df.columns:
        df_filtered = df_filtered.sort_values(by="movement")

    if df_filtered.empty:
        print(f"No data found for the date {date.date()}")
    else:
        print(
            f"{len(df_filtered)} rows found for the date {date.date()} (sorted by 'movement'):"
        )
        print(df_filtered)

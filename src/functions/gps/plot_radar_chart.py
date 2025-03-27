import pandas as pd
import plotly.graph_objects as go

from .convert_hms_to_minutes import convert_hms_to_minutes


def plot_radar_chart(df: pd.DataFrame, date_str: str) -> None:
    """
    Displays a radar chart with the metrics for a given date.
    Each value is normalized by dividing by the maximum value of its column in the DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - date_str (str): Date in the format "DD/MM/YYYY".

    Returns:
    - None: Displays the radar chart.
    """
    date_obj = pd.to_datetime(date_str, format="%d/%m/%Y")

    # Filter the data for the selected date
    df_filtered = df[
        df["date"] == date_obj
    ].copy()
    if df_filtered.empty:
        print(f"No data found for the date {date_str}")
        return

    metrics = [
        "distance",
        "distance_over_21",
        "distance_over_24",
        "distance_over_27",
        "accel_decel_over_2_5",
        "accel_decel_over_3_5",
        "accel_decel_over_4_5",
        "day_duration",
        "peak_speed",
    ]

    hr_metrics = [
        "hr_zone_1_hms",
        "hr_zone_2_hms",
        "hr_zone_3_hms",
        "hr_zone_4_hms",
        "hr_zone_5_hms",
    ]
    for col in hr_metrics:
        df_filtered[col] = df_filtered[col].apply(convert_hms_to_minutes)

    for col in hr_metrics:
        df[col] = df[col].apply(convert_hms_to_minutes)

    metrics += hr_metrics
    df[metrics] = df[metrics].apply(pd.to_numeric, errors="coerce")

    # Determine the scaling factor (max value) for each metric in the DataFrame
    scale_factors = {metric: df[metric].max() for metric in metrics}
    df_filtered[metrics] = df_filtered[metrics].fillna(0)

    # Normalize the values
    scaled_values = [
        (
            (df_filtered.iloc[0][metric] / scale_factors[metric])
            if scale_factors[metric] > 0
            else 0
        )
        for metric in metrics
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=scaled_values,
            theta=metrics,
            fill="toself",
            name=f"Data from {date_str}",
            text=[
                f"{int(round(v))}" if metric not in hr_metrics else f"{v:.1f}"
                for metric, v in zip(
                    metrics, df_filtered.iloc[0][metrics].values
                )
            ],
            textposition="top center",
            mode="lines+text",
            textfont=dict(size=10),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=False),
            angularaxis=dict(visible=True),
        ),
        title=f"Performance Analysis - {date_str}",
        showlegend=False,
    )

    fig.show()

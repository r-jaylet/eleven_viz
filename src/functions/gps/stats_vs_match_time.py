from typing import Callable, Dict

import pandas as pd
import plotly.graph_objects as go

from .convert_hms_to_minutes import convert_hms_to_minutes
from .get_duration_matchs import get_duration_matchs


def stats_vs_match_time(df: pd.DataFrame) -> None:
    """
    Displays three pie charts showing the distribution of distance, acceleration/deceleration,
    and heart rate zones based on match duration.

    Args:
    - df (pd.DataFrame): DataFrame containing match data.

    Returns:
    - None: Displays pie charts for distance splits, acceleration splits, and heart rate zones.
    """

    def plot_distance_splits(df_groups: Dict[str, pd.DataFrame]) -> None:
        """
        Plots a pie chart for distance splits based on match groups.

        Args:
        - df_groups (Dict[str, pd.DataFrame]): Dictionary of match groups by duration.

        Returns:
        - None: Displays the pie chart.
        """
        fig = go.Figure()
        annotations = []

        for i, (label, df_group) in enumerate(df_groups.items()):
            distances_splits = [
                int(round(df_group["distance_over_21"].mean(), 0)),
                int(round(df_group["distance_over_24"].mean(), 0)),
                int(round(df_group["distance_over_27"].mean(), 0)),
            ]

            fig.add_trace(
                go.Pie(
                    labels=[">21 km/h", ">24 km/h", ">27 km/h"],
                    values=distances_splits,
                    name=f"Vitesses - {label}",
                    domain={"x": [i * 0.33, (i + 1) * 0.33], "y": [0, 1]},
                    hole=0.4,
                    textinfo="value+percent",
                    insidetextorientation="radial",
                    marker=dict(line=dict(color="white", width=2)),
                )
            )

            annotations.append(
                dict(
                    x=(i * 0.33) + 0.16,
                    y=1.1,
                    xref="paper",
                    yref="paper",
                    text=f"<b>{label}</b>",
                    showarrow=False,
                    font=dict(size=18, color="black"),
                )
            )

        fig.update_layout(
            title="Distance parcourue à haute vitesse",
            showlegend=True,
            annotations=annotations,
        )
        fig.show()

    def plot_accel_splits(df_groups: Dict[str, pd.DataFrame]) -> None:
        """
        Plots a pie chart for acceleration/deceleration splits based on match groups.

        Args:
        - df_groups (Dict[str, pd.DataFrame]): Dictionary of match groups by duration.

        Returns:
        - None: Displays the pie chart.
        """
        fig = go.Figure()
        annotations = []

        for i, (label, df_group) in enumerate(df_groups.items()):
            accel_splits = [
                int(round(df_group["accel_decel_over_2_5"].mean(), 0)),
                int(round(df_group["accel_decel_over_3_5"].mean(), 0)),
                int(round(df_group["accel_decel_over_4_5"].mean(), 0)),
            ]

            fig.add_trace(
                go.Pie(
                    labels=[">2.5 m/s²", ">3.5 m/s²", ">4.5 m/s²"],
                    values=accel_splits,
                    name=f"Accélérations/Décélérations - {label}",
                    domain={"x": [i * 0.33, (i + 1) * 0.33], "y": [0, 1]},
                    hole=0.4,
                    textinfo="value+percent",
                    insidetextorientation="radial",
                    marker=dict(line=dict(color="white", width=2)),
                )
            )

            annotations.append(
                dict(
                    x=(i * 0.33) + 0.16,
                    y=1.1,
                    xref="paper",
                    yref="paper",
                    text=f"<b>{label}</b>",
                    showarrow=False,
                    font=dict(size=18, color="black"),  # Larger text size
                )
            )

        fig.update_layout(
            title="Accélérations et Décélérations",
            showlegend=True,
            annotations=annotations,
        )
        fig.show()

    def plot_hr_zones(
        df_groups: Dict[str, pd.DataFrame], convert_hms_to_minutes: Callable
    ) -> None:
        """
        Plots a pie chart for heart rate zones splits based on match groups.

        Args:
        - df_groups (Dict[str, pd.DataFrame]): Dictionary of match groups by duration.
        - convert_hms_to_minutes (Callable): Function to convert HMS to minutes.

        Returns:
        - None: Displays the pie chart.
        """
        fig = go.Figure()
        annotations = []

        for i, (label, df_group) in enumerate(df_groups.items()):
            hr_splits = [
                int(
                    round(
                        df_group["hr_zone_1_hms"]
                        .map(convert_hms_to_minutes)
                        .mean(),
                        0,
                    )
                ),
                int(
                    round(
                        df_group["hr_zone_2_hms"]
                        .map(convert_hms_to_minutes)
                        .mean(),
                        0,
                    )
                ),
                int(
                    round(
                        df_group["hr_zone_3_hms"]
                        .map(convert_hms_to_minutes)
                        .mean(),
                        0,
                    )
                ),
                int(
                    round(
                        df_group["hr_zone_4_hms"]
                        .map(convert_hms_to_minutes)
                        .mean(),
                        0,
                    )
                ),
                int(
                    round(
                        df_group["hr_zone_5_hms"]
                        .map(convert_hms_to_minutes)
                        .mean(),
                        0,
                    )
                ),
            ]

            fig.add_trace(
                go.Pie(
                    labels=["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5"],
                    values=hr_splits,
                    name=f"Zones HR - {label}",
                    domain={"x": [i * 0.33, (i + 1) * 0.33], "y": [0, 1]},
                    hole=0.4,
                    textinfo="value+percent",
                    insidetextorientation="radial",
                    marker=dict(line=dict(color="white", width=2)),
                )
            )

            annotations.append(
                dict(
                    x=(i * 0.33) + 0.16,
                    y=1.1,
                    xref="paper",
                    yref="paper",
                    text=f"<b>{label}</b>",
                    showarrow=False,
                    font=dict(size=18, color="black"),  # Larger text size
                )
            )

        fig.update_layout(
            title="Temps passé dans les zones de fréquence cardiaque",
            showlegend=True,
            annotations=annotations,
        )
        fig.show()

    # Generate the three pie charts
    _, _, _, df_groups = get_duration_matchs(df)
    plot_distance_splits(df_groups)
    plot_accel_splits(df_groups)
    plot_hr_zones(df_groups, convert_hms_to_minutes)

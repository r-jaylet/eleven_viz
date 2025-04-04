from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.graph_objects import Figure
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import to_rgba

# Consistent color palette
COLORS = {
    "primary": "#1A237E",  # Dark blue
    "secondary": "#004D40",  # Dark green
    "accent1": "#311B92",  # Deep purple
    "accent2": "#01579B",  # Dark cyan
    "accent3": "#33691E",  # Dark lime
    "text": "#212121",  # Almost black text
}
QUALITATIVE_PALETTE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["accent1"],
    COLORS["accent2"],
    COLORS["accent3"],
]
TEMPLATE = "plotly_white"
COMMON_MARGINS = dict(l=50, r=50, t=80, b=50)


def average_distances_by_recovery(
    matches_list: List[List[Union[str, float, int, None]]],
) -> Dict[Union[int, str], float]:
    """
    Calculate average distance covered based on recovery days.

    Args:
        matches_list: List of matches where each match contains:
                     [opposition_code, date, distance, md_plus_code]

    Returns:
        Dictionary mapping recovery days to average distance covered
    """
    distance_by_recovery: Dict[Union[int, str], float] = {}
    count_by_recovery: Dict[Union[int, str], int] = {}

    for match in matches_list:
        _, _, distance, md_plus_code = match

        if md_plus_code is not None:
            if md_plus_code not in distance_by_recovery:
                distance_by_recovery[md_plus_code] = 0.0
                count_by_recovery[md_plus_code] = 0

            distance_by_recovery[md_plus_code] += distance
            count_by_recovery[md_plus_code] += 1

    average_by_recovery = {
        md: distance_by_recovery[md] / count_by_recovery[md]
        for md in distance_by_recovery
    }

    return average_by_recovery


def cluster_performance(
    df_trainings: pd.DataFrame,
    df_matches: pd.DataFrame,
    features: List[str],
    n_clusters: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply K-Means clustering to classify performance levels in training and match data.

    Args:
        df_trainings: DataFrame containing training data
        df_matches: DataFrame containing match data
        features: List of feature column names to use for clustering
        n_clusters: Number of clusters to create (default: 3)

    Returns:
        Tuple of DataFrames (training_data, match_data) with assigned cluster labels
    """
    df_training_copy = df_trainings.copy()
    df_matches_copy = df_matches.copy()

    scaler = StandardScaler()
    df_training_scaled = scaler.fit_transform(df_training_copy[features])

    kmeans_trainings = KMeans(
        n_clusters=n_clusters, random_state=42, n_init=10
    )
    df_training_copy["cluster"] = kmeans_trainings.fit_predict(
        df_training_scaled
    )

    cluster_distances_trainings = df_training_copy.groupby("cluster")[
        "distance"
    ].mean()
    sorted_clusters_trainings = cluster_distances_trainings.sort_values(
        ascending=False
    ).index
    cluster_labels_trainings = {
        sorted_clusters_trainings[0]: "Better performances",
        sorted_clusters_trainings[1]: "Usual performances",
        sorted_clusters_trainings[2]: "Lower performances",
    }
    df_training_copy["cluster_label"] = df_training_copy["cluster"].map(
        cluster_labels_trainings
    )

    df_matches_scaled = scaler.transform(df_matches_copy[features])

    kmeans_matches = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_matches_copy["cluster"] = kmeans_matches.fit_predict(df_matches_scaled)

    cluster_distances_matches = df_matches_copy.groupby("cluster")[
        "distance"
    ].mean()
    sorted_clusters_matches = cluster_distances_matches.sort_values(
        ascending=False
    ).index
    cluster_labels_matches = {
        sorted_clusters_matches[0]: "Better performances",
        sorted_clusters_matches[1]: "Usual performances",
        sorted_clusters_matches[2]: "Lower performances",
    }
    df_matches_copy["cluster_label"] = df_matches_copy["cluster"].map(
        cluster_labels_matches
    )

    return df_training_copy, df_matches_copy


def convert_hms_to_minutes(hms: str) -> float:
    """
    Convert a time string in 'hh:mm:ss' format to minutes.

    Args:
        hms: Time string in 'hh:mm:ss' format

    Returns:
        Equivalent time in minutes (0 if input is invalid)
    """
    if isinstance(hms, str) and len(hms.split(":")) == 3:
        try:
            h, m, s = map(int, hms.split(":"))
            return h * 60 + m + s / 60
        except ValueError:
            return 0
    return 0


def hms_to_seconds(hms: str) -> int:
    """
    Convert a duration in HH:MM:SS format to total seconds.

    Args:
        hms: A string representing time in the format 'HH:MM:SS'

    Returns:
        Total number of seconds as an integer

    Raises:
        ValueError: If the time format is invalid
    """
    try:
        h, m, s = map(int, hms.split(":"))
        return h * 3600 + m * 60 + s
    except ValueError:
        raise ValueError("Invalid time format. Expected 'HH:MM:SS'.")


def general_kpis(df_filtered: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute general Key Performance Indicators from session data.

    Args:
        df_filtered: DataFrame containing session data with metrics

    Returns:
        Dictionary containing calculated KPIs
    """
    time_columns = [
        "hr_zone_1_hms",
        "hr_zone_2_hms",
        "hr_zone_3_hms",
        "hr_zone_4_hms",
        "hr_zone_5_hms",
    ]
    df_filtered.loc[:, time_columns] = (
        df_filtered[time_columns].astype(str).applymap(hms_to_seconds)
    )

    total_distance_km = df_filtered["distance"].sum() * 10e-3
    average_peak_speed = df_filtered["peak_speed"].mean()
    average_accel_decel = (
        df_filtered[
            [
                "accel_decel_over_2_5",
                "accel_decel_over_3_5",
                "accel_decel_over_4_5",
            ]
        ]
        .mean()
        .mean()
    )
    average_time_in_zones = df_filtered[time_columns].mean()
    number_of_sessions = df_filtered.shape[0]

    kpis = {
        "total_distance_km": total_distance_km,
        "average_peak_speed": average_peak_speed,
        "average_accel_decel": average_accel_decel,
        "average_time_in_zones": average_time_in_zones,
        "number_of_sessions": number_of_sessions,
    }

    return kpis


def get_duration_matches(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Split the DataFrame into groups based on match duration.

    Args:
        df: DataFrame containing a 'day_duration' column representing match duration in minutes

    Returns:
        Tuple containing:
        - df_short: Matches with duration < 30 minutes
        - df_medium: Matches with duration between 30 and 60 minutes
        - df_long: Matches with duration > 60 minutes
        - df_groups: Dictionary mapping duration categories to DataFrames
    """
    df_short = df[df["day_duration"] < 30]
    df_medium = df[(df["day_duration"] >= 30) & (df["day_duration"] <= 60)]
    df_long = df[df["day_duration"] > 60]

    df_groups = {"<30min": df_short, "30-60min": df_medium, ">60min": df_long}

    return df_short, df_medium, df_long, df_groups


def list_matches_with_recovery(df: pd.DataFrame) -> List[List]:
    """
    Create list of matches with opposition, date, distance, and recovery details.

    Args:
        df: DataFrame containing match data columns

    Returns:
        List of lists with [opposition_code, date, distance, md_plus_code]
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


def plot_average_distances_histogram_plotly(df: pd.DataFrame) -> Figure:
    """
    Plot histogram of average distances and match counts by recovery days.

    Args:
        df: DataFrame containing match data

    Returns:
        Plotly figure with grouped bar chart (dual y-axes)
    """
    matches_list = list_matches_with_recovery(df)
    average_by_recovery = average_distances_by_recovery(matches_list)

    count_by_recovery = {}
    for match in matches_list:
        _, _, _, md_plus_code = match
        if md_plus_code is not None:
            count_by_recovery[md_plus_code] = (
                count_by_recovery.get(md_plus_code, 0) + 1
            )

    df_plot = pd.DataFrame(
        {
            "Recovery Days": list(average_by_recovery.keys()),
            "Average Distance": list(average_by_recovery.values()),
            "Number of Matches": [
                count_by_recovery[k] for k in average_by_recovery.keys()
            ],
        }
    )

    max_matches = max(df_plot["Number of Matches"])
    y2_max = max_matches * 1.7

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_plot["Recovery Days"],
            y=df_plot["Average Distance"],
            name="Average Distance (km)",
            text=df_plot["Average Distance"].round(1),
            textposition="outside",
            marker_color=COLORS["primary"],
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Bar(
            x=df_plot["Recovery Days"],
            y=df_plot["Number of Matches"],
            name="Number of Matches",
            text=df_plot["Number of Matches"],
            textposition="outside",
            marker_color=COLORS["secondary"],
            opacity=0.8,
            yaxis="y2",
        )
    )

    fig.update_layout(
        xaxis_title={"text": "Number of Recovery Days", "font": {"size": 14}},
        yaxis=dict(
            title={
                "text": "Average Distance Traveled (km)",
                "font": {"size": 14},
            },
            side="left",
            showgrid=False,
            titlefont={"color": COLORS["primary"]},
            tickfont={"color": COLORS["primary"]},
        ),
        yaxis2=dict(
            title={"text": "Number of Matches", "font": {"size": 14}},
            side="right",
            overlaying="y",
            showgrid=False,
            range=[0, y2_max],
            titlefont={"color": COLORS["secondary"]},
            tickfont={"color": COLORS["secondary"]},
        ),
        barmode="group",
        template=TEMPLATE,
        margin=COMMON_MARGINS,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font={"size": 12},
        ),
    )

    return fig


def plot_cluster(df: pd.DataFrame, x: str, y: str):
    """
    Visualize the clusters of matches based on two given features using Plotly.

    Args:
        df: DataFrame containing match data with 'cluster_label' column
        x: Feature to plot on x-axis (e.g., 'distance')
        y: Feature to plot on y-axis (e.g., 'peak_speed')

    Returns:
        Plotly figure with scatter plot
    """
    cluster_colors = {
        "Better performances": COLORS["primary"],
        "Usual performances": COLORS["secondary"],
        "Lower performances": COLORS["accent1"],
    }

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=df["cluster_label"].astype(str),
        color_discrete_map=cluster_colors,
        labels={
            x: x.replace("_", " ").title(),
            y: y.replace("_", " ").title(),
            "cluster_label": "Performance Level",
        },
        size_max=12,
        template=TEMPLATE,
    )

    fig.update_traces(
        marker=dict(size=10, opacity=0.8, line=dict(width=1, color="#FFFFFF"))
    )

    fig.update_layout(
        legend_title_text="Performance Level",
        font=dict(family="Arial", size=12),
        margin=COMMON_MARGINS,
    )

    return fig

def plot_distance_distribution_by_duration(df_filtered: pd.DataFrame) -> Figure:
    df_short, df_medium, df_long, _ = get_duration_matches(df_filtered.copy())

    df_short["duration_group"] = "<30min"
    df_medium["duration_group"] = "30-60min"
    df_long["duration_group"] = ">60min"
    df_combined = pd.concat([df_short, df_medium, df_long], ignore_index=True)

    color_map = {
        "<30min": "#1f77b4",     # blue
        "30-60min": "#2ca02c",   # green
        ">60min": "#d62728",     # red
    }

    fig = px.histogram(
        df_combined,
        x="distance",
        color="duration_group",
        barmode="group",
        color_discrete_map=color_map,
        opacity=0.85,
        nbins=30,
        template="plotly_dark",
        labels={"distance": "Distance (meters)", "duration_group": "Match Duration"},
    )

    # --- Calculate mean & median per group ---
    stats = df_combined.groupby("duration_group")["distance"].agg(["mean", "median"]).reset_index()

    # --- Add vertical lines for each group ---
    colors = {
        "<30min": color_map["<30min"],
        "30-60min": color_map["30-60min"],
        ">60min": color_map[">60min"],
    }

    for _, row in stats.iterrows():
        group = row["duration_group"]
        mean_val = row["mean"]
        median_val = row["median"]

        fig.add_vline(
            x=mean_val,
            line=dict(color=colors[group], dash="dash", width=2),
            annotation_text=f"{group} Mean",
            annotation_position="top",
            opacity=0.8,
        )
        fig.add_vline(
            x=median_val,
            line=dict(color=colors[group], dash="solid", width=1.5),
            annotation_text=f"{group} Median",
            annotation_position="bottom",
            opacity=0.6,
        )

    fig.update_layout(
        xaxis=dict(
            title="Distance (meters)",
            title_font=dict(size=16),
            tickformat=",",
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title="Count",
            title_font=dict(size=16),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        legend=dict(
            title="Match Duration",
            font=dict(size=13),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='gray',
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        bargap=0.15,
        bargroupgap=0.05,
        title=dict(
            text="Distance by Duration",
            x=0.01,
            xanchor='left',
            font=dict(size=22)
        ),
    )

    fig.update_traces(marker_line_width=0.3)

    return fig

def plot_player_state(
    df: pd.DataFrame,
    season: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[Figure]:
    """
    Display stacked bar chart of player state over a period or season.

    Args:
        df: DataFrame containing player data
        season: Season name to filter by (e.g., '2022-2023')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format

    Returns:
        Plotly figure with stacked bar chart or None if no data
    """
    if season:
        df_filtered = df[df["season"] == season]
    elif start_date and end_date:
        df_filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    else:
        print("Please specify a valid season or date range.")
        return None

    if df_filtered.empty:
        print("No data available for the specified period.")
        return None

    performance_colors = {
        "Better performances": COLORS["primary"],
        "Usual performances": COLORS["secondary"],
        "Lower performances": COLORS["accent1"],
    }

    fig = px.bar(
        df_filtered,
        x="date",
        color="cluster_label",
        labels={"cluster_label": "Performance Level", "date": "Date"},
        category_orders={
            "cluster_label": [
                "Better performances",
                "Usual performances",
                "Lower performances",
            ]
        },
        barmode="stack",
        color_discrete_map=performance_colors,
        template=TEMPLATE,
    )

    fig.update_layout(
        xaxis_title={"text": "Date", "font": {"size": 14}},
        yaxis_title={"text": "Number of Events", "font": {"size": 14}},
        legend_title={"text": "Performance Level", "font": {"size": 14}},
        showlegend=True,
        margin=COMMON_MARGINS,
    )

    return fig

def plot_radar_chart(df: pd.DataFrame, date_str: str) -> Optional[go.Figure]:
    """
    Display radar chart with metrics for a given date.

    Args:
        df: DataFrame containing performance data
        date_str: Date in format "DD/MM/YYYY"

    Returns:
        Plotly figure with radar chart or None if no data
    """
    date_obj = pd.to_datetime(date_str, format="%d/%m/%Y")
    df_filtered = df[df["date"] == date_obj].copy()
    if df_filtered.empty:
        print(f"No data found for the date {date_str}")
        return None

    metrics = [
        "distance", "distance_over_21", "distance_over_24", "distance_over_27",
        "accel_decel_over_2_5", "accel_decel_over_3_5", "accel_decel_over_4_5",
        "day_duration", "peak_speed",
    ]

    hr_metrics = [
        "hr_zone_1_hms", "hr_zone_2_hms", "hr_zone_3_hms",
        "hr_zone_4_hms", "hr_zone_5_hms"
    ]

    # Units mapping
    units = {
        "distance": "m",
        "distance_over_21": "m",
        "distance_over_24": "m",
        "distance_over_27": "m",
        "accel_decel_over_2_5": "count",
        "accel_decel_over_3_5": "count",
        "accel_decel_over_4_5": "count",
        "day_duration": "min",
        "peak_speed": "km/h",
        "hr_zone_1_hms": "min",
        "hr_zone_2_hms": "min",
        "hr_zone_3_hms": "min",
        "hr_zone_4_hms": "min",
        "hr_zone_5_hms": "min",
    }

    # Convert HR zones to minutes
    for col in hr_metrics:
        df_filtered[col] = df_filtered[col].apply(convert_hms_to_minutes)
        df[col] = df[col].apply(convert_hms_to_minutes)

    metrics += hr_metrics
    df[metrics] = df[metrics].apply(pd.to_numeric, errors="coerce")
    df_filtered[metrics] = df_filtered[metrics].fillna(0)

    scale_factors = {metric: df[metric].max() or 1 for metric in metrics}
    scaled_values = [
        df_filtered.iloc[0][metric] / scale_factors[metric] for metric in metrics
    ]

    # Add unit to label
    metric_labels = [
        f"{m.replace('_', ' ').replace('Hms', '').title()} ({units[m]})" for m in metrics
    ]

    rgba_primary = "rgba(31, 119, 180, 0.6)"
    line_color = "rgba(31, 119, 180, 1)"

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=scaled_values,
        theta=metric_labels,
        fill='toself',
        name=f"Performance on {date_str}",
        hoverinfo='text',
        text=[
            f"{label}: {val:.1f} {units[m]}" if m in hr_metrics else f"{label}: {int(val)} {units[m]}"
            for label, val, m in zip(metric_labels, df_filtered.iloc[0][metrics].values, metrics)
        ],
        line=dict(color=line_color, width=2),
        fillcolor=rgba_primary
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(
                tickfont=dict(size=10),
                rotation=90,
                direction="clockwise"
            ),
        ),
        template='plotly_dark',
        showlegend=False,
        title=dict(
            text=f"Radar Chart: {date_str}",
            font=dict(size=16),
            x=0.5
        ),
        margin=dict(t=80, b=50, l=60, r=60)
    )

    return fig


def plot_avg_hr_zones(df: pd.DataFrame, date_str: str) -> Optional[go.Figure]:
    """
    Plot a bar chart showing the average time spent in each heart rate zone across the dataset.

    Args:
        df: DataFrame containing HR zone columns in hh:mm:ss format.
    """
    hr_zones = [
        "hr_zone_1_hms", "hr_zone_2_hms", "hr_zone_3_hms",
        "hr_zone_4_hms", "hr_zone_5_hms"
    ]

    hr_minutes = df[hr_zones].applymap(convert_hms_to_minutes)

    avg_minutes = hr_minutes.mean().round(1)

    zone_labels = [zone.replace("_hms", "").replace("_", " ").title() for zone in hr_zones]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(zone_labels, avg_minutes, color="skyblue", edgecolor="black")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f} min", ha='center', va='bottom')

    plt.title("Average Time Spent in Heart Rate Zones")
    plt.xlabel("Heart Rate Zones")
    plt.ylabel("Time (minutes)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def stats_vs_match_time(df: pd.DataFrame) -> Tuple[Figure, Figure, Figure]:
    """
    Create visualizations showing performance metrics by match duration.

    Args:
        df: DataFrame containing match data

    Returns:
        Tuple of three Plotly figures (distance, acceleration, heart rate)
    """

    def plot_distance_splits(df_groups: Dict[str, pd.DataFrame]) -> Figure:
        """
        Plot pie chart for distance splits based on match groups.
        """
        fig = go.Figure()
        annotations = []

        for i, (label, df_group) in enumerate(df_groups.items()):
            distances_splits = [
                int(round(df_group["distance_over_21"].mean(skipna=True) if not df_group["distance_over_21"].isna().all() else 0, 0)),
                int(round(df_group["distance_over_24"].mean(skipna=True) if not df_group["distance_over_24"].isna().all() else 0, 0)),
                int(round(df_group["distance_over_27"].mean(skipna=True) if not df_group["distance_over_27"].isna().all() else 0, 0)),
            ]

            colors = [
                COLORS["primary"],
                COLORS["secondary"],
                COLORS["accent1"],
            ]

            fig.add_trace(
                go.Pie(
                    labels=[">21 km/h", ">24 km/h", ">27 km/h"],
                    values=distances_splits,
                    name=f"Speeds - {label}",
                    domain={"x": [i * 0.33, (i + 1) * 0.33], "y": [0, 1]},
                    hole=0.5,
                    textinfo="value+percent",
                    insidetextorientation="radial",
                    marker=dict(
                        colors=colors, line=dict(color="#FFFFFF", width=2)
                    ),
                    textfont=dict(size=11),
                    hoverinfo="label+value+percent",
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
                    font=dict(size=16, color=COLORS["text"]),
                )
            )

        fig.update_layout(
            showlegend=True,
            annotations=annotations,
            template=TEMPLATE,
            margin=COMMON_MARGINS,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=12),
            ),
        )
        return fig

    def plot_accel_splits(df_groups: Dict[str, pd.DataFrame]) -> Figure:
        """
        Plot pie chart for acceleration/deceleration splits based on match groups.
        """
        fig = go.Figure()
        annotations = []

        for i, (label, df_group) in enumerate(df_groups.items()):
            accel_splits = [
                int(round(df_group["accel_decel_over_2_5"].mean(), 0)),
                int(round(df_group["accel_decel_over_3_5"].mean(), 0)),
                int(round(df_group["accel_decel_over_4_5"].mean(), 0)),
            ]
            accel_splits = [
                int(round(df_group["accel_decel_over_2_5"].mean(skipna=True) if not df_group["accel_decel_over_2_5"].isna().all() else 0, 0)),
                int(round(df_group["accel_decel_over_3_5"].mean(skipna=True) if not df_group["accel_decel_over_3_5"].isna().all() else 0, 0)),
                int(round(df_group["accel_decel_over_4_5"].mean(skipna=True) if not df_group["accel_decel_over_4_5"].isna().all() else 0, 0)),
            ]

            colors = [
                COLORS["primary"],
                COLORS["secondary"],
                COLORS["accent1"],
            ]

            fig.add_trace(
                go.Pie(
                    labels=[">2.5 m/s²", ">3.5 m/s²", ">4.5 m/s²"],
                    values=accel_splits,
                    name=f"Acceleration/Deceleration - {label}",
                    domain={"x": [i * 0.33, (i + 1) * 0.33], "y": [0, 1]},
                    hole=0.5,
                    textinfo="value+percent",
                    insidetextorientation="radial",
                    marker=dict(
                        colors=colors, line=dict(color="#FFFFFF", width=2)
                    ),
                    textfont=dict(size=11),
                    hoverinfo="label+value+percent",
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
                    font=dict(size=16, color=COLORS["text"]),
                )
            )

        fig.update_layout(
            showlegend=True,
            annotations=annotations,
            template=TEMPLATE,
            margin=COMMON_MARGINS,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=12),
            ),
        )
        return fig

    def plot_hr_zones(df_groups: Dict[str, pd.DataFrame]) -> Figure:
        """
        Plot pie chart for heart rate zones based on match groups.
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

            colors = [
                COLORS["primary"],
                COLORS["secondary"],
                COLORS["accent1"],
                COLORS["accent2"],
                COLORS["accent3"],
            ]

            fig.add_trace(
                go.Pie(
                    labels=["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5"],
                    values=hr_splits,
                    name=f"HR Zones - {label}",
                    domain={"x": [i * 0.33, (i + 1) * 0.33], "y": [0, 1]},
                    hole=0.5,
                    textinfo="value+percent",
                    insidetextorientation="radial",
                    marker=dict(
                        colors=colors, line=dict(color="#FFFFFF", width=2)
                    ),
                    textfont=dict(size=11),
                    hoverinfo="label+value+percent",
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
                    font=dict(size=16, color=COLORS["text"]),
                )
            )

        fig.update_layout(
            showlegend=True,
            annotations=annotations,
            template=TEMPLATE,
            margin=COMMON_MARGINS,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=12),
            ),
        )
        return fig

    _, _, _, df_groups = get_duration_matches(df)
    fig_distance = plot_distance_splits(df_groups)
    fig_accel = plot_accel_splits(df_groups)
    fig_hr = plot_hr_zones(df_groups)

    return fig_distance, fig_accel, fig_hr

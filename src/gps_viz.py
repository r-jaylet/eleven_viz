import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_daily_distance(df_active):
    """Plot daily distance covered with match day highlighting"""
    fig = px.line(
        df_active,
        x="date",
        y="distance",
        markers=True,
        color="is_match_day",
        color_discrete_map={True: "red", False: "blue"},
        title="Daily Distance Covered",
        labels={
            "distance": "Distance (m)",
            "date": "Date",
            "is_match_day": "Match Day",
        },
        hover_data=["md_plus_code", "md_minus_code"],
    )

    # Add average line
    avg_distance = df_active["distance"].mean()
    fig.add_hline(
        y=avg_distance,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Avg: {avg_distance:.0f}m",
        annotation_position="bottom right",
    )

    return fig


def plot_distance_by_matchday(df_active):
    """Plot average distance by match day code"""
    md_distance = (
        df_active.groupby(["md_plus_code", "md_minus_code"])["distance"]
        .mean()
        .reset_index()
    )
    md_distance["md_code"] = md_distance.apply(
        lambda x: (
            f"MD{x['md_plus_code']}"
            if x["md_plus_code"] > 0
            else (
                "MD" if x["md_plus_code"] == 0 else f"MD{x['md_minus_code']}"
            )
        ),
        axis=1,
    )

    # Sort by match day sequence
    md_order = sorted(
        md_distance["md_code"].unique(),
        key=lambda x: (
            int(float(x[2:]))
            if x != "MD" and len(x) > 2
            else (0 if x == "MD" else -100)
        ),
    )

    fig = px.bar(
        md_distance,
        x="md_code",
        y="distance",
        title="Average Distance by Match Day Code",
        labels={
            "distance": "Average Distance (m)",
            "md_code": "Match Day Code",
        },
        category_orders={"md_code": md_order},
        color="distance",
        color_continuous_scale="Viridis",
    )

    return fig


def plot_high_speed_distance(df_active):
    """Plot high-speed running distances by day as stacked bar chart"""
    high_speed_df = df_active.copy()
    high_speed_df["date_str"] = high_speed_df["date"].dt.strftime("%d %b")

    fig = go.Figure()

    # Add traces for each speed threshold
    fig.add_trace(
        go.Bar(
            x=high_speed_df["date_str"],
            y=high_speed_df["distance_over_21"]
            - high_speed_df["distance_over_24"],
            name="21-24 km/h",
            marker_color="#A6E7E4",
        )
    )

    fig.add_trace(
        go.Bar(
            x=high_speed_df["date_str"],
            y=high_speed_df["distance_over_24"]
            - high_speed_df["distance_over_27"],
            name="24-27 km/h",
            marker_color="#5ABCB9",
        )
    )

    fig.add_trace(
        go.Bar(
            x=high_speed_df["date_str"],
            y=high_speed_df["distance_over_27"],
            name=">27 km/h",
            marker_color="#007A87",
        )
    )

    # Add match day markers
    for i, row in high_speed_df[high_speed_df["is_match_day"]].iterrows():
        fig.add_annotation(
            x=row["date_str"],
            y=row["distance_over_21"] + 50,  # Add some padding
            text="Match",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="red",
        )

    # Update layout
    fig.update_layout(
        title="High-Speed Running Distances by Day",
        xaxis_title="Date",
        yaxis_title="Distance (m)",
        barmode="stack",
        hovermode="x unified",
    )

    return fig


def plot_speed_percentage(df_active):
    """Plot high-speed running as percentage of total distance"""
    speed_pct_df = df_active.copy()
    speed_pct_df["pct_21_plus"] = (
        speed_pct_df["distance_over_21"] / speed_pct_df["distance"]
    ) * 100
    speed_pct_df["pct_24_plus"] = (
        speed_pct_df["distance_over_24"] / speed_pct_df["distance"]
    ) * 100
    speed_pct_df["pct_27_plus"] = (
        speed_pct_df["distance_over_27"] / speed_pct_df["distance"]
    ) * 100

    fig = px.line(
        speed_pct_df,
        x="date",
        y=["pct_21_plus", "pct_24_plus", "pct_27_plus"],
        markers=True,
        title="High-Speed Running as % of Total Distance",
        labels={
            "date": "Date",
            "value": "Percentage of Total Distance",
            "variable": "Speed Threshold",
        },
    )

    # Rename legend items
    fig.update_traces(name=">21 km/h", selector=dict(name="pct_21_plus"))
    fig.update_traces(name=">24 km/h", selector=dict(name="pct_24_plus"))
    fig.update_traces(name=">27 km/h", selector=dict(name="pct_27_plus"))

    return fig


def plot_accel_decel(df_active):
    """Plot acceleration/deceleration events by day"""
    fig = px.line(
        df_active,
        x="date",
        y=[
            "accel_decel_over_2_5",
            "accel_decel_over_3_5",
            "accel_decel_over_4_5",
        ],
        markers=True,
        title="Acceleration/Deceleration Events by Day",
        labels={
            "date": "Date",
            "value": "Number of Events",
            "variable": "Acceleration/Deceleration",
        },
    )

    # Rename legend items
    fig.update_traces(
        name=">2.5 m/s²", selector=dict(name="accel_decel_over_2_5")
    )
    fig.update_traces(
        name=">3.5 m/s²", selector=dict(name="accel_decel_over_3_5")
    )
    fig.update_traces(
        name=">4.5 m/s²", selector=dict(name="accel_decel_over_4_5")
    )

    return fig


def plot_peak_speed(df_active):
    """Plot peak speed by day"""
    fig = px.line(
        df_active,
        x="date",
        y="peak_speed",
        markers=True,
        color="is_match_day",
        color_discrete_map={True: "red", False: "blue"},
        title="Peak Speed by Day",
        labels={
            "peak_speed": "Peak Speed (km/h)",
            "date": "Date",
            "is_match_day": "Match Day",
        },
    )

    # Add average line
    avg_peak = df_active["peak_speed"].mean()
    fig.add_hline(
        y=avg_peak,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Avg: {avg_peak:.1f} km/h",
        annotation_position="bottom right",
    )

    return fig


def plot_speed_zones(df_active):
    """Plot speed zones distribution as pie chart"""
    speed_zones = pd.DataFrame(
        {
            "Zone": ["<21 km/h", "21-24 km/h", "24-27 km/h", ">27 km/h"],
            "Average Distance": [
                df_active["distance"].mean()
                - df_active["distance_over_21"].mean(),
                df_active["distance_over_21"].mean()
                - df_active["distance_over_24"].mean(),
                df_active["distance_over_24"].mean()
                - df_active["distance_over_27"].mean(),
                df_active["distance_over_27"].mean(),
            ],
        }
    )

    fig = px.pie(
        speed_zones,
        values="Average Distance",
        names="Zone",
        title="Average Distance Distribution by Speed Zone",
        color_discrete_sequence=px.colors.sequential.Viridis,
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")

    return fig


def plot_heart_rate_zones(df_active):
    """Plot heart rate zone distribution by day"""
    hr_cols = [
        "hr_zone_1_hms_seconds",
        "hr_zone_2_hms_seconds",
        "hr_zone_3_hms_seconds",
        "hr_zone_4_hms_seconds",
        "hr_zone_5_hms_seconds",
    ]

    hr_active = df_active.copy()
    hr_active["total_hr_seconds"] = hr_active[hr_cols].sum(axis=1)

    # Calculate percentages
    for i, col in enumerate(hr_cols, 1):
        hr_active[f"hr_zone_{i}_pct"] = (
            hr_active[col] / hr_active["total_hr_seconds"]
        ) * 100

    # Create stacked bar chart for heart rate zones
    hr_long = pd.melt(
        hr_active,
        id_vars=["date"],
        value_vars=[f"hr_zone_{i}_pct" for i in range(1, 6)],
        var_name="zone",
        value_name="percentage",
    )

    hr_long["zone"] = hr_long["zone"].str.replace("_pct", "")
    hr_long["zone_name"] = hr_long["zone"].map(
        {
            "hr_zone_1": "Zone 1 (Very Light)",
            "hr_zone_2": "Zone 2 (Light)",
            "hr_zone_3": "Zone 3 (Moderate)",
            "hr_zone_4": "Zone 4 (Hard)",
            "hr_zone_5": "Zone 5 (Maximum)",
        }
    )

    hr_long["date_str"] = pd.to_datetime(hr_long["date"]).dt.strftime("%d %b")

    # Filter out days with no HR data
    hr_long = hr_long[hr_long["percentage"] > 0]

    if hr_long.empty:
        return None

    fig = px.bar(
        hr_long,
        x="date_str",
        y="percentage",
        color="zone_name",
        title="Heart Rate Zone Distribution by Day",
        labels={
            "percentage": "Time Percentage",
            "date_str": "Date",
            "zone_name": "Heart Rate Zone",
        },
        color_discrete_sequence=[
            "#91cf60",
            "#d9ef8b",
            "#fee08b",
            "#fc8d59",
            "#d73027",
        ],
    )

    fig.update_layout(barmode="stack")
    return fig


def plot_weekly_distance(df):
    """Plot weekly total distance"""
    weekly_data = (
        df.groupby("week_num")
        .agg(
            {
                "distance": "sum",
                "is_match_day": "sum",  # Count of match days per week
            }
        )
        .reset_index()
    )

    # Add week starting date
    week_dates = df.groupby("week_num")["date"].min().reset_index()
    weekly_data = pd.merge(weekly_data, week_dates, on="week_num")
    weekly_data["week_start"] = weekly_data["date"].dt.strftime("%d %b")

    fig = px.bar(
        weekly_data,
        x="week_num",
        y="distance",
        title="Weekly Total Distance",
        labels={"distance": "Total Distance (m)", "week_num": "Week"},
        text=weekly_data["week_start"],
        color="is_match_day",
        color_continuous_scale="Reds",
        hover_data=["is_match_day"],
    )

    fig.update_layout(xaxis=dict(tickmode="linear"), hovermode="x")
    fig.update_traces(
        textposition="outside",
        hovertemplate="Week %{x}<br>Start Date: %{text}<br>Total Distance: %{y:.0f}m<br>Matches: %{customdata[0]}",
    )

    return fig

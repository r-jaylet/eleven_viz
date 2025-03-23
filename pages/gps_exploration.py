import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_preprocessing import load_gps


def show():

    try:
        df, df_active = load_gps()

        # Dashboard title
        st.title("Player GPS Performance Dashboard")
        st.markdown("### Training and Match Data Analysis")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Distance Analysis",
                "Speed Metrics",
                "Heart Rate Analysis",
                "Week Overview",
                "Match vs Training",
            ]
        )

        # Tab 1: Distance Analysis
        with tab1:
            st.subheader("Distance Metrics Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Total distance over time
                fig_dist = px.line(
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
                fig_dist.add_hline(
                    y=avg_distance,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Avg: {avg_distance:.0f}m",
                    annotation_position="bottom right",
                )

                st.plotly_chart(fig_dist, use_container_width=True)

            with col2:
                # Distance by match day code
                md_distance = (
                    df_active.groupby(["md_plus_code", "md_minus_code"])[
                        "distance"
                    ]
                    .mean()
                    .reset_index()
                )
                md_distance["md_code"] = md_distance.apply(
                    lambda x: (
                        f"MD{x['md_plus_code']}"
                        if x["md_plus_code"] > 0
                        else (
                            "MD"
                            if x["md_plus_code"] == 0
                            else f"MD{x['md_minus_code']}"
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

                fig_md_dist = px.bar(
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

                st.plotly_chart(fig_md_dist, use_container_width=True)

            # High-speed distance analysis
            st.subheader("High-Speed Distance Analysis")

            # Create a stacked bar chart for high-speed distances
            high_speed_df = df_active.copy()
            high_speed_df["date_str"] = high_speed_df["date"].dt.strftime(
                "%d %b"
            )

            fig_high_speed = go.Figure()

            # Add traces for each speed threshold
            fig_high_speed.add_trace(
                go.Bar(
                    x=high_speed_df["date_str"],
                    y=high_speed_df["distance_over_21"]
                    - high_speed_df["distance_over_24"],
                    name="21-24 km/h",
                    marker_color="#A6E7E4",
                )
            )

            fig_high_speed.add_trace(
                go.Bar(
                    x=high_speed_df["date_str"],
                    y=high_speed_df["distance_over_24"]
                    - high_speed_df["distance_over_27"],
                    name="24-27 km/h",
                    marker_color="#5ABCB9",
                )
            )

            fig_high_speed.add_trace(
                go.Bar(
                    x=high_speed_df["date_str"],
                    y=high_speed_df["distance_over_27"],
                    name=">27 km/h",
                    marker_color="#007A87",
                )
            )

            # Add match day markers
            for i, row in high_speed_df[
                high_speed_df["is_match_day"]
            ].iterrows():
                fig_high_speed.add_annotation(
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
            fig_high_speed.update_layout(
                title="High-Speed Running Distances by Day",
                xaxis_title="Date",
                yaxis_title="Distance (m)",
                barmode="stack",
                hovermode="x unified",
            )

            st.plotly_chart(fig_high_speed, use_container_width=True)

            # Speed distance percentage by day
            col1, col2 = st.columns(2)

            with col1:
                # Calculate percentages
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

                fig_pct = px.line(
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
                fig_pct.update_traces(
                    name=">21 km/h", selector=dict(name="pct_21_plus")
                )
                fig_pct.update_traces(
                    name=">24 km/h", selector=dict(name="pct_24_plus")
                )
                fig_pct.update_traces(
                    name=">27 km/h", selector=dict(name="pct_27_plus")
                )

                st.plotly_chart(fig_pct, use_container_width=True)

            with col2:
                # Acceleration/deceleration analysis
                fig_accel = px.line(
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
                fig_accel.update_traces(
                    name=">2.5 m/s²",
                    selector=dict(name="accel_decel_over_2_5"),
                )
                fig_accel.update_traces(
                    name=">3.5 m/s²",
                    selector=dict(name="accel_decel_over_3_5"),
                )
                fig_accel.update_traces(
                    name=">4.5 m/s²",
                    selector=dict(name="accel_decel_over_4_5"),
                )

                st.plotly_chart(fig_accel, use_container_width=True)

        # Tab 2: Speed Metrics
        with tab2:
            st.subheader("Speed Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Peak speed by day
                fig_peak = px.line(
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
                fig_peak.add_hline(
                    y=avg_peak,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Avg: {avg_peak:.1f} km/h",
                    annotation_position="bottom right",
                )

                st.plotly_chart(fig_peak, use_container_width=True)

            with col2:
                # Speed zones distribution
                speed_zones = pd.DataFrame(
                    {
                        "Zone": [
                            "<21 km/h",
                            "21-24 km/h",
                            "24-27 km/h",
                            ">27 km/h",
                        ],
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

                fig_zones = px.pie(
                    speed_zones,
                    values="Average Distance",
                    names="Zone",
                    title="Average Distance Distribution by Speed Zone",
                    color_discrete_sequence=px.colors.sequential.Viridis,
                )

                fig_zones.update_traces(
                    textposition="inside", textinfo="percent+label"
                )

                st.plotly_chart(fig_zones, use_container_width=True)

            # High-speed running and sprint analysis
            st.subheader("High-Speed Running Analysis")

            # Create a scatter plot matrix for speed metrics
            speed_metrics = [
                "distance_over_21",
                "distance_over_24",
                "distance_over_27",
                "accel_decel_over_2_5",
                "peak_speed",
            ]

            fig_scatter = px.scatter_matrix(
                df_active,
                dimensions=speed_metrics,
                color="is_match_day",
                title="Relationships Between Speed Metrics",
                labels={
                    "distance_over_21": "Dist >21 km/h (m)",
                    "distance_over_24": "Dist >24 km/h (m)",
                    "distance_over_27": "Dist >27 km/h (m)",
                    "accel_decel_over_2_5": "Accel/Decel >2.5",
                    "peak_speed": "Peak Speed (km/h)",
                    "is_match_day": "Match Day",
                },
                color_discrete_map={True: "red", False: "blue"},
            )

            fig_scatter.update_layout(height=700)

            st.plotly_chart(fig_scatter, use_container_width=True)

            # Speed metrics by match day code
            col1, col2 = st.columns(2)

            with col1:
                # High-speed running by match day code
                md_speed = (
                    df_active.groupby(["md_plus_code", "md_minus_code"])[
                        [
                            "distance_over_21",
                            "distance_over_24",
                            "distance_over_27",
                        ]
                    ]
                    .mean()
                    .reset_index()
                )

                md_speed["md_code"] = md_speed.apply(
                    lambda x: (
                        f"MD{x['md_plus_code']}"
                        if x["md_plus_code"] > 0
                        else (
                            "MD"
                            if x["md_plus_code"] == 0
                            else f"MD{x['md_minus_code']}"
                        )
                    ),
                    axis=1,
                )

                # Sort by match day sequence
                md_order = sorted(
                    md_speed["md_code"].unique(),
                    key=lambda x: (
                        int(float(x[2:]))
                        if x != "MD" and len(x) > 2
                        else (0 if x == "MD" else -100)
                    ),
                )

                fig_md_speed = go.Figure()

                fig_md_speed.add_trace(
                    go.Bar(
                        x=md_speed["md_code"],
                        y=md_speed["distance_over_21"],
                        name=">21 km/h",
                        marker_color="#A6E7E4",
                    )
                )

                fig_md_speed.add_trace(
                    go.Bar(
                        x=md_speed["md_code"],
                        y=md_speed["distance_over_24"],
                        name=">24 km/h",
                        marker_color="#5ABCB9",
                    )
                )

                fig_md_speed.add_trace(
                    go.Bar(
                        x=md_speed["md_code"],
                        y=md_speed["distance_over_27"],
                        name=">27 km/h",
                        marker_color="#007A87",
                    )
                )

                fig_md_speed.update_layout(
                    title="High-Speed Running by Match Day Code",
                    xaxis_title="Match Day Code",
                    yaxis_title="Average Distance (m)",
                    xaxis={
                        "categoryorder": "array",
                        "categoryarray": md_order,
                    },
                    barmode="group",
                )

                st.plotly_chart(fig_md_speed, use_container_width=True)

            with col2:
                # Acceleration/deceleration by match day code
                md_accel = (
                    df_active.groupby(["md_plus_code", "md_minus_code"])[
                        [
                            "accel_decel_over_2_5",
                            "accel_decel_over_3_5",
                            "accel_decel_over_4_5",
                        ]
                    ]
                    .mean()
                    .reset_index()
                )

                md_accel["md_code"] = md_accel.apply(
                    lambda x: (
                        f"MD{x['md_plus_code']}"
                        if x["md_plus_code"] > 0
                        else (
                            "MD"
                            if x["md_plus_code"] == 0
                            else f"MD{x['md_minus_code']}"
                        )
                    ),
                    axis=1,
                )

                fig_md_accel = go.Figure()

                fig_md_accel.add_trace(
                    go.Bar(
                        x=md_accel["md_code"],
                        y=md_accel["accel_decel_over_2_5"],
                        name=">2.5 m/s²",
                        marker_color="#FFD580",
                    )
                )

                fig_md_accel.add_trace(
                    go.Bar(
                        x=md_accel["md_code"],
                        y=md_accel["accel_decel_over_3_5"],
                        name=">3.5 m/s²",
                        marker_color="#FFA500",
                    )
                )

                fig_md_accel.add_trace(
                    go.Bar(
                        x=md_accel["md_code"],
                        y=md_accel["accel_decel_over_4_5"],
                        name=">4.5 m/s²",
                        marker_color="#FF4500",
                    )
                )

                fig_md_accel.update_layout(
                    title="Acceleration/Deceleration by Match Day Code",
                    xaxis_title="Match Day Code",
                    yaxis_title="Average Events",
                    xaxis={
                        "categoryorder": "array",
                        "categoryarray": md_order,
                    },
                    barmode="group",
                )

                st.plotly_chart(fig_md_accel, use_container_width=True)

        # Tab 3: Heart Rate Analysis
        with tab3:
            st.subheader("Heart Rate Zone Analysis")

            # Convert heart rate data to long format for visualization
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

            hr_long["date_str"] = pd.to_datetime(hr_long["date"]).dt.strftime(
                "%d %b"
            )

            # Filter out days with no HR data
            hr_long = hr_long[hr_long["percentage"] > 0]

            if not hr_long.empty:
                fig_hr = px.bar(
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

                fig_hr.update_layout(barmode="stack")

                st.plotly_chart(fig_hr, use_container_width=True)

                # Heart rate zone vs distance
                col1, col2 = st.columns(2)

                with col1:
                    # Time in different heart rate zones
                    hr_time = hr_active.copy()

                    hr_time_data = []
                    for i in range(1, 6):
                        hr_time_data.append(
                            {
                                "Zone": f"Zone {i}",
                                "Average Time (min)": hr_time[
                                    f"hr_zone_{i}_hms_seconds"
                                ].mean()
                                / 60,
                            }
                        )

                    hr_time_df = pd.DataFrame(hr_time_data)

                    fig_hr_time = px.bar(
                        hr_time_df,
                        x="Zone",
                        y="Average Time (min)",
                        title="Average Time in Heart Rate Zones",
                        color="Zone",
                        color_discrete_sequence=[
                            "#91cf60",
                            "#d9ef8b",
                            "#fee08b",
                            "#fc8d59",
                            "#d73027",
                        ],
                    )

                    st.plotly_chart(fig_hr_time, use_container_width=True)

                with col2:
                    # Calculate correlation between heart rate zones and distance
                    hr_dist_corr = hr_active[
                        [
                            "distance",
                            "distance_over_21",
                            "distance_over_24",
                            "distance_over_27",
                            "hr_zone_1_hms_seconds",
                            "hr_zone_2_hms_seconds",
                            "hr_zone_3_hms_seconds",
                            "hr_zone_4_hms_seconds",
                            "hr_zone_5_hms_seconds",
                        ]
                    ].corr()

                    # Extract correlations between distance metrics and HR zones
                    hr_dist_corr_subset = hr_dist_corr.loc[
                        [
                            "distance",
                            "distance_over_21",
                            "distance_over_24",
                            "distance_over_27",
                        ],
                        [
                            "hr_zone_1_hms_seconds",
                            "hr_zone_2_hms_seconds",
                            "hr_zone_3_hms_seconds",
                            "hr_zone_4_hms_seconds",
                            "hr_zone_5_hms_seconds",
                        ],
                    ]

                    # Rename for better display
                    hr_dist_corr_subset.columns = [
                        "Zone 1",
                        "Zone 2",
                        "Zone 3",
                        "Zone 4",
                        "Zone 5",
                    ]
                    hr_dist_corr_subset.index = [
                        "Total Dist",
                        ">21 km/h",
                        ">24 km/h",
                        ">27 km/h",
                    ]

                    fig_corr = px.imshow(
                        hr_dist_corr_subset,
                        title="Correlation: Distance vs Heart Rate Zones",
                        labels=dict(
                            x="Heart Rate Zone",
                            y="Distance Metric",
                            color="Correlation",
                        ),
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                    )

                    fig_corr.update_layout(height=400)

                    st.plotly_chart(fig_corr, use_container_width=True)

                # Heart rate intensity vs match day code
                hr_md = hr_active.copy()

                hr_md["md_code"] = hr_md.apply(
                    lambda x: (
                        f"MD{x['md_plus_code']}"
                        if x["md_plus_code"] > 0
                        else (
                            "MD"
                            if x["md_plus_code"] == 0
                            else f"MD{x['md_minus_code']}"
                        )
                    ),
                    axis=1,
                )

                # Calculate high-intensity heart rate time (Zone 4 + Zone 5)
                hr_md["high_intensity_hr"] = (
                    hr_md["hr_zone_4_hms_seconds"]
                    + hr_md["hr_zone_5_hms_seconds"]
                )

                # Group by match day code
                hr_md_grouped = (
                    hr_md.groupby("md_code")["high_intensity_hr"]
                    .mean()
                    .reset_index()
                )

                # Convert to minutes
                hr_md_grouped["high_intensity_hr_min"] = (
                    hr_md_grouped["high_intensity_hr"] / 60
                )

                # Sort by match day sequence
                md_order = sorted(
                    hr_md_grouped["md_code"].unique(),
                    key=lambda x: (
                        int(float(x[2:]))
                        if x != "MD" and len(x) > 2
                        else (0 if x == "MD" else -100)
                    ),
                )

                fig_hr_md = px.bar(
                    hr_md_grouped,
                    x="md_code",
                    y="high_intensity_hr_min",
                    title="High-Intensity Heart Rate Time by Match Day Code",
                    labels={
                        "high_intensity_hr_min": "Time in Zone 4-5 (min)",
                        "md_code": "Match Day Code",
                    },
                    category_orders={"md_code": md_order},
                    color="high_intensity_hr_min",
                    color_continuous_scale="Reds",
                )

                st.plotly_chart(fig_hr_md, use_container_width=True)
            else:
                st.warning("No heart rate data available for visualization.")

        # Tab 4: Week Overview
        with tab4:
            st.subheader("Weekly Load Analysis")

            # Group data by week
            weekly_data = (
                df.groupby("week_num")
                .agg(
                    {
                        "distance": "sum",
                        "distance_over_21": "sum",
                        "distance_over_24": "sum",
                        "distance_over_27": "sum",
                        "accel_decel_over_2_5": "sum",
                        "accel_decel_over_3_5": "sum",
                        "accel_decel_over_4_5": "sum",
                        "day_duration": "sum",
                        "peak_speed": "max",
                        "is_match_day": "sum",  # Count of match days per week
                    }
                )
                .reset_index()
            )

            # Add week starting date
            week_dates = df.groupby("week_num")["date"].min().reset_index()
            weekly_data = pd.merge(weekly_data, week_dates, on="week_num")
            weekly_data["week_start"] = weekly_data["date"].dt.strftime(
                "%d %b"
            )

            # Weekly distance
            col1, col2 = st.columns(2)

            with col1:
                fig_weekly_dist = px.bar(
                    weekly_data,
                    x="week_num",
                    y="distance",
                    title="Weekly Total Distance",
                    labels={
                        "distance": "Total Distance (m)",
                        "week_num": "Week",
                    },
                    text=weekly_data["week_start"],
                    color="is_match_day",
                    color_continuous_scale="Reds",
                    hover_data=["is_match_day"],
                )

                fig_weekly_dist.update_layout(
                    xaxis=dict(tickmode="linear"), hovermode="x"
                )

                fig_weekly_dist.update_traces(
                    textposition="outside",
                    hovertemplate="Week %{x}<br>Start Date: %{text}<br>Total Distance: %{y:.0f}m<br>Matches: %{customdata[0]}",
                )

                st.plotly_chart(fig_weekly_dist, use_container_width=True)

    except FileNotFoundError:
        st.error("GPS data file not found.")

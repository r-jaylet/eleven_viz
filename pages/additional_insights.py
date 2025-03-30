import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.additional_viz import (
    compute_load,
    compute_load_vs_recovery,
    plot_load,
    plot_load_over_time,
    plot_load_vs_recovery,
    plot_weekly_danger,
)
from src.data_preprocessing import (
    load_gps,
    load_priority,
    load_recovery_status,
)


def show():
    st.title("Load & Recovery Insights")

    # Load data
    df_gps, df_active = load_gps(
        "data/players_data/marc_cucurella/CFC GPS Data.csv"
    )
    df_recovery = load_recovery_status(
        "data/players_data/marc_cucurella/CFC Recovery status Data.csv"
    )
    df_priority = load_priority(
        "data/players_data/marc_cucurella/CFC Individual Priority Areas.csv"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Training Load", "Load vs Recovery", "Risk Analysis", "Objectives"]
    )

    with tab1:
        st.subheader("Training Load Evolution")

        load_params = st.expander("Load Calculation Parameters")
        with load_params:
            col1, col2, col3 = st.columns(3)
            with col1:
                alpha = st.slider("Distance Weight (α)", 0.5, 3.0, 2.0, 0.1)
            with col2:
                beta = st.slider("Accel/Decel Weight (β)", 0.5, 3.0, 1.0, 0.1)
            with col3:
                gamma = st.slider("High Speed Weight (γ)", 0.5, 3.0, 3.0, 0.1)

            rolling_window = st.slider("Rolling Window (days)", 1, 14, 7)

        df_gps, threshold, df_load_by_date = compute_load(
            df_gps,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            top_x_percent=5,
        )

        fig_load = plot_load(df_gps)
        st.plotly_chart(fig_load, use_container_width=True)

        fig_load = plot_load_over_time(
            df_load_by_date,
            rolling_window=rolling_window,
        )

        st.plotly_chart(fig_load, use_container_width=True)

        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

        with metrics_col1:
            st.metric("Average Load", f"{df_load_by_date['load'].mean():.3f}")
        with metrics_col2:
            st.metric("Max Load", f"{df_load_by_date['load'].max():.3f}")
        with metrics_col3:
            st.metric("High Load Threshold", f"{threshold:.3f}")

    with tab2:
        st.subheader("Load vs. Recovery Analysis")

        recovery_params = st.expander("Recovery Analysis Parameters")
        with recovery_params:
            window_size = st.slider("Recovery Window Size (days)", 1, 14, 7)

        merged_df = compute_load_vs_recovery(
            df_recovery=df_recovery,
            df_gps=df_gps,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            window_size=window_size,
        )
        fig_load_vs_recovery = plot_load_vs_recovery(
            merged_df=merged_df,
        )

        st.plotly_chart(fig_load_vs_recovery, use_container_width=True)

    with tab3:
        st.subheader("Weekly Risk Analysis")

        fig_weekly_danger = plot_weekly_danger(merged_df)
        st.plotly_chart(fig_weekly_danger, use_container_width=True)

        # Information about risk levels
        st.info(
            """
        **Risk Level Interpretation:**
        - **Green**: Normal training load - no days exceeding threshold
        - **Yellow**: Warning level - 1-2 days exceeding high load threshold
        - **Red**: High risk - 3+ days exceeding high load threshold
        
        Consistent high risk weeks may lead to overtraining and increased injury risk.
        """
        )

    with tab4:
        st.subheader("Individual Objectives & Priority Areas")

        table = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(df_priority.columns),
                        fill_color="#1E88E5",
                        font=dict(color="white", size=14),
                        align="center",
                        height=40,
                    ),
                    cells=dict(
                        values=[
                            df_priority[col] for col in df_priority.columns
                        ],
                        fill_color=["#F5F5F5", "#EEEEEE"] * 4,
                        font=dict(color="#212121", size=12),
                        align="center",
                        height=40,
                    ),
                )
            ]
        )

        table.update_layout(
            margin=dict(l=10, r=10, b=10, t=10),
            height=500,
        )

        st.plotly_chart(table, use_container_width=True)

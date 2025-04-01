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
        "data/players_data/marc_cucurella/CFC GPS Data.csv",
        season=st.session_state.selected_season,
    )
    df_recovery = load_recovery_status(
        "data/players_data/marc_cucurella/CFC Recovery status Data.csv",
        season=st.session_state.selected_season,
    )
    df_priority = load_priority(
        "data/players_data/marc_cucurella/CFC Individual Priority Areas.csv",
    )

    # Simplified tabs
    tabs = st.tabs(["Training Load", "Recovery Analysis", "Objectives"])

    # TRAINING LOAD TAB
    with tabs[0]:
        st.header("Training Load")

        col1, col2, col3 = st.columns(3)
        with col1:
            alpha = st.slider("Distance Weight", 1.0, 3.0, 2.0, 0.5)
        with col2:
            beta = st.slider("Accel/Decel Weight", 1.0, 3.0, 1.0, 0.5)
        with col3:
            gamma = st.slider("High Speed Weight", 1.0, 3.0, 3.0, 0.5)

        rolling_window = st.slider("Average Window (days)", 3, 14, 7)

        # Calculate load
        df_gps, threshold, df_load_by_date = compute_load(
            df_gps,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            top_x_percent=5,
        )

        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Load", f"{df_load_by_date['load'].mean():.2f}")
        with col2:
            st.metric("Max Load", f"{df_load_by_date['load'].max():.2f}")
        with col3:
            st.metric("High Load Threshold", f"{threshold:.2f}")

        # Load visualization
        st.subheader("Load Distribution")
        fig_load = plot_load(df_gps)
        st.plotly_chart(fig_load, use_container_width=True)

        st.subheader("Load Over Time")
        fig_load_time = plot_load_over_time(
            df_load_by_date,
            rolling_window=rolling_window,
        )
        st.plotly_chart(fig_load_time, use_container_width=True)

    # RECOVERY ANALYSIS TAB
    with tabs[1]:
        st.header("Recovery Analysis")

        # Recovery vs load
        st.subheader("Load vs Recovery")
        window_size = st.slider("Recovery Window", 5, 10, 7)

        merged_df = compute_load_vs_recovery(
            df_recovery=df_recovery,
            df_gps=df_gps,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            window_size=window_size,
        )

        fig_load_vs_recovery = plot_load_vs_recovery(merged_df)
        st.plotly_chart(fig_load_vs_recovery, use_container_width=True)

        # Risk analysis
        st.subheader("Weekly Risk Assessment")
        fig_weekly_danger = plot_weekly_danger(merged_df)
        st.plotly_chart(fig_weekly_danger, use_container_width=True)

        # Simple risk legend
        st.markdown(
            """
        **Risk Levels:**
        - **Green**: Normal load
        - **Yellow**: Warning - monitor closely
        - **Red**: High risk - reduce load
        """
        )

    # OBJECTIVES TAB
    with tabs[2]:
        st.header("Individual Objectives")

        # Display priority areas as a simple table
        st.dataframe(df_priority, use_container_width=True)

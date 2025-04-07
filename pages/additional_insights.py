import streamlit as st

from src.additional_viz import (
    compute_load,
    compute_load_vs_recovery,
    plot_load,
    plot_load_over_time,
    plot_load_vs_recovery,
    plot_weekly_danger,
    plot_player_load_vs_expression, 
    plot_pre_post_match_recovery_dumbbell, 
    plot_recovery_vs_load_peaks, 
    plot_readiness_snapshot_before_matches
)
from src.data_preprocessing import (
    load_gps,
    load_priority,
    load_recovery_status,
    load_physical_capabilities
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
    df_physical = load_physical_capabilities(
        "data/players_data/marc_cucurella/CFC Physical Capability Data.csv",
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
        st.markdown("This graph breaks down how training load is related to key physical metrics: total distance, high-speed distance (>21 km/h), and accel/decel efforts (>2.5 m/s²). ")
        fig_load = plot_load(df_gps)
        st.plotly_chart(fig_load, use_container_width=True)

        st.subheader("Load Over Time")
        st.markdown("""
        This chart tracks the player’s daily training load over time. Training load is calculated using a composite formula that combines distance, high-speed distance, and accel/decel counts, each passed through a sigmoid transformation and weighted by coefficients (α, β, γ). 
        This approach gives a normalized intensity score that reflects both volume and intensity of physical effort across sessions.""")
        fig_load_time = plot_load_over_time(
            df_load_by_date,
            rolling_window=rolling_window,
        )
        st.plotly_chart(fig_load_time, use_container_width=True)

        # Load vs expression visualization 
        st.subheader("Plaver Load vs Expression Development")
        st.markdown("""
        This graph visualizes the relationship between a player’s physical training load and their expression development (performance benchmark) over time. It combines three curves:
        - Composite Load (Normalized): A daily average derived from normalized GPS metrics (distance, accelerations, high-speed distance, peak speed).
        - Smoothed Load (7-day Avg): A rolling average of the composite load to highlight weekly load trends.
        - BenchmarkPct (Expression): The player's benchmark performance percentage from capability assessments (e.g. sprint, jump, strength tests).
        This chart helps coaches assess whether increased training loads correlate with improvements in physical capabilities.  """)
        fig_load_vs_expression = plot_player_load_vs_expression(df_gps, df_physical)
        st.plotly_chart(fig_load_vs_expression, use_container_width=True)

    # RECOVERY ANALYSIS TAB
    with tabs[1]:
        st.header("Recovery Analysis")

        # Recovery vs load
        st.subheader("Load vs Recovery")

        st.markdown("""
        This graph shows the relationship between player training load, recovery score, and a computed risk index (in red) over time.
        Load is calculated using a weighted formula that includes distance, high-speed running, and acceleration/deceleration events from GPS data.
        Recovery is derived from scores like emboss_baseline_score, averaged using a rolling window.
        Risk captures when a player might be under too much physical stress without enough recovery time, it is calculated as: (load / max_load) * (1 - recovery / max_recovery)
        """
        )

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
        st.markdown("""
        This chart shows a weekly summary of training overload risk, based on how many days per week the player's load was among the top 10% (most demanding). 
        The function computes a danger threshold from GPS-derived load scores, then counts how many days per week exceed that threshold. This helps visualize how frequently the player hits risky load zones.
        """
        )
        
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

        # Pre to post match recovery status : 
        st.subheader("Comparison of pre- vs post-match recovery scores for each match using a dumbbell chart")

        st.markdown(
            """
            This graph compares recovery scores before and after each match using a color-coded dumbbell chart. Each line connects the pre- and post-match values, with color indicating the impact level:
            🟥 High (>0.3 drop), 🟨 Moderate (0.1-0.3), 🟩 Low (<0.1).
            It pulls recovery scores 1 day before and after matches (±1 day tolerance), computes the difference, and visually tracks how much recovery declines post-game.
            A large drop from pre to post likely reflects intense physical effort and accumulated muscle stress.
            """
        )

        fig_pre_post_match_recovery = plot_pre_post_match_recovery_dumbbell(df_gps, df_recovery, recovery_metric="soreness_baseline_composite")
        st.plotly_chart(fig_pre_post_match_recovery, use_container_width=True)

        # markdown : 
        st.markdown(
            """
            **Impact Legend (based on drop in recovery score after match):**
            - 🟥 **High Impact**: Drop > 0.3 &nbsp; → &nbsp; **Red line** — intense effort or low post-match recovery  
            - 🟨 **Moderate Impact**: Drop between 0.1 and 0.3 &nbsp; → &nbsp; **Amber line** — manageable fatigue  
            - 🟩 **Low Impact**: Drop < 0.1 &nbsp; → &nbsp; **Green line** — good resilience or light match load
            """,
            unsafe_allow_html=True
        )

        # Recovery score vs training load peaks
        st.subheader("Recovery Score vs Training Load Peaks")

        st.markdown(
            """
            This visualization compares the rolling training load (area in dark blue) with the player's daily recovery scores (yellow line and dots). It helps identify whether spikes in load are followed by dips in recovery quality.
            **Rolling Load** is calculated as a composite of GPS metrics (distance, acceleration/deceleration, peak speed, etc.), averaged over a selectable rolling window (e.g., 7 days).
            **Recovery Score** comes from selected wellness metrics like soreness_baseline_composite, sleep_baseline_composite, etc.
            You can customize the recovery metric and the load smoothing window to explore different trends.

            """
        )

        # Recovery metric selector
        recovery_metric = st.selectbox(
            "Select Recovery Metric",
            options=[
                "soreness_baseline_composite",
                "sleep_baseline_composite",
                "subjective_baseline_composite"
            ],
            index=0,
        )

        # Rolling window selector
        rolling_window = st.slider("Rolling Load Window (days)", 3, 14, 7)

        # Generate and display the figure
        fig_recovery_vs_load = plot_recovery_vs_load_peaks(
            gps_df=df_gps,
            recovery_df=df_recovery,
            recovery_metric=recovery_metric,
            rolling_window=rolling_window,
        )

        st.plotly_chart(fig_recovery_vs_load, use_container_width=True)

        # readiness before each match : 
        st.subheader("Readiness Snapshot Before Each Match")
        st.markdown("This graph shows a readiness score for each match, calculated as the average of the player’s benchmark performance score (latest benchmarkPct) and recovery score (from 1–2 days before the match). " \
        "Bars are colored based on thresholds: 🟥 low readiness, 🟨 moderate, and 🟩 high readiness. " \
        "It depicts how well-prepared the player was heading into each match.")

        fig_readiness_snapshot = plot_readiness_snapshot_before_matches(
            gps_df=df_gps,
            recovery_df=df_recovery,
            capability_df=df_physical,
            recovery_metric="soreness_baseline_composite",
            readiness_thresholds=(-0.1, 0.1),  
        )

        st.plotly_chart(fig_readiness_snapshot, use_container_width=True)

    # OBJECTIVES TAB
    with tabs[2]:
        st.header("Individual Objectives")

        # Display priority areas as a simple table
        st.dataframe(df_priority, use_container_width=True)

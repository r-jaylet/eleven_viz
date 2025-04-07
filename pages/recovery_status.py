import streamlit as st

from src.data_preprocessing import load_recovery_status
from src.recovery_viz import (
    create_category_completeness_bar,
    create_completeness_radar,
    create_composite_line,
    create_correlation_heatmap,
    plot_global_recovery_score,
    plot_recovery_metrics_by_category,
    plot_weekly_recovery_heatmap
)


def show():
    st.title("Recovery Status")

    try:
        # Load data
        df = load_recovery_status(
            "data/players_data/marc_cucurella/CFC Recovery status Data.csv",
            season=st.session_state.selected_season,
        )

        # Simple date filter in sidebar
        min_date = df["sessionDate"].min().date()
        max_date = df["sessionDate"].max().date()

        start_date = st.sidebar.date_input("From", min_date)
        end_date = st.sidebar.date_input("To", max_date)

        # Filter data
        date_filtered_df = df[
            (df["sessionDate"].dt.date >= start_date)
            & (df["sessionDate"].dt.date <= end_date)
        ]

        # Convert dates for visualization functions
        start_date_str = start_date.strftime("%d/%m/%Y")
        end_date_str = end_date.strftime("%d/%m/%Y")

        # Main tabs
        tabs = st.tabs(["Overview", "Detailed Analysis"])

        # OVERVIEW TAB
        with tabs[0]:
    
            # Global recovery score
            st.subheader("Recovery Summary")
            st.markdown(""" 
            This graph shows the evolution of the global recovery score over time, combining daily recovery values and a 7-day moving average. 
            The blue line with markers represents the raw emboss_baseline_score collected each day, while the bold orange line smooths these fluctuations.
            """)
            fig = plot_global_recovery_score(df)
            st.plotly_chart(fig, use_container_width=True)

            
            # Recovery metrics by category - simplified view
            st.subheader("Recovery by Category")
            st.markdown(""" 
            This multi-panel figure shows time series plots of recovery data by category, highlighting both data completeness (solid lines) and composite scores (dotted lines) over time. 
                        Each subplot corresponds to a specific recovery category like sleep, soreness, or subjective
            """)
            fig = plot_recovery_metrics_by_category(
                df, start_date_str, end_date_str
            )
            st.plotly_chart(fig, use_container_width=True)

        # DETAILED ANALYSIS TAB
        with tabs[1]:
            st.header("Category Analysis")

            # Filter data by type
            completeness_df = date_filtered_df[
                date_filtered_df["metric_type"] == "completeness"
            ]
            composite_df = date_filtered_df[
                date_filtered_df["metric_type"] == "composite"
            ]

            # Two-column layout for category analysis
            col1, col2 = st.columns(2)

            with col1:
                # Category completeness bar chart
                st.subheader("Completeness by Category")
                st.markdown("""
                This bar chart shows the average data completeness across recovery categories such as sleep, soreness, subjective, and various musculoskeletal metrics. 
                Each bar represents the proportion of days where data was available for that category, giving a snapshot of data coverage over time. 
                It's useful for identifying which wellness domains are consistently tracked and which may need improved data collection practices—e.g. “subjective” has the highest completeness, while bio and msk_load_tolerance are less consistently recorded.""")
                if not completeness_df.empty:
                    fig = create_category_completeness_bar(completeness_df)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Category selector with composite scores
                st.subheader("Category Scores")
                st.markdown("""This graph displays composite recovery scores across multiple wellness categories (e.g. soreness, sleep, subjective, bio) over time. """)
                if not composite_df.empty:
                    fig = create_composite_line(composite_df)
                    st.plotly_chart(fig, use_container_width=True)

            # Correlation analysis at the bottom
            if len(date_filtered_df) > 5:
                st.subheader("Metric Correlations")

                # Create pivot table for correlation
                pivot_df = date_filtered_df.pivot_table(
                    index="sessionDate", columns="metric", values="value"
                ).fillna(0)

                # Only include columns with sufficient variation
                valid_cols = [
                    col for col in pivot_df.columns if pivot_df[col].std() > 0
                ]

                if len(valid_cols) > 1:
                    corr_matrix = pivot_df[valid_cols].corr()
                    fig = create_correlation_heatmap(corr_matrix)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Weekly recovery heatmap
            st.subheader("Weekly recovery heatmap")
            st.markdown("""
            This heatmap visualizes how recovery scores vary by day and week. 
            Each row represents a week (e.g., 2024-W14) and each column a day of the week, with color indicating the average emboss_baseline_score for that day. 
            Green tones reflect higher recovery, while red indicates poor recovery.""")
            weekly_heatmap = plot_weekly_recovery_heatmap(df)
            st.plotly_chart(weekly_heatmap, use_container_width=True)

    except FileNotFoundError:
        st.error("Recovery data file not found")
    except Exception as e:
        st.error(f"Error: {str(e)}")

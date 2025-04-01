import streamlit as st

from src.data_preprocessing import load_recovery_status
from src.recovery_viz import (
    create_category_completeness_bar,
    create_completeness_radar,
    create_composite_line,
    create_correlation_heatmap,
    plot_global_recovery_score,
    plot_recovery_metrics_by_category,
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
            st.header("Recovery Summary")

            # Two-column layout for main metrics
            col1, col2 = st.columns(2)

            with col1:
                # Global recovery score
                fig = plot_global_recovery_score(df)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Radar chart for latest data
                latest_date = date_filtered_df["sessionDate"].max()
                latest_df = date_filtered_df[
                    date_filtered_df["sessionDate"] == latest_date
                ]
                completeness_df = latest_df[
                    latest_df["metric_type"] == "completeness"
                ]

                if not completeness_df.empty:
                    fig = create_completeness_radar(completeness_df)
                    st.plotly_chart(fig, use_container_width=True)

            # Recovery metrics by category - simplified view
            st.subheader("Recovery by Category")
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
                if not completeness_df.empty:
                    fig = create_category_completeness_bar(completeness_df)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Category selector with composite scores
                st.subheader("Category Scores")
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

    except FileNotFoundError:
        st.error("Recovery data file not found")
    except Exception as e:
        st.error(f"Error: {str(e)}")

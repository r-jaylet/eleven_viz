import streamlit as st

from src.data_preprocessing import load_recovery_status
from src.recovery_viz import (
    create_category_completeness_bar,
    create_category_completeness_time,
    create_category_composite_time,
    create_completeness_radar,
    create_composite_line,
    create_correlation_heatmap,
    create_daily_completeness_line,
    create_daily_tracking,
    plot_global_recovery_score,
    plot_recovery_metrics_by_category,
)


def show():
    try:
        st.title("Recovery Status Dashboard")

        # Data loading and preparation
        df = load_recovery_status(
            "data/players_data/marc_cucurella/CFC Recovery status Data.csv"
        )

        # Date range filter
        st.sidebar.header("Date Range Filter")
        min_date = df["sessionDate"].min().date()
        max_date = df["sessionDate"].max().date()

        start_date = st.sidebar.date_input("Start Date", min_date)
        end_date = st.sidebar.date_input("End Date", max_date)

        # Convert dates to string format for viz functions
        start_date_str = start_date.strftime("%d/%m/%Y")
        end_date_str = end_date.strftime("%d/%m/%Y")

        # Filter data by date
        date_filtered_df = df[
            (df["sessionDate"].dt.date >= start_date)
            & (df["sessionDate"].dt.date <= end_date)
        ]

        st.header("Recovery Overview")

        col1, col2 = st.columns(2)

        with col1:
            fig = plot_global_recovery_score(df)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Latest Completeness Radar
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

        # Recovery Metrics by Category
        st.subheader("Recovery Metrics by Category")
        fig = plot_recovery_metrics_by_category(
            df, start_date_str, end_date_str
        )
        st.plotly_chart(fig, use_container_width=True)

        st.header("Detailed Analysis")

        tab1, tab2, tab3 = st.tabs(
            ["Completeness", "Composites", "Daily Tracking"]
        )

        with tab1:

            completeness_df = date_filtered_df[
                date_filtered_df["metric_type"] == "completeness"
            ]

            col1, col2 = st.columns(2)

            with col1:
                fig = create_category_completeness_bar(completeness_df)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = create_daily_completeness_line(completeness_df)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            composite_df = date_filtered_df[
                date_filtered_df["metric_type"] == "composite"
            ]
            if not composite_df.empty:
                st.subheader("Composite Scores by Category")
                fig = create_composite_line(composite_df)
                st.plotly_chart(fig, use_container_width=True)

            if not completeness_df.empty:
                st.subheader("Category Deep Dive")
                categories = sorted(completeness_df["category"].unique())
                selected_category = st.selectbox("Select Category", categories)

                col1, col2 = st.columns(2)

                with col1:
                    cat_completeness = completeness_df[
                        completeness_df["category"] == selected_category
                    ]
                    if not cat_completeness.empty:
                        fig = create_category_completeness_time(
                            cat_completeness, selected_category
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    cat_composite = composite_df[
                        composite_df["category"] == selected_category
                    ]
                    if not cat_composite.empty:
                        fig = create_category_composite_time(
                            cat_composite, selected_category
                        )
                        st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Daily tracking overview
            if not date_filtered_df.empty:
                daily_stats = (
                    date_filtered_df.groupby("sessionDate")
                    .agg(mean=("value", "mean"), count=("value", "count"))
                    .reset_index()
                )

                fig = create_daily_tracking(daily_stats)
                st.plotly_chart(fig, use_container_width=True)

            # Correlation heatmap between metrics
            if len(date_filtered_df) > 5:  # Only show if enough data points
                pivot_df = date_filtered_df.pivot_table(
                    index="sessionDate", columns="metric", values="value"
                ).fillna(0)

                # Only include columns with sufficient variation
                valid_cols = [
                    col for col in pivot_df.columns if pivot_df[col].std() > 0
                ]
                if len(valid_cols) > 1:
                    corr_matrix = pivot_df[valid_cols].corr()

                    st.subheader("Metric Correlations")
                    fig = create_correlation_heatmap(corr_matrix)
                    st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error(
            "Recovery data file not found. Please check the data directory."
        )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

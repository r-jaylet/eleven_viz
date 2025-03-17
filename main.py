"""
Streamlit dashboard application for data visualization.
"""

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Import functions from data_functions module
from viz_functions import filter_data, get_summary_stats, load_csv_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Cache data loading to improve performance
@st.cache_data
def get_data(file_path, date_columns=None):
    """
    Load and cache data to improve dashboard performance.
    """
    return load_csv_data(file_path, date_columns=date_columns)


def main():
    """
    Main function to run the Streamlit application.
    """
    # App title and description
    st.title("📊 Data Visualization Dashboard")
    st.markdown("Upload a CSV file to visualize your data.")

    # Sidebar for file uploading and filters
    with st.sidebar:
        st.header("Data Configuration")

        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            # Save the file temporarily
            temp_path = Path("temp_data.csv")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Try to infer date columns
            try:
                sample_df = pd.read_csv(uploaded_file, nrows=5)
                date_cols = [
                    col
                    for col in sample_df.columns
                    if sample_df[col].dtype == "object"
                    and "date" in col.lower()
                ]
                if date_cols:
                    st.info(
                        f"Auto-detected potential date columns: {', '.join(date_cols)}"
                    )
            except Exception as e:
                logger.error(f"Error in date column detection: {str(e)}")
                date_cols = []

            # Option to select date columns
            selected_date_cols = st.multiselect(
                "Select date columns",
                options=sample_df.columns,
                default=date_cols,
            )

            # Load the data
            try:
                df = get_data(temp_path, date_columns=selected_date_cols)
                st.success(
                    f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns."
                )

                # Show filter options
                st.header("Data Filters")
                filters = {}

                # Create up to 3 filters
                for i in range(3):
                    if st.checkbox(f"Add filter #{i+1}", key=f"filter_{i}"):
                        col = st.selectbox(
                            "Select column",
                            options=df.columns,
                            key=f"filter_col_{i}",
                        )

                        # Different filter UI based on data type
                        if pd.api.types.is_numeric_dtype(df[col]):
                            min_val, max_val = float(df[col].min()), float(
                                df[col].max()
                            )
                            filter_range = st.slider(
                                f"Range for {col}",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                key=f"filter_range_{i}",
                            )
                            filters[col] = {
                                "min": filter_range[0],
                                "max": filter_range[1],
                            }
                        else:
                            unique_values = df[col].dropna().unique().tolist()
                            selected_values = st.multiselect(
                                f"Select values for {col}",
                                options=unique_values,
                                default=(
                                    unique_values[:5]
                                    if len(unique_values) > 5
                                    else unique_values
                                ),
                                key=f"filter_values_{i}",
                            )
                            if selected_values:
                                filters[col] = selected_values

                # Apply filters
                if filters:
                    filtered_df = filter_data(df, filters)
                else:
                    filtered_df = df

                # Store dataframe in session state for access in the main area
                st.session_state["data"] = filtered_df

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                logger.error(f"Error loading data: {str(e)}")
                return

        # Display app info
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This dashboard allows you to upload and visualize CSV data."
        )
        st.markdown("Made with Streamlit and Python.")

    # Main area - only show if data is loaded
    if "data" in st.session_state:
        filtered_df = st.session_state["data"]

        # Data overview section
        st.header("Data Overview")
        tab1, tab2, tab3 = st.tabs(
            ["Preview", "Summary Stats", "Visualizations"]
        )

        with tab1:
            st.subheader("Data Preview")
            st.dataframe(filtered_df.head(10), use_container_width=True)

            # Data dimensions and info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", filtered_df.shape[0])
            with col2:
                st.metric("Columns", filtered_df.shape[1])

            # Column data types
            st.subheader("Column Data Types")
            dtypes_df = pd.DataFrame(
                {
                    "Column": filtered_df.columns,
                    "Data Type": filtered_df.dtypes.astype(str),
                    "Non-Null Count": filtered_df.count().values,
                    "Missing Values": filtered_df.isna().sum().values,
                    "Missing %": (
                        filtered_df.isna().sum() / len(filtered_df) * 100
                    )
                    .round(2)
                    .values,
                }
            )
            st.dataframe(dtypes_df, use_container_width=True)

        with tab2:
            st.subheader("Summary Statistics")
            summary = get_summary_stats(filtered_df)
            if not summary.empty:
                st.dataframe(summary.round(2), use_container_width=True)
            else:
                st.info("No numeric columns found for summary statistics.")

        with tab3:
            st.subheader("Data Visualizations")

            # Let user select visualization type and columns
            viz_type = st.selectbox(
                "Select visualization type",
                options=[
                    "Histogram",
                    "Scatter Plot",
                    "Line Chart",
                    "Bar Chart",
                    "Box Plot",
                ],
            )

            # Get numeric and categorical columns for appropriate selectors
            numeric_cols = filtered_df.select_dtypes(
                include=["number"]
            ).columns.tolist()
            categorical_cols = filtered_df.select_dtypes(
                exclude=["number"]
            ).columns.tolist()

            if viz_type == "Histogram":
                if numeric_cols:
                    x_col = st.selectbox(
                        "Select column for histogram", options=numeric_cols
                    )
                    color_col = st.selectbox(
                        "Select column for color (optional)",
                        options=[None] + categorical_cols,
                    )
                    bins = st.slider(
                        "Number of bins", min_value=5, max_value=100, value=20
                    )

                    fig = px.histogram(
                        filtered_df,
                        x=x_col,
                        color=color_col,
                        nbins=bins,
                        title=f"Histogram of {x_col}",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns available for histogram.")

            elif viz_type == "Scatter Plot":
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox(
                            "Select X-axis column", options=numeric_cols
                        )
                    with col2:
                        y_col = st.selectbox(
                            "Select Y-axis column",
                            options=(
                                [c for c in numeric_cols if c != x_col]
                                if len(numeric_cols) > 1
                                else numeric_cols
                            ),
                        )

                    color_col = st.selectbox(
                        "Select column for color (optional)",
                        options=[None] + categorical_cols + numeric_cols,
                    )

                    fig = px.scatter(
                        filtered_df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        title=f"Scatter Plot: {x_col} vs {y_col}",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(
                        "At least two numeric columns are required for a scatter plot."
                    )

            elif viz_type == "Line Chart":
                # Check if we have any date columns
                date_cols = [
                    col
                    for col in filtered_df.columns
                    if pd.api.types.is_datetime64_dtype(filtered_df[col])
                ]

                x_options = date_cols + numeric_cols + categorical_cols
                if x_options:
                    x_col = st.selectbox(
                        "Select X-axis column", options=x_options
                    )

                    y_cols = st.multiselect(
                        "Select Y-axis column(s)",
                        options=numeric_cols,
                        default=[numeric_cols[0]] if numeric_cols else [],
                    )

                    if y_cols:
                        # Create line chart with Plotly
                        fig = px.line(
                            filtered_df,
                            x=x_col,
                            y=y_cols,
                            title=f"Line Chart: {', '.join(y_cols)} by {x_col}",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select at least one Y-axis column.")
                else:
                    st.warning("No suitable columns found for line chart.")

            elif viz_type == "Bar Chart":
                if categorical_cols and numeric_cols:
                    x_col = st.selectbox(
                        "Select X-axis (categorical)", options=categorical_cols
                    )
                    y_col = st.selectbox(
                        "Select Y-axis (numeric)", options=numeric_cols
                    )

                    # Option for aggregation
                    agg_func = st.selectbox(
                        "Aggregation function",
                        options=[
                            "sum",
                            "mean",
                            "count",
                            "median",
                            "min",
                            "max",
                        ],
                    )

                    # Create aggregated dataframe
                    agg_df = (
                        filtered_df.groupby(x_col)[y_col]
                        .agg(agg_func)
                        .reset_index()
                    )
                    agg_df = agg_df.sort_values(y_col, ascending=False).head(
                        20
                    )  # Top 20 for readability

                    fig = px.bar(
                        agg_df,
                        x=x_col,
                        y=y_col,
                        title=f"Bar Chart: {agg_func.capitalize()} of {y_col} by {x_col}",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(
                        "Bar charts require both categorical and numeric columns."
                    )

            elif viz_type == "Box Plot":
                if numeric_cols:
                    y_col = st.selectbox(
                        "Select numeric column", options=numeric_cols
                    )
                    x_col = st.selectbox(
                        "Select grouping column (optional)",
                        options=[None] + categorical_cols,
                    )

                    fig = px.box(
                        filtered_df,
                        x=x_col,
                        y=y_col,
                        title=f"Box Plot of {y_col}"
                        + (f" by {x_col}" if x_col else ""),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns available for box plot.")


if __name__ == "__main__":
    main()

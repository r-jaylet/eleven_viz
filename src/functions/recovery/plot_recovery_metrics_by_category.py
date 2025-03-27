import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp


def plot_recovery_metrics_by_category(
    df: pd.DataFrame, start_date: str, end_date: str
) -> None:
    """
    Generates time series for each category with the metrics 'completeness' and 'composite',
    filtered by the date range provided.

    Args:
    - df (pd.DataFrame): DataFrame containing recovery data with columns 'sessionDate', 'category', 'metric', and 'value'.
    - start_date (str): Start date for filtering the data, in the format 'dd/mm/yyyy'.
    - end_date (str): End date for filtering the data, in the format 'dd/mm/yyyy'.
    """
    df["sessionDate"] = pd.to_datetime(df["sessionDate"], format="%d/%m/%Y")

    df = df[
        (df["sessionDate"] >= pd.to_datetime(start_date, format="%d/%m/%Y"))
        & (df["sessionDate"] <= pd.to_datetime(end_date, format="%d/%m/%Y"))
    ]

    # List of categories
    categories = [
        "bio",
        "msk_joint_range",
        "msk_load_tolerance",
        "subjective",
        "soreness",
        "sleep",
    ]

    fig = sp.make_subplots(rows=3, cols=2, subplot_titles=categories)

    # Iterate over categories to add the curves
    for i, category in enumerate(categories):
        df_cat = df[df["category"] == category]

        df_completeness = df_cat[
            df_cat["metric"].str.endswith("_completeness")
        ]
        df_composite = df_cat[df_cat["metric"].str.endswith("_composite")]

        row, col = (i // 2) + 1, (i % 2) + 1

        fig.add_trace(
            go.Scatter(
                x=df_completeness["sessionDate"],
                y=df_completeness["value"],
                mode="lines",
                name=f"{category} - Completeness",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=df_composite["sessionDate"],
                y=df_composite["value"],
                mode="lines",
                name=f"{category} - Composite",
                line=dict(dash="dot"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Completeness",
            line=dict(color="black", width=2),
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Composite",
            line=dict(color="black", dash="dot", width=2),
            showlegend=True,
        )
    )

    fig.update_layout(
        title="Evolution of Recovery Metrics by Category",
        height=800,
        template="plotly_white",
        showlegend=True,
    )

    fig.show()

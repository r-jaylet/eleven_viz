import matplotlib.pyplot as plt
import pandas as pd

from src.functions.additional_insights.compute_load_and_plot import (
    compute_load_and_plot,
)
from src.functions.recovery.plot_global_recovery_score import (
    plot_global_recovery_score,
)


def plot_load_vs_recovery(
    df_recovery: pd.DataFrame,
    df_gps: pd.DataFrame,
    start_date: str = "01/01/2023",
    end_date: str = "31/12/2030",
    alpha: float = 2,
    beta: float = 1,
    gamma: float = 3,
    top_x_percent: float = 5,
    show_plot: bool = True,
    window_size: int = 7,
) -> None:
    """
    Plots the relationship between the global recovery score and load for a given date range.
    It uses the recovery score and load data calculated from the provided dataframes.
    Also calculates a danger score based on load and recovery score.

    Args:
    - df_recovery (pd.DataFrame): DataFrame containing recovery data with columns 'sessionDate', 'metric', and 'value'.
    - df_gps (pd.DataFrame): DataFrame containing GPS data with columns 'date', 'distance', 'distance_over_21', and 'accel_decel_over_2_5'.
    - start_date (str): Start date for filtering the data (default is "01/01/2023").
    - end_date (str): End date for filtering the data (default is "31/12/2030").
    - alpha (float): Exponent applied to distance in load calculation (default is 2).
    - beta (float): Exponent applied to accelerations/decelerations in load calculation (default is 1).
    - gamma (float): Exponent applied to distance over 21 km/h in load calculation (default is 3).
    - top_x_percent (float): Percentage for determining the top x% threshold for load (default is 5).
    - show_plot (bool): Whether to display the plot (default is True).

    Returns:
    - None: The function displays the plot but does not return any value.
    """

    # Calculate the global recovery score and get the filtered dataframe
    df_recovery_filtered = plot_global_recovery_score(
        df_recovery,
        start_date=start_date,
        end_date=end_date,
        show_plot=False,
        window_size=window_size,
    )

    # Calculate the load and get the filtered dataframe
    threshold, df_load_by_date = compute_load_and_plot(
        df_gps,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        top_x_percent=top_x_percent,
        show_plot=False,
        window_size=window_size,
    )

    # Merge the recovery score and load data on the date
    df_load_by_date["date"] = pd.to_datetime(
        df_load_by_date["date_str"], format="%d/%m/%Y"
    )
    df_recovery_filtered["sessionDate"] = pd.to_datetime(
        df_recovery_filtered["sessionDate"], format="%d/%m/%Y"
    )
    merged_df = pd.merge(
        df_recovery_filtered,
        df_load_by_date,
        left_on="sessionDate",
        right_on="date",
        how="inner",
    )

    # Normalize and calculate the danger score
    max_load = merged_df["window_mean"].max()
    max_recovery_score = merged_df["rolling_mean"].max()
    merged_df["danger_score"] = (merged_df["window_mean"] / max_load) * (
        1 - merged_df["rolling_mean"] / max_recovery_score
    )

    # plot section
    if show_plot:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Load", color="tab:blue")
        ax1.plot(
            merged_df["date"],
            merged_df["window_mean"],
            color="tab:blue",
            label="Load",
        )
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Recovery Score", color="tab:orange")
        ax2.plot(
            merged_df["date"],
            merged_df["rolling_mean"],
            color="tab:orange",
            label="Recovery Score",
        )
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(
            ("outward", 60)
        )  # Shift ax3 to avoid overlap
        ax3.set_ylabel("Danger Score", color="tab:red")
        ax3.plot(
            merged_df["date"],
            merged_df["danger_score"],
            color="tab:red",
            label="Danger Score",
        )
        ax3.tick_params(axis="y", labelcolor="tab:red")

        plt.title("Load, Recovery Score and Danger Score vs Date")
        ax1.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        plt.show()

    return merged_df

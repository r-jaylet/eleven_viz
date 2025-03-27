import pandas as pd

from .hms_to_seconds import hms_to_seconds


def general_kpis(df_filtered: pd.DataFrame) -> dict:
    """
    Computes and returns general KPIs based on the filtered DataFrame.

    :param df_filtered: DataFrame containing session data with distance, peak speed, acceleration/deceleration,
                        and heart rate zone durations.
    :return: A dictionary containing the KPIs.
    """
    time_columns = [
        "hr_zone_1_hms",
        "hr_zone_2_hms",
        "hr_zone_3_hms",
        "hr_zone_4_hms",
        "hr_zone_5_hms",
    ]
    df_filtered.loc[:, time_columns] = (
        df_filtered[time_columns].astype(str).applymap(hms_to_seconds)
    )

    total_distance_km = df_filtered["distance"].sum() * 10e-3
    average_peak_speed = df_filtered["peak_speed"].mean()
    average_accel_decel = (
        df_filtered[
            [
                "accel_decel_over_2_5",
                "accel_decel_over_3_5",
                "accel_decel_over_4_5",
            ]
        ]
        .mean()
        .mean()
    )
    average_time_in_zones = df_filtered[time_columns].mean()
    number_of_sessions = df_filtered.shape[0]

    kpis = {
        "total_distance_km": total_distance_km,
        "average_peak_speed": average_peak_speed,
        "average_accel_decel": average_accel_decel,
        "average_time_in_zones": average_time_in_zones,
        "number_of_sessions": number_of_sessions,
    }

    return kpis

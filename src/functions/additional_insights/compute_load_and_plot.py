from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_load_and_plot(
    df_gps: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    top_x_percent: float = 5.0,
    show_plot: bool = True,
    window_size: int = 7,  # Paramètre pour la taille de la fenêtre de moyenne mobile
) -> Tuple[float, pd.DataFrame]:
    """
    Calcule la charge (load) d'un joueur à partir des données GPS après transformation sigmoïde
    et affiche des graphiques d'analyse. La fonction retourne également le seuil du top x%
    des charges les plus élevées et un DataFrame avec la charge moyenne par jour.

    Args:
        df_gps (pd.DataFrame): DataFrame contenant les colonnes 'date',
                               'distance', 'distance_over_21' et 'accel_decel_over_2_5'.
        alpha (float, optional): Exposant appliqué à la distance dans le calcul de la charge.
                                 Par défaut 1.0.
        beta (float, optional): Exposant appliqué aux accélérations/décélérations.
                                Par défaut 1.0.
        gamma (float, optional): Exposant appliqué à la distance parcourue à plus de 21 km/h.
                                 Par défaut 1.0.
        top_x_percent (float, optional): Pourcentage utilisé pour déterminer le seuil des
                                         charges les plus élevées. Par défaut 5.0.
        show_plot (bool, optional): Si True, affiche les graphiques. Par défaut True.
        window_size (int, optional): Taille de la fenêtre pour la moyenne mobile. Par défaut 7 jours.

    Returns:
        Tuple[float, pd.DataFrame]:
            - Seuil du top x% des plus grandes charges.
            - DataFrame avec la charge moyenne par jour (format de date 'dd/mm/yyyy') et la moyenne mobile.
    """

    # Supprimer les lignes avec des valeurs manquantes dans les variables concernées
    df_gps_cleaned = df_gps.dropna(
        subset=["distance", "distance_over_21", "accel_decel_over_2_5"]
    ).copy()

    # Appliquer la fonction sigmoïde
    def sigmoid(x: pd.Series) -> pd.Series:
        return 1 / (1 + np.exp(-(x - x.mean()) / x.std()))

    df_gps_cleaned["distance_sigmoid"] = sigmoid(df_gps_cleaned["distance"])
    df_gps_cleaned["distance_over_21_sigmoid"] = sigmoid(
        df_gps_cleaned["distance_over_21"]
    )
    df_gps_cleaned["accel_decel_over_2_5_sigmoid"] = sigmoid(
        df_gps_cleaned["accel_decel_over_2_5"]
    )

    # Calcul de la charge (load)
    df_gps_cleaned["load"] = (
        (df_gps_cleaned["distance_sigmoid"] ** alpha)
        * (df_gps_cleaned["distance_over_21_sigmoid"] ** gamma)
        * (df_gps_cleaned["accel_decel_over_2_5_sigmoid"] ** beta)
    ) ** (1 / (alpha + beta + gamma))

    # Affichage des graphiques si show_plot est True
    if show_plot:
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))

        # Load vs Distance
        axs[0].scatter(
            df_gps_cleaned["distance"],
            df_gps_cleaned["load"],
            color="b",
            alpha=0.5,
        )
        axs[0].set_title("Load vs Distance")
        axs[0].set_xlabel("Distance")
        axs[0].set_ylabel("Load")

        # Load vs Distance over 21 km/h
        axs[1].scatter(
            df_gps_cleaned["distance_over_21"],
            df_gps_cleaned["load"],
            color="g",
            alpha=0.5,
        )
        axs[1].set_title("Load vs Distance Over 21 km/h")
        axs[1].set_xlabel("Distance Over 21 km/h")
        axs[1].set_ylabel("Load")

        # Load vs Accel/Decel > 2.5 m/s²
        axs[2].scatter(
            df_gps_cleaned["accel_decel_over_2_5"],
            df_gps_cleaned["load"],
            color="r",
            alpha=0.5,
        )
        axs[2].set_title("Load vs Accel/Decel > 2.5 m/s²")
        axs[2].set_xlabel("Accel/Decel > 2.5 m/s²")
        axs[2].set_ylabel("Load")

        # Fonction de répartition empirique (CDF) de la charge
        sorted_load = np.sort(df_gps_cleaned["load"])
        cdf = np.arange(1, len(sorted_load) + 1) / len(sorted_load)
        axs[3].plot(
            sorted_load, cdf, marker="o", linestyle="-", color="purple"
        )
        axs[3].set_title(
            "Fonction de répartition empirique (CDF) de la charge"
        )
        axs[3].set_xlabel("Charge (Load)")
        axs[3].set_ylabel("Probabilité cumulée")

        plt.tight_layout()
        plt.show()

    # Créer un DataFrame avec la charge en fonction de la date
    df_gps_cleaned["date"] = pd.to_datetime(
        df_gps_cleaned["date"], dayfirst=True
    )
    df_gps_cleaned["date_str"] = df_gps_cleaned["date"].dt.strftime("%d/%m/%Y")

    # Calcul de la charge moyenne par jour
    df_load_by_date = df_gps_cleaned.groupby("date_str", as_index=False)[
        "load"
    ].mean()

    # Convertir 'date_str' en datetime et trier avant de renvoyer
    df_load_by_date["date"] = pd.to_datetime(
        df_load_by_date["date_str"], format="%d/%m/%Y"
    )
    df_load_by_date = df_load_by_date.sort_values(by="date")

    # Ajouter la moyenne mobile de la charge
    df_load_by_date["window_mean"] = (
        df_load_by_date["load"]
        .rolling(window=window_size, min_periods=1)
        .mean()
    )

    # Déterminer le seuil du top x% des plus grandes charges
    threshold_value = np.percentile(
        df_gps_cleaned["load"], 100 - top_x_percent
    )

    return threshold_value, df_load_by_date

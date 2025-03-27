from typing import Dict, List, Union


def average_distances_by_recovery(
    matches_list: List[List[Union[str, float, int, None]]],
) -> Dict[Union[int, str], float]:
    """
    Calculates the average distance covered based on recovery days (md_plus_code).

    :param matches_list: A list of matches where each match is represented as a list containing:
                         [opposition_code (str), date (any), distance (float), md_plus_code (int or None)]
    :return: A dictionary where keys are recovery days (md_plus_code) and values are the average distance covered.
    """
    distance_by_recovery: Dict[Union[int, str], float] = {}
    count_by_recovery: Dict[Union[int, str], int] = {}

    for match in matches_list:
        opposition_code, date, distance, md_plus_code = match

        if md_plus_code is not None:
            if md_plus_code not in distance_by_recovery:
                distance_by_recovery[md_plus_code] = 0.0
                count_by_recovery[md_plus_code] = 0

            distance_by_recovery[md_plus_code] += distance
            count_by_recovery[md_plus_code] += 1

    average_by_recovery = {
        md: distance_by_recovery[md] / count_by_recovery[md]
        for md in distance_by_recovery
    }

    return average_by_recovery

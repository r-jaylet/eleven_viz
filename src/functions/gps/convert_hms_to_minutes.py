def convert_hms_to_minutes(hms: str) -> float:
    """
    Converts a time string in 'hh:mm:ss' format to minutes.

    :param hms: Time string in 'hh:mm:ss' format.
    :return: Equivalent time in minutes. Returns 0 if the input is invalid.
    """
    if isinstance(hms, str) and len(hms.split(":")) == 3:
        try:
            h, m, s = map(int, hms.split(":"))
            return h * 60 + m + s / 60
        except ValueError:
            return 0
    return 0

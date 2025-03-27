def hms_to_seconds(hms: str) -> int:
    """
    Converts a duration in HH:MM:SS format into total seconds.

    :param hms: A string representing time in the format 'HH:MM:SS'.
    :return: The total number of seconds as an integer.
    """
    try:
        h, m, s = map(int, hms.split(":"))
        return h * 3600 + m * 60 + s
    except ValueError:
        raise ValueError("Invalid time format. Expected 'HH:MM:SS'.")

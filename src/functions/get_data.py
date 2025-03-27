import pandas as pd

def get_data(path: str, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """
    Loads a CSV file into a Pandas DataFrame.

    :param path: Path to the CSV file.
    :param encoding: Encoding format for reading the file (default: "ISO-8859-1").
    :return: DataFrame containing the loaded data.
    """
    return pd.read_csv(path, encoding=encoding)

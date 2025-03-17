"""
This module contains functions for loading and processing data from CSV files.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_csv_data(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
    date_columns: Optional[List[str]] = None,
    index_col: Optional[Union[str, int]] = None,
) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file
        columns: List of columns to load (if None, load all columns)
        date_columns: List of columns to parse as dates
        index_col: Column to set as index

    Returns:
        pandas.DataFrame: Loaded data

    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file cannot be parsed as CSV
    """
    try:
        file_path = Path(file_path)
        logger.info(f"Loading data from {file_path}")

        # Check if file exists
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load data
        df = pd.read_csv(
            file_path,
            usecols=columns,
            parse_dates=date_columns,
            index_col=index_col,
        )

        logger.info(f"Successfully loaded data with shape {df.shape}")
        return df

    except pd.errors.EmptyDataError:
        logger.error(f"Empty file: {file_path}")
        raise
    except pd.errors.ParserError:
        logger.error(f"Parser error when reading file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error when loading {file_path}: {str(e)}")
        raise


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        pandas.DataFrame: Summary statistics
    """
    logger.info("Generating summary statistics")

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        logger.warning("No numeric columns found in DataFrame")
        return pd.DataFrame()

    # Calculate statistics
    summary = df[numeric_cols].describe().T
    summary["missing"] = df[numeric_cols].isna().sum()
    summary["missing_pct"] = (df[numeric_cols].isna().sum() / len(df)) * 100

    return summary


def filter_data(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Filter DataFrame based on column conditions.

    Args:
        df: Input DataFrame
        filters: Dictionary with column names as keys and filter conditions as values
                e.g., {'column1': [value1, value2], 'column2': {'min': 0, 'max': 100}}

    Returns:
        pandas.DataFrame: Filtered DataFrame
    """
    filtered_df = df.copy()

    for col, condition in filters.items():
        if col not in filtered_df.columns:
            logger.warning(
                f"Column {col} not found in DataFrame. Skipping filter."
            )
            continue

        if isinstance(condition, list):
            # Filter by list of values
            filtered_df = filtered_df[filtered_df[col].isin(condition)]
        elif isinstance(condition, dict):
            # Filter by range
            if "min" in condition:
                filtered_df = filtered_df[filtered_df[col] >= condition["min"]]
            if "max" in condition:
                filtered_df = filtered_df[filtered_df[col] <= condition["max"]]

    logger.info(f"Filtered data from {len(df)} to {len(filtered_df)} rows")
    return filtered_df

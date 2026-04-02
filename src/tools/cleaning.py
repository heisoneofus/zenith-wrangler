from __future__ import annotations

import pandas as pd


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def fill_missing(
    df: pd.DataFrame,
    strategy: str = "auto",
    fill_value: dict[str, any] | None = None,
) -> pd.DataFrame:
    """
    Fill missing values in dataframe.

    Args:
        df: Input dataframe
        strategy: 'auto' (median for numeric, mode for categorical),
                  'mean', 'median', 'mode', 'forward', 'backward', or 'constant'
        fill_value: Dict mapping column names to specific fill values (overrides strategy)
    """
    result = df.copy()
    fill_value = fill_value or {}

    for column in result.columns:
        if not result[column].isna().any():
            continue

        # Use explicit fill value if provided
        if column in fill_value:
            result[column] = result[column].fillna(fill_value[column])
            continue

        # Apply strategy
        if strategy == "auto":
            if pd.api.types.is_numeric_dtype(result[column]):
                median_val = result[column].median()
                result[column] = result[column].fillna(median_val if pd.notna(median_val) else 0)
            else:
                mode = result[column].mode(dropna=True)
                fill_val = mode.iloc[0] if not mode.empty else "Unknown"
                result[column] = result[column].fillna(fill_val)
        elif strategy == "mean":
            if pd.api.types.is_numeric_dtype(result[column]):
                mean_val = result[column].mean()
                result[column] = result[column].fillna(mean_val if pd.notna(mean_val) else 0)
        elif strategy == "median":
            if pd.api.types.is_numeric_dtype(result[column]):
                median_val = result[column].median()
                result[column] = result[column].fillna(median_val if pd.notna(median_val) else 0)
        elif strategy == "mode":
            mode = result[column].mode(dropna=True)
            fill_val = mode.iloc[0] if not mode.empty else ("Unknown" if not pd.api.types.is_numeric_dtype(result[column]) else 0)
            result[column] = result[column].fillna(fill_val)
        elif strategy == "forward":
            result[column] = result[column].fillna(method="ffill")
        elif strategy == "backward":
            result[column] = result[column].fillna(method="bfill")
        elif strategy == "constant":
            default_val = 0 if pd.api.types.is_numeric_dtype(result[column]) else "Unknown"
            result[column] = result[column].fillna(default_val)

    return result


def remove_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
    method: str = "iqr",
) -> pd.DataFrame:
    """
    Remove outliers from dataframe.

    Args:
        df: Input dataframe
        columns: List of column names to check for outliers (None = all numeric columns)
        factor: IQR multiplier for outlier detection (default 1.5)
        method: 'iqr' or 'zscore' (z-score uses factor as standard deviation threshold)
    """
    result = df.copy()
    numeric_cols = columns or [col for col in result.columns if pd.api.types.is_numeric_dtype(result[col])]

    # Build combined mask to avoid cumulative filtering bug
    mask = pd.Series([True] * len(result), index=result.index)

    for column in numeric_cols:
        if column not in result.columns:
            continue
        series = result[column].dropna()
        if series.empty or len(series) < 4:
            continue

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            mask &= (result[column].isna()) | ((result[column] >= lower) & (result[column] <= upper))
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue
            z_scores = ((result[column] - mean) / std).abs()
            mask &= (result[column].isna()) | (z_scores <= factor)

    return result[mask]

from __future__ import annotations

import ast

import pandas as pd


def pivot_data(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: str = "mean",
) -> pd.DataFrame:
    return pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc).reset_index()


def aggregate_by(
    df: pd.DataFrame,
    group_by: list[str],
    metrics: list[str] | dict[str, str | list[str]],
    agg: str | dict[str, str] = "sum",
) -> pd.DataFrame:
    """
    Aggregate data by grouping dimensions.

    Args:
        df: Input dataframe
        group_by: List of columns to group by
        metrics: List of columns to aggregate, or dict mapping columns to aggregation functions
        agg: Aggregation function(s) - string (applies to all metrics) or dict mapping columns to functions
    """
    if isinstance(metrics, dict):
        # metrics is already a dict like {'col1': 'sum', 'col2': ['mean', 'std']}
        return df.groupby(group_by, dropna=False).agg(metrics).reset_index()
    elif isinstance(agg, dict):
        # metrics is a list, agg is a dict specifying per-column aggregations
        return df.groupby(group_by, dropna=False)[metrics].agg(agg).reset_index()
    else:
        # Simple case: single aggregation function for all metrics
        return df.groupby(group_by, dropna=False)[metrics].agg(agg).reset_index()


def flatten_nested(df: pd.DataFrame, max_depth: int = 1) -> pd.DataFrame:
    """
    Flatten nested JSON/dict structures in dataframe columns.

    Args:
        df: Input dataframe
        max_depth: Maximum nesting depth to flatten (default 1)
    """
    result = df.copy()

    def _parse_maybe_nested(value):
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return value
            if text[0] in "[{":
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, (dict, list)):
                        return parsed
                except (ValueError, SyntaxError):
                    return value
        return value

    # Detect columns with nested structures (dicts, lists of dicts, lists of scalars)
    nested_cols = []
    for col in result.columns:
        sample = result[col].dropna()
        if sample.empty:
            continue

        parsed_sample = sample.map(_parse_maybe_nested)
        first_nested = next((val for val in parsed_sample if isinstance(val, (dict, list))), None)
        if first_nested is None:
            continue

        result[col] = result[col].map(_parse_maybe_nested)

        if isinstance(first_nested, dict):
            nested_cols.append((col, "dict"))
        elif isinstance(first_nested, list):
            first_item = next((item for item in first_nested if item is not None), None)
            if isinstance(first_item, dict):
                nested_cols.append((col, "list_of_dicts"))
            else:
                nested_cols.append((col, "list_of_scalars"))

    for column, col_type in nested_cols:
        try:
            if col_type == "dict":
                expanded = pd.json_normalize(result[column], max_level=max_depth).add_prefix(f"{column}.")
                result = result.drop(columns=[column]).join(expanded)
            elif col_type == "list_of_dicts":
                # Explode list into separate rows, then normalize
                exploded = result.explode(column)
                normalized = pd.json_normalize(exploded[column], max_level=max_depth).add_prefix(f"{column}.")
                # Merge by position to avoid index-multiplication on duplicated exploded indices
                exploded = exploded.drop(columns=[column]).reset_index(drop=True)
                normalized = normalized.reset_index(drop=True)
                exploded = exploded.join(normalized)
                result = exploded
            elif col_type == "list_of_scalars":
                result = result.explode(column)
        except Exception:
            # Skip columns that can't be flattened
            continue

    return result

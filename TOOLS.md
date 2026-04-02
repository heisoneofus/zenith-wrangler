# TOOLS

This document describes the orchestrator tool catalog used by `zenith-wrangler`.

## `read_csv`
- **Name:** `read_csv`
- **Category:** loader
- **Purpose:** Load CSV data into a pandas dataframe.
- **When to use:** Source dataset is a `.csv` file.
- **When not to use:** Source is Excel/Parquet, or data is already loaded in context.
- **Requires context:** none
- **Produces context:** `dataframe`
- **Parameters:**
  - `path` (`str`, required): file path
  - `sample_rows` (`int`, optional, `>=1`): row limit for sampling
- **Output:** pandas dataframe
- **Failure modes:** missing file, invalid CSV encoding/format, invalid `sample_rows`
- **Example call:** `{ "tool_name": "read_csv", "params": { "path": "data/sales.csv" } }`

## `read_excel`
- **Name:** `read_excel`
- **Category:** loader
- **Purpose:** Load Excel data (`.xlsx`, `.xls`) into a pandas dataframe.
- **When to use:** Source dataset is spreadsheet-based.
- **When not to use:** Source is CSV/Parquet or workbook is unsupported/corrupted.
- **Requires context:** none
- **Produces context:** `dataframe`
- **Parameters:**
  - `path` (`str`, required): file path
  - `sample_rows` (`int`, optional, `>=1`): row limit for sampling
- **Output:** pandas dataframe
- **Failure modes:** missing file, unsupported workbook, invalid `sample_rows`
- **Example call:** `{ "tool_name": "read_excel", "params": { "path": "data/sales.xlsx", "sample_rows": 500 } }`

## `read_parquet`
- **Name:** `read_parquet`
- **Category:** loader
- **Purpose:** Load Parquet data into a pandas dataframe.
- **When to use:** Source dataset is a `.parquet` file.
- **When not to use:** Source is CSV/Excel, or parquet metadata/data is corrupted.
- **Requires context:** none
- **Produces context:** `dataframe`
- **Parameters:**
  - `path` (`str`, required): file path
  - `sample_rows` (`int`, optional, `>=1`): row limit for sampling
- **Output:** pandas dataframe
- **Failure modes:** missing file, unreadable parquet schema/data, invalid `sample_rows`
- **Example call:** `{ "tool_name": "read_parquet", "params": { "path": "data/sales.parquet" } }`

## `drop_duplicates`
- **Name:** `drop_duplicates`
- **Category:** cleaning
- **Purpose:** Remove duplicate rows.
- **When to use:** Duplicate records inflate metrics or produce repeated visuals.
- **When not to use:** Duplicate rows represent meaningful repeated events.
- **Requires context:** `dataframe`
- **Produces context:** `dataframe`
- **Parameters:** none
- **Output:** deduplicated dataframe
- **Failure modes:** missing dataframe context
- **Example call:** `{ "tool_name": "drop_duplicates", "params": {} }`

## `fill_missing`
- **Name:** `fill_missing`
- **Category:** cleaning
- **Purpose:** Impute missing values with strategy-aware defaults.
- **When to use:** Null-heavy fields block aggregation/visualization.
- **When not to use:** Missingness itself is analytically important and should remain explicit.
- **Requires context:** `dataframe`
- **Produces context:** `dataframe`
- **Parameters:**
  - `strategy` (`auto|mean|median|mode|forward|backward|constant`, optional, default `auto`)
  - `fill_value` (`dict[str, any]`, optional): column-specific constants (usually with `constant`)
- **Output:** dataframe with reduced missingness
- **Failure modes:** invalid strategy literal, incompatible fill values for column types, missing dataframe context
- **Example call:** `{ "tool_name": "fill_missing", "params": { "strategy": "constant", "fill_value": { "region": "Unknown" } } }`

## `remove_outliers`
- **Name:** `remove_outliers`
- **Category:** cleaning
- **Purpose:** Remove outlier rows using IQR or Z-score.
- **When to use:** Extreme values distort KPI trends or chart readability.
- **When not to use:** Outliers are meaningful anomalies that should be retained.
- **Requires context:** `dataframe`
- **Produces context:** `dataframe`
- **Parameters:**
  - `columns` (`list[str]`, optional): numeric columns to process (all numeric if omitted)
  - `factor` (`float`, optional, `>0`, default `1.5`): sensitivity factor
  - `method` (`iqr|zscore`, optional, default `iqr`)
- **Output:** filtered dataframe
- **Failure modes:** invalid method/factor, non-numeric target columns, missing dataframe context
- **Example call:** `{ "tool_name": "remove_outliers", "params": { "columns": ["sales"], "method": "zscore", "factor": 3.0 } }`

## `pivot_data`
- **Name:** `pivot_data`
- **Category:** transform
- **Purpose:** Build a pivoted matrix for matrix/heatmap workflows.
- **When to use:** Need row/column cross-tab style data structure.
- **When not to use:** Data is already in desired wide format.
- **Requires context:** `dataframe`
- **Produces context:** `dataframe` (derived dataframe reference; does not overwrite baseline cleaned dataframe)
- **Parameters:**
  - `index` (`str`, required): row key
  - `columns` (`str`, required): column key
  - `values` (`str`, required): values column
  - `aggfunc` (`mean|sum|median|count|min|max`, optional, default `mean`)
  - `dataframe_ref` (`str`, optional): source dataframe reference (defaults to active dataframe)
  - `output_dataframe_ref` (`str`, optional): name for the derived dataframe artifact
- **Output:** pivoted dataframe
- **Failure modes:** missing columns, non-aggregable values, invalid agg function, missing dataframe context
- **Example call:** `{ "tool_name": "pivot_data", "params": { "index": "region", "columns": "month", "values": "sales", "aggfunc": "mean" } }`

## `aggregate_by`
- **Name:** `aggregate_by`
- **Category:** transform
- **Purpose:** Group and aggregate metrics for KPI rollups and chart-ready summaries.
- **When to use:** Need grouped totals/means/counts per dimension.
- **When not to use:** Granular row-level detail must be preserved.
- **Requires context:** `dataframe`
- **Produces context:** `dataframe` (derived dataframe reference; does not overwrite baseline cleaned dataframe)
- **Parameters:**
  - `group_by` (`list[str]`, required): grouping columns
  - `metrics` (`list[str] | dict[str, str | list[str]]`, required): metric columns or per-column aggregation map
  - `agg` (`str | dict[str, str]`, optional, default `sum`): global or per-column aggregator
  - `dataframe_ref` (`str`, optional): source dataframe reference (defaults to active dataframe)
  - `output_dataframe_ref` (`str`, optional): name for the derived dataframe artifact
- **Output:** grouped dataframe
- **Failure modes:** missing grouping/metric columns, unsupported aggregators, missing dataframe context
- **Example call:** `{ "tool_name": "aggregate_by", "params": { "group_by": ["region"], "metrics": ["sales"], "agg": "sum" } }`

## `flatten_nested`
- **Name:** `flatten_nested`
- **Category:** transform
- **Purpose:** Flatten nested dict/list-of-dict payloads into tabular columns.
- **When to use:** Dataset includes JSON-like nested structures.
- **When not to use:** Data is already flat or nesting should remain intact for downstream logic.
- **Requires context:** `dataframe`
- **Produces context:** `dataframe` (derived dataframe reference; does not overwrite baseline cleaned dataframe)
- **Parameters:**
  - `max_depth` (`int`, optional, `>=0`, default `1`): flatten recursion depth
  - `dataframe_ref` (`str`, optional): source dataframe reference (defaults to active dataframe)
  - `output_dataframe_ref` (`str`, optional): name for the derived dataframe artifact
- **Output:** flattened dataframe
- **Failure modes:** malformed nested values, unsupported deep structures, missing dataframe context
- **Example call:** `{ "tool_name": "flatten_nested", "params": { "max_depth": 2 } }`

## `create_figure`
- **Name:** `create_figure`
- **Category:** visualization
- **Purpose:** Create a Plotly figure from a validated `VisualSpec`.
- **When to use:** Need chart objects for dashboard rendering.
- **When not to use:** No meaningful numeric/categorical fields are available.
- **Requires context:** `dataframe`
- **Produces context:** `figure`
- **Parameters:**
  - `spec` (`VisualSpec`, required): chart metadata (`title`, `chart_type`, axis fields, optional color/aggregation)
  - `dataframe_ref` (`str`, optional): explicit dataframe reference to visualize. If omitted, tool uses baseline cleaned dataframe.
- **Output:** Plotly figure object
- **Failure modes:** missing required chart columns/axes, invalid chart spec, empty post-aggregation dataset
- **Example call:** `{ "tool_name": "create_figure", "params": { "spec": { "title": "Sales by Region", "chart_type": "bar", "x": "region", "y": "sales" } } }`

## `build_dashboard`
- **Name:** `build_dashboard`
- **Category:** dashboard
- **Purpose:** Build Dash app/layout from a validated `DashboardSpec`.
- **When to use:** Final assembly step after selecting visuals/filters.
- **When not to use:** Data pipeline is incomplete or chart specifications are still unstable.
- **Requires context:** `dataframe`
- **Produces context:** `dashboard`
- **Parameters:**
  - `design` (`DashboardSpec`, required): dashboard title/layout/visual/filter plan
  - `dataframe_ref` (`str`, optional): explicit dataframe reference for dashboard/filter context. If omitted, tool uses baseline cleaned dataframe.
- **Output:** `DashboardResult` containing Dash app + resolved spec
- **Failure modes:** invalid visual/filter references, rendering callback errors at runtime, missing dataframe context
- **Example call:** `{ "tool_name": "build_dashboard", "params": { "design": { "title": "Sales Dashboard", "layout": "grid", "visuals": [] } } }`

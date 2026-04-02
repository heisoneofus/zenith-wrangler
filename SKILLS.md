# SKILLS

Reusable orchestration patterns for `zenith-wrangler`.

## Skill: Basic Tabular KPI Dashboard
- **Problem shape:** Flat business table with core measures (`revenue`, `orders`, `margin`) and 1-3 dimensions.
- **Signals:** Numeric KPI columns, categorical dimensions, low nesting.
- **Recommended tool sequence:**
  1. `read_csv` / `read_excel` / `read_parquet`
  2. `drop_duplicates`
  3. `fill_missing`
  4. `aggregate_by`
  5. `create_figure` (bar/line KPI visuals)
  6. `build_dashboard`
- **Expected outputs:** Clean grouped dataframe, KPI figures, dashboard with filters.
- **Common mistakes:** Aggregating before handling nulls, over-filtering dimensions, inconsistent metric naming.
- **Fallback behavior:** If aggregation fails, skip to direct `create_figure` on cleaned raw dataframe and build a minimal dashboard.

## Skill: Time-Series Dashboard
- **Problem shape:** Date/time indexed metrics with trend and seasonality questions.
- **Signals:** Date-like columns, periodic metrics, request for trend/forecast-like views.
- **Recommended tool sequence:**
  1. `read_*`
  2. `fill_missing` (often `forward`/`backward`)
  3. `remove_outliers` (optional)
  4. `aggregate_by` (daily/weekly/monthly grouping when needed)
  5. `create_figure` (line/area)
  6. `build_dashboard`
- **Expected outputs:** Time-grain-consistent dataframe, trend figures, dashboard with date filter.
- **Common mistakes:** Using non-parsed date fields, mixed frequencies in one chart, removing true spikes as “outliers”.
- **Fallback behavior:** If temporal grouping is unreliable, render charts directly from raw timestamp order and flag assumptions.

## Skill: Dirty Public CSV Cleanup
- **Problem shape:** Messy open-data CSV with nulls, duplicates, and inconsistent typing.
- **Signals:** High NA ratio, duplicated rows, suspicious extreme values, object-typed numeric columns.
- **Recommended tool sequence:**
  1. `read_csv`
  2. `drop_duplicates`
  3. `fill_missing` (`auto` first, then `constant` if needed)
  4. `remove_outliers`
  5. `aggregate_by` or direct `create_figure`
  6. `build_dashboard`
- **Expected outputs:** Stabilized dataframe suitable for charting and KPI summaries.
- **Common mistakes:** Applying outlier removal before type/NA cleanup, too-aggressive filtering on small samples.
- **Fallback behavior:** If cleaning introduces too much loss, revert to minimal cleanup (`drop_duplicates` + `fill_missing(auto)`) and continue.

## Skill: Nested JSON Normalization
- **Problem shape:** Columns contain dict or list-of-dict payloads.
- **Signals:** Object columns with nested braces/lists, missing expected flat fields.
- **Recommended tool sequence:**
  1. `read_*`
  2. `flatten_nested`
  3. `fill_missing`
  4. `aggregate_by` / `pivot_data` (optional)
  5. `create_figure`
  6. `build_dashboard`
- **Expected outputs:** Flattened tabular dataframe with extracted nested attributes.
- **Common mistakes:** Flattening too deeply without need, naming collisions, skipping post-flatten null handling.
- **Fallback behavior:** If flattening partially fails, keep successful extracted columns, preserve original payload columns, proceed with available fields.

## Skill: Heatmap / Matrix Analysis
- **Problem shape:** Need cross-dimensional intensity comparisons (region vs month, product vs segment).
- **Signals:** Two strong categorical dimensions plus numeric measure.
- **Recommended tool sequence:**
  1. `read_*`
  2. `fill_missing`
  3. `pivot_data`
  4. `create_figure` (heatmap)
  5. `build_dashboard`
- **Expected outputs:** Pivoted matrix dataframe and heatmap-oriented dashboard.
- **Common mistakes:** Choosing overly high-cardinality dimensions, passing non-numeric values as matrix measure.
- **Fallback behavior:** If pivot fails, use `aggregate_by` and render bar chart alternatives while preserving analytical intent.

## Skill: Dashboard Update / Regeneration
- **Problem shape:** Existing analysis/spec needs refreshed visuals or layout with same data context.
- **Signals:** Update prompt asks for chart type swaps, filter additions, layout changes.
- **Recommended tool sequence:**
  1. Reuse existing dataset context (or `read_*` if required)
  2. Optional cleanup/transforms only when update depends on them
  3. `create_figure` for revised visual specs
  4. `build_dashboard`
- **Expected outputs:** Updated dashboard preserving prior intent and compatible filters.
- **Common mistakes:** Re-running unnecessary heavy transforms, changing KPI definitions unintentionally.
- **Fallback behavior:** If requested visuals are incompatible with available fields, generate closest valid chart types and report substitutions.

## Skill: Sparse / Fallback Dashboard Generation
- **Problem shape:** Very small, sparse, or partially broken dataset where ideal plan is impossible.
- **Signals:** Few rows, many null columns, failed transform/aggregation attempts.
- **Recommended tool sequence:**
  1. `read_*`
  2. Minimal cleanup (`drop_duplicates`, `fill_missing(auto)`)
  3. Optional `aggregate_by` only if stable
  4. Conservative `create_figure` (single-axis bar/histogram)
  5. `build_dashboard`
- **Expected outputs:** Functional “best-effort” dashboard with explicit limitations.
- **Common mistakes:** Forcing complex multi-step transforms on unstable data, suppressing error context.
- **Fallback behavior:** Build a minimal dashboard with available valid figures and surface constraints/errors clearly.

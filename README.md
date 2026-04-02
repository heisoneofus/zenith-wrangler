# Zenith Wrangler

Zenith Wrangler is a microservice that turns raw datasets into interactive dashboards using an LLM-guided analysis and an automated Plotly + Dash pipeline.

## Quick Start

### Install dependencies

```
shell uv sync
``` 

### Run with local Dash app output (default)
```
shell uv run zenith-wrangler --data path/to/dataset.csv --description path/to/context.md
shell uv run zenith-wrangler --data path/to/dataset.csv --llm-api-key <YOUR_API_KEY>
``` 

### Update an existing session/dashboard plan

```
shell uv run zenith-wrangler --update --session logs/session_20260401_143022.log --prompt "Change the time series to a bar chart"
``` 

### Optional HTML fallback output
```
shell uv run zenith-wrangler --data path/to/dataset.csv --output-format html
``` 

### Export tool catalog artifact (for prompt assembly pipelines)
```
shell uv run zenith-wrangler --dump-tool-catalog
``` 

Use `--catalog-output-dir path/to/dir` to override the default `outputs/tool_catalogs` destination.

You can also inspect all available CLI options with:
```
shell uv run zenith-wrangler --help
``` 

## Pipeline and architecture

The core decision engine is preserved:

1. Load and profile dataset
2. Run LLM-guided analysis (`Analyzer`) for metrics, data quality, and dashboard design
3. Build dynamic execution plan (`Orchestrator`) with tool calls for cleaning/transforms/figure generation
4. Build Dash layout and callbacks from generated `DashboardSpec` and `VisualSpec`
5. Render in a local Dash server (`--output-format server`, default)

This means the app still decides dashboard content dynamically from the agentic pipeline; Dash is the delivery layer.

Tool orchestration now uses a schema-backed registry with `pydantic` input validation (`ToolSpec`), including machine-readable catalog export for prompt/tooling integration.

Nested-like columns are now handled more robustly across analysis and transforms:
- Analyzer guidance explicitly flags dict-like/list-like/JSON-like and CSV stringified nested values (for example `"['Drama', 'Crime']"` or `'{"a": 1}'`) and recommends `flatten_nested` when beneficial.
- Orchestrator parsing maps flatten/nested/json/list-like recommendations to `flatten_nested` automatically.
- `flatten_nested` safely parses stringified nested values with Python standard library parsing and explodes list-of-scalar columns into rows.

Intelligence/planning behavior has been expanded for broader tool usage and chart diversity:
- Analyzer guidance now flags identifier-like fields (for example `user_id`, `transaction_id`, UUID/GUID) so they are not treated as continuous KPIs by default.
- Analyzer/orchestrator now support explicit transform recommendations for `aggregate_by` (grouped KPI rollups) and `pivot_data` (matrix/heatmap-ready data).
- Visualization planning now allows a wider chart mix, including `pie` and `heatmap` workflows where appropriate.

## Outputs

- Live Dash app on `localhost:<port>` in default server mode
- Session logs in `logs/` (including dashboard spec)
- Transformed dataset artifacts in `outputs/` when cleaning/transforms run
- Optional HTML dashboard in `outputs/` only when `--output-format html` is used

## Tooling and orchestration references

- `TOOLS.md`: full catalog of available tools, parameter schemas, usage guidance, outputs, and failure modes.
- `SKILLS.md`: reusable orchestration playbooks for common dashboard/data-shaping workflows.

## Resilience and fallback behavior

- The service falls back to heuristic analysis if no OpenAI API key is available.
- Invalid/missing chart fields are validated before figure creation.
- In Dash mode, chart-specific rendering failures are shown as controlled error figures in the UI.
- If pipeline execution fails in server mode, a Dash error page is shown with details instead of a silent crash.
- Update mode requires accessible dataset context for server rendering.

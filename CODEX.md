# Zenith Wrangler - AI-Powered Data Dashboard Generator

## Overview

Zenith Wrangler is an autonomous microservice that transforms raw datasets into interactive dashboards through intelligent analysis and automated visualization generation. It leverages LLM reasoning (GPT-5.4) to understand data characteristics, determine optimal transformations, and generate production-ready Plotly Dash dashboards.

## Core Functionality

### Input
- **Dataset**: `.csv`, `.xlsx`, `.xls`, `.parquet` files via CLI path argument
- **Optional Description**: YAML, Markdown, or plain text file providing context about the dataset
- **Mode**: Standard creation or `--update` mode for dashboard refinement

### Output
- Interactive Plotly Dash dashboard (`.html` or live server)
- Session log file with complete LLM reasoning chain and decisions
- Transformed dataset artifacts (if transformations applied)

## Architecture

### Phase 1: Analysis Pipeline

The system performs sequential LLM-driven analysis:

1. **Metrics Identification**
   - Analyze dataset schema, distributions, and relationships
   - Identify primary KPIs and supporting secondary metrics
   - Consider domain context from optional description file

2. **Data Quality Assessment**
   - Detect missing values, duplicates, outliers
   - Identify required cleaning operations
   - Determine transformation needs (normalization, aggregation, pivoting, etc.)

3. **Dashboard Design Planning**
   - Select optimal layout structure (grid, tabs, sections)
   - Choose visualization types per metric (time series, bar, scatter, heatmap, etc.)
   - Define interactive elements (filters, dropdowns, date ranges, cross-filtering)

### Phase 2: Orchestrator Execution

An LLM orchestrator agent with tool-calling capabilities executes:

**Available Tools:**
- `read_csv()`, `read_excel()`, `read_parquet()` - Data loading
- `drop_duplicates()`, `fill_missing()`, `remove_outliers()` - Cleaning operations
- `pivot_data()`, `aggregate_by()`, `flatten_nested()` - Transformations
- `create_figure()` - Generate Plotly chart objects
- `build_dashboard()` - Assemble Dash layout with callbacks
- `export_dashboard()` - Generate standalone HTML or launch server

**Execution Flow:**
1. Load dataset using appropriate reader
2. Apply cleaning operations identified in Phase 1
3. Execute transformations in dependency order
4. Generate visualization components
5. Construct dashboard with interactivity
6. Export final output

### Phase 3: Update Mode

When invoked with `--update` flag:
- Skip data loading and transformation steps
- Load existing dashboard configuration
- Apply user-requested visual/layout modifications
- Regenerate dashboard with updates
- Preserve data pipeline artifacts

## Session Logging

Each execution creates a timestamped session log containing:

```
logs/session_YYYYMMDD_HHMMSS.log
├── Input Configuration
├── Phase 1: Analysis
│   ├── Metrics Analysis (prompt + response)
│   ├── Data Quality Assessment (prompt + response)
│   └── Dashboard Design (prompt + response)
├── Phase 2: Orchestration
│   ├── Tool Call 1: [tool_name] + reasoning
│   ├── Tool Call 2: [tool_name] + reasoning
│   └── ...
├── Phase 3: Output Generation
└── Execution Summary
```


Format: Structured JSON or human-readable markdown with clear section delineation.

## CLI Interface

```shell script
# Standard mode
python wrangler.py --data path/to/dataset.csv --description path/to/context.yaml

# Update mode
python wrangler.py --update --session logs/session_20260401_143022.log --prompt "Change the time series to a bar chart and add a region filter"

# Advanced options
python wrangler.py --data data.parquet --output-format server --port 8050 --log-level debug
```


### Arguments
- `--data`: Path to input dataset (required in standard mode)
- `--description`: Path to optional context file
- `--update`: Enable update mode
- `--session`: Session log to update (required with `--update`)
- `--prompt`: Update instructions (required with `--update`)
- `--output-format`: `html` (default) or `server`
- `--port`: Port for server mode (default: 8050)
- `--log-level`: `info` (default), `debug`, `error`

## Implementation Guidelines

### LLM Integration
- Use structured output mode with JSON schemas for analysis phases
- Implement function calling for orchestrator tool execution
- Include chain-of-thought reasoning in prompts
- Set appropriate temperature (0.3 for analysis, 0.1 for tool selection)

### Tool System Design
```python
@dataclass
class ToolSpec:
    name: str
    description: str
    category: str
    input_model: type[BaseModel]
    output_kind: str
    requires_context: tuple[str, ...]
    produces_context: tuple[str, ...]
    usage_guidance: str
    public_safe: bool
    deterministic: bool
    version: str
    examples: list[dict[str, Any]]
    execute: Callable

    def validate_params(self, raw: dict[str, Any]) -> BaseModel: ...
    def to_prompt_dict(self) -> dict[str, Any]: ...
    def to_json_schema(self) -> dict[str, Any]: ...

class ToolRegistry:
    """Schema-backed registry of available tools"""
    def register(self, tool: ToolSpec) -> None: ...
    def get(self, name: str) -> ToolSpec: ...
    def list_tools(self) -> list[str]: ...
    def list_tool_specs(self) -> list[ToolSpec]: ...
    def export_tool_catalog(self) -> list[dict[str, Any]]: ...

class Orchestrator:
    """LLM-powered agent that selects and executes tools"""
    def plan_execution(self, analysis: dict) -> list[ToolCall]
    def execute_plan(self, plan: list[ToolCall]) -> ExecutionResult
```

Tool calls are validated through `pydantic` input models before execution, and normalized params are logged for traceability.

See root-level docs for discoverability and prompt/orchestration guidance:
- `TOOLS.md` for tool semantics and failure modes
- `SKILLS.md` for reusable multi-step orchestration patterns


### Scalability Considerations
- Stream-process large datasets (chunked reading)
- Implement sampling strategy for LLM analysis (analyze representative subset)
- Cache intermediate results
- Parallel execution of independent tool calls

### Error Handling
- Graceful fallbacks for LLM API failures
- Data validation at each transformation step
- User-friendly error messages with recovery suggestions
- Automatic retry logic with exponential backoff

### Dashboard Standards
- Responsive layout (mobile, tablet, desktop)
- Accessible color schemes (colorblind-safe palettes)
- Loading indicators for interactive elements
- Clear axis labels, titles, and legends
- Consistent spacing and typography (use Dash Bootstrap Components)

## Technology Stack

**Core:**
- Python 3.11+
- OpenAI Python SDK (GPT-5.2+)
- Plotly + Dash
- Pandas (primary data processing library)

**Recommended Libraries:**
- `pyarrow` - Parquet support
- `openpyxl` - Excel handling
- `pydantic` - Data validation and schema enforcement
- `click` - CLI framework
- `loguru` - Enhanced logging
- `dash-bootstrap-components` - UI components

---

## Code Review Summary (2026-04-02)

### Issues Identified and Resolved

**1. Data Cleaning Tools (src/tools/cleaning.py)**
- **fill_missing()**: Enhanced with parameterizable strategies ('auto', 'mean', 'median', 'mode', 'forward', 'backward', 'constant') and column-specific fill values via dict. Previously used hardcoded median/mode with no edge case handling for all-NaN columns.
- **remove_outliers()**: Fixed critical cumulative filtering bug where successive column filters compounded instead of combining. Now builds unified mask before filtering. Added Z-score method as alternative to IQR. Added minimum row count validation (n>=4) and zero-variance checks.

**2. Data Transformation Tools (src/tools/transforms.py)**
- **aggregate_by()**: Extended to support dict-based aggregations allowing different functions per column (e.g., `{'revenue': 'sum', 'margin': 'mean'}`). Previously limited to single aggregation function across all metrics.
- **flatten_nested()**: Handles dict columns, list-of-dict columns, and CSV stringified nested payloads via safe standard-library parsing (`ast.literal_eval`) with graceful degradation on malformed values. Added max_depth parameter for controlling flattening recursion and list-of-scalar explosion behavior for chart/filter friendliness.

**3. Visualization Tools (src/tools/visualization.py)**
- **create_figure()**: Added empty dataframe validation before processing. Added post-aggregation validation to prevent rendering empty charts, returning error placeholders instead.

**4. Tool Registry (src/agents/orchestrator.py)**
- Updated all tool parameter signatures in `build_registry()` to match refactored implementations:
  - `fill_missing`: added `strategy` and `fill_value` parameters
  - `remove_outliers`: added `factor` and `method` parameters
  - `aggregate_by`: updated to accept dict types for `metrics` and `agg`
  - `flatten_nested`: added `max_depth` parameter

**5. Intelligence Layer and Visualization Variety (src/agents/analyzer.py, src/agents/orchestrator.py, src/tools/visualization.py)**
- Analyzer now explicitly identifies identifier-like columns (e.g., `user_id`, `transaction_id`, UUID/GUID-style fields) and avoids using them as continuous KPI axes by default.
- Analyzer quality guidance now includes explicit transform intent for grouped rollups (`aggregate_by`) and matrix reshaping (`pivot_data`) when beneficial.
- Orchestrator operation parsing now maps aggregate/pivot-like recommendations into concrete `aggregate_by`/`pivot_data` tool calls with schema-safe defaults.
- Orchestrator also sanitizes visual specs to reduce inappropriate ID-axis usage when identifier-like fields appear in LLM output.
- Visualization support now includes `pie` charts in `VisualSpec` + figure generation, enabling broader dashboard variety beyond timeline/bar defaults.

### Flexibility Improvements

The refactored tools now handle diverse datasets through:
- **Parameterized strategies** instead of hardcoded logic
- **Multi-type parameter support** (str|dict for aggregations, enabling per-column control)
- **Graceful degradation** with try-except blocks in complex operations (flatten_nested)
- **Validation layers** preventing empty/invalid data from reaching visualization stage
- **Edge case handling**:
  - All-NaN columns → fills with 0/"Unknown"
  - Zero variance columns → skips outlier detection
  - Minimal row counts (<4) → skips statistical operations
  - Empty post-aggregation → returns error figure

### Dataset Compatibility

The system can now robustly process:
- Datasets with 100% missing values in columns (fills with type-appropriate defaults)
- High-cardinality categorical data (mode falls back to "Unknown" when unavailable)
- Complex nested JSON structures (both `{...}` dicts and `[{...}, {...}]` list-of-dicts)
- Edge cases in outlier detection (zero IQR/std, insufficient data points)
- Post-aggregation empty results (graceful error rendering vs crashes)
- Multi-strategy missing value imputation (forward/backward fill, statistical, constant)

---

## Project Structure

```
zenith-wrangler/
├── wrangler.py                 # CLI entrypoint
├── src/
│   ├── agents/
│   │   ├── analyzer.py         # Phase 1 LLM calls
│   │   └── orchestrator.py     # Phase 2 tool execution
│   ├── tools/
│   │   ├── loaders.py          # Data reading tools
│   │   ├── cleaning.py         # Data quality tools
│   │   ├── transforms.py       # Transformation tools
│   │   └── visualization.py    # Dashboard generation tools
│   ├── dashboard/
│   │   ├── builder.py          # Dash app construction
│   │   └── templates.py        # Layout templates
│   ├── logging/
│   │   └── session.py          # Session logging utilities
│   └── config.py               # Configuration management
├── logs/                        # Session logs
├── outputs/                     # Generated dashboards
├── tests/
├── requirements.txt
└── README.md
```


## Development Phases

**Phase 1:** Core analysis pipeline + basic tool registry
**Phase 2:** Orchestrator with tool execution + logging
**Phase 3:** Dashboard generation + HTML export
**Phase 4:** Update mode + session management
**Phase 5:** Optimization for large datasets + advanced visualizations

## Success Metrics

- Successfully processes 95%+ of standard tabular datasets
- Generates dashboards in <5 minutes for datasets <100MB
- LLM reasoning logs enable full audit trail
- Update mode reduces iteration time by 80%+
# Zenith Wrangler

Turn raw datasets into an analyst-facing dashboard navigator.

Zenith Wrangler profiles a dataset, proposes a dashboard plan, records the reasoning behind each visual, and renders a local Dash workspace with plan, canvas, and provenance surfaces. It can run in heuristic mode or use an LLM for smarter planning, and it now persists machine-readable session state and execution traces for revision workflows.

## What It Does

- Loads CSV, Excel, or Parquet data
- Profiles schema, metrics, and data-quality issues
- Proposes transforms and chart selections before execution
- Executes the approved plan with tool-level tracing and guardrails
- Renders a local dashboard workspace with filters, rationale, and version history
- Supports dashboard revisions through patch-based update requests

## Quick Start

### Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

### Install

```bash
git clone https://github.com/heisoneofus/zenith-wrangler.git
cd zenith-wrangler
uv sync
```

### Run The Included Sample

```bash
uv run zenith-wrangler --data sample_data/sample.csv
```

Then open:

```text
http://localhost:8050
```

## Common Workflows

### Review A Plan Before Execution

```bash
uv run zenith-wrangler --data sample_data/sample.csv --review-only --output-format html
```

This writes a session log plus sidecar artifacts:

- `logs/session_*.log`
- `logs/session_*.state.json`
- `logs/session_*.trace.json`

### Run With Your Own Data

```bash
uv run zenith-wrangler --data your_file.csv
uv run zenith-wrangler --data your_file.xlsx
uv run zenith-wrangler --data your_file.parquet
```

### Provide Extra Analyst Context

```bash
uv run zenith-wrangler \
  --data your_file.csv \
  --description description.md
```

### Use LLM Planning

```bash
uv run zenith-wrangler \
  --data your_file.csv \
  --llm-api-key YOUR_KEY
```

Without an API key, Zenith Wrangler falls back to heuristic planning.

### Revise A Prior Session

```bash
uv run zenith-wrangler \
  --update \
  --session logs/session_20260404_010101.log \
  --prompt "Change Margin Over Time to a scatter chart and add filter for segment"
```

## CLI Notes

- `--data` is required in standard mode.
- `--description` adds domain context for analysis.
- `--review-only` writes the proposal and session artifacts without executing the plan.
- `--update` reuses a prior session and applies structured dashboard patch operations.
- `--output-format` supports `server` and `html`.
- Use `--output-format server` for live workspace controls like `Approve Plan`, chart regeneration prompts, and export buttons.
- Use `--output-format html` for a static artifact export; it renders the visuals but does not run Dash callbacks.

## Project Structure

```text
src/
  agents/       planner, executor substrate, critic, and patching
  dashboard/    Dash workspace generation
  logging/      markdown logs plus JSON session artifacts
  tools/        loaders, cleaning, transforms, visualization
  tooling/      schema-backed tool specs
  models.py     shared typed artifacts

sample_data/    example input datasets
tests/          unit and integration coverage
```

## Running Tests

Use the managed environment so dependencies match the project:

```bash
uv run pytest -q
```

## Current Focus

- Analyst-facing navigator UX
- Inspectable provenance and trace artifacts
- Safer transform execution with bounded repair
- Patch-based revision workflows

## Limitations

- Local-first, single-user workflow
- Dashboard revision prompts are structured heuristics first, not a full conversational editor
- Static HTML exports do not include the full interactive navigator workspace

## License

MIT

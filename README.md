# Zenith Wrangler

Turn raw datasets into working dashboards automatically.

Zenith Wrangler takes a dataset, understands its structure, and builds a dashboard on the fly using LLM-assisted logic or heuristic fallback. No manual chart wiring, no predefined schemas.

It’s built for fast exploration when you don’t want to spend time shaping data manually.

---

## What it does

- Reads a dataset (CSV for now)
- Infers structure and relationships
- Uses LLM or heuristics to plan visualizations
- Launches a local Dash app with generated charts

You give it data. It gives you a dashboard.

---

## 5-minute quick start (no API key required)

This is the easiest way to verify everything works.

### 1. Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) installed

If you don’t have `uv`:

```bash
pip install uv
```

---

### 2. Install & run

```bash
git clone https://github.com/heisoneofus/zenith-wrangler.git
cd zenith-wrangler

uv sync
uv run zenith-wrangler --data sample_data/sample.csv
```

Then open:

```
http://localhost:8050
```

You should see a generated dashboard.

---

## Using your own data

```bash
uv run zenith-wrangler --data your_file.csv
```

Optional context file:

```bash
uv run zenith-wrangler \
  --data your_file.csv \
  --context description.md
```

---

## Using LLM mode (optional)

If you want smarter chart planning:

```bash
uv run zenith-wrangler \
  --data your_file.csv \
  --llm-api-key YOUR_KEY
```

Without a key, the app falls back to heuristic mode.

---

## Project structure

```
src/
  core/        # data processing + orchestration
  planning/    # LLM / heuristic logic
  dashboard/   # Dash app generation
  config.py    # runtime config

sample_data/   # example inputs
tests/         # sanity checks
```

---

## Known limitations

- CSV only (for now)
- Works best on clean, tabular data
- Large datasets may slow down dashboard generation
- LLM mode depends on API reliability

---

## Running tests

```bash
pytest
```

---

## Why this exists

Most dashboards are built manually even when the data is already structured.

Zenith Wrangler flips that:
- infer first
- build second
- refine later

---

## Roadmap

- Better dataset profiling
- More chart types
- Smarter LLM planning
- Support for more formats (Parquet, DBs)

---

## License

MIT

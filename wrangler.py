from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import click
import pandas as pd
from loguru import logger

from src.agents.analyzer import Analyzer
from src.agents.orchestrator import Orchestrator, build_registry, export_tool_catalog
from src.config import AppConfig, LLMConfig
from src.logging.session import init_session_logger, load_dashboard_spec, load_session_metadata
from src.models import DashboardSpec
from src.tools import loaders


def _read_description(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Description file not found: {path}")
    return path.read_text(encoding="utf-8")


def _load_for_analysis(data_path: Path, sample_rows: int) -> pd.DataFrame:
    loader = loaders.detect_loader(data_path)
    if loader == "read_excel":
        return loaders.read_excel(data_path, sample_rows=sample_rows)
    if loader == "read_parquet":
        return loaders.read_parquet(data_path, sample_rows=sample_rows)
    return loaders.read_csv(data_path, sample_rows=sample_rows)


def _apply_update_prompt(spec: DashboardSpec, prompt: str) -> DashboardSpec:
    lowered = prompt.lower()
    ChartType = Literal["line", "bar", "scatter", "histogram", "box", "heatmap", "area", "pie"]
    chart_map: dict[str, ChartType] = {
        "bar chart": "bar",
        "line chart": "line",
        "scatter": "scatter",
        "heatmap": "heatmap",
        "heat map": "heatmap",
        "box plot": "box",
        "histogram": "histogram",
        "area": "area",
        "pie chart": "pie",
        "pie": "pie",
    }
    updated_visuals = []
    new_chart_type: ChartType | None = None
    for key, value in chart_map.items():
        if key in lowered:
            new_chart_type = value
            break
    for visual in spec.visuals:
        if new_chart_type:
            visual.chart_type = new_chart_type
        updated_visuals.append(visual)
    spec.visuals = updated_visuals
    return spec


def _stringify_for_parquet(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return value
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _prepare_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    object_columns = result.select_dtypes(include=["object", "str"]).columns
    for column in object_columns:
        non_null = result[column].dropna()
        if non_null.empty:
            continue
        inferred = pd.api.types.infer_dtype(non_null, skipna=True)
        if inferred.startswith("mixed"):
            result[column] = result[column].map(_stringify_for_parquet)
    return result


def _write_tool_catalog_artifact(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = output_dir / f"tool_catalog_{artifact_version}.json"
    artifact_payload = {
        "schema_version": "1.0",
        "artifact_version": artifact_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tools": export_tool_catalog(),
    }
    artifact_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")
    return artifact_path


@click.command()
@click.option("--data", type=click.Path(path_type=Path), required=False, help="Path to input dataset.")
@click.option(
    "--description", type=click.Path(path_type=Path), required=False, help="Optional context file."
)
@click.option("--update", is_flag=True, help="Enable update mode.")
@click.option("--session", type=click.Path(path_type=Path), required=False, help="Session log to update.")
@click.option("--prompt", type=str, required=False, help="Update instructions.")
@click.option("--output-format", type=click.Choice(["server", "html"]), default="server")
@click.option("--port", type=int, default=8050)
@click.option("--log-level", type=click.Choice(["info", "debug", "error"]), default="info")
@click.option("--llm-api-key", type=str, required=False, help="LLM API key override for this run.")
@click.option("--dump-tool-catalog", is_flag=True, help="Write a versioned tool catalog artifact and exit.")
@click.option(
    "--catalog-output-dir",
    type=click.Path(path_type=Path),
    required=False,
    help="Output directory for tool catalog artifact (defaults to outputs/tool_catalogs).",
)
def main(
    data: Path | None,
    description: Path | None,
    update: bool,
    session: Path | None,
    prompt: str | None,
    output_format: str,
    port: int,
    log_level: str,
    llm_api_key: str | None,
    dump_tool_catalog: bool,
    catalog_output_dir: Path | None,
) -> None:
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=log_level.upper())

    root_dir = Path(__file__).resolve().parent
    config = AppConfig.default(root_dir)
    if llm_api_key:
        config = AppConfig(
            root_dir=config.root_dir,
            logs_dir=config.logs_dir,
            outputs_dir=config.outputs_dir,
            llm=LLMConfig(
                model=config.llm.model,
                analysis_temperature=config.llm.analysis_temperature,
                tool_temperature=config.llm.tool_temperature,
                api_key=llm_api_key,
            ),
            sample_rows=config.sample_rows,
        )
    session_logger = init_session_logger(config.logs_dir)
    session_logger.section("Input Configuration")
    session_logger.log_kv(
        {
            "data": str(data) if data else "",
            "description": str(description) if description else "",
            "update": update,
            "dump_tool_catalog": dump_tool_catalog,
            "output_format": output_format,
            "port": port,
        }
    )

    if dump_tool_catalog:
        target_dir = catalog_output_dir or (config.outputs_dir / "tool_catalogs")
        artifact_path = _write_tool_catalog_artifact(target_dir)
        session_logger.section("Tool Catalog Export")
        session_logger.log_kv({"tool_catalog_artifact": str(artifact_path)})
        click.echo(f"Tool catalog artifact written to: {artifact_path}")
        return

    if update:
        if session is None or prompt is None:
            raise click.ClickException("--session and --prompt are required in update mode.")
        session_logger.section("Phase 3: Update Mode")
        spec_payload = load_dashboard_spec(session)
        design = DashboardSpec(**spec_payload)
        updated_design = _apply_update_prompt(design, prompt)
        session_logger.log_json("Update Prompt", {"prompt": prompt})
        session_logger.log_json("Updated Dashboard Spec", updated_design.model_dump())

        data_path = data
        if data_path is None:
            metadata = load_session_metadata(session)
            candidate = metadata.get("transformed_dataset") or metadata.get("data")
            if candidate:
                data_path = Path(candidate)

        df = pd.DataFrame()
        if data_path and data_path.exists():
            df = _load_for_analysis(data_path, config.sample_rows)

        if output_format == "server" and df.empty:
            raise click.ClickException("Update mode needs a dataset for server output.")

        from src.dashboard.builder import build_dashboard
        from src.tools.visualization import build_error_app, create_figure, export_dashboard

        figures: list = []
        dashboard = None
        output_path = config.outputs_dir / f"dashboard_{session_logger.path.stem}.html"
        try:
            figures = [create_figure(df, spec) for spec in updated_design.visuals] if not df.empty else []
            dashboard = build_dashboard(df, updated_design) if not df.empty else None
            export_dashboard(
                output_format=output_format,
                output_path=output_path,
                title=updated_design.title,
                figures=figures,
                app=dashboard.app if dashboard else None,
                port=port,
            )
        except Exception as exc:
            if output_format in {"server", "dash"}:
                error_app = build_error_app(
                    title="Dashboard Update Failed",
                    message="The update could not be rendered. Review details below and retry.",
                    details=str(exc),
                )
                export_dashboard(
                    output_format=output_format,
                    output_path=output_path,
                    title="Dashboard Update Failed",
                    figures=[],
                    app=error_app,
                    port=port,
                )
            else:
                raise click.ClickException(f"Failed to generate updated dashboard: {exc}") from exc

        session_logger.log_dashboard_spec(updated_design.model_dump())
        session_logger.section("Execution Summary")
        session_logger.log_kv(
            {
                "status": "updated",
                "output_path": "server mode" if output_format in {"server", "dash"} else str(output_path),
            }
        )
        return

    if data is None:
        raise click.ClickException("--data is required in standard mode.")

    description_text = _read_description(description)
    sample_df = _load_for_analysis(data, config.sample_rows)

    analyzer = Analyzer(config)
    session_logger.section("Phase 1: Analysis")
    analysis = analyzer.run_analysis(sample_df, description_text, session_logger)
    session_logger.log_json("Analysis Report", analysis.model_dump())

    registry = build_registry()
    orchestrator = Orchestrator(config, registry)
    plan = orchestrator.plan_execution(analysis, data)

    session_logger.section("Phase 2: Orchestration")
    output_path = config.outputs_dir / f"dashboard_{session_logger.path.stem}.html"
    result = orchestrator.execute_plan(
        plan=plan,
        output_format=output_format,
        output_path=output_path,
        port=port,
        logger_ctx=session_logger,
    )

    session_logger.section("Phase 3: Output Generation")
    if result.dataframe is not None and result.transformations_applied:
        transformed_path = config.outputs_dir / f"transformed_{session_logger.path.stem}.parquet"
        safe_dataframe = _prepare_dataframe_for_parquet(result.dataframe)
        safe_dataframe.to_parquet(transformed_path, index=False)
        session_logger.log_kv({"transformed_dataset": str(transformed_path)})

    if result.dashboard is not None:
        session_logger.log_dashboard_spec(result.dashboard.spec.model_dump())

    session_logger.log_kv(
        {
            "output_path": str(result.output_path) if result.output_path else "server mode",
            "transformations": ", ".join(result.transformations_applied) or "none",
        }
    )


if __name__ == "__main__":
    main()

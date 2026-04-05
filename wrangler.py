from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from src.agents.orchestrator import export_tool_catalog
from src.config import AppConfig, LLMConfig
from src.services import ApplicationService


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
@click.option("--review-only", is_flag=True, help="Write the proposed plan/session state and exit before execution.")
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
    review_only: bool,
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
    service = ApplicationService(root_dir=root_dir, llm_api_key=llm_api_key)

    if dump_tool_catalog:
        target_dir = catalog_output_dir or (config.outputs_dir / "tool_catalogs")
        artifact_path = _write_tool_catalog_artifact(target_dir)
        click.echo(f"Tool catalog artifact written to: {artifact_path}")
        return

    if update:
        if session is None or prompt is None:
            raise click.ClickException("--session and --prompt are required in update mode.")
        service.update_from_log_path(
            session_log=session,
            prompt=prompt,
            data_path=data,
            output_format=output_format,
            port=port,
        )
        return

    if data is None:
        raise click.ClickException("--data is required in standard mode.")

    result = service.generate_from_path(
        data_path=data,
        description_path=description,
        output_format=output_format,
        port=port,
        review_only=review_only,
    )

    if review_only:
        click.echo(f"Plan proposal written to: {result.session_log}")
        return


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.io as pio

from src.agents.analyzer import Analyzer
from src.agents.navigator import NavigatorAgent
from src.agents.orchestrator import Orchestrator, build_registry
from src.agents.patcher import apply_dashboard_patch, parse_update_prompt
from src.config import AppConfig, LLMConfig
from src.dashboard.builder import build_dashboard
from src.logging.session import SessionLogger, init_session_logger, load_dashboard_spec, load_session_metadata, load_session_state
from src.models import AnalysisReport, DashboardSpec, ExecutionTrace, SessionState
import src.services.artifacts as artifacts
from src.tools import loaders
from src.tools.visualization import create_figure, error_figure, export_dashboard


@dataclass
class AnalyzeResult:
    session_id: str
    analysis: AnalysisReport
    dashboard_spec: DashboardSpec
    artifacts: list[dict[str, str]]


@dataclass
class SessionSummary:
    session_id: str
    status: str
    title: str
    created_at: str
    updated_at: str


@dataclass
class SessionDetail:
    session_id: str
    status: str
    analysis: dict[str, Any] | None
    dashboard_spec: dict[str, Any]
    figures: list[dict[str, Any]]
    artifacts: list[dict[str, str]]


@dataclass
class GenerateResult:
    session_id: str
    analysis: AnalysisReport
    dashboard_spec: DashboardSpec
    figures: list[dict[str, Any]]
    session_status: str
    artifacts: list[dict[str, str]]


@dataclass
class UpdateResult:
    session_id: str
    dashboard_spec: DashboardSpec
    figures: list[dict[str, Any]]
    session_status: str
    artifacts: list[dict[str, str]]


@dataclass
class CliRunResult:
    session_id: str
    session_log: Path
    session_state: SessionState
    analysis: AnalysisReport | None
    figures: list[dict[str, Any]]
    rendered_output: Path | None


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


class ApplicationService:
    def __init__(self, root_dir: Path, llm_api_key: str | None = None):
        base_config = AppConfig.default(root_dir)
        if llm_api_key:
            self.config = AppConfig(
                root_dir=base_config.root_dir,
                logs_dir=base_config.logs_dir,
                outputs_dir=base_config.outputs_dir,
                llm=LLMConfig(
                    model=base_config.llm.model,
                    analysis_temperature=base_config.llm.analysis_temperature,
                    tool_temperature=base_config.llm.tool_temperature,
                    api_key=llm_api_key,
                ),
                sample_rows=base_config.sample_rows,
            )
        else:
            self.config = base_config

    def analyze_uploaded_dataset(
        self,
        *,
        filename: str,
        content: bytes,
        context_text: str | None = None,
    ) -> AnalyzeResult:
        session_logger = init_session_logger(self.config.logs_dir)
        session_id = session_logger.path.stem
        source_path = self._write_source_artifact(session_id, filename, content)
        context_path = self._write_context_artifact(session_id, context_text)
        description_text = _read_description(context_path)

        session_logger.section("Input Configuration")
        session_logger.log_kv(
            {
                "data": str(source_path),
                "description": str(context_path) if context_path else "",
                "api_mode": "analyze",
            }
        )

        sample_df = _load_for_analysis(source_path, self.config.sample_rows)
        analyzer = Analyzer(self.config)
        session_logger.section("Phase 1: Analysis")
        analysis = analyzer.run_analysis(sample_df, description_text, session_logger)
        session_logger.log_json("Analysis Report", analysis.model_dump())

        session_state = SessionState(
            session_id=session_id,
            status="planned",
            data_path=str(source_path),
            description_path=str(context_path) if context_path else "",
            analysis=analysis,
            trace=ExecutionTrace(session_id=session_id, current_stage="approve_edit", status="planned"),
            active_spec=analysis.design.model_copy(deep=True),
            spec_versions=[analysis.design.model_copy(deep=True)],
            decisions=["Analysis-only API request generated a draft dashboard spec."],
        )
        session_logger.log_dashboard_spec(analysis.design.model_dump())
        session_logger.log_session_state(session_state)
        session_logger.log_execution_trace(session_state.trace)
        self._write_json_artifact(artifacts.dashboard_spec_path(self.config, session_id), analysis.design.model_dump())

        return AnalyzeResult(
            session_id=session_id,
            analysis=analysis,
            dashboard_spec=analysis.design,
            artifacts=self.list_artifacts(session_id, state=session_state),
        )

    def generate_uploaded_dataset(
        self,
        *,
        filename: str,
        content: bytes,
        context_text: str | None = None,
    ) -> GenerateResult:
        session_logger = init_session_logger(self.config.logs_dir)
        session_id = session_logger.path.stem
        source_path = self._write_source_artifact(session_id, filename, content)
        context_path = self._write_context_artifact(session_id, context_text)
        description_text = _read_description(context_path)

        session_logger.section("Input Configuration")
        session_logger.log_kv(
            {
                "data": str(source_path),
                "description": str(context_path) if context_path else "",
                "api_mode": "generate",
                "output_format": "json",
            }
        )

        sample_df = _load_for_analysis(source_path, self.config.sample_rows)
        analyzer = Analyzer(self.config)
        session_logger.section("Phase 1: Analysis")
        analysis = analyzer.run_analysis(sample_df, description_text, session_logger)
        session_logger.log_json("Analysis Report", analysis.model_dump())

        registry = build_registry()
        orchestrator = Orchestrator(self.config, registry)
        navigator = NavigatorAgent(orchestrator)
        proposal_result = navigator.propose(
            analysis=analysis,
            data_path=source_path,
            description_path=context_path,
            user_goal=source_path.name,
            session_id=session_id,
        )
        session_state = navigator.approve(proposal_result.session_state, reason="Auto-approved by API generation.")

        session_logger.section("Phase 2: Orchestration")
        execution_result = orchestrator.execute_plan(
            plan=proposal_result.plan,
            output_format="html",
            output_path=self.config.outputs_dir / f"dashboard_{session_id}.html",
            port=8050,
            logger_ctx=session_logger,
            trace=session_state.trace,
            defer_export=True,
        )
        session_state = navigator.review_execution(session_state, execution_result)

        review_dataframe = execution_result.baseline_dataframe if execution_result.baseline_dataframe is not None else execution_result.dataframe
        figures = self._render_figures(review_dataframe, session_state)

        if execution_result.dataframe is not None and execution_result.transformations_applied:
            transformed_path = artifacts.transformed_dataset_path(self.config, session_id)
            safe_dataframe = _prepare_dataframe_for_parquet(execution_result.dataframe)
            safe_dataframe.to_parquet(transformed_path, index=False)
            session_state.transformed_dataset = str(transformed_path)
            session_logger.log_kv({"transformed_dataset": str(transformed_path)})

        session_state.output_path = ""
        session_logger.section("Phase 3: Output Generation")
        session_logger.log_kv(
            {
                "output_path": "api-json",
                "transformations": ", ".join(execution_result.transformations_applied) or "none",
                "guardrail_warnings": ", ".join(execution_result.warnings) or "none",
                "status": session_state.status,
            }
        )
        self._persist_session_artifacts(session_logger, session_state, figures=figures)

        return GenerateResult(
            session_id=session_id,
            analysis=analysis,
            dashboard_spec=session_state.active_spec,
            figures=figures,
            session_status=session_state.status,
            artifacts=self.list_artifacts(session_id, state=session_state),
        )

    def generate_from_path(
        self,
        *,
        data_path: Path,
        description_path: Path | None = None,
        output_format: str = "json",
        port: int = 8050,
        review_only: bool = False,
    ) -> CliRunResult:
        session_logger = init_session_logger(self.config.logs_dir)
        session_id = session_logger.path.stem
        description_text = _read_description(description_path)

        session_logger.section("Input Configuration")
        session_logger.log_kv(
            {
                "data": str(data_path),
                "description": str(description_path) if description_path else "",
                "review_only": review_only,
                "output_format": output_format,
                "port": port,
            }
        )

        sample_df = _load_for_analysis(data_path, self.config.sample_rows)
        analyzer = Analyzer(self.config)
        session_logger.section("Phase 1: Analysis")
        analysis = analyzer.run_analysis(sample_df, description_text, session_logger)
        session_logger.log_json("Analysis Report", analysis.model_dump())

        registry = build_registry()
        orchestrator = Orchestrator(self.config, registry)
        navigator = NavigatorAgent(orchestrator)
        proposal_result = navigator.propose(
            analysis=analysis,
            data_path=data_path,
            description_path=description_path,
            user_goal=data_path.name,
            session_id=session_id,
        )
        session_logger.log_json("Plan Proposal", proposal_result.proposal.model_dump())
        session_state = proposal_result.session_state

        if review_only:
            session_state.status = "planned"
            session_state.trace.current_stage = "approve_edit"
            session_logger.section("Plan Review")
            session_logger.log_kv({"status": "review_only", "next_step": "Run without --review-only to execute the approved plan."})
            self._persist_session_artifacts(session_logger, session_state)
            return CliRunResult(
                session_id=session_id,
                session_log=session_logger.path,
                session_state=session_state,
                analysis=analysis,
                figures=[],
                rendered_output=None,
            )

        session_state = navigator.approve(session_state)
        session_logger.section("Phase 2: Orchestration")
        execution_result = orchestrator.execute_plan(
            plan=proposal_result.plan,
            output_format="html",
            output_path=self.config.outputs_dir / f"dashboard_{session_id}.html",
            port=port,
            logger_ctx=session_logger,
            trace=session_state.trace,
            defer_export=True,
        )
        session_state = navigator.review_execution(session_state, execution_result)

        review_dataframe = execution_result.baseline_dataframe if execution_result.baseline_dataframe is not None else execution_result.dataframe
        figure_objects = self._build_figures(review_dataframe, session_state)
        figures = self._serialize_figures(figure_objects)

        if execution_result.dataframe is not None and execution_result.transformations_applied:
            transformed_path = artifacts.transformed_dataset_path(self.config, session_id)
            safe_dataframe = _prepare_dataframe_for_parquet(execution_result.dataframe)
            safe_dataframe.to_parquet(transformed_path, index=False)
            session_state.transformed_dataset = str(transformed_path)
            session_logger.log_kv({"transformed_dataset": str(transformed_path)})

        rendered_output = None
        if output_format in {"html", "server", "dash"} and review_dataframe is not None:
            rendered_output = self._render_dashboard_output(
                df=review_dataframe,
                session_state=session_state,
                figures=figure_objects,
                output_format=output_format,
                output_path=self.config.outputs_dir / f"dashboard_{session_id}.html",
                port=port,
            )

        session_state.output_path = "" if rendered_output is None else str(rendered_output)
        session_logger.section("Phase 3: Output Generation")
        session_logger.log_kv(
            {
                "output_path": str(rendered_output) if rendered_output else "server mode" if output_format in {"server", "dash"} else "api-json",
                "transformations": ", ".join(execution_result.transformations_applied) or "none",
                "guardrail_warnings": ", ".join(execution_result.warnings) or "none",
            }
        )
        self._persist_session_artifacts(session_logger, session_state, figures=figures)

        return CliRunResult(
            session_id=session_id,
            session_log=session_logger.path,
            session_state=session_state,
            analysis=analysis,
            figures=figures,
            rendered_output=rendered_output,
        )

    def update_session(self, *, session_id: str, prompt: str) -> UpdateResult:
        cli_result = self.update_from_log_path(
            session_log=artifacts.log_path(self.config, session_id),
            prompt=prompt,
            output_format="json",
        )

        return UpdateResult(
            session_id=session_id,
            dashboard_spec=cli_result.session_state.active_spec,
            figures=cli_result.figures,
            session_status=cli_result.session_state.status,
            artifacts=self.list_artifacts(session_id, state=cli_result.session_state),
        )

    def update_from_log_path(
        self,
        *,
        session_log: Path,
        prompt: str,
        data_path: Path | None = None,
        output_format: str = "json",
        port: int = 8050,
    ) -> CliRunResult:
        if not session_log.exists():
            raise FileNotFoundError(session_log)

        session_logger = SessionLogger(path=session_log)
        session_state = load_session_state(session_log)
        session_id = session_log.stem

        session_logger.section("Phase 3: Update Mode")
        patch = parse_update_prompt(prompt, session_state.active_spec.model_copy(deep=True))
        updated_spec = apply_dashboard_patch(session_state.active_spec.model_copy(deep=True), patch)
        session_state.active_spec = updated_spec
        session_state.spec_versions.append(updated_spec.model_copy(deep=True))
        session_state.pending_patch = patch
        session_state.decisions.append(f"Update prompt applied: {prompt}")

        resolved_data_path = data_path or self._resolve_update_data_path(session_state)
        df = _load_for_analysis(resolved_data_path, self.config.sample_rows)
        figure_objects = self._build_figures(df, session_state)
        figures = self._serialize_figures(figure_objects)

        rendered_output = None
        if output_format in {"html", "server", "dash"}:
            rendered_output = self._render_dashboard_output(
                df=df,
                session_state=session_state,
                figures=figure_objects,
                output_format=output_format,
                output_path=self.config.outputs_dir / f"dashboard_{session_id}.html",
                port=port,
            )
        session_state.output_path = "" if rendered_output is None else str(rendered_output)

        session_logger.log_json("Update Prompt", {"prompt": prompt})
        session_logger.log_json("Dashboard Patch", patch.model_dump())
        session_logger.log_json("Updated Dashboard Spec", updated_spec.model_dump())
        self._persist_session_artifacts(session_logger, session_state, figures=figures)

        return CliRunResult(
            session_id=session_id,
            session_log=session_log,
            session_state=session_state,
            analysis=session_state.analysis,
            figures=figures,
            rendered_output=rendered_output,
        )

    def list_sessions(self) -> list[SessionSummary]:
        state_files = sorted(self.config.logs_dir.glob("session_*.state.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        seen: set[str] = set()
        summaries: list[SessionSummary] = []

        for state_file in state_files:
            session_id = state_file.name.removesuffix(".state.json")
            state = load_session_state(self.config.logs_dir / f"{session_id}.log")
            summaries.append(
                SessionSummary(
                    session_id=session_id,
                    status=state.status,
                    title=state.active_spec.title,
                    created_at=state.created_at,
                    updated_at=state.updated_at,
                )
            )
            seen.add(session_id)

        log_files = sorted(self.config.logs_dir.glob("session_*.log"), key=lambda item: item.stat().st_mtime, reverse=True)
        for log_file in log_files:
            session_id = log_file.stem
            if session_id in seen:
                continue
            metadata = load_session_metadata(log_file)
            try:
                title = load_dashboard_spec(log_file).get("title", "Zenith Wrangler Dashboard")
            except Exception:
                title = "Zenith Wrangler Dashboard"
            summaries.append(
                SessionSummary(
                    session_id=session_id,
                    status=metadata.get("status", "unknown"),
                    title=title,
                    created_at=str(log_file.stat().st_ctime),
                    updated_at=str(log_file.stat().st_mtime),
                )
            )
        return summaries

    def get_session_detail(self, session_id: str) -> SessionDetail:
        state = self._load_state(session_id)
        detail_spec = self._load_dashboard_spec_artifact(session_id)
        figures = self._load_figures_artifact(session_id)
        return SessionDetail(
            session_id=session_id,
            status=state.status if state else "unknown",
            analysis=state.analysis.model_dump() if state and state.analysis else None,
            dashboard_spec=detail_spec,
            figures=figures,
            artifacts=self.list_artifacts(session_id, state=state),
        )

    def resolve_artifact(self, session_id: str, artifact_type: artifacts.ArtifactType) -> Path | None:
        state = self._load_state(session_id)
        return artifacts.resolve_artifact_path(self.config, session_id, artifact_type, state=state)

    def list_artifacts(self, session_id: str, state: SessionState | None = None) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for artifact_type in artifacts.ARTIFACT_CONTENT_TYPES:
            path = artifacts.resolve_artifact_path(self.config, session_id, artifact_type, state=state)
            if path is None:
                continue
            items.append(
                {
                    "type": artifact_type,
                    "path": str(path),
                    "url": f"/artifacts/{session_id}/{artifact_type}",
                    "content_type": artifacts.ARTIFACT_CONTENT_TYPES[artifact_type],
                }
            )
        return items

    def _write_source_artifact(self, session_id: str, filename: str, content: bytes) -> Path:
        suffix = Path(filename).suffix or ".csv"
        path = artifacts.source_path(self.config, session_id, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _write_context_artifact(self, session_id: str, context_text: str | None) -> Path | None:
        if not context_text:
            return None
        path = artifacts.context_path(self.config, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(context_text, encoding="utf-8")
        return path

    def _write_json_artifact(self, path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _persist_session_artifacts(
        self,
        session_logger: SessionLogger,
        session_state: SessionState,
        *,
        figures: list[dict[str, Any]] | None = None,
    ) -> None:
        session_logger.log_dashboard_spec(session_state.active_spec.model_dump())
        session_logger.log_session_state(session_state)
        session_logger.log_execution_trace(session_state.trace)
        self._write_json_artifact(
            artifacts.dashboard_spec_path(self.config, session_state.session_id),
            session_state.active_spec.model_dump(),
        )
        if figures is not None:
            self._write_json_artifact(artifacts.figures_path(self.config, session_state.session_id), figures)

    def _load_state(self, session_id: str) -> SessionState | None:
        log_path = artifacts.log_path(self.config, session_id)
        if not log_path.exists():
            return None
        try:
            return load_session_state(log_path)
        except FileNotFoundError:
            return None

    def _load_dashboard_spec_artifact(self, session_id: str) -> dict[str, Any]:
        artifact_path = artifacts.dashboard_spec_path(self.config, session_id)
        if artifact_path.exists():
            return json.loads(artifact_path.read_text(encoding="utf-8"))
        log_path = artifacts.log_path(self.config, session_id)
        if not log_path.exists():
            raise FileNotFoundError(session_id)
        return load_dashboard_spec(log_path)

    def _load_figures_artifact(self, session_id: str) -> list[dict[str, Any]]:
        artifact_path = artifacts.figures_path(self.config, session_id)
        if not artifact_path.exists():
            return []
        return json.loads(artifact_path.read_text(encoding="utf-8"))

    def _render_figures(self, df: pd.DataFrame | None, session_state: SessionState) -> list[dict[str, Any]]:
        return self._serialize_figures(self._build_figures(df, session_state))

    def _build_figures(self, df: pd.DataFrame | None, session_state: SessionState) -> list[Any]:
        if df is None:
            return []
        figures: list[Any] = []
        for spec in session_state.active_spec.visuals:
            try:
                figure = create_figure(df, spec, theme=session_state.active_spec.theme)
            except Exception as exc:
                figure = error_figure(spec.title, f"Unable to render chart: {exc}", theme=session_state.active_spec.theme)
            figures.append(figure)
        return figures

    def _serialize_figures(self, figures: list[Any]) -> list[dict[str, Any]]:
        return [serialize_figure(figure) for figure in figures]

    def _render_dashboard_output(
        self,
        *,
        df: pd.DataFrame,
        session_state: SessionState,
        figures: list[Any],
        output_format: str,
        output_path: Path,
        port: int,
    ) -> Path | None:
        dashboard = build_dashboard(df, session_state.active_spec, session_state=session_state)
        return export_dashboard(
            output_format=output_format,
            output_path=output_path,
            title=session_state.active_spec.title,
            figures=figures,
            app=dashboard.app,
            port=port,
        )

    def _resolve_update_data_path(self, session_state: SessionState) -> Path:
        if session_state.data_path and Path(session_state.data_path).exists():
            return Path(session_state.data_path)
        if session_state.transformed_dataset and Path(session_state.transformed_dataset).exists():
            return Path(session_state.transformed_dataset)
        raise FileNotFoundError(f"No dataset artifact available for session '{session_state.session_id}'.")


def serialize_figure(figure: Any) -> dict[str, Any]:
    return json.loads(pio.to_json(figure))

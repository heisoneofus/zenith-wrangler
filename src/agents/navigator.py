from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.agents.orchestrator import ExecutionResult, Orchestrator, ToolCall
from src.models import (
    AnalysisReport,
    DashboardPatch,
    DashboardSpec,
    ExecutionTrace,
    PlanProposal,
    SessionState,
    ToolEvent,
    VisualSpec,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quality_notes(analysis: AnalysisReport) -> list[str]:
    notes: list[str] = []
    for issue in analysis.quality.issues:
        columns = ", ".join(issue.columns) if issue.columns else "dataset-wide"
        notes.append(f"{issue.type} ({issue.severity}) on {columns}")
    return notes or ["No material quality issues detected during profiling."]


def _profile_notes(analysis: AnalysisReport) -> list[str]:
    notes = [
        f"Primary metrics: {', '.join(analysis.metrics.primary_metrics) or 'none'}",
        f"Dimensions: {', '.join(analysis.metrics.dimensions) or 'none'}",
        f"Time fields: {', '.join(analysis.metrics.time_fields) or 'none'}",
        f"Sampled rows: {analysis.sampled_rows}",
    ]
    if analysis.metrics.notes:
        notes.append(analysis.metrics.notes)
    return notes


def _proposal_mode(analysis: AnalysisReport) -> str:
    return "heuristic" if "Heuristic" in analysis.design.notes else "llm"


def _provenance_visuals(analysis: AnalysisReport) -> list[VisualSpec]:
    visuals: list[VisualSpec] = []
    metric_summary = ", ".join(analysis.metrics.primary_metrics or analysis.metrics.secondary_metrics) or "available metrics"
    transform_summary = ", ".join(analysis.quality.suggested_operations) or "no transforms"
    for index, visual in enumerate(analysis.design.visuals, start=1):
        rationale = (
            f"Visual {index} prioritizes {visual.chart_type} analysis for {metric_summary}. "
            f"Recommended transform path: {transform_summary}."
        )
        visuals.append(
            visual.model_copy(
                update={
                    "source_dataframe_ref": visual.source_dataframe_ref or "baseline",
                    "rationale": visual.rationale or rationale,
                    "warnings": list(visual.warnings),
                    "confidence": visual.confidence if visual.confidence is not None else 0.72,
                    "status": "proposed",
                }
            )
        )
    return visuals


class Planner:
    def propose_plan(self, analysis: AnalysisReport, user_goal: str, session_id: str) -> PlanProposal:
        design = analysis.design.model_copy(
            update={
                "visuals": _provenance_visuals(analysis),
                "plan_summary": (
                    "Profile the dataset, review transform recommendations, approve the proposed dashboard, "
                    "then execute with a post-render critic pass."
                ),
                "data_quality_summary": _quality_notes(analysis),
                "transform_history": list(analysis.quality.suggested_operations),
                "assumptions": [
                    "Non-interactive CLI runs auto-approve the current plan unless `--review-only` is used.",
                    "Destructive cleaning steps may be skipped when they remove too many rows.",
                ],
                "approval_status": "draft",
            }
        )
        return PlanProposal(
            session_id=session_id,
            mode=_proposal_mode(analysis),  # type: ignore[arg-type]
            user_goal=user_goal,
            summary=(
                f"Generate an analyst-ready dashboard for {Path(user_goal).name if user_goal else 'the dataset'} "
                "with explicit reasoning, guarded transforms, and revisable visuals."
            ),
            profile_notes=_profile_notes(analysis),
            data_quality_notes=_quality_notes(analysis),
            transform_plan=list(analysis.quality.suggested_operations),
            review_notes=[
                "Inspect transform warnings before trusting rolled-up metrics.",
                "Revise one visual at a time through dashboard patches when possible.",
            ],
            approval_required=True,
            approved=False,
            design=design,
        )


class Critic:
    def review(self, design: DashboardSpec, figures: list, trace: ExecutionTrace) -> DashboardSpec | None:
        repaired_visuals: list[VisualSpec] = []
        needs_repair = False
        for index, visual in enumerate(design.visuals):
            current = visual.model_copy(deep=True)
            figure = figures[index] if index < len(figures) else None
            annotation_text = ""
            if figure is not None and getattr(figure.layout, "annotations", None):
                annotation = figure.layout.annotations[0]
                annotation_text = str(getattr(annotation, "text", annotation))
            if annotation_text and ("Unable" in annotation_text or "No data" in annotation_text or "missing" in annotation_text.lower()):
                needs_repair = True
                fallback_chart = "bar" if current.chart_type != "bar" else "histogram"
                repaired_warning = f"Critic replaced failing `{current.chart_type}` chart with `{fallback_chart}` after render issue: {annotation_text}"
                current.chart_type = fallback_chart  # type: ignore[assignment]
                if fallback_chart == "histogram" and current.y and not current.x:
                    current.x = current.y
                    current.y = None
                    current.aggregation = None
                current.warnings.append(repaired_warning)
                current.rationale = current.rationale or repaired_warning
                current.status = "revised"
                trace.repair_notes.append(repaired_warning)
                trace.events.append(
                    ToolEvent(
                        event_type="repair",
                        tool_name="critic",
                        status="repaired",
                        message=repaired_warning,
                    )
                )
            repaired_visuals.append(current)
        if not needs_repair:
            return None
        updated = design.model_copy(update={"visuals": repaired_visuals, "approval_status": "reviewed"})
        return updated


@dataclass
class NavigatorRunResult:
    plan: list[ToolCall]
    proposal: PlanProposal
    session_state: SessionState
    result: ExecutionResult | None = None


class NavigatorAgent:
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.planner = Planner()
        self.critic = Critic()

    def create_session_state(
        self,
        *,
        session_id: str,
        data_path: Path,
        description_path: Path | None,
        analysis: AnalysisReport,
        proposal: PlanProposal,
        user_goal: str,
    ) -> SessionState:
        trace = ExecutionTrace(
            session_id=session_id,
            current_stage="approve_edit",
            status="planned",
            approvals=[],
        )
        return SessionState(
            session_id=session_id,
            status="planned",
            data_path=str(data_path),
            description_path=str(description_path) if description_path else "",
            user_goal=user_goal,
            analysis=analysis,
            plan=proposal,
            trace=trace,
            active_spec=proposal.design.model_copy(deep=True),
            spec_versions=[proposal.design.model_copy(deep=True)],
            decisions=[
                "Planner generated an inspectable proposal before execution.",
            ],
        )

    def propose(
        self,
        *,
        analysis: AnalysisReport,
        data_path: Path,
        description_path: Path | None,
        user_goal: str,
        session_id: str,
    ) -> NavigatorRunResult:
        proposal = self.planner.propose_plan(analysis, user_goal=user_goal or str(data_path), session_id=session_id)
        plan = self.orchestrator.plan_execution(analysis, data_path)
        session_state = self.create_session_state(
            session_id=session_id,
            data_path=data_path,
            description_path=description_path,
            analysis=analysis,
            proposal=proposal,
            user_goal=user_goal or str(data_path),
        )
        return NavigatorRunResult(plan=plan, proposal=proposal, session_state=session_state)

    def approve(self, session_state: SessionState, reason: str = "Auto-approved by CLI execution.") -> SessionState:
        session_state.status = "approved"
        session_state.updated_at = _utc_now()
        if session_state.plan is not None:
            session_state.plan.approved = True
            session_state.plan.design.approval_status = "approved"
            session_state.active_spec = session_state.plan.design.model_copy(deep=True)
        session_state.trace.current_stage = "execute_review"
        session_state.trace.status = "approved"
        session_state.trace.approvals.append(reason)
        session_state.trace.events.append(
            ToolEvent(event_type="approval", tool_name="planner", status="approved", message=reason)
        )
        session_state.decisions.append(reason)
        return session_state

    def apply_patch(self, session_state: SessionState, patch: DashboardPatch, updated_spec: DashboardSpec) -> SessionState:
        session_state.pending_patch = patch
        session_state.active_spec = updated_spec
        session_state.spec_versions.append(updated_spec.model_copy(deep=True))
        session_state.updated_at = _utc_now()
        session_state.decisions.append(f"Applied dashboard patch with {len(patch.operations)} operation(s).")
        if session_state.plan is not None:
            session_state.plan.design = updated_spec.model_copy(deep=True)
        return session_state

    def review_execution(
        self,
        session_state: SessionState,
        execution_result: ExecutionResult,
    ) -> SessionState:
        session_state.trace.current_stage = "execute_review"
        session_state.trace.status = "executed"
        session_state.warnings.extend(execution_result.warnings)
        if execution_result.dashboard is not None:
            session_state.active_spec = execution_result.dashboard.spec.model_copy(deep=True)
        updated_spec = self.critic.review(session_state.active_spec, execution_result.figures, session_state.trace)
        if updated_spec is not None:
            session_state.active_spec = updated_spec
            session_state.spec_versions.append(updated_spec.model_copy(deep=True))
            session_state.trace.status = "repaired"
            session_state.status = "reviewed"
            session_state.decisions.append("Critic applied one bounded repair pass after execution.")
        else:
            session_state.status = "reviewed"
            session_state.decisions.append("Critic accepted the first render without repairs.")
        session_state.updated_at = _utc_now()
        return session_state


def summarize_dataframe(df: pd.DataFrame) -> list[str]:
    return [
        f"Rows: {len(df):,}",
        f"Columns: {len(df.columns):,}",
        f"Numeric fields: {len(df.select_dtypes(include='number').columns):,}",
    ]

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ArtifactDescriptor(BaseModel):
    type: str
    path: str
    url: str
    content_type: str


class AnalyzeResponse(BaseModel):
    session_id: str
    analysis: dict[str, Any]
    dashboard_spec: dict[str, Any]
    artifacts: list[ArtifactDescriptor] = Field(default_factory=list)


class GenerateResponse(AnalyzeResponse):
    figures: list[dict[str, Any]] = Field(default_factory=list)
    session_status: str


class UpdateRequest(BaseModel):
    session_id: str
    prompt: str


class UpdateResponse(BaseModel):
    session_id: str
    dashboard_spec: dict[str, Any]
    figures: list[dict[str, Any]] = Field(default_factory=list)
    session_status: str
    artifacts: list[ArtifactDescriptor] = Field(default_factory=list)


class SessionSummaryResponse(BaseModel):
    session_id: str
    status: str
    title: str
    created_at: str
    updated_at: str


class SessionsListResponse(BaseModel):
    items: list[SessionSummaryResponse] = Field(default_factory=list)


class SessionDetailResponse(BaseModel):
    session_id: str
    status: str
    analysis: dict[str, Any] | None = None
    dashboard_spec: dict[str, Any]
    figures: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[ArtifactDescriptor] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: str | None = None

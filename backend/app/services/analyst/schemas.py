from typing import Any
from pydantic import BaseModel, Field


class AnalystStatusResponse(BaseModel):
    provider: str = "ollama"
    configured_model: str
    ollama_available: bool
    model_available: bool
    status: str  # "ready" | "ollama_offline" | "model_missing" | "error"
    base_url: str
    available_models: list[str] = Field(default_factory=list)
    error: str | None = None
    instructions: str | None = None


class AnalystModelListResponse(BaseModel):
    provider: str = "ollama"
    models: list[str] = Field(default_factory=list)
    configured_model: str
    available: bool


class AnalystExplainRequest(BaseModel):
    question: str | None = None
    stream: bool = False


class AnalystChatRequest(BaseModel):
    message: str
    stream: bool = False
    history_limit: int = 10


class AnalystChatResponse(BaseModel):
    job_id: str | None = None
    role: str = "assistant"
    message: str
    context_summary: dict[str, Any] = Field(default_factory=dict)
    provider: str = "ollama"
    model: str
    timestamp: str


class AnalystBatchSummaryRequest(BaseModel):
    question: str | None = None
    stream: bool = False

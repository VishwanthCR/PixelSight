from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, AsyncGenerator

try:
    from backend.app.config import (
        PIXELSIGHT_LLM_PROVIDER,
        PIXELSIGHT_OLLAMA_BASE_URL,
        PIXELSIGHT_OLLAMA_MODEL,
        PIXELSIGHT_OLLAMA_TIMEOUT,
        RESULTS_ROOT,
    )
    from backend.app.services.analyst.context_builder import ContextBuilder
    from backend.app.services.analyst.guards import enforce_scientific_terminology, sanitize_user_input
    from backend.app.services.analyst.ollama_client import OllamaClient
    from backend.app.services.analyst.prompt_builder import (
        build_context_message,
        build_explain_prompt,
        build_system_message,
    )
    from backend.app.services.analyst.schemas import (
        AnalystChatResponse,
        AnalystModelListResponse,
        AnalystStatusResponse,
    )
except ImportError:
    from app.config import (
        PIXELSIGHT_LLM_PROVIDER,
        PIXELSIGHT_OLLAMA_BASE_URL,
        PIXELSIGHT_OLLAMA_MODEL,
        PIXELSIGHT_OLLAMA_TIMEOUT,
        RESULTS_ROOT,
    )
    from app.services.analyst.context_builder import ContextBuilder
    from app.services.analyst.guards import enforce_scientific_terminology, sanitize_user_input
    from app.services.analyst.ollama_client import OllamaClient
    from app.services.analyst.prompt_builder import (
        build_context_message,
        build_explain_prompt,
        build_system_message,
    )
    from app.services.analyst.schemas import (
        AnalystChatResponse,
        AnalystModelListResponse,
        AnalystStatusResponse,
    )



class PixelSightAnalystService:
    """
    Core orchestrator for PixelSight Analyst.
    Integrates local Ollama LLM as an interpretation and explanation layer on top of
    PixelSight completed job outputs.
    """

    def __init__(
        self,
        base_url: str = PIXELSIGHT_OLLAMA_BASE_URL,
        model: str = PIXELSIGHT_OLLAMA_MODEL,
        timeout: float = PIXELSIGHT_OLLAMA_TIMEOUT,
        provider: str = PIXELSIGHT_LLM_PROVIDER,
        results_root: Path = RESULTS_ROOT,
    ):
        self.provider = provider
        self.configured_model = model
        self.base_url = base_url
        self.timeout = timeout
        self.results_root = results_root
        self.client = OllamaClient(base_url=self.base_url, timeout=self.timeout)
        # Bounded conversation memory keyed by session/job_id: {session_id: [{"role": ..., "content": ...}]}
        self._conversations: dict[str, list[dict[str, str]]] = {}
        self._max_history_turns = 10

    async def get_status(self) -> AnalystStatusResponse:
        """Query Ollama daemon status, installed models, and readiness."""
        info = await self.client.get_status(self.configured_model)
        return AnalystStatusResponse(
            provider=self.provider,
            configured_model=self.configured_model,
            ollama_available=info["ollama_available"],
            model_available=info["model_available"],
            status=info["status"],
            base_url=self.base_url,
            available_models=info["available_models"],
            error=info["error"],
            instructions=info["instructions"],
        )

    async def list_models(self) -> AnalystModelListResponse:
        """List local models available in Ollama."""
        info = await self.client.get_status(self.configured_model)
        return AnalystModelListResponse(
            provider=self.provider,
            models=info["available_models"],
            configured_model=self.configured_model,
            available=info["ollama_available"],
        )

    def set_model(self, model_name: str) -> None:
        """Allow runtime selection of local model."""
        if model_name and isinstance(model_name, str):
            self.configured_model = model_name.strip()

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a job or batch session."""
        if session_id in self._conversations:
            del self._conversations[session_id]

    def _load_report(self, job_id: str) -> dict[str, Any] | None:
        """Load report.json from disk if present."""
        report_file = self.results_root / job_id / "report" / "report.json"
        if report_file.exists():
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    async def explain_job(
        self,
        job_id: str,
        job_data: dict[str, Any],
        question: str | None = None,
        stream: bool = False,
    ) -> AnalystChatResponse | AsyncGenerator[str, None]:
        """
        Explain a completed PixelSight job.
        Backend builds the scientific context automatically.
        """
        report_data = self._load_report(job_id) or job_data.get("report")
        context = ContextBuilder.build_job_context(job_data, report_data)
        application = context["job"]["application"]

        user_query = question or build_explain_prompt(application, context)
        sanitized_query, is_safe, refusal = sanitize_user_input(user_query)
        if not is_safe:
            return AnalystChatResponse(
                job_id=job_id,
                role="assistant",
                message=refusal or "Request was rejected by safety policy.",
                context_summary=context["job"],
                provider=self.provider,
                model=self.configured_model,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        messages = [
            build_system_message(),
            build_context_message(context),
            {"role": "user", "content": sanitized_query},
        ]

        if stream:
            return self._stream_and_record(job_id, messages, sanitized_query)

        # Non-streaming
        raw_response = await self.client.chat_complete(
            messages=messages,
            model=self.configured_model,
        )
        safe_response = enforce_scientific_terminology(raw_response)

        # Store in conversation history
        self._record_turn(job_id, sanitized_query, safe_response)

        return AnalystChatResponse(
            job_id=job_id,
            role="assistant",
            message=safe_response,
            context_summary=context["job"],
            provider=self.provider,
            model=self.configured_model,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def chat_job(
        self,
        job_id: str,
        job_data: dict[str, Any],
        message: str,
        stream: bool = False,
        history_limit: int = 10,
    ) -> AnalystChatResponse | AsyncGenerator[str, None]:
        """
        Conduct a multi-turn grounded conversation about an actual PixelSight job.
        """
        sanitized_message, is_safe, refusal = sanitize_user_input(message)
        if not is_safe:
            return AnalystChatResponse(
                job_id=job_id,
                role="assistant",
                message=refusal or "Request rejected by scientific safety policy.",
                context_summary={"job_id": job_id},
                provider=self.provider,
                model=self.configured_model,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        report_data = self._load_report(job_id) or job_data.get("report")
        context = ContextBuilder.build_job_context(job_data, report_data)

        # Build prompt: system + context + bounded history + user turn
        history = self._get_history(job_id, limit=history_limit)
        messages = [
            build_system_message(),
            build_context_message(context),
            *history,
            {"role": "user", "content": sanitized_message},
        ]

        if stream:
            return self._stream_and_record(job_id, messages, sanitized_message)

        raw_response = await self.client.chat_complete(
            messages=messages,
            model=self.configured_model,
        )
        safe_response = enforce_scientific_terminology(raw_response)
        self._record_turn(job_id, sanitized_message, safe_response)

        return AnalystChatResponse(
            job_id=job_id,
            role="assistant",
            message=safe_response,
            context_summary=context["job"],
            provider=self.provider,
            model=self.configured_model,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def summarize_batch(
        self,
        batch_id: str,
        jobs: list[dict[str, Any]],
        question: str | None = None,
        stream: bool = False,
    ) -> AnalystChatResponse | AsyncGenerator[str, None]:
        """Generate a scientific executive summary of a batch execution."""
        context = ContextBuilder.build_batch_context(batch_id, jobs)
        user_query = question or (
            f"Please provide an analytical summary of batch execution {batch_id}. "
            "Highlight throughput, job status distribution (completed vs failed), key reconstruction metrics, "
            "and any observed processing anomalies or warnings."
        )
        sanitized_query, is_safe, refusal = sanitize_user_input(user_query)
        if not is_safe:
            return AnalystChatResponse(
                job_id=batch_id,
                role="assistant",
                message=refusal or "Request rejected by safety policy.",
                context_summary=context["batch"],
                provider=self.provider,
                model=self.configured_model,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        messages = [
            build_system_message(),
            build_context_message(context),
            {"role": "user", "content": sanitized_query},
        ]

        if stream:
            return self._stream_and_record(batch_id, messages, sanitized_query)

        raw_response = await self.client.chat_complete(
            messages=messages,
            model=self.configured_model,
        )
        safe_response = enforce_scientific_terminology(raw_response)
        self._record_turn(batch_id, sanitized_query, safe_response)

        return AnalystChatResponse(
            job_id=batch_id,
            role="assistant",
            message=safe_response,
            context_summary=context["batch"],
            provider=self.provider,
            model=self.configured_model,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def _stream_and_record(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        user_query: str,
    ) -> AsyncGenerator[str, None]:
        """Stream response chunks and record turn once finished."""
        full_chunks = []
        async for chunk in self.client.chat_stream(messages, self.configured_model):
            full_chunks.append(chunk)
            yield chunk

        full_text = enforce_scientific_terminology("".join(full_chunks))
        self._record_turn(session_id, user_query, full_text)

    def _get_history(self, session_id: str, limit: int = 10) -> list[dict[str, str]]:
        history = self._conversations.get(session_id, [])
        # Each turn is user + assistant (2 items). Keep up to limit turns.
        max_items = limit * 2
        return history[-max_items:] if len(history) > max_items else list(history)

    def _record_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        if session_id not in self._conversations:
            self._conversations[session_id] = []
        self._conversations[session_id].append({"role": "user", "content": user_text})
        self._conversations[session_id].append({"role": "assistant", "content": assistant_text})
        # Keep bounded to prevent unbounded memory growth
        max_items = self._max_history_turns * 2
        if len(self._conversations[session_id]) > max_items:
            self._conversations[session_id] = self._conversations[session_id][-max_items:]

from __future__ import annotations

import json
from typing import Any, AsyncGenerator
import httpx


class OllamaClient:
    """
    Asynchronous client for interacting with local Ollama runtime.
    Does not require internet access or cloud API keys.
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def get_status(self, configured_model: str) -> dict[str, Any]:
        """
        Check whether Ollama is reachable and whether the configured model is installed.
        """
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    model_found = self._check_model_match(configured_model, models)
                    return {
                        "ollama_available": True,
                        "model_available": model_found,
                        "available_models": models,
                        "status": "ready" if model_found else "model_missing",
                        "error": None if model_found else f"Model '{configured_model}' is not pulled in Ollama.",
                        "instructions": (
                            None
                            if model_found
                            else f"Run `ollama pull {configured_model}` in your terminal to install the model."
                        ),
                    }
                else:
                    return {
                        "ollama_available": False,
                        "model_available": False,
                        "available_models": [],
                        "status": "error",
                        "error": f"Ollama returned HTTP status {resp.status_code}",
                        "instructions": "Ensure Ollama service is functioning properly.",
                    }
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return {
                "ollama_available": False,
                "model_available": False,
                "available_models": [],
                "status": "ollama_offline",
                "error": f"Could not connect to Ollama at {self.base_url}. Service is not running.",
                "instructions": "Start Ollama by running `ollama serve` or launching the Ollama desktop app.",
            }
        except Exception as exc:
            return {
                "ollama_available": False,
                "model_available": False,
                "available_models": [],
                "status": "error",
                "error": str(exc),
                "instructions": "Verify your Ollama configuration in environment variables.",
            }

    @staticmethod
    def _check_model_match(configured: str, available_models: list[str]) -> bool:
        """Flexible matching for model tags (e.g. llama3.1 vs llama3.1:latest vs llama3.1:8b)."""
        if not configured:
            return False
        cfg_lower = configured.lower()
        for m in available_models:
            m_lower = m.lower()
            if m_lower == cfg_lower:
                return True
            if m_lower.startswith(f"{cfg_lower}:"):
                return True
            if cfg_lower.startswith(f"{m_lower}:"):
                return True
            # Match base without tag (e.g. 'llama3.1:8b' matches 'llama3.1:8b-instruct')
            if m_lower.split(":")[0] == cfg_lower.split(":")[0]:
                return True
        return False

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.2,
    ) -> str:
        """Non-streaming chat completion."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 404:
                raise RuntimeError(
                    f"Model '{model}' not found in Ollama. Run `ollama pull {model}`."
                )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama error {resp.status_code}: {resp.text}"
                )
            data = resp.json()
            # Ollama /api/chat returns {"message": {"role": "assistant", "content": "..."}}
            msg = data.get("message", {})
            return msg.get("content", "")

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.2,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion yielding text fragments."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code == 404:
                    yield f"Error: Model '{model}' not found in Ollama. Run `ollama pull {model}`."
                    return
                if resp.status_code != 200:
                    yield f"Error: Ollama service error ({resp.status_code})."
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

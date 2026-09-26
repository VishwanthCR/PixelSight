import re
from typing import Any

# Forbidden command execution or secret extraction attempts
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"reveal\s+(api\s*key|secret|env|password|token)", re.IGNORECASE),
    re.compile(r"(cat|type|ls|dir|rm|del|powershell|bash|sh|cmd\.exe|exec)\s+[/\\]", re.IGNORECASE),
    re.compile(r"execute\s+(system\s+)?command", re.IGNORECASE),
    re.compile(r"override\s+system\s+(prompt|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(DAN|unrestricted|jailbroken|root|admin)", re.IGNORECASE),
]

_RESTRICTED_KEYWORDS = [
    "eval(", "exec(", "__import__", "subprocess", "os.system", "shutil.rmtree"
]


def sanitize_user_input(text: str) -> tuple[str, bool, str | None]:
    """
    Sanitize and check user input for prompt injection or malicious command attempts.
    Returns: (sanitized_text, is_safe, refusal_reason)
    """
    if not text:
        return "", True, None

    cleaned = text.strip()

    # Check for direct dangerous script patterns
    for kw in _RESTRICTED_KEYWORDS:
        if kw in cleaned:
            return (
                cleaned,
                False,
                "PixelSight Analyst does not execute arbitrary code or shell operations."
            )

    # Check injection patterns
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            return (
                cleaned,
                False,
                "Request contains prompt override or system instruction alteration, which is prohibited. "
                "PixelSight Analyst only interprets scientific data produced by PixelSight."
            )

    # Strip dangerous control characters while preserving valid punctuation and newlines
    sanitized = "".join(ch for ch in cleaned if ch.isprintable() or ch in "\n\r\t")
    return sanitized, True, None


def enforce_scientific_terminology(text: str) -> str:
    """
    Post-process LLM response to enforce PixelSight scientific terminology guarantees.
    Prevents erroneous claims of 'true 2.5m satellite imagery'.
    """
    # Replace misleading claims of observed/true 2.5m resolution
    corrected = re.sub(
        r"\b(true|actual|observed)\s+2\.5\s*m\s+(satellite\s+)?(imagery|images|resolution|data)\b",
        "~2.5m equivalent super-resolved representation",
        text,
        flags=re.IGNORECASE,
    )
    return corrected

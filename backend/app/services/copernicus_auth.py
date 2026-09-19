import os
import threading
import time
from typing import Optional
import requests

from backend.app.config import COPERNICUS_TOKEN_URL


class CopernicusAuthError(Exception):
    """Raised when Copernicus authentication fails or credentials are missing."""
    pass


class CopernicusAuthService:
    def __init__(
        self,
        token_url: str = COPERNICUS_TOKEN_URL,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        self.token_url = token_url
        self._explicit_client_id = client_id
        self._explicit_client_secret = client_secret
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self._lock = threading.Lock()

    @property
    def client_id(self) -> str:
        if self._explicit_client_id is not None:
            return self._explicit_client_id.strip()
        return (os.getenv("COPERNICUS_CLIENT_ID") or "").strip()

    @property
    def client_secret(self) -> str:
        if self._explicit_client_secret is not None:
            return self._explicit_client_secret.strip()
        return (os.getenv("COPERNICUS_CLIENT_SECRET") or "").strip()

    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def get_access_token(self, force_refresh: bool = False) -> str:
        """
        Retrieves a valid OAuth2 Bearer token, reusing cached token if within validity window.
        Refreshes token automatically 60 seconds before expiration.
        """
        with self._lock:
            now = time.time()
            if not force_refresh and self._token and now < self._expires_at:
                return self._token

            c_id = self.client_id
            c_sec = self.client_secret

            if not c_id or not c_sec:
                raise CopernicusAuthError(
                    "Copernicus credentials are not configured. Please set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in backend/.env."
                )

            payload = {
                "grant_type": "client_credentials",
                "client_id": c_id,
                "client_secret": c_sec,
            }

            try:
                response = requests.post(
                    self.token_url,
                    data=payload,
                    timeout=20,
                )
            except requests.RequestException as exc:
                raise CopernicusAuthError(
                    f"Network error connecting to Copernicus identity service: {exc}"
                ) from exc

            if response.status_code == 401 or response.status_code == 400:
                raise CopernicusAuthError(
                    "Copernicus authentication failed: Invalid Client ID or Client Secret."
                )
            elif response.status_code != 200:
                raise CopernicusAuthError(
                    f"Copernicus identity service returned unexpected status {response.status_code}."
                )

            try:
                data = response.json()
                token = data.get("access_token")
                expires_in = int(data.get("expires_in", 1800))
            except Exception as exc:
                raise CopernicusAuthError(
                    f"Malformed response from Copernicus token endpoint: {exc}"
                ) from exc

            if not token:
                raise CopernicusAuthError(
                    "Copernicus token response did not contain an access_token."
                )

            # Store token and cache expiration with safety margin
            self._token = token
            self._expires_at = now + max(60, expires_in - 60)
            return self._token

    def clear_cache(self) -> None:
        with self._lock:
            self._token = None
            self._expires_at = 0.0


copernicus_auth = CopernicusAuthService()

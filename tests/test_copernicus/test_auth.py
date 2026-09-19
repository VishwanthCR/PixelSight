from unittest.mock import patch, MagicMock
import pytest
import time

from backend.app.services.copernicus_auth import CopernicusAuthError, CopernicusAuthService


def test_copernicus_auth_missing_credentials():
    service = CopernicusAuthService(client_id="", client_secret="")
    with pytest.raises(CopernicusAuthError) as exc:
        service.get_access_token()
    assert "not configured" in str(exc.value)


def test_copernicus_auth_success_and_caching():
    service = CopernicusAuthService(client_id="dummy_id", client_secret="dummy_secret")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "access_token": "mock_jwt_token_123",
        "expires_in": 1800,
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        token1 = service.get_access_token()
        assert token1 == "mock_jwt_token_123"
        assert mock_post.call_count == 1

        # Second call should reuse cache without network request
        token2 = service.get_access_token()
        assert token2 == "mock_jwt_token_123"
        assert mock_post.call_count == 1


def test_copernicus_auth_invalid_credentials_error():
    service = CopernicusAuthService(client_id="bad_id", client_secret="bad_sec")

    mock_resp = MagicMock()
    mock_resp.status_code = 401

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(CopernicusAuthError) as exc:
            service.get_access_token()
        assert "Invalid Client ID or Client Secret" in str(exc.value)

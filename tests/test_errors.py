"""Tests for Ragora error classes."""

from ragora.models import (
    APIError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RagoraException,
    RateLimitError,
    ServerError,
)


def test_ragora_exception_basic():
    err = RagoraException("something failed", 400)
    assert err.message == "something failed"
    assert err.status_code == 400
    assert err.error is None
    assert err.request_id is None


def test_ragora_exception_with_details():
    api_error = APIError(code="bad_request", message="bad")
    err = RagoraException("fail", 400, error=api_error, request_id="req-123")
    assert err.error == api_error
    assert err.request_id == "req-123"


def test_is_retryable():
    assert RagoraException("", 429).is_retryable is True
    assert RagoraException("", 500).is_retryable is True
    assert RagoraException("", 502).is_retryable is True
    assert RagoraException("", 503).is_retryable is True
    assert RagoraException("", 504).is_retryable is True
    assert RagoraException("", 400).is_retryable is False
    assert RagoraException("", 404).is_retryable is False


def test_is_rate_limited():
    assert RagoraException("", 429).is_rate_limited is True
    assert RagoraException("", 500).is_rate_limited is False


def test_is_auth_error():
    assert RagoraException("", 401).is_auth_error is True
    assert RagoraException("", 403).is_auth_error is True
    assert RagoraException("", 400).is_auth_error is False


def test_str_format():
    err = RagoraException("not found", 404, request_id="req-abc")
    assert str(err) == "[404] not found (Request ID: req-abc)"


def test_str_without_request_id():
    err = RagoraException("bad request", 400)
    assert str(err) == "[400] bad request"


def test_authentication_error():
    err = AuthenticationError("unauthorized", 401)
    assert err.status_code == 401
    assert isinstance(err, RagoraException)


def test_authorization_error():
    err = AuthorizationError("forbidden", 403)
    assert err.status_code == 403
    assert isinstance(err, RagoraException)


def test_not_found_error():
    err = NotFoundError("missing", 404)
    assert err.status_code == 404
    assert isinstance(err, RagoraException)


def test_rate_limit_error():
    err = RateLimitError("slow down", retry_after=30.0)
    assert err.status_code == 429
    assert err.retry_after == 30.0
    assert isinstance(err, RagoraException)


def test_server_error():
    err = ServerError("internal", 503)
    assert err.status_code == 503
    assert isinstance(err, RagoraException)

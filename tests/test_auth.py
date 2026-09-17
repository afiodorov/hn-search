"""GitHub sign-in and the admin gate on deleting recent queries.

The flow runs against a fake GitHub (an httpx mock transport), so what is
tested is this side: the state cookie, the signed user cookie, who counts as
an admin, and that DELETE refuses everyone else.

No lifespan here: the MCP session manager it would start can only run once
per process and test_agents owns it. These routes need nothing from it.
"""

import time

import httpx
import pytest
from fastapi.testclient import TestClient

from hn_search.api import auth
from hn_search.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app, base_url="https://hn.example")


@pytest.fixture
def github(monkeypatch):
    """A configured deployment and a GitHub that accepts one code."""
    monkeypatch.setenv("GITHUB_CLIENT_ID", "id-123")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "secret-456")
    monkeypatch.setenv("ADMIN_GITHUB_USERS", "afiodorov, Other-Admin")
    monkeypatch.delenv("AUTH_TRUSTED_USER_HEADER", raising=False)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url == auth.GITHUB_TOKEN:
            seen["token_request"] = dict(httpx.QueryParams(request.content.decode()))
            if seen["token_request"].get("code") != "good-code":
                return httpx.Response(200, json={"error": "bad_verification_code"})
            return httpx.Response(
                200, json={"access_token": "tok", "token_type": "bearer"}
            )
        if request.url == auth.GITHUB_USER:
            seen["auth_header"] = request.headers.get("authorization")
            return httpx.Response(200, json={"login": "afiodorov"})
        return httpx.Response(404)

    monkeypatch.setattr(auth, "_transport", httpx.MockTransport(handler))
    return seen


@pytest.fixture
def deleted(monkeypatch):
    """What the job manager was asked to forget."""
    calls: list[str] = []
    monkeypatch.setattr(
        "hn_search.api.app.job_manager.delete_recent_query", calls.append
    )
    return calls


def _cookie(login: str, days: int = 1) -> str:
    return auth._sign(login, int(time.time()) + days * 86400)


def test_anonymous_me_and_unconfigured_delete(client, monkeypatch, deleted):
    monkeypatch.delenv("GITHUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("AUTH_TRUSTED_USER_HEADER", raising=False)
    r = client.get("/auth/me")
    assert r.json() == {"login": None, "admin": False, "configured": False}
    r = client.delete("/api/recent", params={"q": "anything"})
    assert r.status_code == 403
    assert "not configured" in r.json()["detail"]
    assert deleted == []
    r = client.get("/auth/login", follow_redirects=False)
    assert r.status_code == 404


def test_login_redirects_to_github_with_a_state_cookie(client, github):
    r = client.get("/auth/login", follow_redirects=False)
    assert r.status_code == 302
    location = r.headers["location"]
    assert location.startswith(auth.GITHUB_AUTHORIZE)
    assert "client_id=id-123" in location
    assert "redirect_uri=https%3A%2F%2Fhn.example%2Foauth2%2Fcallback" in location
    state = httpx.URL(location).params["state"]
    assert r.cookies[auth.STATE_COOKIE] == state


def test_callback_signs_the_user_in(client, github):
    r = client.get("/auth/login", follow_redirects=False)
    state = httpx.URL(r.headers["location"]).params["state"]

    r = client.get(
        auth.CALLBACK_PATH,
        params={"code": "good-code", "state": state},
        cookies={auth.STATE_COOKIE: state},
        follow_redirects=False,
    )
    assert r.status_code == 302 and r.headers["location"] == "/"
    assert github["token_request"]["client_secret"] == "secret-456"
    assert github["auth_header"] == "Bearer tok"
    assert auth.USER_COOKIE in r.cookies

    me = client.get("/auth/me", cookies={auth.USER_COOKIE: r.cookies[auth.USER_COOKIE]})
    assert me.json() == {"login": "afiodorov", "admin": True, "configured": True}

    r = client.post("/auth/logout")
    assert r.status_code == 204


def test_callback_rejects_a_bad_state_or_code(client, github):
    r = client.get(
        auth.CALLBACK_PATH,
        params={"code": "good-code", "state": "x"},
        follow_redirects=False,
    )
    assert r.status_code == 400
    r = client.get(
        auth.CALLBACK_PATH,
        params={"code": "bad-code", "state": "s"},
        cookies={auth.STATE_COOKIE: "s"},
        follow_redirects=False,
    )
    assert r.status_code == 502


def test_delete_needs_an_admin(client, github, deleted):
    r = client.delete("/api/recent", params={"q": "yoga"})
    assert r.status_code == 403 and "Sign in" in r.json()["detail"]

    r = client.delete(
        "/api/recent",
        params={"q": "yoga"},
        cookies={auth.USER_COOKIE: _cookie("someone-else")},
    )
    assert r.status_code == 403 and "not an admin" in r.json()["detail"]

    forged = _cookie("afiodorov").rsplit("|", 1)[0] + "|" + "0" * 64
    r = client.delete(
        "/api/recent", params={"q": "yoga"}, cookies={auth.USER_COOKIE: forged}
    )
    assert r.status_code == 403

    expired = auth._sign("afiodorov", int(time.time()) - 1)
    r = client.delete(
        "/api/recent", params={"q": "yoga"}, cookies={auth.USER_COOKIE: expired}
    )
    assert r.status_code == 403
    assert deleted == []

    r = client.delete(
        "/api/recent",
        params={"q": "yoga"},
        cookies={auth.USER_COOKIE: _cookie("afiodorov")},
    )
    assert r.status_code == 204
    # Admin names are case-insensitive, and the list is trimmed.
    r = client.delete(
        "/api/recent",
        params={"q": "rust vs go"},
        cookies={auth.USER_COOKIE: _cookie("other-admin")},
    )
    assert r.status_code == 204
    assert deleted == ["yoga", "rust vs go"]


def test_me_ignores_a_cookie_once_the_secret_changes(client, github, monkeypatch):
    good = _cookie("afiodorov")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "rotated")
    r = client.get("/auth/me", cookies={auth.USER_COOKIE: good})
    assert r.json()["login"] is None


def test_trusted_proxy_header_names_the_user(client, monkeypatch, deleted):
    """Staging: Caddy copies X-Auth-Request-User from oauth2-proxy."""
    monkeypatch.delenv("GITHUB_CLIENT_ID", raising=False)
    monkeypatch.setenv("AUTH_TRUSTED_USER_HEADER", "X-Auth-Request-User")
    r = client.get("/auth/me")
    assert r.json() == {"login": None, "admin": False, "configured": False}
    r = client.get("/auth/me", headers={"X-Auth-Request-User": "AFiodorov"})
    assert r.json() == {"login": "AFiodorov", "admin": True, "configured": False}
    r = client.delete(
        "/api/recent", params={"q": "yoga"}, headers={"X-Auth-Request-User": "guest"}
    )
    assert r.status_code == 403
    r = client.delete(
        "/api/recent",
        params={"q": "yoga"},
        headers={"X-Auth-Request-User": "afiodorov"},
    )
    assert r.status_code == 204
    assert deleted == ["yoga"]

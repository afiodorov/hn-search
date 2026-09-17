"""Who is asking: GitHub sign-in, and the short list of admins.

Everything on this service is public and read-only except one thing, deleting
a recent query from the shared "recent searches" list, and that needs an
admin: a GitHub login named in `ADMIN_GITHUB_USERS`. Nothing else changes for
a visitor — searching, the recent list, the MCP surface are all as before.

The flow is plain GitHub OAuth, done here rather than by a proxy because prod
runs on Railway with nothing in front of it. `/auth/login` sends the browser
to GitHub, `/oauth2/callback` swaps the code for the login name, and the name
is kept in a cookie signed with HMAC so it cannot be forged. No token is
stored: the only thing this service ever wanted to know is who you are.

Configuration, all environment variables:

  GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET   a GitHub OAuth app whose callback is
                                           https://<this host>/oauth2/callback.
  ADMIN_GITHUB_USERS                       comma-separated logins; default afiodorov
  SESSION_SECRET                           optional; the cookie key. Derived from
                                           the client secret when absent.
  AUTH_TRUSTED_USER_HEADER                 staging only: the header a trusted
                                           proxy in front sets to the GitHub
                                           login (X-Auth-Request-User behind
                                           Caddy + oauth2-proxy, which
                                           overwrites any client-sent value).
                                           Never set this where a client can
                                           reach the app directly.

Without a client id or a trusted header the routes still answer, sign-in is
simply unavailable and nobody is an admin — deletes are refused rather than
left open.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

USER_COOKIE = "hn_user"
STATE_COOKIE = "hn_oauth_state"
CALLBACK_PATH = "/oauth2/callback"
COOKIE_DAYS = 30
DEFAULT_ADMINS = "afiodorov"

GITHUB_AUTHORIZE = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN = "https://github.com/login/oauth/access_token"
GITHUB_USER = "https://api.github.com/user"

router = APIRouter()


def oauth_configured() -> bool:
    return bool(
        os.environ.get("GITHUB_CLIENT_ID") and os.environ.get("GITHUB_CLIENT_SECRET")
    )


def trusted_header() -> str | None:
    return os.environ.get("AUTH_TRUSTED_USER_HEADER") or None


def configured() -> bool:
    """Whether anyone can be signed in at all, by either mechanism."""
    return oauth_configured() or trusted_header() is not None


def admins() -> set[str]:
    raw = os.environ.get("ADMIN_GITHUB_USERS", DEFAULT_ADMINS)
    return {name.strip().lower() for name in raw.split(",") if name.strip()}


def _key() -> bytes:
    secret = os.environ.get("SESSION_SECRET") or os.environ.get(
        "GITHUB_CLIENT_SECRET", ""
    )
    return hashlib.sha256(f"hn-user-cookie:{secret}".encode()).digest()


def _sign(login: str, expires: int) -> str:
    body = f"{login}|{expires}"
    mac = hmac.new(_key(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}|{mac}"


def _verify(value: str | None) -> str | None:
    """The login a cookie names, or None if it is missing, forged or expired."""
    if not value or not oauth_configured():
        return None
    try:
        login, expires, mac = value.split("|")
        expires_at = int(expires)
    except ValueError:
        return None
    expected = hmac.new(
        _key(), f"{login}|{expires}".encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(mac, expected) or expires_at < time.time():
        return None
    return login


def current_user(request: Request) -> str | None:
    header = trusted_header()
    if header:
        return request.headers.get(header) or None
    return _verify(request.cookies.get(USER_COOKIE))


def is_admin(request: Request) -> bool:
    user = current_user(request)
    return user is not None and user.lower() in admins()


def require_admin(request: Request) -> str:
    """The admin's login, or a 403 that says what would have been needed."""
    user = current_user(request)
    if user is None:
        if not configured():
            raise HTTPException(
                status_code=403,
                detail="Only an admin can do that, and GitHub sign-in is not "
                "configured on this deployment.",
            )
        raise HTTPException(status_code=403, detail="Sign in with GitHub first.")
    if user.lower() not in admins():
        raise HTTPException(status_code=403, detail=f"{user} is not an admin.")
    return user


def _secure(request: Request) -> bool:
    # Behind Railway or Caddy the request arrives as http; only a dev loopback
    # is genuinely plain. Same reasoning as llms.txt's scheme fix-up.
    return request.url.hostname not in ("localhost", "127.0.0.1", "::1")


def _callback_url(request: Request) -> str:
    base = request.base_url
    if _secure(request):
        base = base.replace(scheme="https")
    return str(base).rstrip("/") + CALLBACK_PATH


@router.get("/auth/me")
def me(request: Request) -> dict:
    """Who the browser is signed in as, and whether that makes them an admin.

    `configured` is whether a sign-in link is worth showing: false when there
    is no OAuth app, and false behind a trusted proxy too, since there the
    visitor is signed in already or would not have got this far.
    """
    user = current_user(request)
    return {
        "login": user,
        "admin": user is not None and user.lower() in admins(),
        "configured": oauth_configured() and trusted_header() is None,
    }


@router.get("/auth/login")
def login(request: Request) -> Response:
    if not oauth_configured():
        raise HTTPException(status_code=404, detail="GitHub sign-in is not configured.")
    state = secrets.token_urlsafe(24)
    params = {
        "client_id": os.environ["GITHUB_CLIENT_ID"],
        "redirect_uri": _callback_url(request),
        "state": state,
        # No scope: the public profile is all this needs, and the app never
        # keeps the token.
    }
    response = RedirectResponse(
        f"{GITHUB_AUTHORIZE}?{urlencode(params)}", status_code=302
    )
    response.set_cookie(
        STATE_COOKIE,
        state,
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=_secure(request),
    )
    return response


@router.get(CALLBACK_PATH)
async def callback(request: Request, code: str = "", state: str = "") -> Response:
    if not oauth_configured():
        raise HTTPException(status_code=404, detail="GitHub sign-in is not configured.")
    expected = request.cookies.get(STATE_COOKIE)
    if (
        not code
        or not state
        or not expected
        or not hmac.compare_digest(state, expected)
    ):
        raise HTTPException(
            status_code=400, detail="Sign-in state mismatch; try again."
        )

    login = await _github_login(code, _callback_url(request))
    expires = int(time.time()) + COOKIE_DAYS * 86400
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(
        USER_COOKIE,
        _sign(login, expires),
        max_age=COOKIE_DAYS * 86400,
        httponly=True,
        samesite="lax",
        secure=_secure(request),
    )
    response.delete_cookie(STATE_COOKIE)
    return response


@router.post("/auth/logout", status_code=204)
def logout() -> Response:
    response = Response(status_code=204)
    response.delete_cookie(USER_COOKIE)
    return response


# Swappable in tests, so the flow can be exercised without GitHub.
_transport: httpx.AsyncBaseTransport | None = None


async def _github_login(code: str, redirect_uri: str) -> str:
    """Exchange the code for the user's login. The token is used once and dropped."""
    async with httpx.AsyncClient(transport=_transport, timeout=15) as client:
        token_response = await client.post(
            GITHUB_TOKEN,
            data={
                "client_id": os.environ["GITHUB_CLIENT_ID"],
                "client_secret": os.environ["GITHUB_CLIENT_SECRET"],
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Accept": "application/json"},
        )
        token = (
            token_response.json().get("access_token")
            if token_response.is_success
            else None
        )
        if not token:
            raise HTTPException(
                status_code=502, detail="GitHub did not accept the sign-in code."
            )
        user_response = await client.get(
            GITHUB_USER,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        login = user_response.json().get("login") if user_response.is_success else None
        if not login:
            raise HTTPException(status_code=502, detail="GitHub did not return a user.")
        return str(login)

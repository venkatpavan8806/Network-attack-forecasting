"""Who is calling? Resolves every request to a user id.

Website users (the React app) send `Authorization: Bearer <Supabase access
token>`. The token is verified here -- signature, expiry and audience --
before its `sub` (the Supabase user UUID) is trusted as the user id:

  - projects using Supabase's asymmetric JWT signing keys (the default for
    new projects): verified against the project's public JWKS at
    {SUPABASE_URL}/auth/v1/.well-known/jwks.json
  - projects still on the legacy shared secret: set SUPABASE_JWT_SECRET
    (Supabase dashboard -> Project Settings -> JWT Keys) and HS256 tokens
    are verified with it

Capture agents send `Authorization: Bearer nadf_...` -- a sensor token the
user created on the website (see app/db.py: only its hash is stored).

Local development without Supabase (AUTH_MODE=dev, the default when
SUPABASE_URL is unset and we are not running on Render): the caller is
identified by the `X-Dev-User` header (the React app generates a random id
per browser), so even locally every browser gets its own empty workspace.
Dev mode is refused on Render -- a deployed backend must verify real logins.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import jwt
from fastapi import Header, HTTPException

from app import db


@dataclass
class AuthConfig:
    mode: str                  # "supabase" | "dev"
    supabase_url: str | None
    jwt_secret: str | None
    problem: str | None = None  # set when the configuration is unusable


def _project_origin(raw: str | None) -> str | None:
    """https://<ref>.supabase.co from any URL form the dashboard shows
    (".../rest/v1/", trailing slash, ...)."""
    from urllib.parse import urlsplit
    raw = (raw or "").strip()
    if not raw:
        return None
    parts = urlsplit(raw)
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else raw.rstrip("/")


def load_config() -> AuthConfig:
    supabase_url = _project_origin(os.environ.get("SUPABASE_URL"))
    secret = os.environ.get("SUPABASE_JWT_SECRET") or None
    mode = os.environ.get("AUTH_MODE") or ("supabase" if supabase_url else "dev")
    problem = None
    if mode == "supabase" and not supabase_url:
        problem = "AUTH_MODE=supabase but SUPABASE_URL is not set"
    if mode == "dev" and os.environ.get("RENDER"):
        problem = ("refusing to run with AUTH_MODE=dev on Render: set SUPABASE_URL "
                   "(and SUPABASE_JWT_SECRET for legacy projects) -- see DEPLOY.md")
    return AuthConfig(mode=mode, supabase_url=supabase_url, jwt_secret=secret, problem=problem)


CONFIG = load_config()


def reload_config():
    global CONFIG
    CONFIG = load_config()
    _jwk_client.cache_clear()
    return CONFIG


@lru_cache(maxsize=1)
def _jwk_client(supabase_url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(f"{supabase_url}/auth/v1/.well-known/jwks.json", cache_keys=True, lifespan=3600)


def verify_supabase_token(token: str, cfg: AuthConfig | None = None) -> dict:
    """Returns the verified claims, or raises HTTPException(401)."""
    cfg = cfg or CONFIG
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="malformed access token")
    alg = header.get("alg")
    try:
        if alg == "HS256":
            if not cfg.jwt_secret:
                raise HTTPException(status_code=401, detail="HS256 token but SUPABASE_JWT_SECRET is not configured")
            key = cfg.jwt_secret
        elif alg in ("ES256", "RS256", "EdDSA"):
            key = _jwk_client(cfg.supabase_url).get_signing_key_from_jwt(token).key
        else:
            raise HTTPException(status_code=401, detail=f"unsupported token algorithm: {alg}")
        claims = jwt.decode(token, key, algorithms=[alg], audience="authenticated",
                            options={"require": ["exp", "sub"]})
    except HTTPException:
        raise
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="session expired -- please sign in again")
    except (jwt.PyJWTError, jwt.PyJWKClientError) as e:
        raise HTTPException(status_code=401, detail=f"invalid access token: {e}")
    return claims


def _bearer(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def current_user(authorization: str | None = Header(default=None),
                 x_dev_user: str | None = Header(default=None)) -> str:
    """FastAPI dependency: the calling website user's id."""
    if CONFIG.problem:
        raise HTTPException(status_code=503, detail=f"server auth misconfigured: {CONFIG.problem}")
    if CONFIG.mode == "dev":
        user = (x_dev_user or "local-dev").strip()[:64]
        return user or "local-dev"
    token = _bearer(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="not signed in")
    if token.startswith(db.TOKEN_PREFIX):
        raise HTTPException(status_code=401, detail="sensor tokens can only be used by the capture agent")
    return verify_supabase_token(token)["sub"]


def current_sensor(authorization: str | None = Header(default=None)) -> dict:
    """FastAPI dependency for capture-agent endpoints: the sensor (incl.
    its owner's user_id) identified by the agent's token."""
    token = _bearer(authorization)
    sensor = db.sensor_by_token(token) if token else None
    if sensor is None:
        raise HTTPException(status_code=401, detail="unknown or revoked sensor token -- create a new one on the website")
    return sensor

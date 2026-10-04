"""Identifies the signed-in user behind each API request.

The website sends `Authorization: Bearer <Supabase access token>`. The
token's signature, expiry and audience are verified before its `sub` (the
Supabase user id) is trusted:
  - new Supabase projects (asymmetric signing keys): verified against the
    project's public keys at {SUPABASE_URL}/auth/v1/.well-known/jwks.json
  - projects on the legacy shared secret: set SUPABASE_JWT_SECRET

Local development without Supabase (SUPABASE_URL unset, not on Render):
every request is treated as one local user, "local".
"""
from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import urlsplit

import jwt
from fastapi import Header, HTTPException


def _project_origin(raw: str | None) -> str | None:
    """https://<ref>.supabase.co from any form of the project URL."""
    raw = (raw or "").strip()
    if not raw:
        return None
    parts = urlsplit(raw)
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else raw.rstrip("/")


SUPABASE_URL = _project_origin(os.environ.get("SUPABASE_URL"))
JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET") or None
DEV_MODE = SUPABASE_URL is None and not os.environ.get("RENDER")
PROBLEM = None if (SUPABASE_URL or DEV_MODE) else "SUPABASE_URL is not set -- logins cannot be verified"


@lru_cache(maxsize=1)
def _jwks() -> jwt.PyJWKClient:
    return jwt.PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json", cache_keys=True, lifespan=3600)


def verify_token(token: str) -> str:
    """Returns the Supabase user id for a valid access token, else raises 401."""
    try:
        alg = jwt.get_unverified_header(token).get("alg")
        if alg == "HS256":
            if not JWT_SECRET:
                raise HTTPException(status_code=401, detail="HS256 token but SUPABASE_JWT_SECRET is not set")
            key = JWT_SECRET
        elif alg in ("ES256", "RS256", "EdDSA"):
            key = _jwks().get_signing_key_from_jwt(token).key
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
    return claims["sub"]


def current_user(authorization: str | None = Header(default=None)) -> str:
    """FastAPI dependency: the signed-in user's id."""
    if PROBLEM:
        raise HTTPException(status_code=503, detail=PROBLEM)
    if DEV_MODE:
        return "local"
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="not signed in")
    return verify_token(authorization[7:].strip())

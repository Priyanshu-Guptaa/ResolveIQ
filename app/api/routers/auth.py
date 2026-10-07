"""Login endpoint for the hosted deployment."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.dependencies import get_auth_service, get_settings_dep
from app.auth.dependencies import Principal, require_user
from app.auth.service import AuthService
from app.config import Settings

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    username: str
    role: str
    auth_enabled: bool


@router.post("/login", response_model=TokenResponse)
def login(
    body: LoginRequest,
    service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings_dep),
) -> TokenResponse:
    if not settings.auth_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Authentication is not enabled.")
    token = service.authenticate(body.username, body.password)
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password.")
    return TokenResponse(access_token=token)


@router.get("/me", response_model=MeResponse)
def me(principal: Principal = Depends(require_user), settings: Settings = Depends(get_settings_dep)) -> MeResponse:
    return MeResponse(username=principal.username, role=principal.role, auth_enabled=settings.auth_enabled)

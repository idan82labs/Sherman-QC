from __future__ import annotations

from fastapi import APIRouter
from apps.api.services import auth_service
from apps.api.routes._helpers import bind

router = APIRouter(tags=["Authentication"])

bind(router, "/api/auth/login", auth_service.login, ['POST'])
bind(router, "/api/auth/register", auth_service.register, ['POST'])
bind(router, "/api/auth/me", auth_service.get_me, ['GET'])
bind(router, "/api/auth/change-password", auth_service.change_password, ['POST'])
bind(router, "/api/auth/users", auth_service.list_users, ['GET'])
bind(router, "/api/auth/users/{user_id}", auth_service.update_user, ['PUT'])
bind(router, "/api/auth/users/{user_id}", auth_service.delete_user, ['DELETE'])

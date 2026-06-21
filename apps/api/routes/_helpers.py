from __future__ import annotations

from fastapi import APIRouter


def bind(router: APIRouter, path: str, endpoint, methods: list[str]):
    router.add_api_route(path, endpoint, methods=methods)

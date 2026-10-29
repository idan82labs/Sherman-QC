"""Principals and capabilities for the production line HTTP API (CONTRACTS §9).

A :class:`Principal` is either a *user* (legacy JWT, role admin/operator/viewer) or a
*device* (a revocable station kiosk token).  Capabilities are granted by role; a
device only ever receives ``line.read`` and only for the station it is scoped to.

Capability table (``recipe.release`` defaults to admin and is widened only through
``SHERMAN_RELEASE_ROLES``)::

    line.read        viewer, operator, admin; device for its own station
    line.operate     operator, admin
    parts.write      operator, admin
    parts.delete     admin
    parts.import     admin
    recipe.release   SHERMAN_RELEASE_ROLES (default admin)
    devices.manage   admin
    metrics.read     admin
    admin            admin (pseudo-capability used by the legacy route policy)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Optional

__all__ = [
    "ROLES",
    "CAPABILITIES",
    "DEVICE_CAPABILITIES",
    "Principal",
    "roles_for",
    "has_capability",
]

ROLES: tuple[str, ...] = ("admin", "operator", "viewer")

CAPABILITIES: Mapping[str, frozenset[str]] = {
    "line.read": frozenset({"viewer", "operator", "admin"}),
    "line.operate": frozenset({"operator", "admin"}),
    "parts.write": frozenset({"operator", "admin"}),
    "parts.delete": frozenset({"admin"}),
    "parts.import": frozenset({"admin"}),
    "recipe.release": frozenset({"admin"}),
    "devices.manage": frozenset({"admin"}),
    "metrics.read": frozenset({"admin"}),
    "admin": frozenset({"admin"}),
}

#: The only capabilities a station device token can ever hold.
DEVICE_CAPABILITIES: frozenset[str] = frozenset({"line.read"})

PrincipalKind = Literal["user", "device"]


@dataclass(frozen=True)
class Principal:
    """An authenticated caller.

    ``user_id`` is the legacy user id (as a string) for users and the token id for
    devices.  ``station_scope`` is set only for devices.  ``scopes`` is the set of
    capabilities this principal was granted when it was authenticated.
    """

    kind: PrincipalKind
    user_id: str
    username: str
    role: str
    station_scope: Optional[str] = None
    scopes: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def for_user(
        cls,
        user_id: object,
        username: str,
        role: str,
        *,
        release_roles: Iterable[str] = ("admin",),
    ) -> "Principal":
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        grants = frozenset(cap for cap in CAPABILITIES if role in roles_for(cap, release_roles))
        return cls(kind="user", user_id=str(user_id), username=str(username), role=role, scopes=grants)

    @classmethod
    def for_device(cls, token_id: str, label: str, station_id: str) -> "Principal":
        if not station_id:
            raise ValueError("a device principal needs a station scope")
        return cls(
            kind="device",
            user_id=str(token_id),
            username=str(label),
            role="device",
            station_scope=station_id,
            scopes=DEVICE_CAPABILITIES,
        )

    def audit(self) -> dict[str, str]:
        """The ``principal`` block for LineEventV1 (``{kind, id, role}``)."""
        return {"kind": self.kind, "id": self.user_id, "role": self.role}


def roles_for(capability: str, release_roles: Iterable[str] = ("admin",)) -> frozenset[str]:
    """User roles that hold ``capability``.  Unknown capabilities map to no role."""
    if capability == "recipe.release":
        return frozenset(r for r in release_roles if r in ROLES and r != "viewer")
    return CAPABILITIES.get(capability, frozenset())


def has_capability(
    principal: Optional[Principal],
    capability: str,
    station_id: Optional[str] = None,
    *,
    release_roles: Iterable[str] = ("admin",),
) -> bool:
    """Whether ``principal`` may exercise ``capability`` (fail-closed).

    A device holds only ``line.read`` and only when ``station_id`` is given and equals
    its scope; a request that names no station is refused for devices.
    """
    if principal is None or capability not in CAPABILITIES:
        return False
    if principal.kind == "device":
        return (
            capability in DEVICE_CAPABILITIES
            and capability in principal.scopes
            and principal.station_scope is not None
            and station_id is not None
            and station_id == principal.station_scope
        )
    if principal.kind != "user" or principal.role not in ROLES:
        return False
    return principal.role in roles_for(capability, release_roles)

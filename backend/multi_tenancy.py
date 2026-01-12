"""
Multi-Tenancy Module

Provides tenant isolation for SaaS deployments:
- Tenant context management
- Data isolation middleware
- Per-tenant configuration
- Resource quotas and limits

Isolation strategies:
- Schema-per-tenant (PostgreSQL)
- Row-level security
- Tenant ID filtering
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from functools import wraps
from contextlib import contextmanager
import threading
import uuid

logger = logging.getLogger(__name__)


class TenantTier(Enum):
    """Tenant subscription tiers"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class IsolationStrategy(Enum):
    """Data isolation strategies"""
    SHARED_SCHEMA = "shared"  # Row-level tenant_id filtering
    SCHEMA_PER_TENANT = "schema"  # Separate schema per tenant
    DATABASE_PER_TENANT = "database"  # Separate database per tenant


@dataclass
class TenantQuota:
    """Resource quotas for a tenant"""
    max_jobs_per_month: int = 100
    max_storage_gb: float = 10.0
    max_users: int = 5
    max_api_calls_per_hour: int = 1000
    max_batch_size: int = 50
    max_parts_per_batch: int = 100
    retention_days: int = 90

    # Current usage
    jobs_this_month: int = 0
    storage_used_gb: float = 0.0
    api_calls_this_hour: int = 0

    def is_within_quota(self, resource: str) -> bool:
        """Check if resource usage is within quota"""
        if resource == "jobs":
            return self.jobs_this_month < self.max_jobs_per_month
        elif resource == "storage":
            return self.storage_used_gb < self.max_storage_gb
        elif resource == "api_calls":
            return self.api_calls_this_hour < self.max_api_calls_per_hour
        return True

    def to_dict(self) -> Dict:
        return {
            "limits": {
                "max_jobs_per_month": self.max_jobs_per_month,
                "max_storage_gb": self.max_storage_gb,
                "max_users": self.max_users,
                "max_api_calls_per_hour": self.max_api_calls_per_hour,
                "max_batch_size": self.max_batch_size,
                "retention_days": self.retention_days
            },
            "usage": {
                "jobs_this_month": self.jobs_this_month,
                "storage_used_gb": self.storage_used_gb,
                "api_calls_this_hour": self.api_calls_this_hour
            },
            "remaining": {
                "jobs": self.max_jobs_per_month - self.jobs_this_month,
                "storage_gb": self.max_storage_gb - self.storage_used_gb,
                "api_calls": self.max_api_calls_per_hour - self.api_calls_this_hour
            }
        }


@dataclass
class TenantSettings:
    """Tenant-specific settings"""
    # General
    company_name: str = ""
    timezone: str = "UTC"
    locale: str = "en-US"

    # QC Settings
    default_tolerance_mm: float = 0.1
    default_material: str = "aluminum"
    auto_approve_pass: bool = False
    require_approval_for_fail: bool = True

    # Notification settings
    email_notifications: bool = True
    webhook_enabled: bool = False
    webhook_url: str = ""

    # Feature flags
    ai_analysis_enabled: bool = True
    gdt_enabled: bool = True
    spc_enabled: bool = True
    batch_processing_enabled: bool = True

    def to_dict(self) -> Dict:
        return {
            "company_name": self.company_name,
            "timezone": self.timezone,
            "locale": self.locale,
            "qc": {
                "default_tolerance_mm": self.default_tolerance_mm,
                "default_material": self.default_material,
                "auto_approve_pass": self.auto_approve_pass,
                "require_approval_for_fail": self.require_approval_for_fail
            },
            "notifications": {
                "email": self.email_notifications,
                "webhook_enabled": self.webhook_enabled
            },
            "features": {
                "ai_analysis": self.ai_analysis_enabled,
                "gdt": self.gdt_enabled,
                "spc": self.spc_enabled,
                "batch_processing": self.batch_processing_enabled
            }
        }


@dataclass
class Tenant:
    """Tenant entity"""
    id: str
    name: str
    tier: TenantTier = TenantTier.FREE
    active: bool = True

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Settings and quotas
    settings: TenantSettings = field(default_factory=TenantSettings)
    quota: TenantQuota = field(default_factory=TenantQuota)

    # Isolation
    schema_name: Optional[str] = None
    database_url: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "tier": self.tier.value,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "settings": self.settings.to_dict(),
            "quota": self.quota.to_dict()
        }


# Tier-based quota templates
TIER_QUOTAS = {
    TenantTier.FREE: TenantQuota(
        max_jobs_per_month=50,
        max_storage_gb=1.0,
        max_users=2,
        max_api_calls_per_hour=100,
        max_batch_size=10,
        max_parts_per_batch=20,
        retention_days=30
    ),
    TenantTier.STARTER: TenantQuota(
        max_jobs_per_month=500,
        max_storage_gb=10.0,
        max_users=10,
        max_api_calls_per_hour=1000,
        max_batch_size=50,
        max_parts_per_batch=100,
        retention_days=90
    ),
    TenantTier.PROFESSIONAL: TenantQuota(
        max_jobs_per_month=5000,
        max_storage_gb=100.0,
        max_users=50,
        max_api_calls_per_hour=10000,
        max_batch_size=200,
        max_parts_per_batch=500,
        retention_days=365
    ),
    TenantTier.ENTERPRISE: TenantQuota(
        max_jobs_per_month=999999,  # Unlimited
        max_storage_gb=1000.0,
        max_users=999999,
        max_api_calls_per_hour=999999,
        max_batch_size=1000,
        max_parts_per_batch=10000,
        retention_days=3650  # 10 years
    )
}


# Thread-local storage for current tenant context
_tenant_context = threading.local()


class TenantContext:
    """
    Context manager for tenant operations.

    Usage:
        with TenantContext(tenant_id="tenant_123"):
            # All operations in this block are scoped to tenant_123
            jobs = get_jobs()  # Only returns tenant_123's jobs
    """

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self._previous_tenant = None

    def __enter__(self):
        self._previous_tenant = getattr(_tenant_context, 'tenant_id', None)
        _tenant_context.tenant_id = self.tenant_id
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _tenant_context.tenant_id = self._previous_tenant
        return False


def get_current_tenant_id() -> Optional[str]:
    """Get the current tenant ID from context"""
    return getattr(_tenant_context, 'tenant_id', None)


def set_current_tenant(tenant_id: str):
    """Set the current tenant ID in context"""
    _tenant_context.tenant_id = tenant_id


def clear_tenant_context():
    """Clear the tenant context"""
    _tenant_context.tenant_id = None


def require_tenant(func):
    """Decorator to require tenant context"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tenant_id = get_current_tenant_id()
        if not tenant_id:
            raise TenantNotSetError("No tenant context set. Use TenantContext or set_current_tenant().")
        return func(*args, **kwargs)
    return wrapper


class TenantNotSetError(Exception):
    """Raised when tenant context is required but not set"""
    pass


class TenantQuotaExceededError(Exception):
    """Raised when tenant exceeds quota"""
    pass


class TenantNotFoundError(Exception):
    """Raised when tenant is not found"""
    pass


class TenantManager:
    """
    Manager for tenant operations.

    Handles:
    - Tenant CRUD operations
    - Quota management
    - Tenant context validation
    """

    def __init__(
        self,
        isolation_strategy: IsolationStrategy = IsolationStrategy.SHARED_SCHEMA,
        storage_path: Optional[str] = None
    ):
        self.isolation_strategy = isolation_strategy
        self.storage_path = storage_path
        self._tenants: Dict[str, Tenant] = {}

        # Load persisted tenants if storage path provided
        if storage_path:
            self._load_tenants()

    def create_tenant(
        self,
        name: str,
        tier: TenantTier = TenantTier.FREE,
        settings: Optional[TenantSettings] = None
    ) -> Tenant:
        """
        Create a new tenant.

        Args:
            name: Tenant name
            tier: Subscription tier
            settings: Optional custom settings

        Returns:
            Created Tenant
        """
        tenant_id = f"tenant_{uuid.uuid4().hex[:12]}"

        # Get quota based on tier
        quota = TIER_QUOTAS.get(tier, TIER_QUOTAS[TenantTier.FREE])

        tenant = Tenant(
            id=tenant_id,
            name=name,
            tier=tier,
            settings=settings or TenantSettings(company_name=name),
            quota=quota
        )

        # Set up isolation
        if self.isolation_strategy == IsolationStrategy.SCHEMA_PER_TENANT:
            tenant.schema_name = f"tenant_{tenant_id.replace('-', '_')}"

        self._tenants[tenant_id] = tenant
        self._save_tenants()

        logger.info(f"Created tenant: {tenant_id} ({name}) with tier {tier.value}")

        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID"""
        return self._tenants.get(tenant_id)

    def get_tenant_or_raise(self, tenant_id: str) -> Tenant:
        """Get tenant by ID or raise exception"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(f"Tenant not found: {tenant_id}")
        return tenant

    def update_tenant(self, tenant_id: str, updates: Dict) -> Optional[Tenant]:
        """Update tenant properties"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return None

        for key, value in updates.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)

        tenant.updated_at = datetime.now()
        self._save_tenants()

        return tenant

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant (soft delete by default)"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False

        tenant.active = False
        tenant.updated_at = datetime.now()
        self._save_tenants()

        logger.info(f"Deleted tenant: {tenant_id}")
        return True

    def list_tenants(self, active_only: bool = True) -> List[Tenant]:
        """List all tenants"""
        tenants = list(self._tenants.values())
        if active_only:
            tenants = [t for t in tenants if t.active]
        return tenants

    def check_quota(self, tenant_id: str, resource: str) -> bool:
        """
        Check if tenant is within quota for a resource.

        Args:
            tenant_id: Tenant ID
            resource: Resource type (jobs, storage, api_calls)

        Returns:
            True if within quota
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False

        return tenant.quota.is_within_quota(resource)

    def increment_usage(self, tenant_id: str, resource: str, amount: int = 1) -> bool:
        """
        Increment resource usage for tenant.

        Args:
            tenant_id: Tenant ID
            resource: Resource type
            amount: Amount to increment

        Returns:
            True if successful and within quota

        Raises:
            TenantQuotaExceededError: If quota exceeded
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False

        quota = tenant.quota

        if resource == "jobs":
            if quota.jobs_this_month + amount > quota.max_jobs_per_month:
                raise TenantQuotaExceededError(
                    f"Job quota exceeded for tenant {tenant_id}"
                )
            quota.jobs_this_month += amount

        elif resource == "storage":
            if quota.storage_used_gb + amount > quota.max_storage_gb:
                raise TenantQuotaExceededError(
                    f"Storage quota exceeded for tenant {tenant_id}"
                )
            quota.storage_used_gb += amount

        elif resource == "api_calls":
            if quota.api_calls_this_hour + amount > quota.max_api_calls_per_hour:
                raise TenantQuotaExceededError(
                    f"API rate limit exceeded for tenant {tenant_id}"
                )
            quota.api_calls_this_hour += amount

        self._save_tenants()
        return True

    def reset_hourly_usage(self):
        """Reset hourly usage counters (call from scheduler)"""
        for tenant in self._tenants.values():
            tenant.quota.api_calls_this_hour = 0
        self._save_tenants()

    def reset_monthly_usage(self):
        """Reset monthly usage counters (call from scheduler)"""
        for tenant in self._tenants.values():
            tenant.quota.jobs_this_month = 0
        self._save_tenants()

    def upgrade_tier(self, tenant_id: str, new_tier: TenantTier) -> Optional[Tenant]:
        """Upgrade tenant to a new tier"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return None

        old_tier = tenant.tier
        tenant.tier = new_tier

        # Update quota limits (keep current usage)
        new_quota = TIER_QUOTAS.get(new_tier, TIER_QUOTAS[TenantTier.FREE])
        tenant.quota.max_jobs_per_month = new_quota.max_jobs_per_month
        tenant.quota.max_storage_gb = new_quota.max_storage_gb
        tenant.quota.max_users = new_quota.max_users
        tenant.quota.max_api_calls_per_hour = new_quota.max_api_calls_per_hour
        tenant.quota.max_batch_size = new_quota.max_batch_size
        tenant.quota.retention_days = new_quota.retention_days

        tenant.updated_at = datetime.now()
        self._save_tenants()

        logger.info(f"Upgraded tenant {tenant_id} from {old_tier.value} to {new_tier.value}")

        return tenant

    def get_schema_name(self, tenant_id: str) -> str:
        """Get database schema name for tenant"""
        tenant = self.get_tenant(tenant_id)
        if tenant and tenant.schema_name:
            return tenant.schema_name

        if self.isolation_strategy == IsolationStrategy.SCHEMA_PER_TENANT:
            return f"tenant_{tenant_id.replace('-', '_')}"

        return "public"  # Shared schema

    def _load_tenants(self):
        """Load tenants from storage"""
        import json
        from pathlib import Path

        if not self.storage_path:
            return

        path = Path(self.storage_path)
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            for tenant_data in data.get("tenants", []):
                tenant = Tenant(
                    id=tenant_data["id"],
                    name=tenant_data["name"],
                    tier=TenantTier(tenant_data.get("tier", "free")),
                    active=tenant_data.get("active", True),
                    created_at=datetime.fromisoformat(tenant_data["created_at"]),
                    updated_at=datetime.fromisoformat(tenant_data["updated_at"]),
                    schema_name=tenant_data.get("schema_name")
                )
                self._tenants[tenant.id] = tenant

            logger.info(f"Loaded {len(self._tenants)} tenants from storage")

        except Exception as e:
            logger.error(f"Error loading tenants: {e}")

    def _save_tenants(self):
        """Save tenants to storage"""
        import json
        from pathlib import Path

        if not self.storage_path:
            return

        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "tenants": [
                    {
                        **t.to_dict(),
                        "schema_name": t.schema_name
                    }
                    for t in self._tenants.values()
                ]
            }

            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving tenants: {e}")


class TenantAwareQueryBuilder:
    """
    Query builder that automatically adds tenant filtering.

    Usage:
        builder = TenantAwareQueryBuilder("jobs")
        query = builder.select().where(status="completed").build()
        # Automatically adds WHERE tenant_id = current_tenant
    """

    def __init__(self, table_name: str):
        self.table_name = table_name
        self._select_fields = ["*"]
        self._where_clauses = []
        self._params = {}
        self._order_by = None
        self._limit = None
        self._offset = None

    def select(self, *fields) -> 'TenantAwareQueryBuilder':
        if fields:
            self._select_fields = list(fields)
        return self

    def where(self, **conditions) -> 'TenantAwareQueryBuilder':
        for key, value in conditions.items():
            self._where_clauses.append(f"{key} = :{key}")
            self._params[key] = value
        return self

    def order_by(self, field: str, desc: bool = False) -> 'TenantAwareQueryBuilder':
        self._order_by = f"{field} {'DESC' if desc else 'ASC'}"
        return self

    def limit(self, n: int) -> 'TenantAwareQueryBuilder':
        self._limit = n
        return self

    def offset(self, n: int) -> 'TenantAwareQueryBuilder':
        self._offset = n
        return self

    def build(self) -> tuple:
        """
        Build SQL query with automatic tenant filtering.

        Returns:
            Tuple of (query_string, params_dict)
        """
        tenant_id = get_current_tenant_id()
        if not tenant_id:
            raise TenantNotSetError("No tenant context set")

        # Add tenant filter
        self._where_clauses.insert(0, "tenant_id = :tenant_id")
        self._params["tenant_id"] = tenant_id

        fields = ", ".join(self._select_fields)
        query = f"SELECT {fields} FROM {self.table_name}"

        if self._where_clauses:
            query += " WHERE " + " AND ".join(self._where_clauses)

        if self._order_by:
            query += f" ORDER BY {self._order_by}"

        if self._limit:
            query += f" LIMIT {self._limit}"

        if self._offset:
            query += f" OFFSET {self._offset}"

        return query, self._params


# Middleware for FastAPI
def tenant_middleware_factory(tenant_manager: TenantManager):
    """
    Create FastAPI middleware for tenant context.

    Usage:
        app.middleware("http")(tenant_middleware_factory(tenant_manager))
    """
    async def tenant_middleware(request, call_next):
        # Extract tenant from header or token
        tenant_id = request.headers.get("X-Tenant-ID")

        if not tenant_id:
            # Try to get from JWT token
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                # Token parsing would go here
                pass

        if tenant_id:
            # Validate tenant exists and is active
            tenant = tenant_manager.get_tenant(tenant_id)
            if tenant and tenant.active:
                set_current_tenant(tenant_id)

                # Check API rate limit
                try:
                    tenant_manager.increment_usage(tenant_id, "api_calls")
                except TenantQuotaExceededError:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=429, detail="API rate limit exceeded")

        try:
            response = await call_next(request)
        finally:
            clear_tenant_context()

        return response

    return tenant_middleware


# Convenience functions
def create_tenant_manager(
    strategy: str = "shared",
    storage_path: Optional[str] = None
) -> TenantManager:
    """Create tenant manager with specified strategy"""
    strategy_enum = IsolationStrategy(strategy)
    return TenantManager(isolation_strategy=strategy_enum, storage_path=storage_path)

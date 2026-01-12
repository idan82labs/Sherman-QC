"""
Tests for Multi-Tenancy Module
"""

import pytest
from datetime import datetime
from pathlib import Path
import sys
import tempfile

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from multi_tenancy import (
    TenantTier,
    IsolationStrategy,
    TenantQuota,
    TenantSettings,
    Tenant,
    TenantContext,
    TenantManager,
    TenantAwareQueryBuilder,
    get_current_tenant_id,
    set_current_tenant,
    clear_tenant_context,
    require_tenant,
    TenantNotSetError,
    TenantQuotaExceededError,
    TenantNotFoundError,
    TIER_QUOTAS,
    create_tenant_manager,
)


class TestTenantTier:
    """Test TenantTier enum"""

    def test_tier_values(self):
        """Test tier values"""
        assert TenantTier.FREE.value == "free"
        assert TenantTier.STARTER.value == "starter"
        assert TenantTier.PROFESSIONAL.value == "professional"
        assert TenantTier.ENTERPRISE.value == "enterprise"


class TestIsolationStrategy:
    """Test IsolationStrategy enum"""

    def test_strategy_values(self):
        """Test isolation strategy values"""
        assert IsolationStrategy.SHARED_SCHEMA.value == "shared"
        assert IsolationStrategy.SCHEMA_PER_TENANT.value == "schema"
        assert IsolationStrategy.DATABASE_PER_TENANT.value == "database"


class TestTenantQuota:
    """Test TenantQuota dataclass"""

    def test_quota_creation(self):
        """Test quota creation with defaults"""
        quota = TenantQuota()
        assert quota.max_jobs_per_month == 100
        assert quota.max_storage_gb == 10.0
        assert quota.max_users == 5

    def test_quota_within_limit(self):
        """Test quota check within limit"""
        quota = TenantQuota(max_jobs_per_month=100, jobs_this_month=50)
        assert quota.is_within_quota("jobs") is True

    def test_quota_exceeded(self):
        """Test quota check when exceeded"""
        quota = TenantQuota(max_jobs_per_month=100, jobs_this_month=100)
        assert quota.is_within_quota("jobs") is False

    def test_quota_to_dict(self):
        """Test quota serialization"""
        quota = TenantQuota(
            max_jobs_per_month=500,
            jobs_this_month=100
        )
        data = quota.to_dict()

        assert data["limits"]["max_jobs_per_month"] == 500
        assert data["usage"]["jobs_this_month"] == 100
        assert data["remaining"]["jobs"] == 400


class TestTenantSettings:
    """Test TenantSettings dataclass"""

    def test_settings_defaults(self):
        """Test settings with defaults"""
        settings = TenantSettings()
        assert settings.default_tolerance_mm == 0.1
        assert settings.ai_analysis_enabled is True

    def test_settings_custom(self):
        """Test custom settings"""
        settings = TenantSettings(
            company_name="Acme Corp",
            default_tolerance_mm=0.05,
            ai_analysis_enabled=False
        )
        assert settings.company_name == "Acme Corp"
        assert settings.ai_analysis_enabled is False

    def test_settings_to_dict(self):
        """Test settings serialization"""
        settings = TenantSettings(company_name="Test Co")
        data = settings.to_dict()

        assert data["company_name"] == "Test Co"
        assert "qc" in data
        assert "features" in data


class TestTenant:
    """Test Tenant dataclass"""

    def test_tenant_creation(self):
        """Test tenant creation"""
        tenant = Tenant(
            id="tenant_123",
            name="Test Tenant",
            tier=TenantTier.STARTER
        )
        assert tenant.id == "tenant_123"
        assert tenant.tier == TenantTier.STARTER
        assert tenant.active is True

    def test_tenant_to_dict(self):
        """Test tenant serialization"""
        tenant = Tenant(
            id="tenant_456",
            name="Another Tenant",
            tier=TenantTier.PROFESSIONAL
        )
        data = tenant.to_dict()

        assert data["id"] == "tenant_456"
        assert data["tier"] == "professional"
        assert "settings" in data
        assert "quota" in data


class TestTierQuotas:
    """Test tier-based quotas"""

    def test_free_tier_limits(self):
        """Test free tier limits"""
        quota = TIER_QUOTAS[TenantTier.FREE]
        assert quota.max_jobs_per_month == 50
        assert quota.max_storage_gb == 1.0
        assert quota.max_users == 2

    def test_enterprise_tier_limits(self):
        """Test enterprise tier limits"""
        quota = TIER_QUOTAS[TenantTier.ENTERPRISE]
        assert quota.max_jobs_per_month > 100000
        assert quota.max_storage_gb >= 1000.0

    def test_tiers_increase_limits(self):
        """Test that higher tiers have higher limits"""
        free = TIER_QUOTAS[TenantTier.FREE]
        starter = TIER_QUOTAS[TenantTier.STARTER]
        pro = TIER_QUOTAS[TenantTier.PROFESSIONAL]
        enterprise = TIER_QUOTAS[TenantTier.ENTERPRISE]

        assert free.max_jobs_per_month < starter.max_jobs_per_month
        assert starter.max_jobs_per_month < pro.max_jobs_per_month
        assert pro.max_jobs_per_month < enterprise.max_jobs_per_month


class TestTenantContext:
    """Test TenantContext manager"""

    def setup_method(self):
        """Clear context before each test"""
        clear_tenant_context()

    def test_context_sets_tenant(self):
        """Test context sets current tenant"""
        with TenantContext("tenant_123"):
            assert get_current_tenant_id() == "tenant_123"

        # After context, should be cleared
        assert get_current_tenant_id() is None

    def test_nested_contexts(self):
        """Test nested tenant contexts"""
        with TenantContext("tenant_1"):
            assert get_current_tenant_id() == "tenant_1"

            with TenantContext("tenant_2"):
                assert get_current_tenant_id() == "tenant_2"

            # Should restore outer context
            assert get_current_tenant_id() == "tenant_1"

    def test_manual_set_clear(self):
        """Test manual set and clear"""
        set_current_tenant("tenant_abc")
        assert get_current_tenant_id() == "tenant_abc"

        clear_tenant_context()
        assert get_current_tenant_id() is None


class TestRequireTenantDecorator:
    """Test require_tenant decorator"""

    def setup_method(self):
        clear_tenant_context()

    def test_decorator_raises_without_tenant(self):
        """Test decorator raises when no tenant set"""
        @require_tenant
        def protected_func():
            return "success"

        with pytest.raises(TenantNotSetError):
            protected_func()

    def test_decorator_allows_with_tenant(self):
        """Test decorator allows execution with tenant"""
        @require_tenant
        def protected_func():
            return "success"

        with TenantContext("tenant_123"):
            result = protected_func()
            assert result == "success"


class TestTenantManager:
    """Test TenantManager class"""

    @pytest.fixture
    def manager(self):
        return TenantManager()

    def test_manager_creation(self, manager):
        """Test manager creation"""
        assert manager.isolation_strategy == IsolationStrategy.SHARED_SCHEMA
        assert len(manager._tenants) == 0

    def test_create_tenant(self, manager):
        """Test creating tenant"""
        tenant = manager.create_tenant("Test Tenant", TenantTier.STARTER)

        assert tenant.name == "Test Tenant"
        assert tenant.tier == TenantTier.STARTER
        assert tenant.id.startswith("tenant_")
        assert tenant.active is True

    def test_get_tenant(self, manager):
        """Test getting tenant"""
        created = manager.create_tenant("Test")
        retrieved = manager.get_tenant(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_nonexistent_tenant(self, manager):
        """Test getting nonexistent tenant"""
        tenant = manager.get_tenant("nonexistent")
        assert tenant is None

    def test_get_tenant_or_raise(self, manager):
        """Test get_tenant_or_raise"""
        with pytest.raises(TenantNotFoundError):
            manager.get_tenant_or_raise("nonexistent")

    def test_update_tenant(self, manager):
        """Test updating tenant"""
        tenant = manager.create_tenant("Original")
        manager.update_tenant(tenant.id, {"name": "Updated"})

        updated = manager.get_tenant(tenant.id)
        assert updated.name == "Updated"

    def test_delete_tenant(self, manager):
        """Test deleting tenant (soft delete)"""
        tenant = manager.create_tenant("To Delete")
        result = manager.delete_tenant(tenant.id)

        assert result is True
        deleted = manager.get_tenant(tenant.id)
        assert deleted.active is False

    def test_list_tenants(self, manager):
        """Test listing tenants"""
        manager.create_tenant("Tenant 1")
        manager.create_tenant("Tenant 2")

        tenants = manager.list_tenants()
        assert len(tenants) == 2

    def test_list_active_only(self, manager):
        """Test listing active tenants only"""
        t1 = manager.create_tenant("Active")
        t2 = manager.create_tenant("Inactive")
        manager.delete_tenant(t2.id)

        active = manager.list_tenants(active_only=True)
        all_tenants = manager.list_tenants(active_only=False)

        assert len(active) == 1
        assert len(all_tenants) == 2


class TestQuotaManagement:
    """Test quota management"""

    @pytest.fixture
    def manager(self):
        return TenantManager()

    def test_check_quota_within(self, manager):
        """Test quota check within limit"""
        tenant = manager.create_tenant("Test", TenantTier.STARTER)
        assert manager.check_quota(tenant.id, "jobs") is True

    def test_increment_usage(self, manager):
        """Test incrementing usage"""
        tenant = manager.create_tenant("Test", TenantTier.STARTER)

        manager.increment_usage(tenant.id, "jobs", 10)

        updated = manager.get_tenant(tenant.id)
        assert updated.quota.jobs_this_month == 10

    def test_quota_exceeded_error(self, manager):
        """Test quota exceeded raises error"""
        tenant = manager.create_tenant("Test", TenantTier.FREE)
        # Free tier has 50 jobs/month

        # Use up quota
        manager.increment_usage(tenant.id, "jobs", 50)

        # Try to exceed
        with pytest.raises(TenantQuotaExceededError):
            manager.increment_usage(tenant.id, "jobs", 1)

    def test_upgrade_tier(self, manager):
        """Test upgrading tier"""
        tenant = manager.create_tenant("Test", TenantTier.FREE)
        assert tenant.quota.max_jobs_per_month == 50

        manager.upgrade_tier(tenant.id, TenantTier.PROFESSIONAL)

        upgraded = manager.get_tenant(tenant.id)
        assert upgraded.tier == TenantTier.PROFESSIONAL
        assert upgraded.quota.max_jobs_per_month == 5000


class TestSchemaIsolation:
    """Test schema-based isolation"""

    def test_schema_per_tenant(self):
        """Test schema name generation for schema-per-tenant"""
        manager = TenantManager(
            isolation_strategy=IsolationStrategy.SCHEMA_PER_TENANT
        )
        tenant = manager.create_tenant("Test")

        schema = manager.get_schema_name(tenant.id)
        assert schema.startswith("tenant_")

    def test_shared_schema(self):
        """Test shared schema returns public"""
        manager = TenantManager(
            isolation_strategy=IsolationStrategy.SHARED_SCHEMA
        )
        tenant = manager.create_tenant("Test")

        schema = manager.get_schema_name(tenant.id)
        assert schema == "public"


class TestTenantAwareQueryBuilder:
    """Test TenantAwareQueryBuilder"""

    def setup_method(self):
        clear_tenant_context()

    def test_query_without_tenant_raises(self):
        """Test query building without tenant raises error"""
        builder = TenantAwareQueryBuilder("jobs")
        with pytest.raises(TenantNotSetError):
            builder.select().build()

    def test_query_adds_tenant_filter(self):
        """Test query automatically adds tenant filter"""
        with TenantContext("tenant_123"):
            builder = TenantAwareQueryBuilder("jobs")
            query, params = builder.select().build()

            assert "tenant_id = :tenant_id" in query
            assert params["tenant_id"] == "tenant_123"

    def test_query_with_conditions(self):
        """Test query with additional conditions"""
        with TenantContext("tenant_456"):
            builder = TenantAwareQueryBuilder("jobs")
            query, params = (builder
                .select("id", "status")
                .where(status="completed")
                .order_by("created_at", desc=True)
                .limit(10)
                .build())

            assert "tenant_id = :tenant_id" in query
            assert "status = :status" in query
            assert "ORDER BY created_at DESC" in query
            assert "LIMIT 10" in query
            assert params["status"] == "completed"


class TestPersistence:
    """Test tenant persistence"""

    def test_save_and_load(self):
        """Test saving and loading tenants"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            storage_path = f.name

        try:
            # Create and save
            manager1 = TenantManager(storage_path=storage_path)
            t1 = manager1.create_tenant("Tenant 1", TenantTier.STARTER)
            t2 = manager1.create_tenant("Tenant 2", TenantTier.PROFESSIONAL)

            # Load in new manager
            manager2 = TenantManager(storage_path=storage_path)

            assert len(manager2._tenants) == 2
            assert manager2.get_tenant(t1.id) is not None
            assert manager2.get_tenant(t2.id) is not None
        finally:
            Path(storage_path).unlink()


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_create_tenant_manager_shared(self):
        """Test creating manager with shared strategy"""
        manager = create_tenant_manager(strategy="shared")
        assert manager.isolation_strategy == IsolationStrategy.SHARED_SCHEMA

    def test_create_tenant_manager_schema(self):
        """Test creating manager with schema strategy"""
        manager = create_tenant_manager(strategy="schema")
        assert manager.isolation_strategy == IsolationStrategy.SCHEMA_PER_TENANT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

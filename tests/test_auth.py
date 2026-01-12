"""
Tests for Authentication Module
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from auth import (
    PasswordHasher, JWTManager, SQLiteUserManager, User,
    get_user_manager
)


class TestPasswordHasher:
    """Test password hashing"""

    def test_hash_password(self):
        """Test password hashing produces valid hash"""
        password = "testpassword123"
        hash_result = PasswordHasher.hash_password(password)

        assert hash_result is not None
        assert "$" in hash_result  # Contains salt separator
        assert len(hash_result) > 64  # Has salt + hash

    def test_verify_password_correct(self):
        """Test correct password verification"""
        password = "testpassword123"
        hash_result = PasswordHasher.hash_password(password)

        assert PasswordHasher.verify_password(password, hash_result) is True

    def test_verify_password_incorrect(self):
        """Test incorrect password verification"""
        password = "testpassword123"
        hash_result = PasswordHasher.hash_password(password)

        assert PasswordHasher.verify_password("wrongpassword", hash_result) is False

    def test_different_passwords_different_hashes(self):
        """Test that different passwords produce different hashes"""
        hash1 = PasswordHasher.hash_password("password1")
        hash2 = PasswordHasher.hash_password("password2")

        assert hash1 != hash2

    def test_same_password_different_hashes(self):
        """Test that same password produces different hashes (due to salt)"""
        password = "testpassword"
        hash1 = PasswordHasher.hash_password(password)
        hash2 = PasswordHasher.hash_password(password)

        assert hash1 != hash2  # Different salts
        assert PasswordHasher.verify_password(password, hash1)
        assert PasswordHasher.verify_password(password, hash2)


class TestJWTManager:
    """Test JWT token management"""

    @pytest.fixture
    def test_user(self):
        """Create a test user"""
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            password_hash="dummy_hash",
            role="operator",
            is_active=True,
            created_at="2026-01-12T00:00:00",
            last_login=None
        )

    def test_create_token(self, test_user):
        """Test JWT token creation"""
        token = JWTManager.create_token(test_user)

        assert token is not None
        assert token.count(".") == 2  # JWT has 3 parts

    def test_verify_valid_token(self, test_user):
        """Test verifying a valid token"""
        token = JWTManager.create_token(test_user)
        token_data = JWTManager.verify_token(token)

        assert token_data is not None
        assert token_data.user_id == test_user.id
        assert token_data.username == test_user.username
        assert token_data.role == test_user.role

    def test_verify_invalid_token(self):
        """Test verifying an invalid token"""
        token_data = JWTManager.verify_token("invalid.token.here")
        assert token_data is None

    def test_verify_tampered_token(self, test_user):
        """Test that tampered tokens are rejected"""
        token = JWTManager.create_token(test_user)

        # Tamper with the token
        parts = token.split(".")
        parts[1] = parts[1][:-5] + "xxxxx"  # Modify payload
        tampered_token = ".".join(parts)

        token_data = JWTManager.verify_token(tampered_token)
        assert token_data is None


class TestUserManager:
    """Test user management"""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        manager = SQLiteUserManager(db_path)
        yield manager

        # Cleanup
        os.unlink(db_path)

    def test_default_admin_created(self, temp_db):
        """Test that default admin user is created"""
        admin = temp_db.get_user_by_username("admin")

        assert admin is not None
        assert admin.role == "admin"

    def test_create_user(self, temp_db):
        """Test creating a new user"""
        user = temp_db.create_user(
            username="newuser",
            email="new@example.com",
            password="password123",
            role="operator"
        )

        assert user is not None
        assert user.username == "newuser"
        assert user.email == "new@example.com"
        assert user.role == "operator"
        assert user.is_active is True

    def test_create_duplicate_username(self, temp_db):
        """Test that duplicate usernames are rejected"""
        temp_db.create_user(
            username="duplicate",
            email="dup1@example.com",
            password="password123"
        )

        result = temp_db.create_user(
            username="duplicate",
            email="dup2@example.com",
            password="password123"
        )

        assert result is None

    def test_authenticate_success(self, temp_db):
        """Test successful authentication"""
        temp_db.create_user(
            username="authtest",
            email="auth@example.com",
            password="testpass123"
        )

        user = temp_db.authenticate("authtest", "testpass123")

        assert user is not None
        assert user.username == "authtest"

    def test_authenticate_wrong_password(self, temp_db):
        """Test authentication with wrong password"""
        temp_db.create_user(
            username="authtest2",
            email="auth2@example.com",
            password="testpass123"
        )

        user = temp_db.authenticate("authtest2", "wrongpassword")

        assert user is None

    def test_authenticate_nonexistent_user(self, temp_db):
        """Test authentication with non-existent user"""
        user = temp_db.authenticate("nonexistent", "password")
        assert user is None

    def test_update_password(self, temp_db):
        """Test password update"""
        user = temp_db.create_user(
            username="passtest",
            email="pass@example.com",
            password="oldpass"
        )

        success = temp_db.update_password(user.id, "newpass")
        assert success is True

        # Old password should fail
        old_auth = temp_db.authenticate("passtest", "oldpass")
        assert old_auth is None

        # New password should work
        new_auth = temp_db.authenticate("passtest", "newpass")
        assert new_auth is not None

    def test_update_user_role(self, temp_db):
        """Test updating user role"""
        user = temp_db.create_user(
            username="roletest",
            email="role@example.com",
            password="password"
        )

        updated = temp_db.update_user(user.id, role="admin")

        assert updated is not None
        assert updated.role == "admin"

    def test_delete_user(self, temp_db):
        """Test user deletion (deactivation)"""
        user = temp_db.create_user(
            username="deletetest",
            email="delete@example.com",
            password="password"
        )

        success = temp_db.delete_user(user.id)
        assert success is True

        # User should still exist but be inactive
        deleted_user = temp_db.get_user_by_id(user.id)
        assert deleted_user is not None
        assert deleted_user.is_active is False

        # Authentication should fail
        auth_result = temp_db.authenticate("deletetest", "password")
        assert auth_result is None

    def test_list_users(self, temp_db):
        """Test listing users"""
        # Create additional users
        temp_db.create_user("user1", "user1@example.com", "pass1")
        temp_db.create_user("user2", "user2@example.com", "pass2")

        users = temp_db.list_users()

        # Should include admin + 2 new users
        assert len(users) >= 3

    def test_list_users_exclude_inactive(self, temp_db):
        """Test that inactive users are excluded by default"""
        user = temp_db.create_user("inactive", "inactive@example.com", "pass")
        temp_db.delete_user(user.id)

        active_users = temp_db.list_users(include_inactive=False)
        all_users = temp_db.list_users(include_inactive=True)

        assert len(all_users) > len(active_users)


class TestUserModel:
    """Test User model"""

    def test_to_dict_excludes_password(self):
        """Test that to_dict excludes password hash"""
        user = User(
            id=1,
            username="test",
            email="test@example.com",
            password_hash="secret_hash",
            role="operator",
            is_active=True,
            created_at="2026-01-12T00:00:00",
            last_login=None
        )

        user_dict = user.to_dict()

        assert "password_hash" not in user_dict
        assert user_dict["username"] == "test"
        assert user_dict["email"] == "test@example.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

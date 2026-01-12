"""
Authentication Module for Sherman Scan QC System

Provides JWT-based authentication with user management.
Supports both SQLite (development) and PostgreSQL (production).
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
import logging
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-to-a-secure-random-string")
JWT_EXPIRATION_HOURS = int(os.environ.get("JWT_EXPIRATION_HOURS", "24"))
DATABASE_PATH = Path(__file__).parent.parent / "data" / "users.db"
DATABASE_URL = os.environ.get("DATABASE_URL", "")


@dataclass
class User:
    """User model"""
    id: int
    username: str
    email: str
    password_hash: str
    role: str  # admin, operator, viewer
    is_active: bool
    created_at: str
    last_login: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-safe dict (excludes password)"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "last_login": self.last_login
        }


@dataclass
class TokenData:
    """JWT token payload data"""
    user_id: int
    username: str
    role: str
    exp: datetime


class PasswordHasher:
    """Simple password hasher using PBKDF2"""

    ITERATIONS = 100000
    SALT_LENGTH = 32

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password with a random salt"""
        salt = secrets.token_hex(PasswordHasher.SALT_LENGTH)
        hash_bytes = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            PasswordHasher.ITERATIONS
        )
        return f"{salt}${hash_bytes.hex()}"

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        try:
            salt, stored_hash = password_hash.split('$')
            hash_bytes = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                PasswordHasher.ITERATIONS
            )
            return hash_bytes.hex() == stored_hash
        except ValueError:
            return False


class JWTManager:
    """Simple JWT implementation without external dependencies"""

    @staticmethod
    def create_token(user: User) -> str:
        """Create a JWT token for a user"""
        import base64
        import json
        import hmac

        # Header
        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header).encode()
        ).decode().rstrip('=')

        # Payload
        exp_time = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        payload = {
            "user_id": user.id,
            "username": user.username,
            "role": user.role,
            "exp": exp_time.isoformat(),
            "iat": datetime.utcnow().isoformat()
        }
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip('=')

        # Signature
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            SECRET_KEY.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip('=')

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    @staticmethod
    def verify_token(token: str) -> Optional[TokenData]:
        """Verify and decode a JWT token"""
        import base64
        import json
        import hmac

        try:
            parts = token.split('.')
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts

            # Verify signature
            message = f"{header_b64}.{payload_b64}"
            expected_sig = hmac.new(
                SECRET_KEY.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
            expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode().rstrip('=')

            if not hmac.compare_digest(signature_b64, expected_sig_b64):
                return None

            # Decode payload
            # Add padding if needed
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += '=' * padding

            payload = json.loads(base64.urlsafe_b64decode(payload_b64))

            # Check expiration
            exp = datetime.fromisoformat(payload['exp'])
            if exp < datetime.utcnow():
                return None

            return TokenData(
                user_id=payload['user_id'],
                username=payload['username'],
                role=payload['role'],
                exp=exp
            )

        except Exception as e:
            logger.warning(f"Token verification failed: {e}")
            return None


class SQLiteUserManager:
    """SQLite user manager for local development"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initialize user database schema"""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'operator',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_login TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
            """)

            # Create default admin user if no users exist
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if count == 0:
                self._create_default_admin(conn)

    def _create_default_admin(self, conn):
        """Create default admin user"""
        password_hash = PasswordHasher.hash_password("admin123")
        now = datetime.now().isoformat()

        conn.execute("""
            INSERT INTO users (username, email, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("admin", "admin@sherman-qc.local", password_hash, "admin", now))

        logger.info("Created default admin user (username: admin, password: admin123)")

    def _row_to_user(self, row) -> User:
        """Convert database row to User"""
        return User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            role=row["role"],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            last_login=row["last_login"]
        )

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: str = "operator"
    ) -> Optional[User]:
        """Create a new user"""
        password_hash = PasswordHasher.hash_password(password)
        now = datetime.now().isoformat()

        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO users (username, email, password_hash, role, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (username, email, password_hash, role, now))
                last_id = cursor.lastrowid

            # Get user after commit (outside with block)
            return self.get_user_by_id(last_id)

        except sqlite3.IntegrityError as e:
            logger.warning(f"User creation failed: {e}")
            return None

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,)
            ).fetchone()

            if row:
                return self._row_to_user(row)
            return None

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,)
            ).fetchone()

            if row:
                return self._row_to_user(row)
            return None

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user and return user if successful"""
        user = self.get_user_by_username(username)

        if not user:
            return None

        if not user.is_active:
            return None

        if not PasswordHasher.verify_password(password, user.password_hash):
            return None

        # Update last login
        self.update_last_login(user.id)

        return user

    def update_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        now = datetime.now().isoformat()

        with self.get_connection() as conn:
            conn.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (now, user_id)
            )

    def update_password(self, user_id: int, new_password: str) -> bool:
        """Update user's password"""
        password_hash = PasswordHasher.hash_password(new_password)

        with self.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (password_hash, user_id)
            )
            return cursor.rowcount > 0

    def update_user(
        self,
        user_id: int,
        email: Optional[str] = None,
        role: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> Optional[User]:
        """Update user details"""
        updates = []
        params = []

        if email is not None:
            updates.append("email = ?")
            params.append(email)

        if role is not None:
            updates.append("role = ?")
            params.append(role)

        if is_active is not None:
            updates.append("is_active = ?")
            params.append(int(is_active))

        if not updates:
            return self.get_user_by_id(user_id)

        params.append(user_id)

        with self.get_connection() as conn:
            conn.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
                params
            )

        return self.get_user_by_id(user_id)

    def list_users(self, include_inactive: bool = False) -> list:
        """List all users"""
        with self.get_connection() as conn:
            if include_inactive:
                rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM users WHERE is_active = 1 ORDER BY created_at DESC"
                ).fetchall()

            return [self._row_to_user(row) for row in rows]

    def delete_user(self, user_id: int) -> bool:
        """Delete a user (soft delete by deactivating)"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE users SET is_active = 0 WHERE id = ?",
                (user_id,)
            )
            return cursor.rowcount > 0


class PostgreSQLUserManager:
    """PostgreSQL user manager for production"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._init_db()

    @contextmanager
    def get_connection(self):
        """Get database connection"""
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL support. "
                "Install it with: pip install psycopg2-binary"
            )

        conn = psycopg2.connect(self.database_url)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initialize PostgreSQL user schema"""
        try:
            import psycopg2
        except ImportError:
            logger.warning("psycopg2 not installed, PostgreSQL support disabled")
            return

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(255) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        role VARCHAR(50) NOT NULL DEFAULT 'operator',
                        is_active BOOLEAN NOT NULL DEFAULT TRUE,
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                        last_login TIMESTAMP WITH TIME ZONE
                    )
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
                """)

                # Create default admin if no users exist
                cur.execute("SELECT COUNT(*) FROM users")
                count = cur.fetchone()[0]
                if count == 0:
                    self._create_default_admin(cur)

        logger.info("PostgreSQL users database initialized")

    def _create_default_admin(self, cur):
        """Create default admin user"""
        password_hash = PasswordHasher.hash_password("admin123")

        cur.execute("""
            INSERT INTO users (username, email, password_hash, role)
            VALUES (%s, %s, %s, %s)
        """, ("admin", "admin@sherman-qc.local", password_hash, "admin"))

        logger.info("Created default admin user (username: admin, password: admin123)")

    def _row_to_user(self, row: dict) -> User:
        """Convert database row to User"""
        return User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            role=row["role"],
            is_active=row["is_active"],
            created_at=str(row["created_at"]) if row["created_at"] else None,
            last_login=str(row["last_login"]) if row["last_login"] else None
        )

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: str = "operator"
    ) -> Optional[User]:
        """Create a new user"""
        from psycopg2.extras import RealDictCursor

        password_hash = PasswordHasher.hash_password(password)

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        INSERT INTO users (username, email, password_hash, role)
                        VALUES (%s, %s, %s, %s)
                        RETURNING *
                    """, (username, email, password_hash, role))
                    row = cur.fetchone()
                    return self._row_to_user(dict(row))

        except Exception as e:
            logger.warning(f"User creation failed: {e}")
            return None

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        from psycopg2.extras import RealDictCursor

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                row = cur.fetchone()
                if row:
                    return self._row_to_user(dict(row))
                return None

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        from psycopg2.extras import RealDictCursor

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM users WHERE username = %s", (username,))
                row = cur.fetchone()
                if row:
                    return self._row_to_user(dict(row))
                return None

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user and return user if successful"""
        user = self.get_user_by_username(username)

        if not user:
            return None

        if not user.is_active:
            return None

        if not PasswordHasher.verify_password(password, user.password_hash):
            return None

        # Update last login
        self.update_last_login(user.id)

        return user

    def update_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET last_login = NOW() WHERE id = %s",
                    (user_id,)
                )

    def update_password(self, user_id: int, new_password: str) -> bool:
        """Update user's password"""
        password_hash = PasswordHasher.hash_password(new_password)

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET password_hash = %s WHERE id = %s",
                    (password_hash, user_id)
                )
                return cur.rowcount > 0

    def update_user(
        self,
        user_id: int,
        email: Optional[str] = None,
        role: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> Optional[User]:
        """Update user details"""
        updates = []
        params = []

        if email is not None:
            updates.append("email = %s")
            params.append(email)

        if role is not None:
            updates.append("role = %s")
            params.append(role)

        if is_active is not None:
            updates.append("is_active = %s")
            params.append(is_active)

        if not updates:
            return self.get_user_by_id(user_id)

        params.append(user_id)

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE users SET {', '.join(updates)} WHERE id = %s",
                    params
                )

        return self.get_user_by_id(user_id)

    def list_users(self, include_inactive: bool = False) -> list:
        """List all users"""
        from psycopg2.extras import RealDictCursor

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if include_inactive:
                    cur.execute("SELECT * FROM users ORDER BY created_at DESC")
                else:
                    cur.execute(
                        "SELECT * FROM users WHERE is_active = TRUE ORDER BY created_at DESC"
                    )
                rows = cur.fetchall()
                return [self._row_to_user(dict(row)) for row in rows]

    def delete_user(self, user_id: int) -> bool:
        """Delete a user (soft delete by deactivating)"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET is_active = FALSE WHERE id = %s",
                    (user_id,)
                )
                return cur.rowcount > 0


# Type alias for user manager
UserManager = Union[SQLiteUserManager, PostgreSQLUserManager]


def create_user_manager(database_url: Optional[str] = None) -> UserManager:
    """
    Factory function to create appropriate user manager.

    Args:
        database_url: Optional database URL. If not provided, uses DATABASE_URL env var.
                     If no URL is configured, defaults to SQLite.

    Returns:
        Appropriate user manager instance
    """
    url = database_url or DATABASE_URL

    if url and url.startswith("postgresql"):
        logger.info("Using PostgreSQL user database")
        return PostgreSQLUserManager(url)
    else:
        logger.info("Using SQLite user database")
        return SQLiteUserManager()


# Global instances
_user_manager: Optional[UserManager] = None


def get_user_manager() -> UserManager:
    """Get the global user manager instance"""
    global _user_manager
    if _user_manager is None:
        _user_manager = create_user_manager()
    return _user_manager


def init_user_manager(database_url: Optional[str] = None) -> UserManager:
    """Initialize and return the global user manager"""
    global _user_manager
    _user_manager = create_user_manager(database_url)
    return _user_manager


# FastAPI dependency for authentication
async def get_current_user(authorization: str = None) -> Optional[User]:
    """
    FastAPI dependency to get current authenticated user.

    Usage:
        @app.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"message": f"Hello {user.username}"}
    """
    if not authorization:
        return None

    # Extract token from "Bearer <token>"
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization

    token_data = JWTManager.verify_token(token)
    if not token_data:
        return None

    user_manager = get_user_manager()
    return user_manager.get_user_by_id(token_data.user_id)


def require_role(*roles: str):
    """
    Decorator to require specific roles.

    Usage:
        @app.get("/admin-only")
        @require_role("admin")
        async def admin_route(user: User = Depends(get_current_user)):
            ...
    """
    def decorator(func):
        async def wrapper(*args, user: User = None, **kwargs):
            if not user:
                from fastapi import HTTPException
                raise HTTPException(401, "Authentication required")

            if user.role not in roles:
                from fastapi import HTTPException
                raise HTTPException(403, f"Role {user.role} not authorized")

            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator

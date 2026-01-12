"""
Configuration Module for Sherman Scan QC System

Loads configuration from environment variables with sensible defaults.
Supports .env files for development.
"""

import os
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field
import logging

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)


def get_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment"""
    value = os.environ.get(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_int(key: str, default: int) -> int:
    """Get integer from environment"""
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


def get_float(key: str, default: float) -> float:
    """Get float from environment"""
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return default


def get_list(key: str, default: str = "") -> List[str]:
    """Get comma-separated list from environment"""
    value = os.environ.get(key, default)
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = get_int("PORT", 8080)
    workers: int = get_int("WORKERS", 4)
    request_timeout: int = get_int("REQUEST_TIMEOUT", 300)
    production: bool = get_bool("PRODUCTION", False)


@dataclass
class AIConfig:
    """AI provider configuration"""
    anthropic_api_key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = os.environ.get("GOOGLE_API_KEY")
    openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")

    provider: str = os.environ.get("AI_PROVIDER", "claude")

    claude_model: str = os.environ.get("CLAUDE_MODEL", "claude-opus-4-5-20251101")
    gemini_model: str = os.environ.get("GEMINI_MODEL", "gemini-3-pro")
    openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-5.2")

    @property
    def is_enabled(self) -> bool:
        """Check if any AI provider is configured"""
        return bool(
            self.anthropic_api_key or
            self.google_api_key or
            self.openai_api_key
        )

    @property
    def available_providers(self) -> List[str]:
        """Get list of configured providers"""
        providers = []
        if self.anthropic_api_key:
            providers.append("claude")
        if self.google_api_key:
            providers.append("gemini")
        if self.openai_api_key:
            providers.append("openai")
        return providers


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = os.environ.get("DATABASE_URL", "")

    @property
    def is_postgres(self) -> bool:
        return self.url.startswith("postgresql://")

    @property
    def is_sqlite(self) -> bool:
        return not self.url or self.url.startswith("sqlite://")


@dataclass
class StorageConfig:
    """Storage configuration"""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    upload_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("UPLOAD_DIR", "./uploads")
    ))
    output_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("OUTPUT_DIR", "./output")
    ))
    data_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("DATA_DIR", "./data")
    ))

    max_upload_size_mb: int = get_int("MAX_UPLOAD_SIZE_MB", 100)

    def __post_init__(self):
        # Convert relative paths to absolute
        if not self.upload_dir.is_absolute():
            self.upload_dir = self.base_dir / self.upload_dir
        if not self.output_dir.is_absolute():
            self.output_dir = self.base_dir / self.output_dir
        if not self.data_dir.is_absolute():
            self.data_dir = self.base_dir / self.data_dir

        # Create directories
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = os.environ.get("SECRET_KEY", "change-this-to-a-secure-random-string")
    jwt_expiration_hours: int = get_int("JWT_EXPIRATION_HOURS", 24)
    cors_origins: List[str] = field(default_factory=lambda: get_list("CORS_ORIGINS", "*"))


@dataclass
class QCConfig:
    """QC analysis defaults"""
    default_tolerance_mm: float = get_float("DEFAULT_TOLERANCE_MM", 0.1)
    default_material: str = os.environ.get("DEFAULT_MATERIAL", "Al-5053-H32")
    min_region_points: int = get_int("MIN_REGION_POINTS", 100)


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = os.environ.get("LOG_LEVEL", "INFO")
    log_file: Optional[str] = os.environ.get("LOG_FILE")

    def setup(self):
        """Setup logging based on configuration"""
        log_level = getattr(logging, self.level.upper(), logging.INFO)

        handlers = [logging.StreamHandler()]

        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(str(log_path)))

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers
        )


@dataclass
class Settings:
    """Main settings container"""
    server: ServerConfig = field(default_factory=ServerConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        # Setup logging
        self.logging.setup()

        # Log configuration summary
        logger.info(f"Server: {self.server.host}:{self.server.port}")
        logger.info(f"AI Providers: {self.ai.available_providers or ['none (rule-based only)']}")
        logger.info(f"Database: {'PostgreSQL' if self.database.is_postgres else 'SQLite'}")
        logger.info(f"Production Mode: {self.server.production}")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment"""
    global _settings
    _settings = Settings()
    return _settings


# Convenience exports
settings = get_settings()

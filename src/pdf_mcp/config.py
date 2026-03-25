"""Configuration via environment variables using pydantic-settings."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PDF_MCP_", env_file=".env", extra="ignore"
    )

    # Vault (PDF directory)
    vault_path: Path = Path("~/Documents/PDFs")

    # Database
    db_path: Path = Path("~/.local/share/pdf-mcp/pdf.db")

    # Server
    transport: Literal["stdio", "http"] = "stdio"
    host: str = "0.0.0.0"
    port: int = 10201
    mcp_path: str = "/mcp"
    log_level: str = "INFO"

    # Auth (optional, for HTTP transport)
    github_client_id: str | None = None
    github_client_secret: str | None = None
    oauth_base_url: str | None = None
    oauth_allowed_users: str | None = None
    oauth_state_dir: Path | None = None

    @property
    def vault(self) -> Path:
        return self.vault_path.expanduser().resolve()

    @property
    def database_path(self) -> Path:
        return self.db_path.expanduser().resolve()

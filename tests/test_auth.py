"""OAuth hardening: only allowlisted GitHub users get in, and issued tokens
survive a restart.

The server is public (Tailscale Funnel) behind GitHub OAuth, and GitHub will
sign anyone in — the allowlist is the real protection.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from pdf_mcp.config import Settings


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def _ctx(login: str | None):
    token = None if login is None else SimpleNamespace(claims={"login": login})
    return SimpleNamespace(token=token)


class TestAllowlist:
    def _check(self):
        from pdf_mcp.server import _build_middleware

        [mw] = _build_middleware(
            _settings(github_client_id="id", github_client_secret="s", oauth_allowed_users="jkp, other")
        )
        return mw.auth if callable(mw.auth) else mw.auth[0]

    def test_allowlisted_user_gets_in(self) -> None:
        assert self._check()(_ctx("jkp")) is True

    def test_any_other_github_user_is_refused(self) -> None:
        assert self._check()(_ctx("mallory")) is False

    def test_no_token_is_refused(self) -> None:
        assert self._check()(_ctx(None)) is False

    def test_oauth_without_allowlist_refuses_to_start(self) -> None:
        """Fail closed: GitHub OAuth with no allowlist would admit every GitHub user."""
        from pdf_mcp.server import _build_middleware

        with pytest.raises(RuntimeError, match="OAUTH_ALLOWED_USERS"):
            _build_middleware(_settings(github_client_id="id", github_client_secret="s"))

    def test_no_oauth_needs_no_allowlist(self) -> None:
        from pdf_mcp.server import _build_middleware

        assert _build_middleware(_settings()) == []


class TestOAuthStorage:
    def test_none_without_state_dir(self) -> None:
        from pdf_mcp.server import _build_auth_storage

        assert _build_auth_storage(_settings()) is None

    def test_persists_under_state_dir(self, tmp_path: Path) -> None:
        from pdf_mcp.server import _build_auth_storage

        state = tmp_path / "oauth"
        assert _build_auth_storage(_settings(oauth_state_dir=state)) is not None
        assert state.is_dir()

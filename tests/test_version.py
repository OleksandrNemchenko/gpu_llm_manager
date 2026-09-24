"""§6.7 SPEC-GPU-002: GET /api/version і gpu_manager.__version__.

Маршрут 7 таблиці §6 від models не залежить (з models з'являються лише маршрути 1–6), тож основні
перевірки йдуть на застосунку фази 1 без ModelStore (фікстура env). Чи є коміти, тест дізнається з
git для теки пакета gpu_manager — лише читанням (rev-parse), нічого не змінюючи.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from .conftest import ok_json

pytestmark = pytest.mark.usefixtures("isolated_home")

VERSION_FIELDS = {"version", "commit", "date", "dirty"}


def _package_version() -> object:
    """gpu_manager.__version__; число версії тести не фіксують — його піднімає людина (§6.7)."""
    import gpu_manager

    return getattr(gpu_manager, "__version__", None)


def _has_commits() -> bool | None:
    """Чи має git-репозиторій теки пакета gpu_manager хоч один коміт; None — git недоступний."""
    import gpu_manager

    git = shutil.which("git")
    if git is None:
        return None
    package_dir = Path(gpu_manager.__file__).resolve().parent
    result = subprocess.run(
        [git, "-C", str(package_dir), "rev-parse", "--verify", "--quiet", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.unit
@pytest.mark.req("SPEC-GPU-002 §6.7")
def test_package_version_is_non_empty_string():
    """§6.7: gpu_manager.__version__ — непорожній рядок (саме число тести не фіксують)."""
    got = _package_version()
    assert isinstance(got, str) and got.strip(), f"gpu_manager.__version__: expected a non-empty string, got {got!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-002 §6.7")
def test_version_fields(env):
    """§6.7: GET /api/version → {version, commit, date, dirty}."""
    with env.client() as c:
        body = ok_json(c.get("/api/version"), "GET /api/version")
    keys = set(body) if isinstance(body, dict) else body
    assert keys == VERSION_FIELDS, f"/api/version: expected keys {sorted(VERSION_FIELDS)}, got {keys!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-002 §6.7")
def test_version_value_is_package_version(env):
    """§6.7: version = gpu_manager.__version__."""
    expected = _package_version()
    with env.client() as c:
        body = ok_json(c.get("/api/version"), "GET /api/version")
    got = body.get("version")
    assert got == expected, f"/api/version version: expected gpu_manager.__version__ {expected!r}, got {got!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-002 §6.7")
def test_version_without_commits_is_null(env):
    """§6.7: без комітів commit, date, dirty — null."""
    has_commits = _has_commits()
    if has_commits is None:
        pytest.skip("git is not installed: cannot tell whether the repository has commits")
    if has_commits:
        pytest.skip("the repository of gpu_manager has commits: the 'no commits' branch of §6.7 does not apply")
    with env.client() as c:
        body = ok_json(c.get("/api/version"), "GET /api/version")
    got = {k: body.get(k, "<missing>") for k in ("commit", "date", "dirty")}
    assert got == {"commit": None, "date": None, "dirty": None}, f"/api/version without commits: expected nulls, got {got!r}"


@pytest.mark.e2e
@pytest.mark.req("SPEC-GPU-002 §6.7")
@pytest.mark.req("SPEC-GPU-002 §6")
def test_version_available_with_models(models_env):
    """§6: /api/version є й у застосунку з models."""
    with models_env.client() as c:
        body = ok_json(c.get("/api/version"), "GET /api/version with models")
    expected = _package_version()
    assert body.get("version") == expected, f"/api/version with models: expected gpu_manager.__version__ {expected!r}, got {body!r}"

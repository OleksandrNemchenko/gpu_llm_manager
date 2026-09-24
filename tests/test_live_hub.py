"""§4 SPEC-GPU-002: справжній gpu_manager.hub.HubClient проти HuggingFace — без завантаження ваг.

Навіщо: решта тестів ходить через підроблений hub; цей модуль один раз перевіряє справжній клієнт у
мережі. Запускається лише вручну, за GPU_MANAGER_LIVE=1 (§10); інакше — пропуск. Моделі — невеликі
публічні (openai-community/gpt2) і відома gated (meta-llama/Llama-3.2-1B); завантажується лише
метадані й config.json.

Перевірки, що залежать від облікових даних HF або від запису в кеш, ідуть в окремому процесі Python з
HF_HOME у tmp_path і без HF_TOKEN: так збережений на машині токен не впливає на результат, а справжній
кеш HF не зачіпається.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from .model_fakes import expect_manager_error

LIVE = os.environ.get("GPU_MANAGER_LIVE") == "1"

pytestmark = [
    pytest.mark.component,
    pytest.mark.skipif(not LIVE, reason="live HuggingFace test: set GPU_MANAGER_LIVE=1 to run"),
]

PUBLIC_REPO = "openai-community/gpt2"  # публічна, мала, має model.safetensors і старі формати ваг
GATED_PUBLIC_REPO = "meta-llama/Llama-3.2-1B"  # gated: без токена доступу немає
MISSING_REPO = "gpu-manager-spec-tests/definitely-not-a-model-0f3a"
SEARCH_FIELDS = {"repo", "downloads", "likes", "gated", "task", "params", "updated"}
EXCLUDED_SUFFIXES = (".bin", ".h5", ".msgpack", ".ot", ".tflite", ".onnx", ".gguf", ".pth", ".pt", ".mlmodel")
HF_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_OFFLINE", "HF_TOKEN_PATH")


@pytest.fixture(scope="module")
def client() -> Any:
    from gpu_manager.hub import HubClient

    return HubClient(lambda: None)


def _run_isolated(script: str, hf_home: Path) -> dict[str, Any]:
    """Виконує script в окремому Python з HF_HOME=hf_home і без токенів HF; повертає JSON останнього рядка."""
    import gpu_manager

    repo_root = Path(gpu_manager.__file__).resolve().parent.parent
    env = {k: v for k, v in os.environ.items() if k not in HF_ENV_KEYS}
    env["HF_HOME"] = str(hf_home)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=str(repo_root), env=env, capture_output=True, text=True, check=False, timeout=120
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert result.returncode == 0 and lines, (
        f"isolated script failed: exit {result.returncode}, stdout {result.stdout[-400:]!r}, stderr {result.stderr[-400:]!r}"
    )
    return json.loads(lines[-1])


@pytest.mark.req("SPEC-GPU-002 §4")
def test_live_search_shape_and_popularity_order(client):
    """§4: search(query, limit) — не більше limit результатів з полями §4, за популярністю (downloads спадають)."""
    items = client.search("gpt2", 5)
    downloads = [item.get("downloads") for item in items]
    ok = (
        isinstance(items, list)
        and 1 <= len(items) <= 5
        and all(SEARCH_FIELDS <= set(item) for item in items)
        and downloads == sorted(downloads, reverse=True)
    )
    assert ok, f"HubClient.search('gpt2', 5): expected 1..5 items with {sorted(SEARCH_FIELDS)} by downloads desc, got {items!r}"


@pytest.mark.req("SPEC-GPU-002 §4")
@pytest.mark.req("SPEC-GPU-002 §3.1")
def test_live_files_public_repo(client):
    """§4: files(repo, revision) → (sha, {файл: розмір} після select_files, gated=false)."""
    sha, files, gated = client.files(PUBLIC_REPO, None)
    bad = sorted(name for name in files if name.endswith(EXCLUDED_SUFFIXES))
    sizes_ok = all(isinstance(size, int) and size > 0 for size in files.values())
    ok = (
        isinstance(sha, str)
        and len(sha) == 40
        and "config.json" in files
        and any(name.endswith(".safetensors") for name in files)
        and not bad
        and sizes_ok
        and gated is False
    )
    assert ok, f"HubClient.files({PUBLIC_REPO!r}): expected 40-hex sha, selected files with sizes, gated False; got sha {sha!r}, gated {gated!r}, excluded left {bad}, files {files!r}"


@pytest.mark.req("SPEC-GPU-002 §4")
@pytest.mark.req("SPEC-GPU-002 §3.1")
def test_live_files_already_selected(client):
    """§4: список files() уже пропущено через select_files — повторний відбір його не змінює."""
    from gpu_manager.hub import select_files

    _, files, _ = client.files(PUBLIC_REPO, None)
    again = select_files(list(files))
    assert again == sorted(files), f"select_files over files(): expected unchanged {sorted(files)!r}, got {again!r}"


@pytest.mark.req("SPEC-GPU-002 §4")
def test_live_missing_repo_not_found(client):
    """§4: немає моделі → ManagerError hf_not_found."""
    expect_manager_error("hf_not_found", client.files, MISSING_REPO, None)


@pytest.mark.req("SPEC-GPU-002 §4")
def test_live_config_without_writing_cache(tmp_path):
    """§4: config(repo, revision) → config.json як dict, без запису в кеш HF."""
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    script = (
        "import json\n"
        "from gpu_manager.hub import HubClient\n"
        f"cfg = HubClient(lambda: None).config({PUBLIC_REPO!r}, None)\n"
        "print(json.dumps({'is_dict': isinstance(cfg, dict), 'model_type': cfg.get('model_type') if isinstance(cfg, dict) else None}))\n"
    )
    out = _run_isolated(script, hf_home)
    cached = sorted(str(p.relative_to(hf_home)) for p in hf_home.rglob("*") if "models--" in str(p))
    assert out == {"is_dict": True, "model_type": "gpt2"} and cached == [], (
        f"HubClient.config({PUBLIC_REPO!r}): expected a dict with model_type 'gpt2' and nothing cached, got {out!r}, cached {cached}"
    )


@pytest.mark.req("SPEC-GPU-002 §4")
def test_live_gated_without_token_refused(tmp_path):
    """§4: gated-модель без доступу (токена немає) → check_access піднімає ManagerError hf_gated."""
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    script = (
        "import json\n"
        "from gpu_manager.hub import HubClient\n"
        "from gpu_manager.messages import ManagerError\n"
        "try:\n"
        f"    HubClient(lambda: None).check_access({GATED_PUBLIC_REPO!r})\n"
        "    print(json.dumps({'code': None}))\n"
        "except ManagerError as exc:\n"
        "    print(json.dumps({'code': getattr(exc, 'code', None)}))\n"
    )
    out = _run_isolated(script, hf_home)
    assert out == {"code": "hf_gated"}, f"check_access({GATED_PUBLIC_REPO!r}) without a token: expected hf_gated, got {out!r}"

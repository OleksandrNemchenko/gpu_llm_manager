"""§6 SPEC-GPU-002 (маршрути 1–6): HTTP API моделей через build_app(cfg, manager, models=store).

Захист і формат відмов — як у SPEC-GPU-001 §7 (Host → 421, Content-Type → 415, Origin → 403; відмова —
HTTP 400 {"code", "error"}, текст українською). Усі запити — через starlette TestClient у режимі
контекстного менеджера (lifespan працює). Сервіс сам викликає poll() кожні gpu.sample_interval_s і
stop_all() при зупинці (§9); підроблені процеси самі не завершуються, тож фонове опитування стану
черги під час тесту не змінює.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

import anyio
import pytest

from .conftest import BASE_URL, STRANGER, has_cyrillic, ok_json, refusal
from .model_fakes import (
    CARD_MIB,
    GATED_REPO,
    HUGE_RESERVE_GIB,
    MIB,
    MISSING_REPO,
    NO_WEIGHTS_FILES,
    NO_WEIGHTS_REPO,
    REPO,
    REPO_B,
    gib_close,
    put_snapshot,
)

pytestmark = [pytest.mark.e2e, pytest.mark.usefixtures("isolated_home")]

SEARCH = "/api/models/search"
INFO = "/api/models/info"
MODELS = "/api/models"
DOWNLOAD = "/api/models/download"
CANCEL = "/api/models/cancel"
DELETE = "/api/models/delete"
POST_ROUTES = [DOWNLOAD, CANCEL, DELETE]
INFO_FIELDS = {"repo", "revision", "gated", "files", "size_gib", "fraction", "fit", "local"}
LOCAL_SIZES = {"model-00001-of-00002.safetensors": 768 * MIB, "model-00002-of-00002.safetensors": 256 * MIB}


def _put_local(env: Any, repo: str = REPO_B) -> None:
    """Локальна модель ≈ 1 GiB (розріджені файли знімка)."""
    put_snapshot(env.hf_home, repo, ["config.json", *LOCAL_SIZES], sizes=LOCAL_SIZES)


def _download(client: Any, repo: str = REPO, user: str = "alice") -> Any:
    return client.post(DOWNLOAD, json={"repo": repo, "user": user})


# --- 1. search ------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §6.1")
def test_http_search_returns_hub_results(models_env):
    """§6.1: GET /api/models/search?q=&limit= — як search: результати hub у його порядку."""
    env = models_env
    with env.client() as c:
        body = ok_json(c.get(SEARCH, params={"q": "llama", "limit": 3}), "GET /api/models/search")
    got = [item.get("repo") for item in body] if isinstance(body, list) else body
    expected = [item["repo"] for item in env.hub.pool[:3]]
    assert got == expected, f"search?limit=3: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §6.1")
@pytest.mark.req("SPEC-GPU-002 §5.4.4")
def test_http_search_limit_clamped(models_env):
    """§6.1, §5.4.4: limit понад 50 обмежується до 50."""
    with models_env.client() as c:
        body = ok_json(c.get(SEARCH, params={"q": "llama", "limit": 500}), "GET /api/models/search")
    assert isinstance(body, list) and len(body) == 50, f"search?limit=500: expected 50 results, got {len(body) if isinstance(body, list) else body!r}"


# --- 2. info -----------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §6.2")
def test_http_info_fields(models_env):
    """§6.2: GET /api/models/info?repo= — як info."""
    with models_env.client() as c:
        body = ok_json(c.get(INFO, params={"repo": REPO}), "GET /api/models/info")
    missing = INFO_FIELDS - set(body)
    assert not missing and body.get("repo") == REPO, f"info over HTTP: expected fields {sorted(INFO_FIELDS)} for {REPO!r}, got {body!r}"


@pytest.mark.req("SPEC-GPU-002 §6.2")
def test_http_info_fraction_parameter(models_env):
    """§6.2: параметр fraction доходить до info."""
    with models_env.client() as c:
        body = ok_json(c.get(INFO, params={"repo": REPO, "fraction": 0.5}), "GET /api/models/info")
    assert body.get("fraction") == 0.5, f"info?fraction=0.5: expected fraction 0.5, got {body.get('fraction')!r}"


@pytest.mark.req("SPEC-GPU-002 §6.2")
def test_http_info_per_card_keyed_by_card_size(models_env):
    """§6.2, §3: fit.per_card у JSON — за обсягом карти (ключ "46068")."""
    with models_env.client() as c:
        body = ok_json(c.get(INFO, params={"repo": REPO}), "GET /api/models/info")
    per_card = (body.get("fit") or {}).get("per_card") or {}
    assert str(CARD_MIB) in per_card, f"info fit.per_card: expected key {str(CARD_MIB)!r}, got {sorted(per_card)!r}"


@pytest.mark.req("SPEC-GPU-002 §6.2")
@pytest.mark.req("SPEC-GPU-002 §5.4.3")
@pytest.mark.parametrize("fraction", [0, 1.5])
def test_http_info_bad_fraction(models_env, fraction):
    """§6.2, §5.4.3: fraction поза (0, 1] → 400 bad_fraction."""
    with models_env.client() as c:
        refusal(c.get(INFO, params={"repo": REPO, "fraction": fraction}), "bad_fraction")


@pytest.mark.req("SPEC-GPU-002 §6.2")
@pytest.mark.req("SPEC-GPU-002 §9")
def test_http_info_without_repo_is_bad_request(models_env):
    """§6.2, §9: немає repo в запиті HTTP → 400 bad_request."""
    with models_env.client() as c:
        refusal(c.get(INFO), "bad_request")


# --- 3. список ------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §6.3")
def test_http_models_lists_local_and_downloads(models_env):
    """§6.3: GET /api/models → {local: [...], downloads: [...]}."""
    env = models_env
    _put_local(env, REPO_B)
    with env.client() as c:
        ok_json(_download(c), "POST /api/models/download")
        body = ok_json(c.get(MODELS), "GET /api/models")
    local = [m.get("repo") for m in body.get("local", [])]
    downloads = [d.get("repo") for d in body.get("downloads", [])]
    assert local == [REPO_B] and downloads == [REPO], (
        f"/api/models: expected local [{REPO_B!r}] and downloads [{REPO!r}], got local {local!r}, downloads {downloads!r}"
    )


# --- 4. download ---------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §6.4")
def test_http_download_returns_state(models_env):
    """§6.4: POST /api/models/download {repo, user} (revision необов'язкова) → стан завантаження."""
    env = models_env
    with env.client() as c:
        body = ok_json(_download(c), "POST /api/models/download")
    got = (body.get("repo"), body.get("user"), body.get("status"), len(env.spawn.for_repo(REPO)))
    assert got == (REPO, "alice", "downloading", 1), f"download over HTTP: expected ({REPO!r}, 'alice', 'downloading', 1 process), got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §6.4")
@pytest.mark.req("SPEC-GPU-002 §5.1.6")
def test_http_download_visible_in_journal(models_env):
    """§6.4, §5.1.6: дія download видна в спільному журналі GET /api/journal."""
    with models_env.client() as c:
        ok_json(_download(c), "POST /api/models/download")
        entries = ok_json(c.get("/api/journal"), "GET /api/journal")
    got = [(e.get("action"), e.get("repo")) for e in entries]
    assert ("download", REPO) in got, f"/api/journal: expected a ('download', {REPO!r}) entry, got {got!r}"


@pytest.mark.req("SPEC-GPU-002 §6.4")
@pytest.mark.req("SPEC-GPU-002 §9")
@pytest.mark.parametrize("body", [{"user": "alice"}, {"repo": REPO}], ids=["no-repo", "no-user"])
@pytest.mark.parametrize("path", POST_ROUTES)
def test_http_models_post_missing_field_is_bad_request(models_env, path, body):
    """§6.4–6.6 (SPEC-GPU-001 §7.2): відсутнє обов'язкове поле → 400 bad_request."""
    with models_env.client() as c:
        refusal(c.post(path, json=body), "bad_request")


@pytest.mark.req("SPEC-GPU-002 §6.4")
@pytest.mark.parametrize("path", POST_ROUTES)
def test_http_models_post_malformed_json_is_bad_request(models_env, path):
    """§6.4–6.6 (SPEC-GPU-001 §7.2): некоректний JSON → 400 bad_request."""
    with models_env.client() as c:
        refusal(c.post(path, content=b'{"repo": ', headers={"Content-Type": "application/json"}), "bad_request")


# --- 5. cancel -----------------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §6.5")
def test_http_cancel_returns_state(models_env):
    """§6.5: POST /api/models/cancel {repo, user} → стан завантаження (cancelled)."""
    with models_env.client() as c:
        ok_json(_download(c), "POST /api/models/download")
        body = ok_json(c.post(CANCEL, json={"repo": REPO, "user": "alice"}), "POST /api/models/cancel")
    got = (body.get("repo"), body.get("status"))
    assert got == (REPO, "cancelled"), f"cancel over HTTP: expected ({REPO!r}, 'cancelled'), got {body!r}"


# --- 6. delete ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §6.6")
def test_http_delete_returns_result(models_env):
    """§6.6: POST /api/models/delete {repo, user} → {deleted, repo, freed_gib}."""
    env = models_env
    _put_local(env, REPO_B)
    with env.client() as c:
        body = ok_json(c.post(DELETE, json={"repo": REPO_B, "user": "bob"}), "POST /api/models/delete")
    ok = body.get("deleted") is True and body.get("repo") == REPO_B and gib_close(body.get("freed_gib"), sum(LOCAL_SIZES.values()))
    assert ok, f"delete over HTTP: expected {{'deleted': True, 'repo': {REPO_B!r}, 'freed_gib': ≈1.0}}, got {body!r}"


# --- Зупинка сервісу (§9) ---------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §9")
@pytest.mark.req("SPEC-GPU-002 §5.1.10")
def test_http_shutdown_stops_downloads(models_env):
    """§9: при зупинці сервіс викликає stop_all() — процес завантаження зупинено (SIGTERM або SIGKILL)."""
    env = models_env
    with env.client() as c:
        ok_json(_download(c), "POST /api/models/download")
    proc = env.spawn.last(REPO)
    assert proc.stopped, f"after service shutdown: expected the download process stopped, got events {proc.events!r}"


@pytest.mark.req("SPEC-GPU-002 §9")
@pytest.mark.req("SPEC-GPU-002 §5.1.10")
def test_http_shutdown_keeps_saved_downloads(models_env):
    """§9, §5.1.10: зупинка сервісу не змінює збереженого стану — після перезапуску завантаження триває."""
    env = models_env
    with env.client() as c:
        ok_json(_download(c), "POST /api/models/download")
    restarted = env.restart()
    restarted.store.poll()
    got = (restarted.status(REPO), restarted.spawn.repos)
    assert got == ("downloading", [REPO]), f"after shutdown and restart: expected ('downloading', [{REPO!r}]), got {got!r}"


# --- Відмови (§8, формат — SPEC-GPU-001 §7.2) ---------------------------------------------------------------------------------------


def _delete_active(env: Any, c: Any) -> Any:
    ok_json(_download(c), "POST /api/models/download")
    put_snapshot(env.hf_home, REPO, ["config.json"])
    return c.post(DELETE, json={"repo": REPO, "user": "alice"})


def _search_hub_down(env: Any, c: Any) -> Any:
    env.hub.unavailable = True
    return c.get(SEARCH, params={"q": "llama"})


def _download_no_weights(env: Any, c: Any) -> Any:
    env.hub.add(NO_WEIGHTS_REPO, NO_WEIGHTS_FILES)
    return _download(c, NO_WEIGHTS_REPO)


# code → (перевизначення конфігу, дія над (env, client), що має відмовити цим кодом).
HTTP_REFUSALS: dict[str, tuple[dict[str, Any], Callable[[Any, Any], Any]]] = {
    "unknown_user": ({}, lambda env, c: _download(c, REPO, STRANGER)),
    "hf_gated": ({}, lambda env, c: _download(c, GATED_REPO)),
    "hf_not_found": ({}, lambda env, c: c.get(INFO, params={"repo": MISSING_REPO})),
    "hf_unavailable": ({}, _search_hub_down),
    "bad_fraction": ({}, lambda env, c: c.get(INFO, params={"repo": REPO, "fraction": 0})),
    "disk_full": ({"models.min_free_disk_gib": HUGE_RESERVE_GIB}, lambda env, c: _download(c)),
    "download_active": ({}, _delete_active),
    "download_not_active": ({}, lambda env, c: c.post(CANCEL, json={"repo": REPO, "user": "alice"})),
    "model_not_local": ({}, lambda env, c: c.post(DELETE, json={"repo": REPO, "user": "alice"})),
    "no_vllm_weights": ({}, _download_no_weights),
}


def _refuse(make_models_env: Any, code: str) -> Any:
    overrides, action = HTTP_REFUSALS[code]
    env = make_models_env(overrides)
    with env.client() as c:
        return action(env, c)


@pytest.mark.req("SPEC-GPU-002 §8")
@pytest.mark.req("SPEC-GPU-002 §6")
@pytest.mark.parametrize("code", sorted(HTTP_REFUSALS))
def test_http_models_refusal_is_400_with_code(make_models_env, code):
    """§6, §8: відмова — HTTP 400, у тілі code з таблиці §8."""
    refusal(_refuse(make_models_env, code), code)


@pytest.mark.req("SPEC-GPU-002 §6")
@pytest.mark.parametrize("code", sorted(HTTP_REFUSALS))
def test_http_models_refusal_body_shape(make_models_env, code):
    """§6 (SPEC-GPU-001 §7.2): тіло відмови — {"code", "error"}."""
    resp = _refuse(make_models_env, code)
    try:
        body = resp.json()
    except ValueError:
        body = resp.text[:300]
    keys = sorted(body) if isinstance(body, dict) else body
    assert keys == ["code", "error"], f"{code}: expected body keys ['code', 'error'], got {keys!r}"


@pytest.mark.req("SPEC-GPU-002 §6")
@pytest.mark.parametrize("code", sorted(HTTP_REFUSALS))
def test_http_models_refusal_text_is_ukrainian(make_models_env, code):
    """§6 (SPEC-GPU-001 §7.2): error — текст українською."""
    resp = _refuse(make_models_env, code)
    try:
        text = resp.json().get("error")
    except (ValueError, AttributeError):
        text = None
    assert isinstance(text, str) and has_cyrillic(text), f"{code}: expected a Ukrainian error text, got {text!r}"


@pytest.mark.req("SPEC-GPU-002 §5.1.4")
@pytest.mark.req("SPEC-GPU-002 §6.4")
def test_http_gated_download_starts_nothing(models_env):
    """§5.1.4, §6.4: відмова hf_gated через HTTP не запускає процесу."""
    env = models_env
    with env.client() as c:
        refusal(_download(c, GATED_REPO), "hf_gated")
    assert env.spawn.processes == [], f"after hf_gated over HTTP: expected no process, got {env.spawn.repos!r}"


# --- Маршрути не блокують сервер (§6) --------------------------------------------------------------------------------------------------

# Найдовше, скільки підроблений hub тримає запит, с. Це межа зависання тесту, а не поріг специфікації:
# сервер, що блокується, відповість на /api/overview лише після неї, і тест це покаже.
HOLD_S = 5.0


class _HeldHubCall:
    """Підміняє метод підробленого hub: виклик ставить entered і чекає release (не довше HOLD_S).

    Синхронізація — подіями, без sleep: тест відпускає hub лише після відповіді /api/overview.
    returned — метод hub уже повернувся (відпустив тест або минув HOLD_S).
    """

    def __init__(self, hub: Any, method: str) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.returned = threading.Event()
        original = getattr(hub, method)

        def held(*args: Any, **kwargs: Any) -> Any:
            self.entered.set()
            self.release.wait(HOLD_S)
            self.returned.set()
            return original(*args, **kwargs)

        setattr(hub, method, held)


async def _lifespan(app: Any, started: Any, stop: Any) -> None:
    """Протокол lifespan ASGI, як у TestClient у режимі `with`: startup, чекання stop, shutdown."""
    sent: list[str] = []

    async def receive() -> dict[str, Any]:
        if not sent:
            sent.append("startup")
            return {"type": "lifespan.startup"}
        await stop.wait()
        return {"type": "lifespan.shutdown"}

    async def send(message: dict[str, Any]) -> None:
        if message.get("type") in ("lifespan.startup.complete", "lifespan.startup.failed"):
            started.set()

    await app({"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.0"}, "state": {}}, receive, send)


def _overview_while_held(env: Any, method: str, request: Callable[[Any], Any]) -> dict[str, Any]:
    """В одному циклі подій: request чекає на hub.method, тим часом — GET /api/overview.

    Повертає entered (маршрут дійшов до hub), overview (відповідь), overview_s (скільки вона тривала, с) і
    held (hub ще тримав запит, коли /api/overview відповів).
    """
    import httpx

    from gpu_manager.app import build_app

    app = build_app(env.cfg, env.manager, models=env.store)
    hold = _HeldHubCall(env.hub, method)
    out: dict[str, Any] = {}

    async def scenario() -> None:
        started, stop, route_done = anyio.Event(), anyio.Event(), anyio.Event()
        async with anyio.create_task_group() as tg:
            tg.start_soon(_lifespan, app, started, stop)
            with anyio.fail_after(HOLD_S):
                await started.wait()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE_URL) as client:

                async def slow_route() -> None:
                    try:
                        out["route"] = await request(client)
                    finally:
                        route_done.set()

                tg.start_soon(slow_route)
                out["entered"] = await anyio.to_thread.run_sync(hold.entered.wait, HOLD_S)
                t0 = time.monotonic()
                out["overview"] = await client.get("/api/overview")
                out["overview_s"] = time.monotonic() - t0
                out["held"] = not hold.returned.is_set()
                hold.release.set()
                with anyio.fail_after(2 * HOLD_S):
                    await route_done.wait()
            stop.set()

    anyio.run(scenario)
    return out


# Маршрути §6.1, §6.2, §6.4, що чекають на HF: (метод hub, який тримається; запит).
HF_BOUND_ROUTES: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "search": ("search", lambda c: c.get(SEARCH, params={"q": "llama", "limit": 3})),
    "info": ("files", lambda c: c.get(INFO, params={"repo": REPO})),
    "download": ("files", lambda c: c.post(DOWNLOAD, json={"repo": REPO, "user": "alice"})),
}


@pytest.mark.req("SPEC-GPU-002 §6")
@pytest.mark.parametrize("route", sorted(HF_BOUND_ROUTES))
def test_http_models_route_waiting_on_hf_does_not_block_overview(models_env, route):
    """§6: поки маршрут моделей чекає на HF, GET /api/overview відповідає — ще до того, як hub відпущено."""
    method, request = HF_BOUND_ROUTES[route]
    out = _overview_while_held(models_env, method, request)
    status = getattr(out.get("overview"), "status_code", None)
    ok = out.get("entered") is True and status == 200 and out.get("held") is True
    assert ok, (
        f"{route}: expected GET /api/overview -> HTTP 200 while hub.{method} is still held (up to {HOLD_S} s); "
        f"got route reached hub={out.get('entered')!r}, overview HTTP {status}, answered in {out.get('overview_s', float('nan')):.2f} s, "
        f"hub still held at that moment={out.get('held')!r}"
    )


# --- Захист (SPEC-GPU-001 §7.1) --------------------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-002 §6")
@pytest.mark.parametrize("path", POST_ROUTES)
def test_http_models_post_non_json_refused_415(models_env, path):
    """§6 (SPEC-GPU-001 §7.1.2): POST з Content-Type не application/json → 415."""
    with models_env.client() as c:
        resp = c.post(path, content=json.dumps({"repo": REPO, "user": "alice"}), headers={"Content-Type": "text/plain"})
    assert resp.status_code == 415, f"POST {path} as text/plain: expected HTTP 415, got {resp.status_code}"


@pytest.mark.req("SPEC-GPU-002 §6")
def test_http_models_non_json_download_starts_nothing(models_env):
    """§6 (SPEC-GPU-001 §7.1.2): відхилений за Content-Type download не запускає процесу."""
    env = models_env
    with env.client() as c:
        c.post(DOWNLOAD, content=json.dumps({"repo": REPO, "user": "alice"}), headers={"Content-Type": "text/plain"})
    assert env.spawn.processes == [], f"text/plain download: expected no process, got {env.spawn.repos!r}"


@pytest.mark.req("SPEC-GPU-002 §6")
@pytest.mark.parametrize("path", POST_ROUTES)
def test_http_models_post_foreign_origin_refused_403(models_env, path):
    """§6 (SPEC-GPU-001 §7.1.3): POST з чужим Origin → 403."""
    with models_env.client() as c:
        resp = c.post(path, json={"repo": REPO, "user": "alice"}, headers={"Origin": "http://evil.example"})
    assert resp.status_code == 403, f"POST {path} with a foreign Origin: expected HTTP 403, got {resp.status_code}"


@pytest.mark.req("SPEC-GPU-002 §6")
def test_http_models_foreign_origin_download_starts_nothing(models_env):
    """§6 (SPEC-GPU-001 §7.1.3): відхилений за Origin download не запускає процесу."""
    env = models_env
    with env.client() as c:
        c.post(DOWNLOAD, json={"repo": REPO, "user": "alice"}, headers={"Origin": "http://evil.example"})
    assert env.spawn.processes == [], f"foreign-Origin download: expected no process, got {env.spawn.repos!r}"


@pytest.mark.req("SPEC-GPU-002 §6")
@pytest.mark.parametrize("path", [MODELS, SEARCH, INFO])
def test_http_models_foreign_host_refused_421(models_env, path):
    """§6 (SPEC-GPU-001 §7.1.1): Host не з host_names → 421 і на маршрутах моделей."""
    with models_env.client() as c:
        resp = c.get(path, params={"q": "x", "repo": REPO}, headers={"Host": "evil.example"})
    assert resp.status_code == 421, f"GET {path} with Host evil.example: expected HTTP 421, got {resp.status_code}"


# --- Без models маршрутів немає ---------------------------------------------------------------------------------------------------------------------


ABSENT_ROUTES = [
    ("GET", MODELS, None),
    ("GET", SEARCH, {"q": "llama"}),
    ("GET", INFO, {"repo": REPO}),
    ("POST", DOWNLOAD, {"repo": REPO, "user": "alice"}),
    ("POST", CANCEL, {"repo": REPO, "user": "alice"}),
    ("POST", DELETE, {"repo": REPO, "user": "alice"}),
]


@pytest.mark.req("SPEC-GPU-002 §6")
@pytest.mark.parametrize(("method", "path", "data"), ABSENT_ROUTES, ids=[f"{m}-{p}" for m, p, _ in ABSENT_ROUTES])
def test_http_models_routes_absent_without_models(env, method, path, data):
    """§6: build_app(cfg, manager) без models — маршрутів 1–6 немає (404)."""
    with env.client() as c:
        resp = c.get(path, params=data) if method == "GET" else c.post(path, json=data)
    assert resp.status_code == 404, f"{method} {path} without models: expected HTTP 404, got {resp.status_code}: {resp.text[:200]}"

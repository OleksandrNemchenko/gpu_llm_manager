"""§7.1–7.2 SPEC-GPU-001: захист HTTP (Host, Content-Type, Origin) і маршрути вебсервісу.

Усі запити — через starlette TestClient (без мережевих портів). Типовий Host клієнта —
127.0.0.1:1200, він дозволений; інші значення Host задаються заголовком у конкретному запиті.
"""

from __future__ import annotations

import json

import pytest

from .conftest import (
    MEM_TOTAL_MIB,
    N_GPUS,
    PUBLIC_HOST,
    card,
    gpu_proc,
    ok_json,
    overview,
    refusal,
    release,
    reserve,
)

pytestmark = pytest.mark.e2e

CARD_FIELDS = {
    "index", "name", "uuid", "pci_bus_id", "memory_total_mib", "power_limit_w", "ecc_enabled",
    "compute_capability", "temp_slowdown_c", "temperature_c", "util_pct", "power_w", "memory_used_mib",
    "error", "status", "users", "processes", "reservation",
}
PROCESS_FIELDS = ("pid", "user", "name", "cmdline", "used_mib", "kind")

# host_names тестового конфігу: 127.0.0.1, 203.0.113.7 (server.hosts) і gpu.lan (extra_host_names).
ALLOWED_HOSTS = ["127.0.0.1:1200", "127.0.0.1", "203.0.113.7:1200", "gpu.lan", "gpu.lan:1200"]
FOREIGN_HOSTS = ["evil.example", "evil.example:1200", "gpu.lan.evil.example", "localhost:1200"]
ALLOWED_ORIGINS = ["http://127.0.0.1:1200", "http://203.0.113.7:1200", "http://gpu.lan", "http://gpu.lan:1200"]
FOREIGN_ORIGINS = [
    "http://evil.example",
    "http://evil.example:1200",
    "https://127.0.0.1:1200",
    "null",
    "http://gpu.lan.evil.example",
]
NON_JSON_TYPES = ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"]


# --- §7.1.1 Host ---------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §7.1.1")
@pytest.mark.parametrize("host", ALLOWED_HOSTS)
def test_allowed_host_header_accepted(env, host):
    """§7.1.1: Host з host_names — з портом або без — приймається."""
    with env.client() as c:
        resp = c.get("/healthz", headers={"Host": host})
    assert resp.status_code == 200, f"Host {host!r}: expected HTTP 200, got {resp.status_code}: {resp.text[:200]}"


@pytest.mark.req("SPEC-GPU-001 §7.1.1")
@pytest.mark.parametrize("host", FOREIGN_HOSTS)
def test_foreign_host_header_refused_421(env, host):
    """§7.1.1: Host не з host_names (з портом або без) → 421."""
    with env.client() as c:
        resp = c.get("/healthz", headers={"Host": host})
    assert resp.status_code == 421, f"Host {host!r}: expected HTTP 421, got {resp.status_code}"


@pytest.mark.req("SPEC-GPU-001 §7.1.1")
@pytest.mark.parametrize("path", ["/", "/api/overview", "/api/journal", "/api/history"])
def test_foreign_host_refused_on_every_get_route(env, path):
    """§7.1.1: перевірка Host діє на всіх маршрутах, не лише на /healthz."""
    with env.client() as c:
        resp = c.get(path, headers={"Host": "evil.example"})
    assert resp.status_code == 421, f"GET {path} with Host evil.example: expected HTTP 421, got {resp.status_code}"


@pytest.mark.req("SPEC-GPU-001 §7.1.1")
def test_foreign_host_post_has_no_effect(env):
    """§7.1.1: POST з чужим Host → 421 і бронювання не створюється."""
    with env.client() as c:
        resp = c.post("/api/reserve", json={"gpu": 0, "user": "alice"}, headers={"Host": "evil.example"})
        reservation = card(c, 0)["reservation"]
    assert resp.status_code == 421 and reservation is None, (
        f"POST with foreign Host: expected HTTP 421 and no reservation, got {resp.status_code}, {reservation!r}"
    )


# --- §7.1.2 Content-Type ----------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §7.1.2")
@pytest.mark.parametrize("content_type", NON_JSON_TYPES)
@pytest.mark.parametrize("path", ["/api/reserve", "/api/release"])
def test_post_non_json_content_type_refused_415(env, path, content_type):
    """§7.1.2: POST /api/* з Content-Type, що не application/json, → 415."""
    body = json.dumps({"gpu": 0, "user": "alice"})
    with env.client() as c:
        resp = c.post(path, content=body, headers={"Content-Type": content_type})
    assert resp.status_code == 415, f"POST {path} as {content_type!r}: expected HTTP 415, got {resp.status_code}"


@pytest.mark.req("SPEC-GPU-001 §7.1.2")
def test_post_non_json_content_type_has_no_effect(env):
    """§7.1.2: відхилений за Content-Type запит не бронює карту, хоча тіло — валідний JSON."""
    body = json.dumps({"gpu": 0, "user": "alice"})
    with env.client() as c:
        c.post("/api/reserve", content=body, headers={"Content-Type": "text/plain"})
        reservation = card(c, 0)["reservation"]
    assert reservation is None, f"text/plain reserve: expected no reservation, got {reservation!r}"


# --- §7.1.3–7.1.4 Origin -------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §7.1.3")
@pytest.mark.parametrize("origin", FOREIGN_ORIGINS)
def test_post_foreign_origin_refused_403(env, origin):
    """§7.1.3: POST /api/* з Origin, що не http://<дозволений host[:port]>, → 403."""
    with env.client() as c:
        resp = c.post("/api/release", json={"gpu": 0, "user": "alice"}, headers={"Origin": origin})
    assert resp.status_code == 403, f"POST with Origin {origin!r}: expected HTTP 403, got {resp.status_code}"


@pytest.mark.req("SPEC-GPU-001 §7.1.3")
def test_post_foreign_origin_has_no_effect(env):
    """§7.1.3: відхилений за Origin запит не бронює карту."""
    with env.client() as c:
        c.post("/api/reserve", json={"gpu": 0, "user": "alice"}, headers={"Origin": "http://evil.example"})
        reservation = card(c, 0)["reservation"]
    assert reservation is None, f"foreign-Origin reserve: expected no reservation, got {reservation!r}"


@pytest.mark.req("SPEC-GPU-001 §7.1.3")
@pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
def test_post_allowed_origin_accepted(env, origin):
    """§7.1.3: Origin http://<дозволений host[:port]> приймається."""
    with env.client() as c:
        resp = c.post("/api/release", json={"gpu": 0, "user": "alice"}, headers={"Origin": origin})
    assert resp.status_code == 200, f"POST with Origin {origin!r}: expected HTTP 200, got {resp.status_code}: {resp.text[:200]}"


@pytest.mark.req("SPEC-GPU-001 §7.1.4")
def test_post_without_origin_accepted(env):
    """§7.1.4: POST без Origin дозволено (агенти, curl)."""
    with env.client() as c:
        resp = release(c, 0, "alice")
    assert resp.status_code == 200, f"POST without Origin: expected HTTP 200, got {resp.status_code}: {resp.text[:200]}"


# --- §7.2.1–7.2.2 сторінка й healthz -------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §7.2.1")
def test_index_serves_html(env):
    """§7.2.1: GET / віддає HTML-сторінку."""
    with env.client() as c:
        resp = c.get("/")
    ctype = resp.headers.get("content-type", "")
    assert resp.status_code == 200 and ctype.startswith("text/html"), (
        f"GET /: expected HTTP 200 text/html, got {resp.status_code} {ctype!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §7.2.2")
def test_healthz_body_is_ok_true(env):
    """§7.2.2: GET /healthz → {"ok": true}."""
    with env.client() as c:
        body = ok_json(c.get("/healthz"), "GET /healthz")
    assert body == {"ok": True}, f"/healthz: expected {{'ok': True}}, got {body!r}"


@pytest.mark.req("SPEC-GPU-001 §7.2.2")
def test_healthz_body_ends_with_newline(env):
    """§7.2.2: тіло /healthz закінчується символом \\n."""
    with env.client() as c:
        raw = c.get("/healthz").content
    assert raw.endswith(b"\n"), f"/healthz body: expected a trailing newline, got {raw!r}"


# --- §7.2.3 overview ----------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §7.2.3")
def test_overview_top_level_keys(env):
    """§7.2.3: /api/overview → {ts, users, public_host, gpus}."""
    with env.client() as c:
        data = overview(c)
    missing = {"ts", "users", "public_host", "gpus"} - set(data)
    assert not missing, f"/api/overview: missing keys {sorted(missing)}, got {sorted(data)}"


@pytest.mark.req("SPEC-GPU-001 §7.2.3")
def test_overview_public_host_first_non_loopback_with_port(env):
    """§7.2: public_host — перша адреса з server.hosts, що не loopback, з портом."""
    with env.client() as c:
        public = overview(c)["public_host"]
    assert public == PUBLIC_HOST, f"public_host: expected {PUBLIC_HOST!r}, got {public!r}"


@pytest.mark.req("SPEC-GPU-001 §7.2.3")
@pytest.mark.parametrize(
    "hosts",
    [
        ["127.0.0.1", "127.0.0.2", "198.51.100.20"],
        ["::1", "198.51.100.20"],
        ["localhost", "198.51.100.20"],
    ],
    ids=["ipv4-loopback-range", "ipv6-loopback", "localhost-name"],
)
def test_public_host_skips_every_loopback(make_env, hosts):
    """§7.2: loopback — уся 127.0.0.0/8, ::1 і localhost (RFC 1122); public_host — перша адреса поза ними."""
    # 198.51.100.20 — адреса з документаційного діапазону (RFC 5737), не справжня.
    env = make_env({"server.hosts": hosts})
    with env.client(base_url="http://198.51.100.20:1200") as c:
        public = overview(c)["public_host"]
    assert public == "198.51.100.20:1200", f"hosts {hosts!r}: expected public_host '198.51.100.20:1200', got {public!r}"


@pytest.mark.req("SPEC-GPU-001 §7.2.3")
def test_overview_lists_every_gpu_in_index_order(env):
    """§7.2.3: gpus містить усі N карт бекенда в порядку індексів."""
    with env.client() as c:
        indexes = [g.get("index") for g in overview(c)["gpus"]]
    assert indexes == list(range(N_GPUS)), f"gpus: expected indexes {list(range(N_GPUS))}, got {indexes!r}"


@pytest.mark.req("SPEC-GPU-001 §7.2.3")
def test_overview_card_has_all_fields(env):
    """§7.2: карта в gpus має всі перелічені поля."""
    with env.client() as c:
        item = card(c, 0)
    missing = CARD_FIELDS - set(item)
    assert not missing, f"card fields: missing {sorted(missing)}, got {sorted(item)}"


@pytest.mark.req("SPEC-GPU-001 §7.2.3")
def test_overview_card_static_info_from_backend(env, backend):
    """§7.2, §10: статичні поля карти — з GpuInfo бекенда."""
    info = backend.infos[2]
    expected = {
        "index": 2,
        "name": info.name,
        "uuid": info.uuid,
        "pci_bus_id": info.pci_bus_id,
        "memory_total_mib": MEM_TOTAL_MIB,
        "power_limit_w": info.power_limit_w,
        "ecc_enabled": info.ecc_enabled,
        "compute_capability": info.compute_capability,
        "temp_slowdown_c": info.temp_slowdown_c,
    }
    with env.client() as c:
        item = card(c, 2)
    got = {k: item.get(k) for k in expected}
    assert got == expected, f"card 2 static info: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-001 §7.2.3")
def test_overview_card_telemetry_from_backend(env, backend):
    """§7.2, §10: поля телеметрії карти — з останнього GpuSample; error = null для справного заміру."""
    backend.set_reading(2, temperature_c=55.0, util_pct=73.0, power_w=210.0, memory_used_mib=3000)
    expected = {"temperature_c": 55.0, "util_pct": 73.0, "power_w": 210.0, "memory_used_mib": 3000, "error": None}
    with env.client() as c:
        item = card(c, 2)
    got = {k: item.get(k) for k in expected}
    assert got == expected, f"card 2 telemetry: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-001 §7.2.3")
def test_overview_card_error_for_failed_telemetry(env, backend):
    """§7.2, §3.3: для помилкового заміру поле error непорожнє."""
    backend.set_error(3, "NVML: GPU is lost")
    with env.client() as c:
        error = card(c, 3)["error"]
    assert error, f"card 3 with a failed sample: expected a non-empty error, got {error!r}"


@pytest.mark.req("SPEC-GPU-001 §7.2.3")
def test_overview_card_process_fields(env, backend):
    """§7.2: processes — [{pid, user, name, cmdline, used_mib, kind}] з GpuProcess бекенда."""
    backend.set_processes(1, gpu_proc(4242, "bob", used_mib=2048, cmdline="python3 serve.py --port 8001"))
    expected = {
        "pid": 4242,
        "user": "bob",
        "name": "python3",
        "cmdline": "python3 serve.py --port 8001",
        "used_mib": 2048,
        "kind": "compute",
    }
    with env.client() as c:
        procs = card(c, 1)["processes"]
    got = [{k: p.get(k) for k in PROCESS_FIELDS} for p in procs]
    assert got == [expected], f"card 1 processes: expected [{expected!r}], got {got!r}"


@pytest.mark.req("SPEC-GPU-001 §7.2.3")
def test_overview_unreserved_card_reservation_null(env):
    """§7.2: reservation незаброньованої карти — null."""
    with env.client() as c:
        item = card(c, 0)
    assert "reservation" in item and item["reservation"] is None, (
        f"unreserved card: expected reservation null, got {item.get('reservation', '<missing>')!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §10")
def test_lifespan_startup_performs_tick(env, backend):
    """§10: під час старту (lifespan) застосунок сам робить tick() — телеметрія є без ручного tick()."""
    before = backend.sample_calls
    with env.client() as c:
        started = backend.sample_calls  # одразу після завершення старту lifespan
        item = card(c, 3)
    assert started > before and item.get("temperature_c") == 43.0 and item.get("status") == "free", (
        f"right after startup: expected sample() called during startup and telemetry of gpu 3 (43.0 C, free), "
        f"got sample() calls {before} -> {started}, temperature_c={item.get('temperature_c')!r}, "
        f"status={item.get('status')!r}"
    )


# --- §7.2.6–7.2.7 відповіді reserve / release --------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §7.2.6")
def test_reserve_response_shape(env):
    """§7.2.6: відповідь reserve — {reserved: true, gpu: карта, warnings: [...]}."""
    with env.client() as c:
        body = ok_json(reserve(c, 0, "alice", purpose="x"), "reserve gpu 0")
    gpu_card = body.get("gpu") or {}
    assert (
        body.get("reserved") is True
        and gpu_card.get("index") == 0
        and gpu_card.get("status") == "reserved"
        and isinstance(body.get("warnings"), list)
    ), f"reserve response: expected reserved true, gpu card #0 with status reserved, warnings list; got {body!r}"


@pytest.mark.req("SPEC-GPU-001 §7.2.7")
def test_release_response_shape(env):
    """§7.2.7: відповідь release — {released, previous: зняте бронювання, gpu: карта}."""
    with env.client() as c:
        ok_json(reserve(c, 0, "alice", purpose="x"), "reserve gpu 0")
        body = ok_json(release(c, 0, "alice"), "release gpu 0")
    previous = body.get("previous") or {}
    gpu_card = body.get("gpu") or {}
    assert (
        body.get("released") is True
        and previous.get("user") == "alice"
        and previous.get("purpose") == "x"
        and gpu_card.get("index") == 0
        and gpu_card.get("status") == "free"
    ), f"release response: expected released true, previous alice/x, gpu card #0 free; got {body!r}"


# --- bad_request (§7.2) -----------------------------------------------------------------------------------------------------


@pytest.mark.req("SPEC-GPU-001 §7.2")
@pytest.mark.parametrize("path", ["/api/reserve", "/api/release"])
def test_malformed_json_is_bad_request(env, path):
    """§7.2: некоректний JSON → 400 bad_request."""
    with env.client() as c:
        resp = c.post(path, content=b'{"gpu": 0, "user": ', headers={"Content-Type": "application/json"})
        refusal(resp, "bad_request")


@pytest.mark.req("SPEC-GPU-001 §7.2")
@pytest.mark.parametrize("path", ["/api/reserve", "/api/release"])
@pytest.mark.parametrize("body", [{"user": "alice"}, {"gpu": 0}], ids=["no-gpu", "no-user"])
def test_missing_required_field_is_bad_request(env, path, body):
    """§7.2: відсутнє обов'язкове поле (gpu або user) → 400 bad_request."""
    with env.client() as c:
        refusal(c.post(path, json=body), "bad_request")


@pytest.mark.req("SPEC-GPU-001 §7.2")
@pytest.mark.parametrize("path", ["/api/reserve", "/api/release"])
def test_non_object_body_is_bad_request(env, path):
    """§7.2: JSON-масив замість об'єкта — обов'язкових полів немає → 400 bad_request."""
    with env.client() as c:
        refusal(c.post(path, json=[0, "alice"]), "bad_request")

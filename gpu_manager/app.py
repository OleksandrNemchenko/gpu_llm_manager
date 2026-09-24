"""Збирає один ASGI-застосунок: сторінка, JSON API і MCP (/mcp) на одному порту, плюс щосекундне опитування карт."""

from __future__ import annotations

import ipaddress
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import anyio.to_thread
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware

from .config import Config
from .core import GpuManager
from .credentials import hf_token
from .gateway import gateway_routes
from .gpu import GpuBackend
from .history import History
from .hub import HubClient
from .journal import Journal
from .mcp_tools import build_mcp
from .models import ModelStore
from .prompts import PromptStore
from .reservations import ReservationBook
from .runner import ModelRunner
from .store import StateStore
from .web import HostGuard, web_routes
from .web_models import model_routes
from .web_servers import server_routes

log = logging.getLogger(__name__)
# Стан моделей (/health, /metrics) опитується рідше за карти: старт триває хвилини.
_RUNNER_POLL_S = 2.0


def build_manager(cfg: Config, backend: GpuBackend, clock: Callable[[], float] = time.time) -> GpuManager:
    """Ядро з файлами в cfg.data_dir. backend і clock — шви для тестів (підробні карти й час)."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    history = History(cfg.data_dir / "history.sqlite", cfg.ring_keep_s, cfg.history_db_interval_s,
                      cfg.history_db_keep_days, clock=clock)
    return GpuManager(cfg, backend, ReservationBook(StateStore(cfg.data_dir / "state.json"), clock=clock),
                      Journal(cfg.data_dir / "journal.jsonl", cfg.journal_keep_entries, cfg.journal_keep_days,
                              clock=clock), history, clock=clock)


def build_models(cfg: Config, manager: GpuManager) -> ModelStore:
    """Сховище моделей (фаза 2) з тим самим state.json і журналом, що й у менеджера карт."""
    def token() -> str | None:  # читається щоразу: токен міняють у secrets.json без перезапуску
        return hf_token(cfg.secrets_path)

    return ModelStore(cfg, HubClient(token), manager.store, manager.journal_log,
                      lambda: [g["memory_total_mib"] for g in manager.overview()["gpus"]], token)


def build_runner(cfg: Config, manager: GpuManager, models: ModelStore) -> ModelRunner:
    """Запуск моделей (фаза 3); підключає себе до карт (моделі на картці) і сховища (не видаляти запущену)."""
    runner = ModelRunner(cfg, manager.store, manager.journal_log, models, manager)
    models.in_use = runner.in_use
    manager.models_on = runner.on_gpu
    return runner


async def _poll_forever(poll: Callable[[], None], what: str, interval: float) -> None:
    while True:
        try:
            await anyio.to_thread.run_sync(poll)
        except Exception:
            log.exception("%s poll failed; retrying", what)  # одна невдача не зупиняє опитування
        await anyio.sleep(interval)


async def _sample_forever(manager: GpuManager, interval: float) -> None:
    while True:
        try:
            await anyio.to_thread.run_sync(manager.tick)
        except Exception:
            log.exception("GPU sampling failed; retrying next tick")  # одна невдача не зупиняє опитування
        await anyio.sleep(interval)


def _is_loopback(host: str) -> bool:
    """Loopback за RFC 1122: уся 127.0.0.0/8, ::1 і ім'я localhost."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _public_host(cfg: Config) -> str:
    """Перша не-локальна адреса з конфігу (TailScale) з портом: її показує інструкція MCP."""
    host = next((h for h in cfg.hosts if not _is_loopback(h)), cfg.hosts[0])
    return f"{host}:{cfg.port}"


_GUIDE = Path(__file__).parent / "agent_guide.md"


def render_guide(cfg: Config) -> str:
    """Інструкція для агентів з agent_guide.md з підставленими користувачами, адресою й станом вхідної теки."""
    host = _public_host(cfg).rsplit(":", 1)[0]
    status = ("Status: ready." if cfg.inbox_dir.is_dir()
              else "Status: NOT set up on this server yet; do not create it, tell the human (an admin sets it up).")
    return _GUIDE.read_text(encoding="utf-8").format(
        users=", ".join(cfg.users), inbox=cfg.inbox_dir, host=host, port=cfg.port, inbox_status=status)


def build_app(cfg: Config, manager: GpuManager, models: ModelStore | None = None,
              runner: ModelRunner | None = None, prompts: PromptStore | None = None) -> Starlette:
    """ASGI-застосунок. models — фаза 2, runner — фаза 3; None — без них (так збирають тести фази 1)."""
    mcp = build_mcp(manager, cfg.display_timezone, guide=lambda: render_guide(cfg), models=models, runner=runner,
                    prompts=prompts)
    mcp_app = mcp.streamable_http_app(
        stateless_http=True,  # без MCP-сесій: перезапуск менеджера клієнти не помічають
        json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[f"{h}:{cfg.port}" for h in cfg.host_names] + list(cfg.host_names),
            allowed_origins=[f"http://{h}:{cfg.port}" for h in cfg.host_names],
        ),
    )

    gw_routes, gw_client = gateway_routes(runner, prompts) if runner and prompts else ([], None)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        await anyio.to_thread.run_sync(manager.tick)  # перше відкриття сторінки вже має дані
        async with mcp.session_manager.run(), anyio.create_task_group() as tg:
            tg.start_soon(_sample_forever, manager, cfg.sample_interval_s)
            if models is not None:
                tg.start_soon(_poll_forever, models.poll, "download queue", cfg.sample_interval_s)
            if runner is not None:
                await anyio.to_thread.run_sync(runner.restore)  # підняти моделі, що мали працювати
                tg.start_soon(_poll_forever, runner.poll, "model servers", _RUNNER_POLL_S)
            yield
            tg.cancel_scope.cancel()
        if models is not None:
            models.stop_all()  # у state.json вони лишаються активними й продовжаться після старту
        if gw_client is not None:
            await gw_client.aclose()

    return Starlette(
        routes=[*web_routes(manager, _public_host(cfg)), *(model_routes(models) if models else []),
                *(server_routes(runner) if runner else []),
                *gw_routes, *mcp_app.routes],
        middleware=[Middleware(HostGuard, host_names=cfg.host_names, port=cfg.port)],
        lifespan=lifespan,
    )

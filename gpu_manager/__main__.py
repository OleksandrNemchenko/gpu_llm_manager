"""Точка входу: `python -m gpu_manager [--config config.json]`."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import logging
import socket
import sys
import time
from pathlib import Path
from typing import TextIO

import uvicorn

from .app import build_app, build_manager, build_models, build_runner
from .config import load_config
from .gpu import NvmlBackend
from .prompts import PromptStore

# Linux IP_FREEBIND: дозволяє зайняти адресу TailScale, навіть якщо tailscale0 підніметься після сервісу при старті системи.
_IP_FREEBIND = getattr(socket, "IP_FREEBIND", 15)
# М'яка зупинка не довше за це, секунд: відкрита вкладка браузера тримає зʼєднання, і без межі старий екземпляр
# жив після зупинки сервісу, продовжуючи опитувати карти паралельно з новим.
_GRACEFUL_S = 3
# Скільки новий екземпляр чекає, поки попередній звільнить замок, секунд; далі — вихід, systemd спробує знову.
_LOCK_WAIT_S = 20
_LOCK_POLL_S = 0.5
log = logging.getLogger("gpu_manager")


def single_instance(path: Path) -> TextIO:
    """Ексклюзивний замок на файлі: два менеджери не мають одночасно опитувати карти, писати стан і (у фазі 3)
    керувати тими самими vLLM. Повертає відкритий файл — замок живе, доки він відкритий."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w")  # noqa: SIM115 — файл лишається відкритим увесь час роботи
    deadline = time.monotonic() + _LOCK_WAIT_S
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except BlockingIOError:
            if time.monotonic() > deadline:
                log.error("another gpu_manager instance holds %s; exiting", path)
                sys.exit(1)
            time.sleep(_LOCK_POLL_S)


def bind(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IPV6 if family == socket.AF_INET6 else socket.IPPROTO_IP, _IP_FREEBIND, 1)
    sock.bind((host, port))
    return sock


def main() -> None:
    parser = argparse.ArgumentParser(prog="gpu_manager")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "config.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    lock = single_instance(cfg.data_dir / "manager.lock")
    manager = build_manager(cfg, NvmlBackend())
    models = build_models(cfg, manager)
    prompts = PromptStore(cfg.users, manager.store, manager.journal_log)
    app = build_app(cfg, manager, models, build_runner(cfg, manager, models), prompts)
    # uvicorn слухає одну адресу на сервер; заздалегідь відкриті сокети дають слухати лише localhost і TailScale.
    sockets = [bind(h, cfg.port) for h in cfg.hosts]
    server = uvicorn.Server(uvicorn.Config(app, log_level="info", lifespan="on", access_log=False,
                                           timeout_graceful_shutdown=_GRACEFUL_S))
    asyncio.run(server.serve(sockets=sockets))
    lock.close()


if __name__ == "__main__":
    main()

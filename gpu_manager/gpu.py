"""Телеметрія карт за інтерфейсом GpuBackend: постійні дані, щосекундні заміри й процеси на кожній карті.

Інтерфейс потрібен, щоб тести підставляли підробку замість NVML. NVML читає всі карти без CUDA-контексту,
тож опитування не займає жодної карти."""

from __future__ import annotations

import os
import pwd
from dataclasses import dataclass
from typing import Any, Protocol

_MIB = 1024 * 1024
# Командний рядок довший за це обрізається: сторінці й агентові досить початку.
_CMDLINE_MAX = 300


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    uuid: str
    pci_bus_id: str
    memory_total_mib: int
    power_limit_w: float | None
    ecc_enabled: bool | None
    compute_capability: str | None
    temp_slowdown_c: int | None  # з цієї температури карта сама знижує частоти


@dataclass(frozen=True)
class GpuSample:
    index: int
    ts: float
    temperature_c: int | None
    util_pct: int | None
    power_w: float | None
    memory_used_mib: int | None
    error: str | None = None


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    user: str | None
    name: str
    cmdline: str
    used_mib: int | None
    kind: str  # "compute", "graphics" (напр. рендер UE5) або "compute+graphics"


class GpuBackend(Protocol):
    def info(self) -> list[GpuInfo]: ...

    def sample(self, ts: float) -> list[GpuSample]: ...

    def processes(self) -> dict[int, list[GpuProcess]]: ...


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def describe_process(pid: int, proc_root: str = "/proc") -> tuple[str | None, str, str]:
    """(власник, коротке ім'я, командний рядок) процесу хоста за pid.

    proc_root — корінь procfs (тести підставляють свій). Процес із Docker зазвичай показує власником `root`;
    зниклий процес — (None, "?", "")."""
    base = os.path.join(proc_root, str(pid))
    try:
        uid = os.stat(base).st_uid
    except OSError:
        return None, "?", ""
    try:
        user: str | None = pwd.getpwuid(uid).pw_name
    except KeyError:
        user = str(uid)
    name = _read(os.path.join(base, "comm")).strip() or "?"
    cmdline = _read(os.path.join(base, "cmdline")).replace("\0", " ").strip()[:_CMDLINE_MAX]
    return user, name, cmdline


class NvmlBackend:
    def __init__(self) -> None:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        # Карта, що відпала від шини, до перезавантаження машини не віддає навіть handle. Без неї сервіс падав би
        # при кожному старті, і решта карт з їхніми моделями лишалися б без сторінки, MCP і шлюзу /v1.
        self._handles: list[Any] = []
        self._lost: dict[int, str] = {}  # номер карти -> текст помилки NVML
        for i in range(pynvml.nvmlDeviceGetCount()):
            try:
                self._handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            except pynvml.NVMLError as exc:
                self._handles.append(None)
                self._lost[i] = str(exc)

    def _try(self, fn: Any, *args: Any) -> Any:
        try:
            return fn(*args)
        except self._nvml.NVMLError:
            return None

    def _memory(self, handle: Any) -> Any:
        # v1 рахує ~580 MiB, які драйвер резервує на кожній карті, як зайняті — і вільна карта виглядала
        # зайнятою; v2 показує резерв окремо і збігається з nvidia-smi.
        try:
            return self._nvml.nvmlDeviceGetMemoryInfo(handle, version=self._nvml.nvmlMemory_v2)
        except self._nvml.NVMLError:
            return self._nvml.nvmlDeviceGetMemoryInfo(handle)

    def info(self) -> list[GpuInfo]:
        n = self._nvml
        out = []
        for i, h in enumerate(self._handles):
            try:
                if h is not None:
                    out.append(self._info_one(i, h))
                    continue
            except n.NVMLError:
                pass
            # Нечитна карта лишається під своїм номером: інакше зсунулися б номери решти карт.
            out.append(GpuInfo(index=i, name="unknown", uuid="", pci_bus_id="", memory_total_mib=0, power_limit_w=None,
                               ecc_enabled=None, compute_capability=None, temp_slowdown_c=None))
        return out

    def _info_one(self, i: int, h: Any) -> GpuInfo:
        """Постійні дані однієї карти; назва, UUID і пам'ять обов'язкові (NVMLError — карта нечитна)."""
        n = self._nvml
        limit = self._try(n.nvmlDeviceGetEnforcedPowerLimit, h)
        ecc = self._try(n.nvmlDeviceGetEccMode, h)
        cc = self._try(n.nvmlDeviceGetCudaComputeCapability, h)
        pci = self._try(n.nvmlDeviceGetPciInfo, h)
        return GpuInfo(
            index=i,
            name=_text(n.nvmlDeviceGetName(h)),
            uuid=_text(n.nvmlDeviceGetUUID(h)),
            pci_bus_id=_text(pci.busId) if pci is not None else "",
            memory_total_mib=self._memory(h).total // _MIB,
            power_limit_w=limit / 1000 if limit is not None else None,
            ecc_enabled=bool(ecc[0]) if ecc is not None else None,
            compute_capability=f"{cc[0]}.{cc[1]}" if cc is not None else None,
            temp_slowdown_c=self._try(n.nvmlDeviceGetTemperatureThreshold, h, n.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN),
        )

    def sample(self, ts: float) -> list[GpuSample]:
        n = self._nvml
        out = []
        for i, h in enumerate(self._handles):
            if h is None:
                out.append(GpuSample(i, ts, None, None, None, None, error=self._lost[i]))
                continue
            try:
                util = self._try(n.nvmlDeviceGetUtilizationRates, h)
                power = self._try(n.nvmlDeviceGetPowerUsage, h)
                out.append(
                    GpuSample(
                        index=i,
                        ts=ts,
                        temperature_c=self._try(n.nvmlDeviceGetTemperature, h, n.NVML_TEMPERATURE_GPU),
                        util_pct=util.gpu if util is not None else None,
                        power_w=round(power / 1000, 1) if power is not None else None,
                        memory_used_mib=self._memory(h).used // _MIB,
                    )
                )
            except n.NVMLError as exc:  # напр. карта відпала від шини: повідомити, решту опитати далі
                out.append(GpuSample(i, ts, None, None, None, None, error=str(exc)))
        return out

    def processes(self) -> dict[int, list[GpuProcess]]:
        n = self._nvml
        out: dict[int, list[GpuProcess]] = {}
        for i, h in enumerate(self._handles):
            if h is None:
                out[i] = []
                continue
            seen: dict[int, tuple[str, int | None]] = {}
            for kind, fn in (
                ("compute", n.nvmlDeviceGetComputeRunningProcesses),
                ("graphics", n.nvmlDeviceGetGraphicsRunningProcesses),
            ):
                for p in self._try(fn, h) or []:
                    used = p.usedGpuMemory // _MIB if p.usedGpuMemory is not None else None
                    if p.pid in seen:
                        seen[p.pid] = ("compute+graphics", seen[p.pid][1] or used)
                    else:
                        seen[p.pid] = (kind, used)
            procs = []
            for pid, (kind, used) in sorted(seen.items()):
                user, name, cmdline = describe_process(pid)
                procs.append(GpuProcess(pid, user, name, cmdline, used, kind))
            out[i] = procs
        return out

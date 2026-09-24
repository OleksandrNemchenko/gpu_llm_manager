"""§10 SPEC-GPU-001: справжній бекенд gpu_manager.gpu.NvmlBackend() на картах сервера.

Навіщо: усі інші тести ходять через фальшивий бекенд; цей один перевіряє, що справжній читає NVML.
Кількість і модель карт — з NVML (§1), тест їх не зашиває: еталон — `nvidia-smi --query-gpu=index,name
--format=csv,noheader` (§10); без nvidia-smi звіряти нема з чим — пропуск.
Запускається лише вручну, за змінної оточення GPU_MANAGER_LIVE=1 (§10); інакше — пропуск.

Тест лише читає NVML (info / sample / processes) — жодних обчислень і виділення пам'яті на картах,
тож правило CUDA_VISIBLE_DEVICES=2,3 він не порушує; NVML бачить усі карти незалежно від цієї змінної.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any

import pytest

LIVE = os.environ.get("GPU_MANAGER_LIVE") == "1"

pytestmark = [
    pytest.mark.component,
    pytest.mark.hitl,
    pytest.mark.skipif(not LIVE, reason="live NVML test: set GPU_MANAGER_LIVE=1 to run"),
]

# Команда-еталон §10: той самий перелік карт очима nvidia-smi.
NVIDIA_SMI_QUERY = ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"]
# Запобіжник від зависання nvidia-smi, а не вимога специфікації.
NVIDIA_SMI_TIMEOUT_S = 30
# Поріг «простою» від координатора (повідомлення 2026-09-23); у специфікації порогу немає — TBD.
IDLE_USED_MIB_MAX = 100
# Нижня межа правдоподібної температури: карта під живленням тепліша за 0 °C; у специфікації — TBD.
MIN_PLAUSIBLE_C = 0.0
# Верхня межа, коли NVML не віддав temp_slowdown_c; у специфікації порогу немає — TBD.
FALLBACK_MAX_C = 100.0
# Карта, яку тест «відриває від шини» підміною nvmlDeviceGetHandleByIndex (§10); номер дав координатор.
LOST_INDEX = 1


@pytest.fixture(scope="module")
def snapshot() -> dict[str, Any]:
    """Один знімок NVML: info, процеси до й після sample, sample на справжньому часі."""
    from gpu_manager.gpu import NvmlBackend

    backend = NvmlBackend()
    info = backend.info()
    procs_before = backend.processes()
    samples = backend.sample(time.time())
    procs_after = backend.processes()
    return {"info": info, "samples": samples, "procs": (procs_before, procs_after)}


@pytest.fixture(scope="module")
def smi_cards() -> dict[int, str]:
    """Еталон §10: {індекс: назва} з рядків `<index>, <name>` виводу nvidia-smi.

    Немає nvidia-smi — пропуск (звіряти нема з чим); nvidia-smi є, але впав або не показав карт — помилка.
    """
    if shutil.which(NVIDIA_SMI_QUERY[0]) is None:
        pytest.skip("nvidia-smi not found in PATH: it is the reference for the live NVML comparison (§10)")
    run = subprocess.run(NVIDIA_SMI_QUERY, capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_S, check=False)
    if run.returncode != 0:
        pytest.fail(f"{' '.join(NVIDIA_SMI_QUERY)}: expected exit code 0, got {run.returncode}; stderr {run.stderr!r}")
    cards: dict[int, str] = {}
    for line in run.stdout.splitlines():
        if line.strip():
            index, name = line.split(",", 1)
            cards[int(index)] = name.strip()
    if not cards:
        pytest.fail(f"{' '.join(NVIDIA_SMI_QUERY)}: expected at least one card, got stdout {run.stdout!r}")
    return cards


@pytest.mark.req("SPEC-GPU-001 §10")
@pytest.mark.req("SPEC-GPU-001 §1")
def test_live_nvml_indexes_match_nvidia_smi(snapshot, smi_cards):
    """§1, §10: info() бачить ті самі індекси карт, що й nvidia-smi — кожну рівно раз."""
    got = sorted(g.index for g in snapshot["info"])
    expected = sorted(smi_cards)
    assert got == expected, f"NvmlBackend.info() indexes: expected {expected} (nvidia-smi), got {got!r}"


@pytest.mark.req("SPEC-GPU-001 §10")
@pytest.mark.req("SPEC-GPU-001 §1")
def test_live_nvml_names_match_nvidia_smi(snapshot, smi_cards):
    """§1, §10: назва кожної карти в info() збігається з назвою за тим самим індексом у nvidia-smi."""
    # Лише спільні індекси: розбіжність самих індексів ловить test_live_nvml_indexes_match_nvidia_smi.
    wrong = {g.index: (g.name, smi_cards[g.index]) for g in snapshot["info"] if g.index in smi_cards and g.name != smi_cards[g.index]}
    assert not wrong, f"NvmlBackend.info() names, gpu: (got, expected from nvidia-smi) = {wrong!r}"


@pytest.mark.req("SPEC-GPU-001 §10")
def test_live_nvml_every_card_sampled_without_error(snapshot):
    """§10: sample() читає всі карти — по одному заміру на карту, без помилки."""
    expected = sorted(g.index for g in snapshot["info"])
    got = sorted(s.index for s in snapshot["samples"])
    errors = {s.index: s.error for s in snapshot["samples"] if s.error is not None}
    assert got == expected and not errors, (
        f"NvmlBackend.sample(): expected one error-free sample per card {expected}, got {got}, errors {errors!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §10")
def test_live_nvml_temperatures_plausible(snapshot):
    """§10: температура кожної карти правдоподібна: > 0 °C і не вище власного порогу сповільнення."""
    slowdown = {g.index: g.temp_slowdown_c for g in snapshot["info"]}
    bad = {}
    for s in snapshot["samples"]:
        limit = slowdown.get(s.index)
        upper = float(limit) if isinstance(limit, (int, float)) and limit > 0 else FALLBACK_MAX_C
        if not (isinstance(s.temperature_c, (int, float)) and MIN_PLAUSIBLE_C < s.temperature_c <= upper):
            bad[s.index] = (s.temperature_c, upper)
    assert not bad, f"implausible temperatures, gpu: (reading C, upper bound C) = {bad!r}"


@pytest.mark.req("SPEC-GPU-001 §10")
def test_live_nvml_idle_cards_use_little_memory(snapshot):
    """§10: карта без процесів (і до, і після заміру) має зайнято < 100 MiB."""
    before, after = snapshot["procs"]
    idle = [g.index for g in snapshot["info"] if not before.get(g.index) and not after.get(g.index)]
    used = {s.index: s.memory_used_mib for s in snapshot["samples"] if s.index in idle}
    heavy = {i: mib for i, mib in used.items() if not (isinstance(mib, (int, float)) and mib < IDLE_USED_MIB_MAX)}
    assert not heavy, f"idle cards (no processes): expected < {IDLE_USED_MIB_MAX} MiB used, got {heavy!r} MiB"


@pytest.fixture
def lost_card(monkeypatch: pytest.MonkeyPatch, smi_cards: dict[int, str]) -> dict[str, Any]:
    """Знімок NvmlBackend(), для якого карта LOST_INDEX «відпала від шини» (§10).

    pynvml.nvmlDeviceGetHandleByIndex підмінено: для LOST_INDEX кидає NVMLError_GpuIsLost, для решти карт
    викликає справжню функцію. Підміна діє до кінця тесту — і на конструктор, і на info / sample / processes.
    Тест лише читає NVML, як і решта файлу.
    """
    import pynvml

    from gpu_manager.gpu import NvmlBackend

    if LOST_INDEX not in smi_cards:
        pytest.skip(f"needs a card with index {LOST_INDEX} to lose; nvidia-smi shows indexes {sorted(smi_cards)}")
    original = pynvml.nvmlDeviceGetHandleByIndex

    def get_handle(index: Any, *args: Any, **kwargs: Any) -> Any:
        if int(index) == LOST_INDEX:
            raise pynvml.NVMLError_GpuIsLost()
        return original(index, *args, **kwargs)

    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", get_handle)
    try:
        backend = NvmlBackend()
        info = backend.info()
        samples = backend.sample(time.time())
        procs = backend.processes()
    except pynvml.NVMLError as exc:
        pytest.fail(f"NvmlBackend with gpu {LOST_INDEX} lost: expected the backend to keep working, it raised {exc!r}")
    return {"info": info, "samples": samples, "procs": procs}


@pytest.mark.req("SPEC-GPU-001 §10")
def test_live_nvml_lost_card_kept_in_info(lost_card):
    """§10: карта, для якої NVML кидає NVMLError, лишається в info() з тим самим index, name "unknown", 0 MiB."""
    entries = [g for g in lost_card["info"] if g.index == LOST_INDEX]
    got = [(g.name, g.memory_total_mib) for g in entries]
    assert got == [("unknown", 0)], (
        f"NvmlBackend.info() for lost gpu {LOST_INDEX}: expected one entry (name 'unknown', memory_total_mib 0), "
        f"got {got!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §10")
def test_live_nvml_lost_card_sample_has_error(lost_card):
    """§10: замір втраченої карти в sample() є і має непорожнє error."""
    entries = [s for s in lost_card["samples"] if s.index == LOST_INDEX]
    errors = [s.error for s in entries]
    assert entries and all(errors), (
        f"NvmlBackend.sample() for lost gpu {LOST_INDEX}: expected a sample with a non-empty error, got errors {errors!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §10")
def test_live_nvml_lost_card_processes_empty(lost_card):
    """§10: у processes() втрачена карта — порожній список."""
    procs = lost_card["procs"]
    got = procs.get(LOST_INDEX, "<missing>") if isinstance(procs, dict) else procs
    assert got == [], f"NvmlBackend.processes()[{LOST_INDEX}] for the lost gpu: expected [], got {got!r}"


@pytest.mark.req("SPEC-GPU-001 §10")
def test_live_nvml_lost_card_others_still_read(lost_card, smi_cards):
    """§10: втрачена карта не зупиняє бекенд — кожна інша карта в sample() є і без помилки."""
    expected = sorted(i for i in smi_cards if i != LOST_INDEX)
    got = sorted(s.index for s in lost_card["samples"] if s.index != LOST_INDEX and s.error is None)
    errors = {s.index: s.error for s in lost_card["samples"] if s.index != LOST_INDEX and s.error is not None}
    assert got == expected, (
        f"with gpu {LOST_INDEX} lost: expected error-free samples of gpus {expected}, got {got}, errors {errors!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §10")
def test_live_nvml_process_keys_are_card_indexes(snapshot):
    """§10: processes() — словник з ключами-індексами карт."""
    indexes = {g.index for g in snapshot["info"]}
    before, _ = snapshot["procs"]
    assert isinstance(before, dict), f"NvmlBackend.processes(): expected a dict, got {type(before).__name__}"
    foreign = sorted(k for k in before if k not in indexes)
    assert not foreign, (
        f"NvmlBackend.processes(): expected keys among card indexes {sorted(indexes)}, unexpected {foreign!r}"
    )

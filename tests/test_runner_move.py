"""§2.10a SPEC-GPU-003: перенесення активної моделі на іншу карту або порт — ModelRunner.move(name, user, gpu, port).

Перевіряється: відмова для неактивної моделі; «без змін» не чіпає юніта; unknown_gpu і gpu_memory_low —
до зупинки; успішне перенесення зберігає назву, max_model_len, extra_args і явну частку, а виведену частку
рахує заново; відмова нового старту повертає модель на старе місце з тією самою відмовою; журнал model_move.
Місце нового старту видно в записаному виклику launcher.start (CUDA_VISIBLE_DEVICES і --port).
"""

from __future__ import annotations

import pytest

from .conftest import N_GPUS
from .model_fakes import expect_manager_error
from .runner_fakes import (
    DEFAULT_NAME,
    FRACTION_EMPTY,
    FRACTION_USED_MID,
    GPU,
    GPU_B,
    LINE_REMOTE_CODE,
    PORT_FIRST,
    USED_MID_MIB,
    flag_value,
    fraction_of,
    max_len_of,
    port_of,
)

pytestmark = [pytest.mark.component, pytest.mark.usefixtures("isolated_home")]

NAME = DEFAULT_NAME
EXTRA = ["--enable-prefix-caching"]
MOVE_PORT = PORT_FIRST + 10  # 8010 — явний порт перенесення


@pytest.fixture
def running(runner_env):
    """tiny-llm у стані running на GPU, порт 8000: явна частка 0.5, max_model_len 4096, extra_args EXTRA."""
    env = runner_env
    env.start(fraction=0.5, max_model_len=4096, extra_args=list(EXTRA))
    env.ensure_running(NAME)
    return env


@pytest.mark.req("SPEC-GPU-003 §2.10a")
def test_move_inactive_server_not_found(runner_env):
    """§2.10a: неактивна модель (failed) → server_not_found."""
    env = runner_env
    env.start()
    env.ensure_failed(NAME, LINE_REMOTE_CODE)
    expect_manager_error("server_not_found", env.runner.move, NAME, "alice", gpu=GPU_B)


@pytest.mark.req("SPEC-GPU-003 §2.10a")
@pytest.mark.parametrize(("gpu", "port"), [(None, None), (GPU, PORT_FIRST)], ids=["no-target", "same-place"])
def test_move_without_change_returns_current(running, gpu, port):
    """§2.10a: без змін — повертається поточний стан; юніт не зупиняється й не запускається знову."""
    result = running.runner.move(NAME, "alice", gpu=gpu, port=port)
    current_port = result.get("port") if isinstance(result, dict) else result
    got = (len(running.starts(NAME)), running.launcher.stops, current_port)
    assert got == (1, [], PORT_FIRST), f"move without a change: expected (1 start, no stops, port {PORT_FIRST}), got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.10a")
def test_move_unknown_gpu_refused_before_stop(running):
    """§2.10a: невідома карта → unknown_gpu, а модель не зупинена й працює далі."""
    expect_manager_error("unknown_gpu", running.runner.move, NAME, "alice", gpu=N_GPUS)
    got = (running.launcher.stops, running.status(NAME))
    assert got == ([], "running"), f"after an unknown_gpu move: expected (no stops, 'running'), got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.10a")
@pytest.mark.parametrize(
    ("own", "blocker"),
    [(0.5, 0.6), (None, 0.97)],
    ids=["explicit-0.5-vs-free-0.4", "derived-min-0.05-vs-free-0.03"],
)
def test_move_gpu_memory_low_refused_before_stop(runner_env, own, blocker):
    """§2.10a: на новій карті бракує вільної частки (явна частка моделі або 0.05 для виведеної) →
    gpu_memory_low до зупинки."""
    env = runner_env
    env.start(name="blocker", gpu=GPU_B, fraction=blocker)
    env.start(**({} if own is None else {"fraction": own}))
    env.ensure_running(NAME)
    expect_manager_error("gpu_memory_low", env.runner.move, NAME, "alice", gpu=GPU_B)
    got = (env.unit(NAME) in env.launcher.stops, env.status(NAME))
    assert got == (False, "running"), f"after a gpu_memory_low move: expected (unit not stopped, 'running'), got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.10a")
def test_move_keeps_name_params_and_explicit_fraction(running):
    """§2.10a: stop → start на новій карті й порту з тією ж назвою, max_model_len, extra_args і явною часткою."""
    running.runner.move(NAME, "alice", gpu=GPU_B, port=MOVE_PORT)
    last = running.starts(NAME)[-1]
    got = (
        running.unit(NAME) in running.launcher.stops,
        last.env.get("CUDA_VISIBLE_DEVICES"),
        port_of(last.argv),
        flag_value(last.argv, "--served-model-name"),
        fraction_of(last.argv),
        max_len_of(last.argv),
        last.argv[-len(EXTRA):],
    )
    expected = (True, str(GPU_B), MOVE_PORT, NAME, 0.5, 4096, EXTRA)
    assert got == expected, (
        "moved start: expected (old unit stopped, gpu, port, name, fraction, max_model_len, extra_args) "
        f"{expected!r}, got {got!r}"
    )


@pytest.mark.req("SPEC-GPU-003 §2.10a")
@pytest.mark.req("SPEC-GPU-003 §2.4")
def test_move_recomputes_derived_fraction(runner_env):
    """§2.10a, §2.4: виведену частку перенесення рахує заново для нової карти (там зайнято 10000 MiB → 0.77)."""
    env = runner_env
    env.start()
    env.ensure_running(NAME)
    env.set_used(GPU_B, USED_MID_MIB)
    env.runner.move(NAME, "alice", gpu=GPU_B)
    got = env.fractions(NAME)
    expected = [FRACTION_EMPTY, FRACTION_USED_MID]
    assert got == expected, f"derived fraction before/after the move: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.10a")
def test_move_refused_new_start_restarts_old_place(running):
    """§2.10a: новий старт відмовив (нова карта заброньована іншим) → та сама відмова, а модель знову запущена
    на старій карті й старому порту."""
    running.reserve(GPU_B, "bob")
    expect_manager_error("reserved_by_other", running.runner.move, NAME, "alice", gpu=GPU_B)
    calls = running.starts(NAME)
    last = calls[-1]
    got = (len(calls), last.env.get("CUDA_VISIBLE_DEVICES"), port_of(last.argv), running.status(NAME) in {"starting", "running"})
    expected = (2, str(GPU), PORT_FIRST, True)
    assert got == expected, f"after a refused new start: expected (starts, gpu, port, active) {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-003 §2.10a")
def test_move_journaled(running):
    """§2.10a: журнал model_move з gpu, port, from_gpu, from_port."""
    running.runner.move(NAME, "alice", gpu=GPU_B, port=MOVE_PORT)
    got = [(e.get("gpu"), e.get("port"), e.get("from_gpu"), e.get("from_port")) for e in running.journal("model_move")]
    expected = [(GPU_B, MOVE_PORT, GPU, PORT_FIRST)]
    assert got == expected, f"journal model_move (gpu, port, from_gpu, from_port): expected {expected!r}, got {got!r}"

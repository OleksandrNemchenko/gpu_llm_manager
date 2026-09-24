"""§10 SPEC-GPU-001: gpu_manager.gpu.describe_process(pid, proc_root) над фальшивим /proc у tmp_path.

Фальшивий proc_root — тека з підтеками <pid>, у кожній файли comm і cmdline (аргументи через \\0),
як у справжньому /proc. Власник теки — поточний користувач тестового процесу.
"""

from __future__ import annotations

import pwd
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.component

PID = 4242
CMDLINE_LIMIT = 300  # обрізка cmdline до 300 символів (§10)


def _describe(pid: int, root: Path) -> tuple[Any, ...]:
    from gpu_manager.gpu import describe_process

    return tuple(describe_process(pid, proc_root=str(root)))


def _fake_proc(root: Path, pid: int, *, comm: str = "python3\n", cmdline: bytes = b"python3\x00train.py") -> Path:
    """Створює <root>/<pid>/comm і <root>/<pid>/cmdline; повертає теку процесу."""
    proc_dir = root / str(pid)
    proc_dir.mkdir(parents=True)
    (proc_dir / "comm").write_text(comm, encoding="utf-8")
    (proc_dir / "cmdline").write_bytes(cmdline)
    return proc_dir


@pytest.fixture
def proc_root(tmp_path: Path) -> Path:
    root = tmp_path / "proc"
    root.mkdir()
    return root


@pytest.mark.req("SPEC-GPU-001 §10")
def test_describe_process_name_from_comm(proc_root):
    """§10: name — вміст <proc_root>/<pid>/comm (без кінцевого переносу рядка)."""
    _fake_proc(proc_root, PID, comm="python3\n")
    name = _describe(PID, proc_root)[1]
    assert name == "python3", f"name from comm 'python3\\n': expected 'python3', got {name!r}"


@pytest.mark.req("SPEC-GPU-001 §10")
def test_describe_process_cmdline_nul_separated_to_spaces(proc_root):
    """§10: аргументи cmdline, розділені \\0, стають розділеними пробілами."""
    _fake_proc(proc_root, PID, cmdline=b"python3\x00train.py\x00--lr\x000.1")
    cmdline = _describe(PID, proc_root)[2]
    assert cmdline == "python3 train.py --lr 0.1", f"cmdline: expected 'python3 train.py --lr 0.1', got {cmdline!r}"


@pytest.mark.req("SPEC-GPU-001 §10")
def test_describe_process_cmdline_trailing_nul(proc_root):
    """§10: кінцевий \\0 справжнього /proc не лишає в cmdline жодного \\0."""
    _fake_proc(proc_root, PID, cmdline=b"python3\x00train.py\x00--lr\x000.1\x00")
    cmdline = _describe(PID, proc_root)[2]
    assert "\x00" not in cmdline and cmdline.rstrip(" ") == "python3 train.py --lr 0.1", (
        f"cmdline with a trailing NUL: expected 'python3 train.py --lr 0.1' (trailing space tolerated), got {cmdline!r}"
    )


@pytest.mark.req("SPEC-GPU-001 §10")
def test_describe_process_cmdline_truncated_to_300(proc_root):
    """§10: cmdline обрізається до 300 символів (початок рядка)."""
    _fake_proc(proc_root, PID, cmdline=b"python3\x00" + b"x" * 1000)
    expected = ("python3 " + "x" * 1000)[:CMDLINE_LIMIT]
    cmdline = _describe(PID, proc_root)[2]
    assert cmdline == expected, (
        f"cmdline of 1008 chars: expected the first {CMDLINE_LIMIT} chars, got {len(cmdline)} chars {cmdline[:40]!r}..."
    )


@pytest.mark.req("SPEC-GPU-001 §10")
def test_describe_process_empty_cmdline(proc_root):
    """§10: порожній cmdline (як у потоків ядра) → порожній рядок."""
    _fake_proc(proc_root, PID, cmdline=b"")
    cmdline = _describe(PID, proc_root)[2]
    assert cmdline == "", f"empty cmdline file: expected '', got {cmdline!r}"


@pytest.mark.req("SPEC-GPU-001 §10")
def test_describe_process_owner_is_login_of_pid_dir_owner(proc_root):
    """§10: власник — логін власника теки <proc_root>/<pid>."""
    proc_dir = _fake_proc(proc_root, PID)
    expected = pwd.getpwuid(proc_dir.stat().st_uid).pw_name
    owner = _describe(PID, proc_root)[0]
    assert owner == expected, f"owner of {proc_dir}: expected login {expected!r}, got {owner!r}"


@pytest.mark.req("SPEC-GPU-001 §10")
def test_describe_process_full_tuple(proc_root):
    """§10: результат — кортеж (owner, name, cmdline)."""
    proc_dir = _fake_proc(proc_root, PID, comm="vllm\n", cmdline=b"vllm\x00serve\x00model")
    expected = (pwd.getpwuid(proc_dir.stat().st_uid).pw_name, "vllm", "vllm serve model")
    got = _describe(PID, proc_root)
    assert got == expected, f"describe_process: expected {expected!r}, got {got!r}"


@pytest.mark.req("SPEC-GPU-001 §10")
def test_describe_process_missing_pid(proc_root):
    """§10: неіснуючий pid → (None, "?", "")."""
    _fake_proc(proc_root, PID)
    got = _describe(PID + 1, proc_root)
    assert got == (None, "?", ""), f"missing pid: expected (None, '?', ''), got {got!r}"

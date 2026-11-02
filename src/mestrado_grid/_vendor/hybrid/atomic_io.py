"""Atomic persistence only; bounded retries for Windows file sharing conflicts."""
import ctypes
import datetime
import gzip
import json
import os
from pathlib import Path
import tempfile
import time

RETRY_SECONDS = 30.0
EVENT_LOG = None
STATS = {"writes": 0, "retried_writes": 0, "retries": 0}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def transient(exc):
    # Do not hide disk-full, invalid paths or serialization errors.
    return isinstance(exc, OSError) and getattr(exc, "winerror", None) in (5, 32, 33)


def note(path, retries, elapsed, success, error=None):
    if not retries:
        return
    record = dict(utc=now(), path=str(path), retries=retries, elapsed_s=elapsed,
                  success=success, error=error, operation="atomic_replace")
    if EVENT_LOG:
        try:
            with Path(EVENT_LOG).open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # Diagnostics must not turn a successful commit into a failed search.
            import sys
            print(json.dumps(record, ensure_ascii=False), file=sys.stderr, flush=True)


def atomic_bytes(path, payload, *, timeout=RETRY_SECONDS, replace=None, sleep=time.sleep):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    pending = Path(name)
    started = time.monotonic()
    retries = 0
    replace = os.replace if replace is None else replace
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        while True:
            try:
                replace(pending, path)
                STATS["writes"] += 1
                STATS["retries"] += retries
                STATS["retried_writes"] += bool(retries)
                note(path, retries, time.monotonic()-started, True)
                return
            except OSError as exc:
                elapsed = time.monotonic()-started
                if not transient(exc) or elapsed >= timeout:
                    # Keep both the old complete destination and the uncommitted
                    # payload for diagnosis; never truncate/delete the destination.
                    exc.add_note("Uncommitted payload retained at " + str(pending))
                    note(path, retries, elapsed, False, repr(exc))
                    raise
                retries += 1
                sleep(min(0.01 * 2**min(retries-1, 6), 0.5, timeout-elapsed))
    except BaseException:
        # An incomplete temporary file is not advertised as a committed result.
        raise


def shared_bytes(path):
    """On Windows allow a writer to replace a file while this reader is open."""
    if os.name != "nt":
        return Path(path).read_bytes()
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(Path(path).resolve()), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(fd, "rb") as f:
        return f.read()


def read_json(path):
    return json.loads(shared_bytes(path).decode("utf-8-sig"))


def write_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))


def install(common, support=None):
    def save(path, value):
        write_json(path, common.plain(value))

    def audit(path, value):
        payload = json.dumps(common.plain(value), ensure_ascii=False, allow_nan=False).encode("utf-8")
        atomic_bytes(path, gzip.compress(payload))

    common.save = save
    common.audit = audit
    common.read = read_json
    if support is not None:
        def save_gz(path, value):
            payload = json.dumps(common.plain(value), ensure_ascii=False, allow_nan=False).encode("utf-8")
            atomic_bytes(path, gzip.compress(payload, compresslevel=2))
        support.save = save
        support.read = read_json
        support.save_gz = save_gz
        support.read_gz = lambda path: json.loads(gzip.decompress(shared_bytes(path)))


def process_state(pid):
    """PID plus creation time identifies a Windows process, including launchers."""
    from ctypes import wintypes
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.EnumProcesses.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD,
                                   ctypes.POINTER(wintypes.DWORD)]
    listed = (wintypes.DWORD * 65536)()
    needed = wintypes.DWORD()
    if not psapi.EnumProcesses(listed, ctypes.sizeof(listed), ctypes.byref(needed)):
        raise ctypes.WinError(ctypes.get_last_error())
    if needed.value == ctypes.sizeof(listed):
        raise RuntimeError("Process list exceeded bounded inventory")
    if int(pid) not in listed[:needed.value // ctypes.sizeof(wintypes.DWORD)]:
        return dict(pid=pid, alive=False)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return dict(pid=pid, alive=False)
        raise ctypes.WinError(error)
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        nums = [(t.dwHighDateTime << 32) | t.dwLowDateTime for t in times]
        return dict(pid=pid, alive=code.value == 259, exit_code=code.value,
                    creation_filetime=nums[0], cpu_s=(nums[2]+nums[3])/1e7)
    finally:
        kernel.CloseHandle(handle)

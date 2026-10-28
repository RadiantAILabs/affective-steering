"""Runs a command and kills its whole process group before it can take a 16 GB Mac down with it.

MLX's default memory limit is 1.5x the GPU's recommended working set (17 GB on a 16 GB M1), and past it MLX keeps
allocating into swap. A job that does that starves WindowServer, the watchdog kills WindowServer, and the whole login
session goes with it. This polls the child's physical footprint (the number macOS's own reports use, which counts
Metal buffers) and the kernel's memory pressure level, and SIGKILLs the child's process group when either crosses the
line. It also holds a caffeinate assertion so a long run isn't paused by idle sleep.

    python memguard.py --max-gb 11 -- python converse.py --condition baseline
"""
import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time

_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
_libc = ctypes.CDLL(None)


class _RUsageV2(ctypes.Structure):  # rusage_info_v2 from <sys/resource.h>, up to the field used here
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
        "phys_footprint")] + [("_rest", ctypes.c_uint64 * 40)]


def footprint(pid):
    """Physical footprint in bytes, or None once the process is gone."""
    ru = _RUsageV2()
    if _libproc.proc_pid_rusage(pid, 2, ctypes.byref(ru)) != 0:
        return None
    return ru.phys_footprint


def group_footprint(pgid):
    """Summed footprint of every process in the group, so wrappers like `uv run` don't hide the real job."""
    pids = (ctypes.c_int * 256)()
    n = _libproc.proc_listpgrppids(pgid, pids, ctypes.sizeof(pids))  # returns a pid count, not bytes
    return sum(footprint(p) or 0 for p in pids[: max(n, 0)] if p > 0)


def pressure_level():
    """kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical."""
    v, n = ctypes.c_int(0), ctypes.c_size_t(4)
    _libc.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(v), ctypes.byref(n), None, 0)
    return v.value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=10.0, help="kill when the child's footprint passes this")
    ap.add_argument("--critical-secs", type=float, default=3.0, help="kill after this long at critical pressure")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command given")

    child = subprocess.Popen(cmd, start_new_session=True)
    caf = subprocess.Popen(["caffeinate", "-i", "-w", str(child.pid)])
    limit, peak, critical_since, reason = a.max_gb * 2**30, 0, None, None
    try:
        while child.poll() is None:
            fp = group_footprint(child.pid)
            peak = max(peak, fp)
            if fp > limit:
                reason = f"footprint {fp / 2**30:.1f} GB > --max-gb {a.max_gb:g}"
            elif pressure_level() >= 4:
                critical_since = critical_since or time.time()
                if time.time() - critical_since > a.critical_secs:
                    reason = f"system memory pressure critical for {a.critical_secs:g}s (footprint {fp / 2**30:.1f} GB)"
            else:
                critical_since = None
            if reason:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                break
            time.sleep(a.interval)
    except KeyboardInterrupt:
        os.killpg(child.pid, signal.SIGINT)
        child.wait()
    finally:
        caf.terminate()
    if reason:
        print(f"memguard: killed {cmd[0]}: {reason}", file=sys.stderr, flush=True)
    print(f"memguard: peak footprint {peak / 2**30:.2f} GB, exit {child.returncode}", file=sys.stderr, flush=True)
    sys.exit(137 if reason else child.returncode)


if __name__ == "__main__":
    main()

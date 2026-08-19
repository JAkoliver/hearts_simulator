"""Usable-machine mode: leave CPU/GPU headroom while heavy work runs.

Enable by exporting HEARTS_HEADROOM=<fraction> (e.g. 0.15) in the launcher;
all child processes inherit it. Off (unset/0) = zero overhead, identical
behavior. Three mechanisms, none of which change experiment SEMANTICS -
work runs identically, just slower:

  1. pace(): duty-cycle sleeps sized to the measured work interval, called
     from hot loops (rollout step groups, PPO minibatches, distill batches).
     GPU utilization drops to ~(1-fraction), giving the Windows compositor
     scheduling windows.
  2. apply_process_priority(): BELOW_NORMAL for the calling process (and
     child_priority() for spawned C++ engines), so desktop apps win CPU
     contention.
  3. scaled_workers()/scaled_shards(): shrink CPU pools and concurrent
     SearchEval shard pairs.

VRAM is deliberately NOT capped: the PPO trainer's ~22.7 GB high-water mark
would OOM under a hard fraction cap rather than throttle (see the v5-L
oversubscription incident, docs/speed_ledger.md 2026-07-21). Duty-cycled
compute + priority recovers most desktop usability at unchanged VRAM.
"""
import os
import time

FRACTION = float(os.environ.get('HEARTS_HEADROOM', '0') or 0.0)
enabled = 0.0 < FRACTION < 0.9
_last = None


def apply_process_priority(force=False):
    # 2026-08-18 incident: CPU pools (match gates, fast probes) at NORMAL
    # priority starved the desktop for over an hour (display would not
    # wake; hard power-off). Priority is now lowered whenever a pool asks,
    # headroom enabled or not - it costs nothing when the machine is idle.
    if not enabled and not force and os.environ.get('HEARTS_NO_LOWPRI') == '1':
        return
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def child_priority(pid):
    """Lower a spawned process (e.g. SearchEval.exe) - best effort."""
    if not enabled:
        return
    try:
        import psutil
        psutil.Process(pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


_owed = 0.0
_MIN_SLEEP = 0.06   # sleep in >=60 ms blocks: the Windows compositor needs
_MAX_SLEEP = 0.40   # CHUNKY idle windows (several frames), not milli-gaps


def pace():
    """Accrue FRACTION/(1-FRACTION) of measured work time as sleep debt and
    pay it in >=60 ms chunks so the GPU gets real idle windows."""
    global _last, _owed
    if not enabled:
        return
    now = time.perf_counter()
    if _last is not None and now > _last:
        _owed += (now - _last) * FRACTION / (1.0 - FRACTION)
    if _owed >= _MIN_SLEEP:
        chunk = min(_owed, _MAX_SLEEP)
        time.sleep(chunk)
        _owed -= chunk
    _last = time.perf_counter()


def popen_creationflags():
    """Windows creationflags for spawning C++ engines at reduced priority
    (BELOW_NORMAL_PRIORITY_CLASS); 0 when disabled or non-Windows."""
    if not enabled or os.name != 'nt':
        return 0
    return 0x00004000


def scaled_workers(n):
    return max(1, int(round(n * (1.0 - FRACTION)))) if enabled else n


def scaled_shards(n):
    return max(1, n // 2) if enabled else n


def banner():
    if enabled:
        print(f"HEADROOM MODE: {FRACTION:.0%} reserved (duty-cycled pacing, "
              f"below-normal priority, reduced pool widths)")

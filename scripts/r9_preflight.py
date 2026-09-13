"""League r9 preflight (docs/exploiter_league_r9_prereg.md §3.6/§9).

Run BEFORE EVERY launch; the report is shown to the user together with
the mode they chose. Refuses (exit 1) when:
  - a research process (python train/probe/gate, SearchEval) is running;
  - free VRAM is below the stage's registered need;
  - free disk on E: is below 20 GB.
Usage: python scripts/r9_preflight.py --stage trainer|searchsel|vecprobe|cpu
                                     --mode "full speed"|"headroom 0.25"
"""
import argparse
import shutil
import subprocess
import sys

NEED_GB = {'trainer': 12.0, 'searchsel': 8.0, 'vecprobe': 6.0, 'cpu': 0.0}
RESEARCH = ('train.py', 'defense_probe', 'run_match_gate', 'neutral_raw_eval',
            'SearchEval', 'validate_', 'run_r9_', 'run_r8_')


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                              shell=isinstance(cmd, str)).stdout
    except Exception as e:
        return f'<{e}>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=sorted(NEED_GB))
    ap.add_argument('--mode', required=True,
                    help='the mode the USER chose for this launch (recorded)')
    ap.add_argument('--allow-running', action='store_true',
                    help='stage is a resume of an already-running pipeline unit')
    a = ap.parse_args()
    ok = True
    print(f'R9 PREFLIGHT  stage={a.stage}  mode="{a.mode}"')

    q = sh(['nvidia-smi', '--query-gpu=memory.total,memory.used,utilization.gpu,'
            'temperature.gpu', '--format=csv,noheader,nounits']).strip()
    try:
        tot, used, util, temp = [float(x) for x in q.split(',')]
        free = (tot - used) / 1024
        print(f'GPU: {used/1024:.1f}/{tot/1024:.1f} GB used, {free:.1f} GB free, '
              f'util {util:.0f}%, {temp:.0f}C')
        if free < NEED_GB[a.stage]:
            print(f'  REFUSE: stage needs {NEED_GB[a.stage]:.0f} GB free'); ok = False
        elif util > 30:
            print('  WARN: GPU busy (>30%) - who? (see holders below)')
    except Exception:
        print(f'GPU query failed: {q!r}'); ok = False

    holders = sh("powershell -NoProfile -Command \"(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples | Where-Object {$_.CookedValue -gt 200MB} | Sort-Object CookedValue -Descending | Select-Object -First 6 | ForEach-Object { '{0,8:N0} MB {1}' -f ($_.CookedValue/1MB), $_.InstanceName }\"")
    print('GPU holders (>200 MB dedicated; the driver aggregate LUID row - value above physical VRAM - is skipped):')
    for line in holders.strip().splitlines():
        try:
            mb = float(line.strip().split(' MB')[0].replace(',', ''))
            if mb > tot:
                continue
        except Exception:
            pass
        print('   ', line.strip())

    procs = sh("powershell -NoProfile -Command \"Get-CimInstance Win32_Process | Where-Object {$_.Name -match 'python|SearchEval'} | ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }\"")
    running = [l for l in procs.strip().splitlines()
               if any(k in l for k in RESEARCH) and 'r9_preflight' not in l]
    print(f'research processes running: {len(running)}')
    for l in running:
        print('   ', l[:140])
    if running and not a.allow_running:
        print('  REFUSE: a research stage is still running (one GPU-holding stage at a time)')
        ok = False

    load = sh("powershell -NoProfile -Command \"(Get-CimInstance Win32_Processor).LoadPercentage\"").strip()
    print(f'CPU load: {load}%')
    du = shutil.disk_usage('E:\\')
    print(f'disk E: free {du.free/2**30:.0f} GB')
    if du.free / 2**30 < 20:
        print('  REFUSE: < 20 GB free'); ok = False

    print('PREFLIGHT ' + ('OK - launch permitted in mode "%s"' % a.mode if ok else 'REFUSED'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

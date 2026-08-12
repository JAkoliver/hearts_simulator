# Deploying the site to a VPS (behind a Cloudflare tunnel)

The site is a single FastAPI process with CPU-only inference (verified:
no CUDA anywhere in the serving path; all model loads pin
`map_location='cpu'`). Any 2 GB-RAM Linux VPS runs it. Cloudflare stays
the edge exactly as today - the tunnel just originates from the VPS
instead of a desktop.

## 1. Provision

- Ubuntu 22.04/24.04, 2 GB RAM (torch-CPU needs ~1 GB resident),
  1-2 vCPU, any disk >= 20 GB.
- No inbound ports needed at all: the tunnel is outbound-only. Leave
  the VPS firewall closed except SSH.

## 2. Install

```bash
sudo apt update && sudo apt install -y python3-venv git
git clone https://github.com/JAkoliver/hearts_simulator.git
cd hearts_simulator
python3 -m venv .venv && source .venv/bin/activate
# CPU torch wheel (NOT the cu126 one from requirements.lock):
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
pip install fastapi==0.139.2 'uvicorn[standard]==0.51.0' numpy==2.5.1 pydantic
```

**Build the C++ pybind module** (the server imports `hearts_env`; the
Windows .pyd does not transfer - it must be built on the VPS. Verified
on Ubuntu 26.04 / Python 3.14 / 2 GB RAM, 2026-08-12):

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile \
  && sudo mkswap /swapfile && sudo swapon /swapfile   # compile headroom
sudo apt install -y cmake g++ python3-dev
pip install pybind11
# pip torch has the same include/lib layout CMake expects of ./libtorch:
ln -s "$(pwd)/.venv/lib/python3.14/site-packages/torch" libtorch
python scripts/fetch_cuda_headers.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(.venv/bin/python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -DPython_EXECUTABLE="$(pwd)/.venv/bin/python"
cmake --build build --target hearts_env -j2
cp build/hearts_env.cpython-*.so .
```

Copy from the old host (scp; none of these are in git):
- model weights the server binds: `hearts_web_model.pth`,
  `hearts_equity.pt`, tier nets (`hearts_ai_grandmaster_v4m10.pt`,
  `legacy_v3_pass238/hearts_ai_grandmaster_v3_milestone7.pt`)
- `hearts_web/site_config.py` (production config incl. BACKUP_S3)
- `hearts_web/supporters.json`
- the existing data files, so history/leaderboard carry over:
  `hearts_web/match_logs.jsonl`, `daily_attempts.jsonl`,
  `player_names.jsonl`, `player_keys.jsonl`, `progress_stats.jsonl`,
  `search_shares.jsonl`
- `hearts_web/static/analysis_engine.wasm` + `static/models/` +
  `static/ort/` (client-side analysis assets, intentionally untracked)

## 3. Systemd unit (the stability part)

`/etc/systemd/system/hearts-web.service`:

```ini
[Unit]
Description=Perilune Hearts web server
After=network-online.target

[Service]
User=hearts
WorkingDirectory=/home/hearts/hearts_simulator
ExecStart=/home/hearts/hearts_simulator/.venv/bin/python -u -m uvicorn \
  hearts_web.server:app --host 127.0.0.1 --port 8642
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`sudo systemctl enable --now hearts-web`

## 4. Move the tunnel

On the VPS:
```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cf.deb
sudo dpkg -i cf.deb
```
No interactive login needed: copy `cert.pem` from the old host's
`%USERPROFILE%\.cloudflared\` to `/root/.cloudflared/cert.pem` - it
authorizes tunnel management for the account. Then create a fresh
tunnel and stage it on a TEST hostname before touching the real one:
```bash
cloudflared tunnel create perilune-vps
cloudflared tunnel route dns perilune-vps play.perilune.ai
sudo cloudflared service install   # runs as systemd unit
```
`/etc/cloudflared/config.yml`:
```yaml
tunnel: <tunnel-id>
credentials-file: /etc/cloudflared/<tunnel-id>.json
ingress:
  - hostname: play.perilune.ai
    service: http://127.0.0.1:8642
  - service: http_status:404
```

## 5. Telemetry backup (do this - the logs are training data)

Create an R2 bucket (Cloudflare dashboard -> R2, free tier 10 GB) + an
API token scoped to it; fill `BACKUP_S3` in `site_config.py` (shape in
site_config_example.py). Verify once:
```bash
python -m hearts_web.backup_sync --test    # expects: HTTP 200 - OK
```
The server then uploads every changed data file hourly on its own.

## 6. Cutover checklist

1. VPS server up on 127.0.0.1:8642, `systemctl status hearts-web` green.
2. Data files + models copied; `curl localhost:8642/api/leaderboard`
   shows the real board (history carried over).
3. Stop the tunnel on the old host (RUN_TUNNEL.cmd window / service).
4. Start cloudflared on the VPS; play.perilune.ai now originates there.
5. Play one full solo deal + create one table from a phone (not the
   LAN) to verify the real path.
6. Backup probe: `--test` returns 200; first hourly cycle logs
   `[backup] match_logs.jsonl: ... uploaded`.
7. Old host: keep everything for a week as rollback (rollback = stop
   VPS tunnel, start the old RUN_TUNNEL.cmd - DNS never changes).

## Notes

- The AI seats run the raw net on CPU: single forward pass per
  decision, milliseconds; 1-2 vCPU is plenty at current traffic.
- Deep search in reviews runs in the PLAYER'S browser (wasm + onnx),
  not on the server - a strong-hardware VPS is not needed.
- The live site may run newer weights than any released checkpoint
  (site_config MODEL_PATH decides).
- Test isolation valve carries over: scratch servers must set
  HEARTS_LOG_PATH so test matches never touch the real logs.

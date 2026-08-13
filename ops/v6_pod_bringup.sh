#!/bin/bash
# v6 stage-2 fleet bring-up (docs/v6_prereg.md): push bundle + models to a
# pod, bootstrap (libtorch + build, CRLF-fixed), and start the queue worker
# behind a reverse tunnel. Usage:
#   bash ops/v6_pod_bringup.sh <host> <port> [skip-setup]
# Requires: /tmp/hearts_src.tar (git archive HEAD), model files at repo
# root, .v6_queue_token, a local orchestrator on :8757.
set -euo pipefail
HOST=$1
PORT=$2
SKIP=${3:-}
KEY=~/.ssh/id_ed25519
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 root@$HOST"
TOKEN=$(cat .v6_queue_token)

if [ "$SKIP" != "skip-setup" ]; then
  $SSH 'mkdir -p /workspace'
  scp -i $KEY -P "$PORT" /tmp/hearts_src.tar hearts_ai_search_match.pt \
      hearts_equity.pt shooter_agg_v1b.pt shooter_sel_v1.pt \
      "root@$HOST:/workspace/"
  $SSH 'cd /workspace && rm -rf hearts && mkdir -p hearts \
    && tar -xf hearts_src.tar -C hearts \
    && find hearts -type f \( -name "*.sh" -o -name "*.py" \) -exec sed -i "s/\r$//" {} + \
    && bash hearts/cloud/h100_session/instance_setup.sh'
fi

# worker behind the reverse tunnel, started via a REMOTE SCRIPT FILE
# (launcher discipline: never `chain && cmd &` - the & backgrounds the
# whole chain and it dies with the ssh session; bit us a fourth time)
$SSH "cat > /workspace/start_worker.sh" <<EOF
#!/bin/bash
pkill -f 'cloud/worker.py' 2>/dev/null || true
sleep 1
cd /workspace/hearts
find cloud -name '*.py' -exec sed -i 's/\r\$//' {} +
export HEARTS_QUEUE_URL=http://localhost:8757
export HEARTS_QUEUE_TOKEN=$TOKEN
export HEARTS_GEN_EXE=/workspace/hearts/build/SelfPlayGen
export HEARTS_WORK_DIR=/workspace/work
export HEARTS_SRV_MAX_ROWS=16384
export LD_LIBRARY_PATH=/opt/libtorch/lib
nohup python3 cloud/worker.py > /workspace/worker.log 2>&1 < /dev/null &
disown
echo "worker pid \$!"
EOF
$SSH 'bash /workspace/start_worker.sh; sleep 2; tail -2 /workspace/worker.log'
echo "BRINGUP DONE $HOST:$PORT"

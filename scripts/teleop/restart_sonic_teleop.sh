#!/usr/bin/env bash
# Launch the VR whole-body loco-manip stack over CloudXR: CloudXR runtime (Process A, env `vr_teleop`)
# + a real FIATLUX-*-Teleop env with SONIC legs (Process B = sonic_teleop.py --input vr, env
# `env_isaaclab`). Walk with the LEFT stick + manipulate with the tuned arm teleop, on any task
# (FIATLUX_TASK). For the KEYBOARD whole-body path (no headset), run sonic_teleop.py --input keyboard
# directly instead -- see source/fiatlux_teleop/README.md.
#
# Env overrides: NV_CXR_ENDPOINT_IP (tailnet IP), NV_CXR_MEDIA_PORT, DISPLAY.
set -u
TAILNET_IP="${NV_CXR_ENDPOINT_IP:?set NV_CXR_ENDPOINT_IP to the IP the headset can reach -- the LAN IP of this box (same WiFi, preferred) or its tailnet IP (remote)}"
MEDIA_PORT="${NV_CXR_MEDIA_PORT:-47998}"
TASK="${FIATLUX_TASK:-FIATLUX-Insert-Teleop-v0}"   # e.g. FIATLUX-Carry-Teleop-v0
HAND="${FIATLUX_HAND:-dex3}"                       # dex3 | inspire
# Demo recording (all optional; see sonic_teleop.py --help):
#   FIATLUX_RECORD=1        record the session as a demo bag (score in meta.json)
#   FIATLUX_RECORD_VIDEO=1  also render a follow-cam MP4 (implies RECORD)
#   FIATLUX_RECORD_START=toggle   start with recording OFF (right controller B = upper button toggles)
#   FIATLUX_RECORD_FORMAT=npz     bag as npz instead of hdf5
#   FIATLUX_LAYOUT_SEED=42        room layout: an integer reproduces that exact room,
#                                 'random' draws one. The seed in use is always printed
#                                 and stored in the demo bag's meta.json.
RECORD_ARGS=()
[ "${FIATLUX_RECORD:-0}" = 1 ] && RECORD_ARGS+=(--record bag)
[ "${FIATLUX_RECORD_VIDEO:-0}" = 1 ] && RECORD_ARGS+=(--record-video)
[ -n "${FIATLUX_RECORD_START:-}" ] && RECORD_ARGS+=(--record-start "$FIATLUX_RECORD_START")
[ -n "${FIATLUX_RECORD_FORMAT:-}" ] && RECORD_ARGS+=(--record-format "$FIATLUX_RECORD_FORMAT")
EXTRA_ARGS=()
[ -n "${FIATLUX_LAYOUT_SEED:-}" ] && EXTRA_ARGS+=(--layout_seed "$FIATLUX_LAYOUT_SEED")
REPO="${FIATLUX_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOGDIR="/tmp/fiatlux-xr"; mkdir -p "$LOGDIR"
source ~/miniconda3/etc/profile.d/conda.sh

echo "[1/5] stopping existing sim + runtime..."
for p in $(pgrep -f "scripts/teleop/sonic_teleop.py" || true); do kill "$p" 2>/dev/null || true; done
for p in $(ss -tlnp 2>/dev/null | grep -E ":48322|:49100" | grep -oE "pid=[0-9]+" | grep -oE "[0-9]+" | sort -u); do
  kill "$p" 2>/dev/null || true
done
sleep 5
for p in $(pgrep -f "scripts/teleop/sonic_teleop.py" || true) $(pgrep -f "isaacteleop.cloudxr" || true); do
  kill -9 "$p" 2>/dev/null || true
done
sleep 2

echo "[2/5] clearing stale CloudXR run-state + carb shared memory..."
rm -f ~/.cloudxr/run/cloudxr.pid ~/.cloudxr/run/ipc_cloudxr ~/.cloudxr/run/runtime_started 2>/dev/null || true
rm -f /dev/shm/carb* /dev/shm/sem.carb* /dev/shm/sem.carbonite* 2>/dev/null || true

echo "[3/5] starting CloudXR runtime (--host-client, endpoint $TAILNET_IP:$MEDIA_PORT)..."
conda activate vr_teleop
export CXR_HOST_VOLUME_PATH="$HOME/.cloudxr" CXR_INSTALL_DIR="$HOME/.cloudxr"
export NV_CXR_ENABLE_PUSH_DEVICES=true NV_CXR_ENABLE_TENSOR_DATA=true NV_CXR_FILE_LOGGING=true
export NV_CXR_OUTPUT_DIR="$HOME/.cloudxr/logs" NV_CXR_RUNTIME_DIR="$HOME/.cloudxr/run"
export NV_DEVICE_PROFILE=auto-webrtc XR_RUNTIME_JSON="$HOME/.cloudxr/openxr_cloudxr.json"
export NV_CXR_ENDPOINT_IP="$TAILNET_IP" NV_CXR_MEDIA_PORT="$MEDIA_PORT"
nohup python -u -m isaacteleop.cloudxr --accept-eula --host-client > "$LOGDIR/runtime.log" 2>&1 &
for _ in $(seq 1 30); do ss -tln 2>/dev/null | grep -q ":48322" && break; sleep 2; done
if ss -tln 2>/dev/null | grep -q ":48322"; then echo "   runtime up (48322 + 49100)"; else
  echo "   !! runtime failed -- see $LOGDIR/runtime.log"; exit 1; fi
conda deactivate

echo "[4/5] starting Isaac Lab whole-body teleop sim (sonic_teleop.py --input vr)..."
# Sim env selection (same policy as setup_sim_teleop.sh): explicit SIM_PYTHON > the fresh uv
# .venv if built > the legacy conda env_isaaclab as fallback.
if [ -n "${SIM_PYTHON:-}" ]; then
  SIM_PY="$SIM_PYTHON"; echo "   sim env: SIM_PYTHON ($SIM_PY)"
elif [ -x "$REPO/.venv/bin/python" ]; then
  SIM_PY="$REPO/.venv/bin/python"; echo "   sim env: uv .venv"
else
  conda activate env_isaaclab; SIM_PY=python; echo "   sim env: conda env_isaaclab (legacy)"
fi
source ~/.cloudxr/run/cloudxr.env
cd "$REPO"
export PYTHONPATH="$REPO/source/fiatlux_task:$REPO/source/fiatlux_teleop"
export DISPLAY="${DISPLAY:-:1001}"
echo "   task=$TASK hand=$HAND"
nohup "$SIM_PY" -u scripts/teleop/sonic_teleop.py --task "$TASK" --hand "$HAND" \
    ${RECORD_ARGS[@]+"${RECORD_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    > "$LOGDIR/sonic_teleop.log" 2>&1 &
for _ in $(seq 1 150); do grep -q "Teleop ready" "$LOGDIR/sonic_teleop.log" 2>/dev/null && break; sleep 2; done
if grep -q "Teleop ready" "$LOGDIR/sonic_teleop.log"; then echo "   sim ready"; else
  echo "   sim not ready yet -- watch: tail -f $LOGDIR/sonic_teleop.log"; fi

echo "[5/5] READY."
cat <<EOF
  1. In the Isaac Sim window: AR panel -> Output OpenXR, Runtime System OpenXR Runtime -> Start AR.
  2. On the Pico browser: https://$TAILNET_IP:48322/client/  (accept cert -> Advanced -> Proceed)
     Settings: Server IP $TAILNET_IP, Port 48322, Device Profile Pico 4 Ultra -> Connect.
  3. Walk to the table (LEFT stick) + insert (controller_rel arm teleop: grip-clutch + move, trigger grasp).
  Logs: $LOGDIR/runtime.log , $LOGDIR/sonic_teleop.log
EOF

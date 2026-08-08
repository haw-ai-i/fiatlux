#!/usr/bin/env bash
# Launch the full SONIC VR *walk* stack: CloudXR runtime (Process A, conda env `vr_teleop`) +
# Isaac Lab sim running `scripts/teleop/sonic_drive.py --input vr` (Process B, conda env `env_isaaclab`).
# The Pico thumbsticks drive the pre-trained SONIC locomotion policy (LEFT stick = walk, RIGHT
# stick = turn + crouch). Mirrors restart_xr_teleop.sh, which does the same for arm teleop.
#
# Env overrides: NV_CXR_ENDPOINT_IP (tailnet IP), NV_CXR_MEDIA_PORT, DISPLAY.
set -u
TAILNET_IP="${NV_CXR_ENDPOINT_IP:-100.112.32.21}"
MEDIA_PORT="${NV_CXR_MEDIA_PORT:-47998}"
REPO="$HOME/robotica_project/fiatlux/fiatlux"
LOGDIR="/tmp/fiatlux-xr"; mkdir -p "$LOGDIR"
source ~/miniconda3/etc/profile.d/conda.sh

echo "[1/5] stopping existing sim + runtime..."
for p in $(pgrep -f "scripts/teleop/sonic_drive.py" || true) $(pgrep -f "scripts/teleop/xr_teleop.py" || true); do kill "$p" 2>/dev/null || true; done
for p in $(ss -tlnp 2>/dev/null | grep -E ":48322|:49100" | grep -oE "pid=[0-9]+" | grep -oE "[0-9]+" | sort -u); do
  kill "$p" 2>/dev/null || true
done
sleep 5
for p in $(pgrep -f "scripts/teleop/sonic_drive.py" || true) $(pgrep -f "isaacteleop.cloudxr" || true); do
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

echo "[4/5] starting Isaac Lab SONIC sim (sonic_drive.py --input vr)..."
conda activate env_isaaclab
source ~/.cloudxr/run/cloudxr.env
cd "$REPO"
export PYTHONPATH="$REPO/source/fiatlux_task:$REPO/source/fiatlux_teleop"
export DISPLAY="${DISPLAY:-:1001}"
nohup python -u scripts/teleop/sonic_drive.py --input vr --arm --insert > "$LOGDIR/sonic_vr.log" 2>&1 &
for _ in $(seq 1 150); do grep -q "Start AR" "$LOGDIR/sonic_vr.log" 2>/dev/null && break; sleep 2; done
if grep -q "Start AR" "$LOGDIR/sonic_vr.log"; then echo "   sim ready"; else
  echo "   sim not ready yet -- watch: tail -f $LOGDIR/sonic_vr.log"; fi

echo "[5/5] READY."
cat <<EOF
  1. In the Isaac Sim window: AR panel -> Output OpenXR, Runtime System OpenXR Runtime -> Start AR.
  2. On the Pico browser: https://$TAILNET_IP:48322/client/  (accept cert -> Advanced -> Proceed)
     Settings: Server IP $TAILNET_IP, Port 48322, Device Profile Pico 4 Ultra -> Connect.
  3. Drive: LEFT stick = walk (fwd/strafe), RIGHT stick = turn (X) + crouch (Y).
  Logs: $LOGDIR/runtime.log , $LOGDIR/sonic_vr.log
EOF

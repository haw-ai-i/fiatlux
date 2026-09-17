#!/usr/bin/env bash
# Launch the VR whole-body loco-manip stack over CloudXR: CloudXR runtime (Process A, env `vr_teleop`)
# + a real FIATLUX-*-Teleop env with SONIC legs (Process B = sonic_teleop.py --input vr, env
# `env_isaaclab`). Walk with the LEFT stick + manipulate with the tuned arm teleop, on any task
# (FIATLUX_TASK). For the KEYBOARD whole-body path (no headset), run sonic_teleop.py --input keyboard
# directly instead -- see source/fiatlux_teleop/README.md.
#
# FIATLUX_TASK and NV_CXR_ENDPOINT_IP are required; everything else has a default. Run it with
# either unset and it prints the valid values -- the 12 task ids, or this box's addresses.
# Recording is armed by default (bag + video, right face button starts it); FIATLUX_RECORD=none
# turns it off. See docs/subtask_teleop.md for the full table.
set -u
# WHICH TASK. Required: the old default was FIATLUX-Insert-Teleop-v0, from when that was the
# only teleop env -- it is stationary tabletop manipulation with the pelvis bolted, which this
# driver then unbolts, so falling back to it silently hands you a scene that is not one of the
# 12 subtasks and cannot be scored against the campaign. Better to say so.
if [ -z "${FIATLUX_TASK:-}" ]; then
    echo "set FIATLUX_TASK to the env to drive, e.g."
    echo "    FIATLUX_TASK=FIATLUX-S07-ApproachNewBulb-Teleop-v0 bash ${BASH_SOURCE[0]##*/}"
    echo "  the 12 subtasks (each id + '-Teleop-v0'):"
    sed -n 's/^    ("\(FIATLUX-S[0-9][0-9]-[A-Za-z]*\)".*/    \1-Teleop-v0/p' \
        "$(dirname "${BASH_SOURCE[0]}")/../../source/fiatlux_teleop/fiatlux_teleop/subtask_teleop.py" 2>/dev/null
    exit 1
fi
TASK="$FIATLUX_TASK"
# WHICH ADDRESS THE HEADSET DIALS. Required, and deliberately not guessed: only you know which
# network the headset is on, and getting it wrong is the failure where the client page loads
# fine and CONNECT then hangs -- the page came over a route the media cannot use. So list this
# box's addresses and let the operator pick.
if [ -z "${NV_CXR_ENDPOINT_IP:-}" ]; then
    echo "set NV_CXR_ENDPOINT_IP to the address the headset can reach:"
    _ts="$(tailscale ip -4 2>/dev/null | head -1)"
    [ -n "$_ts" ] && echo "    $_ts   tailnet -- headset anywhere, needs Tailscale on it too"
    ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 \
        | grep -vE '^(172\.1[7-9]\.|100\.)' \
        | while read -r _a; do echo "    $_a   LAN -- headset on this WiFi, nothing to install"; done
    exit 1
fi
TAILNET_IP="$NV_CXR_ENDPOINT_IP"
MEDIA_PORT="${NV_CXR_MEDIA_PORT:-47998}"
HAND="${FIATLUX_HAND:-dex3}"                       # dex3 | inspire
# Demo recording (all optional; see sonic_teleop.py --help):
#   FIATLUX_RECORD=none     do NOT record (default: armed, bag + video)
#   FIATLUX_RECORD_VIDEO=0  skip the follow-cam MP4 (default: on)
#   FIATLUX_RECORD_START=auto     record from launch (default: toggle -- right face button starts it)
#   FIATLUX_RECORD_FORMAT=npz     bag as npz instead of hdf5
#   FIATLUX_WALK_SCALE=1.2        m/s at full left-stick deflection (default 1.0)
#   FIATLUX_LAYOUT_SEED=42        room layout: an integer reproduces that exact room,
#                                 'random' draws one. The seed in use is always printed
#                                 and stored in the demo bag's meta.json.
# Recording is ARMED by default -- the common case is wanting the take, and --record-start
# toggle means nothing is written until the right face button is pressed, so arming costs
# nothing. FIATLUX_RECORD=none opts out. Accepts 1 or bag, since both read as "yes".
RECORD_ARGS=()
case "${FIATLUX_RECORD:-bag}" in
    none|0) ;;
    *) RECORD_ARGS+=(--record bag)
       [ "${FIATLUX_RECORD_VIDEO:-1}" = 1 ] && RECORD_ARGS+=(--record-video) ;;
esac
RECORD_ARGS+=(--record-start "${FIATLUX_RECORD_START:-toggle}")
[ -n "${FIATLUX_RECORD_FORMAT:-}" ] && RECORD_ARGS+=(--record-format "$FIATLUX_RECORD_FORMAT")
EXTRA_ARGS=()
[ -n "${FIATLUX_LAYOUT_SEED:-}" ] && EXTRA_ARGS+=(--layout_seed "$FIATLUX_LAYOUT_SEED")
[ -n "${FIATLUX_WALK_SCALE:-}" ] && EXTRA_ARGS+=(--walk_scale "$FIATLUX_WALK_SCALE")
[ -n "${FIATLUX_CAMERA:-}" ] && EXTRA_ARGS+=(--camera "$FIATLUX_CAMERA")
[ "${FIATLUX_RECORD_SETTLE:-0}" = 1 ] && EXTRA_ARGS+=(--record-settle)
[ -n "${FIATLUX_OUT:-}" ] && EXTRA_ARGS+=(--out "$FIATLUX_OUT")
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # the checkout this script lives in
LOGDIR="/tmp/fiatlux-xr"; mkdir -p "$LOGDIR"
# Conda is only needed for the LEGACY envs (vr_teleop for the runtime, env_isaaclab for the sim);
# the uv .venv path needs none of it. Find its profile script rather than assuming ~/miniconda3,
# and carry on without it -- the checks below fall back to .venv and report what they found.
for _c in "${CONDA_ROOT:-}" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
    [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ] && { . "$_c/etc/profile.d/conda.sh"; break; }
done

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
# WHICH X DISPLAY the Isaac window opens on -- you have to click Start AR in it, so a default
# pointing at a display that does not exist means an invisible window and a headset that can
# never connect. Take the caller's, else the first socket in /tmp/.X11-unix.
if [ -z "${DISPLAY:-}" ]; then
    for _x in /tmp/.X11-unix/X*; do [ -e "$_x" ] && { export DISPLAY=":${_x##*/X}"; break; }; done
fi
[ -n "${DISPLAY:-}" ] || { echo "   no X display found -- set DISPLAY to the desktop running Isaac"; exit 1; }
echo "   display: $DISPLAY"
# Kit prompts for the EULA on stdin, and this launches under nohup with no tty -- the prompt then
# fails with "Unable to bootstrap inner kit kernel: EOF when reading a line" and the only symptom
# upstairs is "sim not ready yet". Passed through if the caller already set it; the default
# matches what setup_sim_teleop.sh in this same directory already uses.
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
# WHICH SONIC POLICY DIRECTORY. SONIC_POLICY_DIR is the documented override, but it wins only when
# the onnx is really under it. The two layouts are branches, not old and new: upstream main keeps the onnx under
# gr00t_wbc/, the gear-sonic-v1.1 branch under decoupled_wbc/ (same files, verified by md5).
# A checkout of one can leave an EMPTY directory for the other, so a path that exists proves
# nothing -- always probe for the .onnx itself.
# Otherwise search, taking the checkout from GR00T_WBC_DIR or from this repo, never from a $HOME.
_ONNX=GR00T-WholeBodyControl-Walk.onnx
if [ ! -f "${SONIC_POLICY_DIR:-}/$_ONNX" ]; then
    unset SONIC_POLICY_DIR
    # $REPO/.. is the project workspace -- the folder holding this checkout, its worktrees and
    # teleop-captures. A tool repo there is one copy shared by every worktree, at the same
    # relative path from each. ../.. is where it landed historically; kept so existing boxes work.
    for _b in "${GR00T_WBC_DIR:-}" "$REPO/../GR00T-WholeBodyControl" \
              "$REPO/../../GR00T-WholeBodyControl" "$HOME/GR00T-WholeBodyControl"; do
        [ -n "$_b" ] && [ -d "$_b" ] || continue
        for _l in decoupled_wbc gr00t_wbc; do
            _c="$_b/$_l/sim2mujoco/resources/robots/g1/policy"
            [ -f "$_c/$_ONNX" ] && { SONIC_POLICY_DIR="$(cd "$_c" && pwd)"; break 2; }
        done
    done
fi
[ -n "${SONIC_POLICY_DIR:-}" ] || { echo "   no $_ONNX found -- set GR00T_WBC_DIR (or SONIC_POLICY_DIR), or run setup_sim_teleop.sh verify"; exit 1; }
export SONIC_POLICY_DIR
echo "   policy dir: $SONIC_POLICY_DIR"
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
  2. In the headset's browser: https://$TAILNET_IP:48322/client/  (accept cert -> Advanced -> Proceed)
     Settings: Server IP $TAILNET_IP, Port 48322, then pick your Device Profile:
       Quest 3S / Quest 3 / Quest 2  -- also switch on "Quest Texture Optimization" and,
                                        on Quest 3/3S, "Quest Color Workaround" (Display P3).
       Pico 4 Ultra
     -> Connect.
  3. Walk to the table (LEFT stick) + insert (controller_rel arm teleop: grip-clutch + move, trigger grasp).
  Logs: $LOGDIR/runtime.log , $LOGDIR/sonic_teleop.log
EOF

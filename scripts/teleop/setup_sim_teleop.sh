#!/usr/bin/env bash
# setup_sim_teleop.sh -- one-command install of everything the SIM whole-body teleop needs.
#
# Two tiers (see source/fiatlux_teleop/README.md for how the stack is wired):
#   KEYBOARD tier -- whole-body teleop (SONIC legs + arm IK) from the desktop, NO headset:
#       sim env (uv) + assets + SONIC policy ONNX. This is the minimum to drive the robot in sim.
#   VR tier -- adds the headset-over-CloudXR path on top (Quest 3S/3/2, Pico 4 Ultra):
#       vr_teleop venv (isaacteleop/CloudXR runtime) + network checks. The CloudXR deps NEVER go
#       into the sim env (known dep conflict; the launcher runs them as two processes).
#
# Usage:
#   ./scripts/teleop/setup_sim_teleop.sh            # keyboard tier (sim + assets + sonic)
#   ./scripts/teleop/setup_sim_teleop.sh vr         # keyboard tier + the VR/CloudXR tier
#   ./scripts/teleop/setup_sim_teleop.sh verify     # check every piece, install nothing
#
# Idempotent: every stage checks before it acts; re-running is safe and cheap.
#
# NO-INTERFERENCE DESIGN (teleop is an OPTION on top of the benchmark, which also uses Isaac Sim):
#   - In-process deps (onnxruntime for SONIC) ship as the OPTIONAL `teleop` extra -- plain
#     `uv sync` (the benchmark workflow) never installs them, and the extra is leaf-only so it
#     cannot re-pin torch/Isaac for benchmark users. If a future teleop dep would move a core pin,
#     it does NOT belong in the extra -- it goes behind a process boundary instead.
#   - Conflicting runtimes (CloudXR/isaacteleop) live in a SEPARATE venv + separate process; they
#     are never installed into the sim env.
#   - External artifacts (SONIC .onnx, ~/.cloudxr) stay outside the benchmark tree, behind env-var
#     paths. This script never edits benchmark files; `verify` is read-only.
#   - Canonical launch is `uv run --extra teleop ...`: uv's exact-sync means a later plain
#     `uv sync` REMOVES the extra's packages -- `uv run --extra` re-syncs at launch, so teleop
#     self-heals and the two workflows cannot fight.
#
# SIM ENV POLICY -- fresh by default, existing by explicit opt-in:
#   Default: a FRESH, self-contained uv env (.venv; Isaac Sim/Lab as pinned wheels -- no external
#   IsaacLab checkouts, so nothing like the ~/SO101_project entanglement can bite it).
#   Opt-in:  if you KNOW you already have a correct install, point at its python explicitly:
#       SIM_PYTHON=~/miniconda3/envs/env_isaaclab/bin/python ./scripts/teleop/setup_sim_teleop.sh
#   The script then validates THAT env (imports + versions + where its isaaclab really lives),
#   skips the uv build, and prints run commands using your python. It never guesses an old env.
#
# Env overrides:
#   SIM_PYTHON         python of an EXISTING sim env to use instead of building the uv env
#   SONIC_POLICY_DIR   where the two SONIC .onnx files live/go
#                      (default: the GR00T-WholeBodyControl checkout the driver already looks in)
#   GR00T_WBC_REPO     git URL used only when the ONNX files are missing
#   GR00T_WBC_REF      branch/tag to clone (default main; gear-sonic-v1.1 is the fuller
#                      branch -- same two policies, plus decoupled_wbc/ and gear_sonic/)
#   VR_VENV            path of the CloudXR venv (default: conda env 'vr_teleop' if conda exists,
#                      else ~/venvs/vr_teleop)
#
# After install -- run it:
#   keyboard:  uv run --extra teleop python scripts/teleop/sonic_teleop.py \
#                  --task FIATLUX-S07-ApproachNewBulb-Teleop-v0 --input keyboard
#              (PYTHONPATH for fiatlux_teleop is exported below; add it to your shell or use the line
#               printed at the end.)
#   VR:        NV_CXR_ENDPOINT_IP=<ip> FIATLUX_TASK=FIATLUX-S07-ApproachNewBulb-Teleop-v0 \
#                  bash scripts/teleop/restart_sonic_teleop.sh
#
# NOTE on envs: docs/getting_started.md's uv env pins the same Isaac Sim 5.1 / Isaac Lab 2.3.2 the
# teleop was built against, and is the only fully-scripted way to build the sim env on a fresh
# machine. The historical conda env 'env_isaaclab' (hand-built) also works and `verify` accepts
# either. VR mode via restart_sonic_teleop.sh still activates conda envs -- on a uv-only machine
# run the two processes by hand (the README shows both commands).
#
# REPO/PATH MAP -- external checkouts this stack touches (why uv is the fresh-machine default:
# the uv env ships Isaac Sim/Lab as wheels and depends on NONE of these):
#   ~/SO101_project/IsaacLab                     Isaac Lab source the LEGACY conda env_isaaclab is
#                                                editable-installed from (another project's folder!
#                                                verify warns about this fragile dependency)
#   ~/robotica_project/GR00T-WholeBodyControl    SONIC policy .onnx files    [SONIC_POLICY_DIR]
#   ~/robotica_project/IsaacTeleop               isaacteleop EXAMPLES only; the runtime comes from
#                                                pip in the vr_teleop env, no checkout needed
#   ~/robotica_project/unitree_robotics/         unitree_sim_isaaclab (robot-protocol DDS sim) --
#                                                used by the sim-to-real benchmark replay, NOT by
#                                                sim teleop; not installed here

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# WHERE THE SONIC .onnx LIVE. An explicit SONIC_POLICY_DIR wins only when the files are actually
# under it, or a stale value triggers a pointless re-clone over a good checkout.
# The two layouts are branches, not old and new: upstream main keeps the onnx under
# gr00t_wbc/, the gear-sonic-v1.1 branch under decoupled_wbc/ (same files, verified by md5).
# A checkout of one can leave an EMPTY directory for the other, so a path that exists proves
# nothing -- always probe for the .onnx itself.
# WORKSPACE is the folder holding this checkout and its worktrees -- a tool repo there is one copy
# shared by all of them.
WORKSPACE="$(cd "$REPO/.." 2>/dev/null && pwd || echo "$REPO/..")"
_onnx_under(){ [ -n "${1:-}" ] && [ -f "$1/GR00T-WholeBodyControl-Walk.onnx" ]; }
if ! _onnx_under "${SONIC_POLICY_DIR:-}"; then
    unset SONIC_POLICY_DIR
    for _b in "${GR00T_WBC_DIR:-}" "$WORKSPACE/GR00T-WholeBodyControl" \
              "$WORKSPACE/../GR00T-WholeBodyControl" "$HOME/GR00T-WholeBodyControl"; do
        for _l in decoupled_wbc gr00t_wbc; do
            _onnx_under "$_b/$_l/sim2mujoco/resources/robots/g1/policy" \
                && { SONIC_POLICY_DIR="$(cd "$_b/$_l/sim2mujoco/resources/robots/g1/policy" && pwd)"; break 2; }
        done
    done
fi
GR00T_WBC_REPO="${GR00T_WBC_REPO:-https://github.com/haw-ai-i/GR00T-WholeBodyControl.git}"
GR00T_WBC_REF="${GR00T_WBC_REF:-main}"   # branch/tag cloned when the .onnx are missing
ONNX_FILES=(GR00T-WholeBodyControl-Balance.onnx GR00T-WholeBodyControl-Walk.onnx)

c_g=$'\033[1;32m'; c_y=$'\033[1;33m'; c_r=$'\033[1;31m'; c_c=$'\033[1;36m'; c_0=$'\033[0m'
ok(){ printf '%s[setup] OK:%s %s\n' "$c_g" "$c_0" "$*"; }
warn(){ printf '%s[setup] WARN:%s %s\n' "$c_y" "$c_0" "$*"; }
log(){ printf '%s[setup]%s %s\n' "$c_c" "$c_0" "$*"; }
die(){ printf '%s[setup] FAIL:%s %s\n' "$c_r" "$c_0" "$*" >&2; exit 1; }

FAILS=0
need(){ "$@" >/dev/null 2>&1; }
check(){ local msg=$1; shift; if "$@" >/dev/null 2>&1; then ok "$msg"; else warn "MISSING: $msg"; FAILS=$((FAILS+1)); fi; }

# ---------- stages ----------

stage_uv(){
    if need command -v uv; then ok "uv present ($(uv --version 2>/dev/null | head -1))"; return; fi
    # Pin the install dir: under a snap-launched shell (e.g. VS Code snap) XDG_DATA_HOME points into
    # ~/snap/code/<rev>/..., and the uv installer would land there and vanish from normal shells.
    log "installing uv (official installer -> $HOME/.local/bin) ..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$HOME/.local/bin" sh || die "uv install failed"
    export PATH="$HOME/.local/bin:$PATH"
    need command -v uv || die "uv still not on PATH -- open a new shell or add ~/.local/bin to PATH"
    ok "uv installed ($(uv --version 2>/dev/null | head -1))"
}

# Validate a user-supplied SIM_PYTHON: imports work, and show versions + where isaaclab lives so
# "I know I have the right version" is checked, not trusted blindly. Returns non-zero if unusable.
check_sim_python(){
    local py="$1"
    [ -x "$py" ] || { warn "SIM_PYTHON '$py' is not an executable"; return 1; }
    OMNI_KIT_ACCEPT_EULA=YES "$py" -c 'import isaaclab, isaacsim, onnxruntime' 2>/dev/null \
        || { warn "SIM_PYTHON '$py' cannot import isaaclab+isaacsim+onnxruntime"; return 1; }
    local info
    info=$(OMNI_KIT_ACCEPT_EULA=YES "$py" - <<'PY' 2>/dev/null
import os, isaaclab
from importlib.metadata import version
def v(p):
    try: return version(p)
    except Exception: return "?"
print(f"isaacsim {v('isaacsim')} | isaaclab {v('isaaclab')} | onnxruntime {v('onnxruntime')} | isaaclab from {os.path.dirname(isaaclab.__file__)}")
PY
)
    ok "SIM_PYTHON valid: $info"
    case "$info" in
        *"isaaclab from $HOME/miniconda3/"*|*"isaaclab from ${SIM_PYTHON%/bin/python*}"*) : ;;
        *) warn "that isaaclab is an EDITABLE install from an external checkout (see path above) -- it breaks if the checkout moves. Your call; the fresh uv env avoids this." ;;
    esac
}

stage_sim_env(){
    if [ -n "${SIM_PYTHON:-}" ]; then
        log "SIM_PYTHON set -- using your existing sim env instead of building the uv env."
        check_sim_python "$SIM_PYTHON" || die "SIM_PYTHON unusable -- fix the path, or unset it to build the fresh uv env"
        return
    fi
    log "uv sync --extra teleop  (Isaac Sim 5.1 + Isaac Lab 2.3.2 + fiatlux_task + onnxruntime;"
    log "first run downloads ~10 GB -- UV_HTTP_TIMEOUT is raised for the big CUDA wheels)"
    ( cd "$REPO" && UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-1200}" uv sync --extra teleop ) || die "uv sync failed"
    ok "sim env ready (.venv)"
}

stage_assets(){
    if [ -f "$REPO/assets/unitree_g1/wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd" ]; then
        ok "assets already downloaded"; return
    fi
    need command -v gsutil || die "gsutil missing (Google Cloud SDK) -- install + authenticate, then re-run. The G1/bulb/ladder USDs live in a GCS bucket (assets/README.md)."
    log "downloading assets from GCS ..."
    ( cd "$REPO" && ./assets/download_assets.sh ) || die "asset download failed (is gsutil authenticated?)"
    ok "assets downloaded"
}

stage_sonic(){
    # SONIC_POLICY_DIR is UNSET when the search above found nothing -- which is exactly the case
    # this stage exists for, and `set -u` turns a bare reference into a crash instead of a clone.
    local missing=0
    for f in "${ONNX_FILES[@]}"; do [ -f "${SONIC_POLICY_DIR:-}/$f" ] || missing=1; done
    if [ "$missing" -eq 0 ]; then ok "SONIC policy ONNX present ($SONIC_POLICY_DIR)"; return; fi
    log "SONIC .onnx not found -- shallow-cloning GR00T-WholeBodyControl for the two policy files ..."
    local dst="${GR00T_WBC_DIR:-$WORKSPACE/GR00T-WholeBodyControl}"
    if [ ! -d "$dst/.git" ]; then
        # Pin the ref: whatever main holds today is not what it will hold in six months, and the
        # two .onnx ARE the dependency. GR00T_WBC_REF overrides -- gear-sonic-v1.1 is the fuller
        # branch (adds decoupled_wbc/, gear_sonic/), same policies, if you need those sources too.
        git clone --depth 1 --branch "$GR00T_WBC_REF" "$GR00T_WBC_REPO" "$dst" \
            || die "clone failed ($GR00T_WBC_REPO @ $GR00T_WBC_REF)"
    fi
    for _l in decoupled_wbc gr00t_wbc; do
        _onnx_under "$dst/$_l/sim2mujoco/resources/robots/g1/policy" \
            && { SONIC_POLICY_DIR="$dst/$_l/sim2mujoco/resources/robots/g1/policy"; break; }
    done
    for f in "${ONNX_FILES[@]}"; do
        [ -f "${SONIC_POLICY_DIR:-}/$f" ] || die "still missing $f under $dst after clone -- set SONIC_POLICY_DIR to wherever the files are"
    done
    ok "SONIC policy ONNX ready"
}

stage_vr(){
    # CloudXR runtime env -- SEPARATE from the sim env, on purpose.
    local pybin
    local conda_root=""
    for _c in "${CONDA_ROOT:-}" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
        [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ] && { conda_root="$_c"; break; }
    done
    if need command -v conda || [ -n "$conda_root" ]; then
        local conda_sh="$conda_root/etc/profile.d/conda.sh"
        [ -f "$conda_sh" ] && . "$conda_sh"
        if conda env list 2>/dev/null | grep -q '^vr_teleop '; then ok "conda env vr_teleop exists"
        else log "creating conda env vr_teleop (python 3.10) ..."; conda create -y -n vr_teleop python=3.10 || die "conda create failed"; fi
        pybin="$conda_root/envs/vr_teleop/bin/python"
    else
        local venv="${VR_VENV:-$HOME/venvs/vr_teleop}"
        [ -d "$venv" ] || { log "creating venv $venv ..."; python3 -m venv "$venv" || die "venv failed"; }
        pybin="$venv/bin/python"
    fi
    if "$pybin" -c 'import isaacteleop' 2>/dev/null; then ok "isaacteleop already installed"
    else
        log "pip install isaacteleop[cloudxr,retargeters] (~=1.3, the hardware-verified line) ..."
        "$pybin" -m pip install 'isaacteleop[cloudxr,retargeters]~=1.3.0' || die "isaacteleop install failed"
    fi
    # The CloudXR runtime itself lands in ~/.cloudxr on the FIRST launch (--accept-eula). Don't
    # auto-launch here (it wants a GPU/X session); tell the operator instead.
    if [ -f "$HOME/.cloudxr/openxr_cloudxr.json" ]; then ok "CloudXR runtime installed (~/.cloudxr)"
    else warn "CloudXR runtime not bootstrapped yet -- first VR launch does it: restart_sonic_teleop.sh (or: python -m isaacteleop.cloudxr --accept-eula --host-client)"; fi
    # Network for the headset: it needs SOME route to this box -- same LAN/WiFi (best
    # latency, use the LAN IP) or Tailscale (the remote option). Neither is "required"; report both.
    local lan_ip; lan_ip=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^(192\.168|10|172)\.' | head -1)
    [ -n "$lan_ip" ] && ok "LAN path: headset on the same WiFi -> NV_CXR_ENDPOINT_IP=$lan_ip" \
                     || warn "no LAN IP found -- check networking"
    if need tailscale status; then ok "remote path also available: tailscale up ($(tailscale ip -4 2>/dev/null | head -1))"
    else log "tailscale not up (fine on a shared LAN; it's only for the remote case, spec §4a)"; fi
    if need command -v ufw && sudo -n ufw status 2>/dev/null | grep -q active; then
        for rule in 47998/udp 48322/tcp 49100/tcp; do sudo -n ufw allow "$rule" >/dev/null 2>&1 || true; done
        ok "ufw ports opened (47998/udp, 48322,49100/tcp)"
    else warn "could not adjust ufw non-interactively -- if the headset can't connect: sudo ufw allow 47998/udp && sudo ufw allow 48322,49100/tcp"; fi
    cat <<'EOF'
  Headset (one-time, manual). The client is a WEB PAGE, so nothing is installed for the
  streaming itself -- any headset whose browser does WebXR works. Tested: Quest 3S, Quest 3,
  Quest 2, Pico 4 Ultra.

  Network path -- pick one:
    SAME WiFi/LAN (preferred, lowest latency): put the headset on this box's network.
       Nothing to install; use the LAN IP printed above as NV_CXR_ENDPOINT_IP.
    REMOTE (different networks) -- Tailscale, sideloaded into the headset's Android:
       Quest 3S / 3 / 2:
         1. Enable Developer Mode for the headset in the Meta Horizon phone app, then
            connect it by USB and `adb install tailscale-android-universal-<ver>.apk`
            (Quest Browser cannot install APKs, unlike the PICO one).
         2. Open Tailscale from the app library (Unknown Sources), join THIS tailnet.
       Pico 4 Ultra:
         1. Settings -> Security -> Install unknown apps -> allow for PICO Browser.
         2. In the PICO Browser download + install the Tailscale APK:
            pkgs.tailscale.com/stable/tailscale-android-universal-<ver>.apk
         3. Open Tailscale on the Pico, join THIS tailnet.
       Either way, use the tailnet IP as NV_CXR_ENDPOINT_IP.

  In the client page (https://<ip>:48322/client/), set Device Profile to your headset. On
  Quest also enable "Quest Texture Optimization"; on Quest 3/3S enable "Quest Color
  Workaround" (Display P3) or the stream looks washed out.
EOF
}

stage_verify(){
    log "verify: keyboard tier"
    if [ -n "${SIM_PYTHON:-}" ]; then
        # user opted into an existing env -- validate exactly that one
        check_sim_python "$SIM_PYTHON" || FAILS=$((FAILS+1))
    elif [ -x "$REPO/.venv/bin/python" ] && OMNI_KIT_ACCEPT_EULA=YES "$REPO/.venv/bin/python" -c 'import isaaclab, onnxruntime' 2>/dev/null; then
        ok "sim env: fresh uv .venv (isaaclab + onnxruntime)"
    else
        warn "MISSING: fresh uv sim env (.venv). Build it:  $0 keyboard"; FAILS=$((FAILS+1))
        need command -v uv || warn "  (uv is not on PATH either -- the keyboard stage installs it)"
        if [ -x "$HOME/miniconda3/envs/env_isaaclab/bin/python" ]; then
            log "  found a legacy conda env_isaaclab -- NOT used by default. If you know it's the right"
            log "  version:  SIM_PYTHON=$HOME/miniconda3/envs/env_isaaclab/bin/python $0 verify"
        fi
    fi
    check "G1 USD asset" test -f "$REPO/assets/unitree_g1/wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd"
    for f in "${ONNX_FILES[@]}"; do check "SONIC $f" test -f "${SONIC_POLICY_DIR:-}/$f"; done
    check "fiatlux_teleop package" test -f "$REPO/source/fiatlux_teleop/fiatlux_teleop/__init__.py"
    log "verify: VR tier"
    local vpy="$HOME/miniconda3/envs/vr_teleop/bin/python"; [ -x "$vpy" ] || vpy="${VR_VENV:-$HOME/venvs/vr_teleop}/bin/python"
    check "isaacteleop (vr env)" "$vpy" -c 'import isaacteleop'
    check "CloudXR runtime (~/.cloudxr/openxr_cloudxr.json)" test -f "$HOME/.cloudxr/openxr_cloudxr.json"
    # network path: LAN or tailscale, either is fine -- informational, never a failure by itself
    local lan_ip; lan_ip=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^(192\.168|10|172)\.' | head -1)
    local ts_ip; ts_ip=$(tailscale ip -4 2>/dev/null | head -1)
    if [ -n "$lan_ip" ] || [ -n "$ts_ip" ]; then
        ok "headset path(s): ${lan_ip:+LAN $lan_ip (same WiFi)}${lan_ip:+${ts_ip:+ | }}${ts_ip:+tailscale $ts_ip (remote)}"
    else
        warn "MISSING: no LAN or tailscale IP -- the headset has no route to this box"; FAILS=$((FAILS+1))
    fi
    if [ "$FAILS" -eq 0 ]; then ok "ALL CHECKS PASSED"; else warn "$FAILS check(s) missing -- run the matching stage(s)"; fi
    return "$FAILS"
}

# ---------- main ----------

print_run_cmd(){
    log "keyboard tier done. Run:"
    if [ -n "${SIM_PYTHON:-}" ]; then
        echo "  cd $REPO && PYTHONPATH=source/fiatlux_task:source/fiatlux_teleop \\"
        echo "  OMNI_KIT_ACCEPT_EULA=YES $SIM_PYTHON scripts/teleop/sonic_teleop.py --task FIATLUX-S07-ApproachNewBulb-Teleop-v0 --input keyboard"
    else
        echo "  cd $REPO && PYTHONPATH=source/fiatlux_task:source/fiatlux_teleop \\"
        echo "  OMNI_KIT_ACCEPT_EULA=YES uv run --extra teleop python scripts/teleop/sonic_teleop.py --task FIATLUX-S07-ApproachNewBulb-Teleop-v0 --input keyboard"
    fi
}

case "${1:-keyboard}" in
    keyboard) [ -n "${SIM_PYTHON:-}" ] || stage_uv
              stage_sim_env; stage_assets; stage_sonic; print_run_cmd ;;
    vr)       [ -n "${SIM_PYTHON:-}" ] || stage_uv
              stage_sim_env; stage_assets; stage_sonic; stage_vr
              log "VR tier done. Launch with:  NV_CXR_ENDPOINT_IP=<ip> FIATLUX_TASK=<subtask>-Teleop-v0 bash scripts/teleop/restart_sonic_teleop.sh"
              log "  (leave either unset and the launcher prints the valid values)" ;;
    verify)   stage_verify ;;
    *)        die "usage: $0 [keyboard|vr|verify]" ;;
esac

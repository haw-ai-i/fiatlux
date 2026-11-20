"""Write anchors.json for a clip: where a holding hand's finger-pad point should be, per label.

One anchor per contact label 1..6 (step1..4, platform, handrail), midway between the two hands in
y; LadderMan adds the per-hand offset (0, +-half the hand spacing, 0). The tracked point on the
hand is its finger-pad point, 0.17 m ahead of the wrist on the palm side, because that point is
on the hold both while the hand is pitched up on its pads and after it has rolled flat. The
anchor sits a little BELOW the surface: LadderMan's lost-contact check requires the hand point to
be above its target.

    python make_anchors.py --clip <dir>      -> <dir>/anchors.json, and prints the overrides
"""
import argparse
import json
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--pad_u", type=float, default=0.17)
ap.add_argument("--palm_pt", type=float, default=0.115)
a = ap.parse_args()
clip = pathlib.Path(a.clip)
h = json.loads((clip / "holds.json").read_text())
sched, holds = h["schedule"], h["holds"]
by = {}
for side in ("left", "right"):
    for s0, s1, name, c in sched[side]:
        by.setdefault("handrail" if name == "hook" else name, {})[side] = c
ys = [by[n][s][1] for n in by for s in by[n]]
yc = 0.5 * (max(ys) + min(ys))
half = 0.5 * (max(ys) - min(ys))
order = ["step1", "step2", "step3", "step4", "platform", "handrail"]
anchors = []
for n in order:
    hz = holds[n]["z"]
    if n not in by:                                        # never held by a hand in this clip
        anchors.append([0.5 * (holds[n]["x"][0] + holds[n]["x"][1]), yc, hz - 0.02]); continue
    cx = by[n]["left"][0]
    flat_pad = cx + (a.pad_u - a.palm_pt)                  # pad point once the hand lies flat
    if n == "step4":
        anchors.append([flat_pad, yc, hz - 0.02])
    elif n == "platform":                                  # arrives on the nose, then rolls flat
        nose = cx - 0.0235
        anchors.append([0.5 * (nose + flat_pad), yc, hz - 0.02])
    else:                                                  # handrail: pivot (as stored), then flat
        flat_pad = (-0.042 + 0.0235) + (a.pad_u - a.palm_pt)
        anchors.append([0.5 * (cx + flat_pad), yc, holds[n]["z"] - 0.03])
(clip / "anchors.json").write_text(json.dumps(dict(anchors=anchors, hand_y_centre=yc, hand_half_spacing=half,
                                                   pad_u=a.pad_u), indent=1))
print("anchors:", [[round(v, 4) for v in x] for x in anchors])
print(f"hands at y {yc+half:+.3f} / {yc-half:+.3f}")
print("overrides:")
print(f"  task.command.contact_target_pos_offset=[[0.0,{half:.4f},0.0],[0.0,{-half:.4f},0.0]]")
print(f"  task.command.contact_eef_pos_offset=[[{a.pad_u},-0.017,0.0],[{a.pad_u},0.017,0.0]]")
print("  +task.command.ladder_config.contact_idx=13")

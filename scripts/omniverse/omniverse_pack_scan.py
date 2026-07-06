#!/usr/bin/env python3
"""Scan a downloaded Omniverse asset-pack ZIP for Fiatlux-relevant assets.

Reads only the ZIP central directory (fast, no full extraction), reports the
pack's overall structure plus keyword-categorized matches (bulbs, lamps, lights,
ladders, ...), and can extract just the matching assets to an output dir.

Usage:
  python scripts/omniverse/omniverse_pack_scan.py <pack.zip>                 # scan + report
  python scripts/omniverse/omniverse_pack_scan.py <pack.zip> --extract DIR   # + extract matches (nested)
  python scripts/omniverse/omniverse_pack_scan.py <pack.zip> --extract DIR --flat  # flatten to DIR/<design>/
  python scripts/omniverse/omniverse_pack_scan.py <pack.zip> --md            # markdown for the log
"""
import os
import sys
import zipfile
from collections import defaultdict

# category -> substrings to match against the asset path (lowercased)
CATEGORIES = {
    "light_bulb": ["bulb", "lightbulb"],
    "lamp": ["lamp", "_lamp", "tablelamp", "floorlamp", "desklamp"],
    "light_fixture": ["light", "luminaire", "sconce", "chandelier", "pendant",
                      "downlight", "spotlight", "lantern", "fixture", "led", "bulbholder",
                      "ceilinglight", "walllight"],
    "socket": ["socket", "bulbholder", "lampholder", "lampsocket", "e26", "e27"],
    "ladder": ["ladder", "stepstand", "stepstool", "stepladder", "scaffold",
               "stair", "rung", "step_stand", "platform_ladder"],
}
ASSET_EXTS = (".usd", ".usda", ".usdc", ".usdz")


def top_level(name):
    parts = name.split("/")
    return parts[0] if parts else name


def common_prefix(paths):
    """Longest common folder prefix across USD paths, stripped for readable asset paths."""
    if not paths:
        return ""
    sp = [p.split("/")[:-1] for p in paths]
    out = []
    for i in range(min(len(s) for s in sp)):
        col = {s[i] for s in sp}
        if len(col) == 1:
            out.append(col.pop())
        else:
            break
    return "/".join(out) + ("/" if out else "")


def detail_report(names, sub):
    """--detail <substr>: list every model USD whose path contains <substr>, grouped by folder."""
    hits = [n for n in names if n.lower().endswith(ASSET_EXTS) and sub in n.lower()]
    byfolder = defaultdict(list)
    for n in hits:
        byfolder[n.rsplit("/", 1)[0]].append(n.rsplit("/", 1)[-1])
    print(f"# detail: '{sub}' — {len(hits)} USD files in {len(byfolder)} folders\n")
    for folder in sorted(byfolder):
        files = sorted(byfolder[folder])
        print(f"{folder}/  ({len(files)})")
        for f in files:
            print(f"    {f}")


def extract_matches(zf, names, wanted, extract_dir, pack, flat):
    """Extract the matched asset folders to extract_dir (flat = curated per-design layout)."""
    n_ex = 0
    if flat:
        # curated flat layout: each matched folder -> extract_dir/<basename>/...,
        # stripping the pack's deep nesting so no separate flatten pass is needed
        os.makedirs(extract_dir, exist_ok=True)
        for w in wanted:                              # w is relative, e.g. Equipment/Ladders/AlumStep_A
            base = w.rsplit("/", 1)[-1]
            if base.startswith("."):                  # .SubUSDs / .thumbs -- stub containers, not designs
                continue
            for n in names:
                if n.endswith("/"):
                    continue
                i = n.find(w + "/")                   # locate the matched folder in the full zip path
                if i < 0:
                    continue
                out = os.path.join(extract_dir, base, n[i + len(w) + 1:])
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with zf.open(n) as src, open(out, "wb") as dst:
                    dst.write(src.read())
                n_ex += 1
        print(f"\n[extracted {n_ex} files (flattened) for {len(wanted)} folders -> {extract_dir}]",
              file=sys.stderr)
    else:
        dest = os.path.join(extract_dir, pack)
        os.makedirs(dest, exist_ok=True)
        for n in names:
            if any(n.startswith(w) or w in n for w in wanted):
                zf.extract(n, dest)
                n_ex += 1
        print(f"\n[extracted {n_ex} files for {len(wanted)} matched folders -> {dest}]",
              file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    zip_path = sys.argv[1]
    extract_dir = None
    if "--extract" in sys.argv:
        extract_dir = sys.argv[sys.argv.index("--extract") + 1]

    pack = os.path.basename(zip_path).rsplit(".", 1)[0]
    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()

    # --detail <substr>: list every model USD whose path contains <substr>, by folder
    if "--detail" in sys.argv:
        sub = sys.argv[sys.argv.index("--detail") + 1].lower()
        detail_report(names, sub)
        zf.close()
        return

    # count only real MODEL usd files — exclude material/texture/thumbnail USDs so
    # per-asset and per-subcategory totals reflect actual models, not sub-files.
    NOISE = ("/materials/", "/textures/", "/.thumbs/")
    usd_names = [n for n in names
                 if n.lower().endswith(ASSET_EXTS) and not any(x in n.lower() for x in NOISE)]

    # overall structure: top-level folders and how many USDs under each
    tops = defaultdict(int)
    for n in usd_names:
        tops[top_level(n)] += 1

    root = common_prefix(usd_names)

    # keyword matches: bucket by the *asset folder* (parent dir of each matched USD)
    matches = {c: set() for c in CATEGORIES}
    for n in usd_names:
        low = n.lower()
        folder = n.rsplit("/", 1)[0]
        if folder.startswith(root):
            folder = folder[len(root):]
        for cat, kws in CATEGORIES.items():
            if any(k in low for k in kws):
                matches[cat].add(folder)

    size_gb = os.path.getsize(zip_path) / 1e9
    out = []
    out.append(f"## {pack}")
    out.append(f"- ZIP size: {size_gb:.1f} GB | entries: {len(names)} | USD files: {len(usd_names)}")
    out.append(f"- Top-level folders ({len(tops)}): " +
               ", ".join(f"{k} ({v} usd)" for k, v in sorted(tops.items(), key=lambda x: -x[1])[:12]))
    out.append(f"- Asset root: `{root}`")
    out.append("")
    out.append("**Fiatlux-relevant matches:**")
    any_hit = False
    for cat in CATEGORIES:
        ms = sorted(matches[cat])
        if ms:
            any_hit = True
            out.append(f"- `{cat}` ({len(ms)}):")
            for m in ms[:40]:
                out.append(f"    - {m}")
            if len(ms) > 40:
                out.append(f"    - ... +{len(ms) - 40} more")
    if not any_hit:
        out.append("- _none found_")

    # full inventory: subcategory -> {asset folder: variant count}
    inv = defaultdict(lambda: defaultdict(int))
    for n in usd_names:
        rel = n[len(root):] if n.startswith(root) else n
        parts = rel.split("/")
        if len(parts) >= 2:
            asset = parts[-2]
            subcat = "/".join(parts[:-2]) or "(root)"
        else:
            asset, subcat = parts[-1], "(root)"
        inv[subcat][asset] += 1
    out.append("")
    out.append("**Full inventory (subcategory → asset types × variants):**")
    for subcat in sorted(inv):
        assets = inv[subcat]
        nvar = sum(assets.values())
        out.append(f"- `{subcat}` — {len(assets)} types, {nvar} USD:")
        line = ", ".join(f"{a}×{c}" for a, c in sorted(assets.items()))
        out.append(f"    {line}")

    text = "\n".join(out)
    print(text)

    if extract_dir:
        flat = "--flat" in sys.argv
        wanted = sorted({m for ms in matches.values() for m in ms})
        extract_matches(zf, names, wanted, extract_dir, pack, flat)
    zf.close()


if __name__ == "__main__":
    main()

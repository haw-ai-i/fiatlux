#!/usr/bin/env python3
"""Extract the shared material library the Warehouse ladders reference from a
downloaded pack zip -- SCOPED to what the ladders actually use, not the whole
~7 GB `Materials/` folder. Into OUT (`.../Materials`) it pulls:
  1. all base .mdl definitions (small);
  2. the base .png textures the ladders' materials reference -- ~63 files
     (~0.7 GB), found by reading the ladders' own material bindings;
  3. the per-design decal materials + textures (Equipment/Ladders/Materials/).

Complements `omniverse_pack_scan.py`: scan extracts the ladder *models* (and each
SimReady ladder's embedded materials); this pulls the *shared* Warehouse
`Materials/` library that lives outside every design folder. Scoping reads the
already-extracted flattened ladders from `dirname(OUT)` to learn which base mdls
(hence textures) to fetch, so run this AFTER scan.

Usage: python scripts/omniverse/omniverse_pack_scan.py <pack.zip> --extract <ROOT> --flat
       python scripts/omniverse/omniverse_pack_extract.py <pack.zip> <ROOT>/Materials
"""

import glob
import os
import re
import shutil
import sys
import zipfile

from pxr import Sdf, Usd, UsdShade


def extract(zf, name, dest):
    """Write one zip member to dest (creating parent dirs)."""
    data = zf.read(name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    return len(data)


def ladder_base_mdls(root):
    """Base-relative .mdl paths the flattened ladders bind (e.g. Metals/Steel_A1.mdl)."""
    used = set()
    for f in glob.glob(os.path.join(root, "*", "*.usd")):
        if f.endswith(("_collision.usd", "_collision_rigid.usd")):
            continue
        try:
            st = Usd.Stage.Open(f)  # keep a ref: PrimRange over a temp stage's root expires
            if not st:
                continue
            for p in Usd.PrimRange(st.GetPseudoRoot()):
                if p.GetTypeName() == "Shader":
                    a = UsdShade.Shader(p).GetSourceAsset("mdl")
                    if a and "/Base/" in str(a):
                        used.add(str(a).strip("@").split("/Base/")[-1])
        except Exception:
            pass
    return used


def textures_of(outbase, mdl_rels):
    """Base-relative .png paths referenced by the given (already-extracted) mdls."""
    texs = set()
    for rel in mdl_rels:
        mp = os.path.join(outbase, rel)
        if not os.path.exists(mp):
            continue
        with open(mp, encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
        for m in re.findall(r'"([^"]*\.png)"', txt):
            texs.add(os.path.normpath(os.path.join(os.path.dirname(rel), m)).replace("\\", "/"))
    return texs


def ladder_input_textures(root):
    """Base-relative .png paths referenced directly by ladder shader inputs (baked texture
    overrides), with the deep ../-chain collapsed away."""
    texs = set()
    for f in glob.glob(os.path.join(root, "*", "*.usd")):
        if f.endswith(("_collision.usd", "_collision_rigid.usd")):
            continue
        try:
            st = Usd.Stage.Open(f)
            if not st:
                continue
            for p in Usd.PrimRange(st.GetPseudoRoot()):
                if p.GetTypeName() != "Shader":
                    continue
                for inp in UsdShade.Shader(p).GetInputs():
                    v = inp.Get()
                    if isinstance(v, Sdf.AssetPath) and v.path and "/Base/" in v.path:
                        texs.add(v.path.split("/Base/")[-1])
        except Exception:
            pass
    return texs


def pull(zf, names, dest_of, label):
    """Extract each name via dest_of(name); skip what's already on disk (idempotent)."""
    got = skipped = 0
    for n in sorted(names):
        dest = dest_of(n)
        if os.path.exists(dest):
            skipped += 1
            continue
        try:
            extract(zf, n, dest)
            got += 1
        except Exception as e:
            print(f"  [FAIL] {n}: {e}", flush=True)
    print(f"  extracted {got} {label}" + (f" ({skipped} already present)" if skipped else ""), flush=True)
    return got


def reconcile_dead_refs(out, missing):
    """Some ladders name a texture with a folder/underscore typo (Aluminum_Brushed/ vs
    Aluminum/, MetalPainted vs Metal_Painted). The real file exists under a differently-spelled
    name -> copy it to the referenced path, matched by normalized name."""
    base = os.path.join(out, "Base")

    def norm(s):
        return s.lower().replace("_", "")

    have = {}
    for dp, _, fs in os.walk(base):
        for fn in fs:
            if fn.endswith(".png"):
                have.setdefault(norm(fn), os.path.join(dp, fn))
    fixed = 0
    for rel in missing:
        src = have.get(norm(os.path.basename(rel)))
        dst = os.path.join(base, rel)
        if src and not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            fixed += 1
    print(f"  reconciled {fixed}/{len(missing)} dead-ref textures (typo names) from real files", flush=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    zip_path = sys.argv[1]
    out = (
        os.path.abspath(sys.argv[2])
        if len(sys.argv) > 2
        else os.path.join(os.path.dirname(os.path.abspath(zip_path)), "Warehouse_ladders", "Materials")
    )
    root = os.path.dirname(out)  # the flattened ladder dir (models already extracted by scan)

    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()

    # 1) all base .mdl definitions (small)
    mdls = [n for n in names if "/Materials/Base/" in n and n.lower().endswith(".mdl") and ".thumbs" not in n]
    print(f"base mdls: {len(mdls)}", flush=True)
    pull(zf, mdls, lambda n: os.path.join(out, "Base", n.split("/Materials/Base/")[-1]), "base .mdl files")

    # 2) base textures SCOPED to what the ladders reference -- both the mdls' own
    #    textures and the ladders' baked shader-input overrides
    used = ladder_base_mdls(root)
    texs = textures_of(os.path.join(out, "Base"), used) | ladder_input_textures(root)
    print(f"ladders use {len(used)} base mdls; {len(texs)} referenced textures", flush=True)
    tex_names = [
        n for n in names if "/Materials/Base/" in n and ".thumbs" not in n and n.split("/Materials/Base/")[-1] in texs
    ]
    missing = sorted(texs - {n.split("/Materials/Base/")[-1] for n in tex_names})
    pull(zf, tex_names, lambda n: os.path.join(out, "Base", n.split("/Materials/Base/")[-1]), "base textures")
    if missing:
        reconcile_dead_refs(out, missing)

    # 3) per-design decal materials + textures (Equipment/Ladders/Materials/)
    decals = [n for n in names if "/Equipment/Ladders/Materials/" in n and not n.endswith("/") and ".thumbs" not in n]
    print(f"decal files: {len(decals)}", flush=True)
    pull(
        zf,
        decals,
        lambda n: os.path.join(out, n.split("/Equipment/Ladders/Materials/")[-1]),
        "decal mdls + textures",
    )

    zf.close()
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()

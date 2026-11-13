"""Split a Psi-0 fine-tune checkpoint into the part that changed, and rebuild it from that part.

The S06 fine-tune (docs/psi0_finetune.md) trains only the action expert; the Qwen3-VL backbone is
frozen. So the released Psi-0 checkpoint plus the ``action_header.*`` tensors is the whole model,
and the 7 GB ``model.safetensors`` does not need to be shipped. Runs in the Psi0 venv.

    # extract (and check that every vlm_model.* tensor equals the base, cast to bf16)
    python scripts/psi0/split_ft_ckpt.py extract --ft <run_dir>/checkpoints/ckpt_2000/model.safetensors \
        --base <psi-dream run_dir>/checkpoints/ckpt_40000/model.safetensors --out action_header.safetensors

    # rebuild a servable model.safetensors
    python scripts/psi0/split_ft_ckpt.py assemble --head action_header.safetensors \
        --base <psi-dream run_dir>/checkpoints/ckpt_40000/model.safetensors \
        --out <run_dir>/checkpoints/ckpt_2000/model.safetensors
"""

import argparse

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HEAD = "action_header."
VLM = "vlm_model."


def extract(args) -> None:
    head, n_vlm, diffs = {}, 0, []
    with safe_open(args.ft, "pt") as ft, safe_open(args.base, "pt") as base:
        base_keys = set(base.keys())
        for key in ft.keys():
            if key.startswith(HEAD):
                head[key] = ft.get_tensor(key).contiguous()
                continue
            assert key.startswith(VLM), f"unexpected key {key}"
            n_vlm += 1
            t = ft.get_tensor(key)
            if key not in base_keys:
                diffs.append((key, "missing in base"))
                continue
            b = base.get_tensor(key).to(t.dtype)
            if b.shape != t.shape or not torch.equal(b, t):
                diffs.append((key, "differs"))
    print(f"{len(head)} action_header tensors, {n_vlm} vlm tensors checked, {len(diffs)} differ from the base")
    for key, why in diffs[:20]:
        print("  ", why, key)
    if diffs and not args.allow_vlm_diff:
        raise SystemExit("VLM differs from the base: the head alone would not reproduce this checkpoint")
    save_file(head, args.out, metadata={"vlm_dtype": str(t.dtype)})
    print(f"wrote {args.out}")


def assemble(args) -> None:
    with safe_open(args.head, "pt") as h:
        tensors = {k: h.get_tensor(k) for k in h.keys()}
        vlm_dtype = getattr(torch, (h.metadata() or {}).get("vlm_dtype", "torch.bfloat16").split(".")[-1])
    with safe_open(args.base, "pt") as base:
        for key in base.keys():
            if key.startswith(VLM):
                tensors[key] = base.get_tensor(key).to(vlm_dtype).contiguous()
    save_file(tensors, args.out)
    print(f"wrote {args.out} ({len(tensors)} tensors)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--ft", required=True)
    e.add_argument("--base", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--allow-vlm-diff", action="store_true")
    a = sub.add_parser("assemble")
    a.add_argument("--head", required=True)
    a.add_argument("--base", required=True)
    a.add_argument("--out", required=True)
    args = p.parse_args()
    {"extract": extract, "assemble": assemble}[args.cmd](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""helper for BEHAVIOR-1K lamp/bulb asset intake.

This script intentionally avoids importing the full OmniGibson simulator stack.
It loads OmniGibson's official asset utility module with a minimal shim so we can
use the official dataset download/key flow and inspect the encrypted dataset
layout for candidate lamp/bulb assets.

It must not be used to commit, upload, or persist decrypted BEHAVIOR assets.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import socket
import sys
import types
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATES = [
    ("light_bulb", "kfmkwd", "replacement_bulb"),
    ("broken_light_bulb", "cugtye", "broken_bulb"),
    ("table_lamp", "ehjsdz", "lamp_fixture"),
]

LAMP_CATEGORY_HINTS = [
    "table_lamp",
    "floor_lamp",
    "wall_mounted_light",
    "room_light",
    "downlight",
    "wall_socket",
]

LOGGER = logging.getLogger("behavior1k_asset_intake")


def _install_omnigibson_shim(data_dir: Path, temp_dir: Path, behavior_repo: Path) -> None:
    """Install minimal modules expected by omnigibson.utils.asset_utils."""
    data_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    og = types.ModuleType("omnigibson")
    og.example_config_path = str(
        behavior_repo / "OmniGibson" / "omnigibson" / "configs"
    )
    og.tempdir = str(temp_dir)
    og.shutdown = lambda: None

    macros = types.ModuleType("omnigibson.macros")
    macros.gm = types.SimpleNamespace(DATA_PATH=str(data_dir))

    utils = types.ModuleType("omnigibson.utils")
    ui_utils = types.ModuleType("omnigibson.utils.ui_utils")
    ui_utils.create_module_logger = lambda module_name: logging.getLogger(module_name)

    sys.modules["omnigibson"] = og
    sys.modules["omnigibson.macros"] = macros
    sys.modules["omnigibson.utils"] = utils
    sys.modules["omnigibson.utils.ui_utils"] = ui_utils


def load_asset_utils(behavior_repo: Path, data_dir: Path, temp_dir: Path):
    """Load OmniGibson's asset_utils.py with a minimal shim."""
    os.environ.setdefault("OMNIGIBSON_NO_OMNIVERSE", "1")
    _install_omnigibson_shim(data_dir=data_dir, temp_dir=temp_dir, behavior_repo=behavior_repo)

    asset_utils_path = (
        behavior_repo / "OmniGibson" / "omnigibson" / "utils" / "asset_utils.py"
    )
    if not asset_utils_path.exists():
        raise FileNotFoundError(asset_utils_path)

    spec = importlib.util.spec_from_file_location(
        "fiatlux_omnigibson_asset_utils", asset_utils_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {asset_utils_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise ImportError(
            "Failed to import OmniGibson asset_utils with the lightweight shim. "
            "asset_utils may now depend on an additional omnigibson.* submodule. "
            "Add a focused shim for that import, or run the official OmniGibson setup "
            "instead of this lightweight helper."
        ) from exc
    return module


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _suffix_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for item in path.rglob("*"):
        if item.is_file():
            suffix = "".join(item.suffixes) or "<none>"
            counts[suffix] = counts.get(suffix, 0) + 1
    return dict(sorted(counts.items()))


def _dataset_layout_exists(dataset_path: Path) -> bool:
    return (dataset_path / "objects").is_dir() and (dataset_path / "metadata").is_dir()


def _online_zip_filename(asset_utils: Any, dataset_name: str) -> str:
    if dataset_name == "behavior-1k-assets":
        return f"behavior-1k-assets-{asset_utils.BEHAVIOR_1K_ASSET_VERSION}.zip"
    if dataset_name == "omnigibson-robot-assets":
        return f"omnigibson-robot-assets-{asset_utils.OMNIGIBSON_ROBOT_ASSETS_VERSION}.zip"
    return f"{dataset_name}.zip"


def download_zipped_dataset(
    asset_utils: Any, dataset_name: str, download_dir: Path
) -> Path:
    """Download a BEHAVIOR zip with a persistent local_dir, then unpack it."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for dataset downloads. "
            "Run with `uv run --with huggingface_hub ...`."
        ) from exc

    download_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(asset_utils.get_dataset_path(dataset_name))
    dataset_path.mkdir(parents=True, exist_ok=True)
    online_filename = _online_zip_filename(asset_utils, dataset_name)

    LOGGER.info("downloading %s into %s", online_filename, download_dir)
    zip_path = Path(
        hf_hub_download(
            repo_id="behavior-1k/zipped-datasets",
            filename=online_filename,
            repo_type="dataset",
            local_dir=str(download_dir),
        )
    )
    LOGGER.info("extracting %s into %s", zip_path, dataset_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path)
    return dataset_path


def _resolve_download_dir(cli_value: Path | None, data_dir: Path, dataset_name: str) -> Path:
    if cli_value is not None:
        return cli_value.resolve()
    return data_dir / "_downloads" / dataset_name


def _relative_files(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    files = [str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()]
    return sorted(files)[:limit]


def _model_record(
    data_dir: Path, category: str, model: str, role: str, max_files: int
) -> dict[str, Any]:
    model_dir = data_dir / "behavior-1k-assets" / "objects" / category / model
    usd_dir = model_dir / "usd"
    expected_usd = usd_dir / f"{model}.usd"
    expected_encrypted_usd = usd_dir / f"{model}.encrypted.usd"

    return {
        "category": category,
        "model": model,
        "role": role,
        "model_dir": str(model_dir),
        "exists": model_dir.exists(),
        "expected_usd": str(expected_usd),
        "expected_usd_exists": expected_usd.exists(),
        "expected_encrypted_usd": str(expected_encrypted_usd),
        "expected_encrypted_usd_exists": expected_encrypted_usd.exists(),
        "size_bytes": _dir_size(model_dir),
        "suffix_counts": _suffix_counts(model_dir),
        "sample_files_limit": max_files,
        "sample_files": _relative_files(model_dir, limit=max_files),
    }


def inspect_dataset(data_dir: Path, max_files: int) -> dict[str, Any]:
    dataset_dir = data_dir / "behavior-1k-assets"
    objects_dir = dataset_dir / "objects"
    metadata_dir = dataset_dir / "metadata"

    categories = []
    if objects_dir.exists():
        categories = sorted(p.name for p in objects_dir.iterdir() if p.is_dir())

    hinted_categories = {}
    for category in LAMP_CATEGORY_HINTS:
        category_dir = objects_dir / category
        hinted_categories[category] = {
            "exists": category_dir.exists(),
            "models": sorted(p.name for p in category_dir.iterdir() if p.is_dir())
            if category_dir.exists()
            else [],
        }

    candidates = [
        _model_record(data_dir, category, model, role, max_files=max_files)
        for category, model, role in DEFAULT_CANDIDATES
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "dataset_dir": str(dataset_dir),
        "dataset_exists": dataset_dir.exists(),
        "metadata_dir": str(metadata_dir),
        "metadata_exists": metadata_dir.exists(),
        "category_count": len(categories),
        "sample_files_limit": max_files,
        "categories": categories,
        "hinted_categories": hinted_categories,
        "candidate_models": candidates,
        "license_gate": {
            "decrypted_assets_persisted": False,
            "gcs_upload_allowed": False,
            "reason": "BEHAVIOR EULA allows use within OmniGibson and forbids redistribution; explicit project approval required before any decrypted asset upload.",
        },
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behavior-repo",
        type=Path,
        default=None,
        help="Path to a BEHAVIOR-1K checkout.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="OmniGibson data path where behavior-1k-assets and omnigibson.key live.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Temporary directory for OmniGibson-compatible helper state.",
    )
    parser.add_argument(
        "--network-timeout",
        type=float,
        default=None,
        help="Optional default socket timeout in seconds for network calls.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_all = subparsers.add_parser(
        "download-all",
        help="Accept EULA, install the BEHAVIOR key, then download encrypted assets.",
    )
    download_all.add_argument(
        "--accept-license",
        action="store_true",
        help="Required. Accept the BEHAVIOR Data Bundle EULA for academic research use.",
    )
    download_all.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help=(
            "Persistent Hugging Face local_dir for the asset zip "
            "(default: DATA_DIR/_downloads/behavior-1k-assets)."
        ),
    )

    download_assets_description = (
        "Accept EULA and download only the encrypted BEHAVIOR asset zip. "
        "This does not install omnigibson.key and does not make assets usable "
        "for decrypted runtime inspection. Use download-all when you need the "
        "normal key-plus-assets order."
    )
    download_assets = subparsers.add_parser(
        "download-assets",
        help=download_assets_description,
        description=download_assets_description,
    )
    download_assets.add_argument(
        "--accept-license",
        action="store_true",
        help="Required. Accept the BEHAVIOR Data Bundle EULA for academic research use.",
    )
    download_assets.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help=(
            "Persistent Hugging Face local_dir for the asset zip "
            "(default: DATA_DIR/_downloads/behavior-1k-assets)."
        ),
    )

    key = subparsers.add_parser(
        "download-key", help="Accept EULA and install the BEHAVIOR key only."
    )
    key.add_argument(
        "--accept-license",
        action="store_true",
        help="Required. Accept the BEHAVIOR Data Bundle EULA for academic research use.",
    )

    inspect = subparsers.add_parser("inspect", help="Inspect encrypted dataset layout.")
    inspect.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output manifest JSON path.",
    )
    inspect.add_argument(
        "--max-files",
        type=int,
        default=80,
        help="Maximum number of sample files to include per candidate model.",
    )

    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    temp_dir = (
        args.temp_dir.resolve()
        if args.temp_dir is not None
        else data_dir / "_fiatlux_temp"
    )
    if args.network_timeout is not None:
        socket.setdefaulttimeout(args.network_timeout)

    needs_behavior_repo = {"download-all", "download-assets", "download-key"}
    behavior_repo = None
    if args.command in needs_behavior_repo:
        if args.behavior_repo is None:
            raise SystemExit(f"--behavior-repo is required for {args.command}")
        behavior_repo = args.behavior_repo.resolve()

    if args.command == "download-all":
        if not args.accept_license:
            raise SystemExit(f"--accept-license is required for {args.command}")
        assert behavior_repo is not None
        asset_utils = load_asset_utils(
            behavior_repo=behavior_repo, data_dir=data_dir, temp_dir=temp_dir
        )
        key_path = Path(asset_utils.get_key_path())
        if key_path.exists():
            LOGGER.info("BEHAVIOR-1K dataset encryption key already installed.")
        else:
            LOGGER.info("installing BEHAVIOR-1K dataset encryption key")
            asset_utils.download_key()

        dataset_path = Path(asset_utils.get_dataset_path("behavior-1k-assets"))
        if _dataset_layout_exists(dataset_path):
            LOGGER.info("BEHAVIOR-1K dataset already installed: %s", dataset_path)
        else:
            download_zipped_dataset(
                asset_utils=asset_utils,
                dataset_name="behavior-1k-assets",
                download_dir=_resolve_download_dir(
                    args.download_dir, data_dir, "behavior-1k-assets"
                ),
            )
        return 0

    if args.command == "download-assets":
        if not args.accept_license:
            raise SystemExit("--accept-license is required for download-assets")
        assert behavior_repo is not None
        asset_utils = load_asset_utils(
            behavior_repo=behavior_repo, data_dir=data_dir, temp_dir=temp_dir
        )
        dataset_path = Path(asset_utils.get_dataset_path("behavior-1k-assets"))
        if _dataset_layout_exists(dataset_path):
            LOGGER.info("BEHAVIOR-1K dataset already exists: %s", dataset_path)
        else:
            download_zipped_dataset(
                asset_utils=asset_utils,
                dataset_name="behavior-1k-assets",
                download_dir=_resolve_download_dir(
                    args.download_dir, data_dir, "behavior-1k-assets"
                ),
            )
            LOGGER.info("Downloaded BEHAVIOR-1K dataset: %s", dataset_path)
        return 0

    if args.command == "download-key":
        if not args.accept_license:
            raise SystemExit("--accept-license is required for download-key")
        assert behavior_repo is not None
        asset_utils = load_asset_utils(
            behavior_repo=behavior_repo, data_dir=data_dir, temp_dir=temp_dir
        )
        key_path = Path(asset_utils.get_key_path())
        LOGGER.debug("before key exists=%s", key_path.exists())
        asset_utils.download_key()
        LOGGER.debug(
            "after key exists=%s size=%s",
            key_path.exists(),
            key_path.stat().st_size if key_path.exists() else 0,
        )
        return 0

    if args.command == "inspect":
        data = inspect_dataset(data_dir, max_files=args.max_files)
        write_json(args.out, data)
        print(args.out)
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

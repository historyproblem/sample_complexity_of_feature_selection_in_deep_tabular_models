"""Install pruning runtime dependencies without replacing the existing environment."""
from __future__ import annotations

import argparse
from importlib import metadata, util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements-pruning-search.txt"
PYTORCH_INDEX = "https://download.pytorch.org/whl/"


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_versions():
    versions = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            versions.setdefault(normalize(name), dist.version)
    return versions


def tensor_install_command(versions, platform=None):
    """Preserve complete existing pairs; let pip resolve a missing matched partner."""
    torch_version, vision_version = versions.get("torch"), versions.get("torchvision")
    if torch_version and vision_version:
        return None
    platform = platform or sys.platform
    if not torch_version and not vision_version:
        if platform.startswith("linux"):
            return ["torch==2.6.0+cu126", "torchvision==0.21.0+cu126",
                    "--index-url", PYTORCH_INDEX + "cu126"]
        return ["torch==2.6.0", "torchvision==0.21.0"]
    requirements = [
        "torch==" + torch_version if torch_version else "torch",
        "torchvision==" + vision_version if vision_version else "torchvision",
    ]
    local_version = (torch_version or vision_version).partition("+")[2]
    if local_version == "cpu" or re.fullmatch(r"cu[0-9]+", local_version):
        requirements += ["--index-url", PYTORCH_INDEX + local_version]
    return requirements


def write_constraints(path, versions):
    path.write_text("".join(f"{name}=={version}\n" for name, version in sorted(versions.items())))


def install_dependencies():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use .venv/bin/python; refusing to install into the system interpreter")
    if util.find_spec("pip") is None:
        subprocess.run([sys.executable, "-m", "ensurepip"], check=True)
    before = installed_versions()
    tensor_args = tensor_install_command(before)
    print("[dependencies] Installing missing runtime requirements; preserving installed versions", flush=True)
    with tempfile.TemporaryDirectory(prefix="pruning-dependencies-") as directory:
        constraints = Path(directory) / "installed-versions.txt"
        write_constraints(constraints, before)
        if tensor_args:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--constraint", str(constraints), *tensor_args],
                check=True,
            )
        # Also protect any packages added with a newly installed torch pair.
        current = installed_versions()
        if not current.get("torch") or not current.get("torchvision"):
            raise RuntimeError("torch/torchvision installation is incomplete")
        write_constraints(constraints, current)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--constraint", str(constraints),
             "--requirement", str(REQUIREMENTS),
             "torch==" + current["torch"], "torchvision==" + current["torchvision"]],
            check=True,
        )
    after = installed_versions()
    changed = {name: (version, after.get(name)) for name, version in before.items()
               if after.get(name) != version}
    if changed:
        raise RuntimeError("Installed versions changed; refusing to start training: " + repr(changed))
    # Check binary/import compatibility in a fresh process, without data or CUDA calls.
    probe = """
import json,sys
sys.path.insert(0, sys.argv[1])
import yaml,hydra,omegaconf,numpy,pandas,PIL,tqdm,torch,torchvision
import net_complexity.training.one_shot_pruning
print(json.dumps({"torch":torch.__version__,"torchvision":torchvision.__version__,
                  "compiled_cuda":torch.version.cuda}))
"""
    subprocess.run([sys.executable, "-c", probe, str(ROOT / "src")], check=True)
    result = {
        "status": "ready", "python": sys.version.split()[0], "executable": sys.executable,
        "existing_versions_preserved": True, "before": before, "after": after,
        "training_started": False, "source_requirements": str(REQUIREMENTS),
    }
    destination = ROOT / "outputs/setup/pruning-dependencies.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print("[dependencies] Imports verified; installation complete", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the setup plan without pip/network/import probes")
    args = parser.parse_args(argv)
    if args.dry_run:
        versions = installed_versions()
        plan = {"executable": sys.executable, "virtualenv": sys.prefix != sys.base_prefix,
                "tensor_install": tensor_install_command(versions),
                "requirements": str(REQUIREMENTS), "preserve_existing_versions": True,
                "training_started": False}
        print(json.dumps(plan, indent=2))
        return plan
    return install_dependencies()


if __name__ == "__main__":
    main()

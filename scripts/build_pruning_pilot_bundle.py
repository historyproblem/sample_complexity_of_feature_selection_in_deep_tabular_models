"""Create a runnable source snapshot and review patch; never include data/checkpoints."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "22c5866681680129eda04eeabefa8335e591da0e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prefixes = ("src/", "configs/", "scripts/", "tests/", "docs/")
    root_files = {"setup.py", "requirements.txt", "pytest.ini", "README.md", "AGENTS.md", "START_HERE.md"}
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    paths = sorted(p for p in set(tracked + untracked)
                   if (p.startswith(prefixes) or p in root_files) and (ROOT / p).is_file()
                   and "__pycache__" not in p and not p.endswith((".ipynb", ".pyc")))
    patch = subprocess.check_output(["git", "diff", "--binary", BASE, "--", *prefixes, *sorted(root_files)],
                                    cwd=ROOT).decode()
    for name in sorted(set(untracked) & set(paths)):
        content = (ROOT / name).read_text().splitlines(keepends=True)
        patch += f"diff --git a/{name} b/{name}\nnew file mode 100644\n"
        patch += "".join(difflib.unified_diff([], content, fromfile="/dev/null", tofile=f"b/{name}"))
    files = {f"pruning_pilot_fixed/{name}": (ROOT / name).read_bytes() for name in paths}
    files["pruning_pilot_fixed/CHANGES_FROM_22c5866.patch"] = patch.encode()
    files["pruning_pilot_fixed/BASE_COMMIT.txt"] = (BASE + "\n").encode()
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in sorted(files.items())}
    files["pruning_pilot_fixed/MANIFEST.sha256.json"] = (json.dumps(hashes, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for name, expected in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    print(f"Created {args.output}: {len(files)} files, {args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()

"""Render every scene in slides/ and explainer/, one after another.

Run it from the animations folder:

    uv run python render_all.py                  # every scene at low quality
    uv run python render_all.py -q h             # every scene at deck quality
    uv run python render_all.py -s               # the last frame of each scene as a still
    uv run python render_all.py torus aim        # only scenes whose file or class name matches

A scene is any class in those two folders whose base class name ends in Scene.
The script finds them by reading the source, so it does not import Manim or
the data.

The scenes run one at a time. Manim writes its LaTeX files into one shared
folder, out/Tex, and deletes the ones it does not need after each formula. Two
renders at once can delete each other's files, which fails with a
FileNotFoundError on a .log file.

The script prints one line per scene with its time and whether it rendered.
It exits with status 1 if any scene failed. The full Manim output of a failed
scene is in out/logs.
"""

import argparse
import ast
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDERS = ["slides", "explainer"]
TEXBIN = "/Library/TeX/texbin"


def find_scenes():
    """Return (file, class name) for every scene class, sorted by file."""
    scenes = []
    for folder in FOLDERS:
        for path in sorted((HERE / folder).glob("*.py")):
            if path.name.startswith("_"):
                continue
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                bases = [b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases]
                if any(b.endswith("Scene") for b in bases):
                    scenes.append((path.relative_to(HERE), node.name))
    return scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="only render scenes whose file or class name contains one of these")
    parser.add_argument("-q", "--quality", default="l", choices=list("lmhpk"), help="Manim quality letter, l by default")
    parser.add_argument("-s", "--still", action="store_true", help="save the last frame as a still instead of a video")
    args = parser.parse_args()

    scenes = find_scenes()
    if args.names:
        wanted = [n.lower() for n in args.names]
        scenes = [(f, c) for f, c in scenes if any(w in str(f).lower() or w in c.lower() for w in wanted)]
    if not scenes:
        sys.exit("No scene matched.")

    env = dict(os.environ)
    if Path(TEXBIN).is_dir() and TEXBIN not in env.get("PATH", ""):
        env["PATH"] = TEXBIN + os.pathsep + env.get("PATH", "")

    logs = HERE / "out" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    failed = []
    for path, name in scenes:
        command = [sys.executable, "-m", "manim", f"-q{args.quality}"]
        if args.still:
            command.append("-s")
        command += [str(path), name]
        start = time.time()
        result = subprocess.run(command, cwd=HERE, env=env, capture_output=True, text=True)
        seconds = time.time() - start
        log = logs / f"{name}.log"
        log.write_text(result.stdout + result.stderr)
        status = "ok" if result.returncode == 0 else f"FAILED, see {log.relative_to(HERE)}"
        print(f"{name:<28} {str(path):<42} {seconds:6.1f} s  {status}", flush=True)
        if result.returncode != 0:
            failed.append(name)

    if failed:
        print(f"{len(failed)} of {len(scenes)} scenes failed.")
        sys.exit(1)
    print(f"All {len(scenes)} scenes rendered.")


if __name__ == "__main__":
    main()

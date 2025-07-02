import subprocess
import pathlib
import json
import sys
import tempfile
import os
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
SELF_PATH = pathlib.Path(__file__).resolve()

BLACK_ARGS = ["black", "--quiet"]
PYLINT_ARGS = ["pylint", str(ROOT), "-f", "json"]


def pylint_score(path: Optional[pathlib.Path] = None) -> float:
    """
    Calculate the mean pylint score for all modules in the given path.
    Returns 10.0 if pylint returns no data.
    """
    target = str(path or ROOT)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        subprocess.run(PYLINT_ARGS + ["-o", tmp.name], check=False)
        try:
            data = json.load(open(tmp.name))
        except Exception:
            data = []
    os.unlink(tmp.name)
    if not data:
        return 10.0
    return sum(m.get("score", 0) for m in data) / len(data)


def auto_format(path: Optional[pathlib.Path] = None) -> None:
    """
    Run Black auto-formatter on the given path quietly.
    """
    subprocess.run(BLACK_ARGS + [str(path or ROOT)], check=False)


def main(threshold: float = 8.5) -> None:
    """
    Checks code quality using pylint, and auto-formats with Black if needed.
    """
    before = pylint_score()
    print(f"Pylint score before: {before:.2f}")
    if before >= threshold:
        print("Quality satisfactory. No formatting needed.")
        return
    print("Running Black auto-formatting...")
    auto_format()
    after = pylint_score()
    print(f"Pylint score after: {after:.2f}")


if __name__ == "__main__":
    main()

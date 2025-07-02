import subprocess
import json
from typing import Any

def pylint_score(path: str = ".") -> float:
    """
    Run pylint and extract the code quality score for the given path.
    Returns a float score (0-10).
    """
    proc = subprocess.run(
        ["pylint", "--score", "y", path, "-sn", "-rn", "--exit-zero"],
        capture_output=True,
        text=True,
    )
    lines = proc.stdout.strip().split("\n")
    score = 0.0
    for line in reversed(lines):
        if "/" in line:
            try:
                score = float(line.split("/")[0].split()[-1])
                break
            except Exception:
                continue
    return score

def black_check(path: str = ".") -> bool:
    """
    Check if Black formatting is needed for the given path.
    Returns True if code is already clean.
    """
    proc = subprocess.run(
        ["black", "--check", "--diff", path],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0

def run_quality() -> dict:
    """
    Run pylint and Black check, print results, and return as dict.
    """
    result = {
        "pylint": pylint_score(),
        "black_clean": black_check(),
    }
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    run_quality()

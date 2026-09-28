from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCTRINE_DIR = ROOT / "mission_plan-new_plan_250519" / "data" / "doctrine"
SCRIPTS = [
    ROOT / "experiments" / "E20_pipeline_contract_verification" / "run_verification.py",
    ROOT / "experiments" / "E21_end_to_end_verification" / "run_e2e.py",
]


def preflight() -> list[str]:
    """Stop early on setup problems that would otherwise change results silently."""
    problems = []

    pdfs = sorted(DOCTRINE_DIR.rglob("*.pdf"))
    pointers = [p for p in pdfs if p.read_bytes()[:5] != b"%PDF-"]
    if not pdfs or pointers:
        problems.append(
            "Doctrine PDFs are Git LFS pointer files, not real PDFs "
            f"({len(pointers)} of {len(pdfs)}). Install Git LFS, then run "
            "'git lfs install' and 'git lfs pull' in the repository root."
        )

    try:
        import pulp

        if "PULP_CBC_CMD" not in pulp.listSolvers(onlyAvailable=True):
            problems.append(
                f"PuLP {pulp.__version__} has no CBC solver, so the MILP step would fall "
                "back to heuristics. Install the tested versions: "
                "python -m pip install -r reproducibility/requirements-lock.txt"
            )
    except ImportError:
        problems.append(
            "PuLP is not installed: python -m pip install -r reproducibility/requirements-lock.txt"
        )

    return problems


def main() -> int:
    problems = preflight()
    if problems:
        for msg in problems:
            print(f"[SETUP ERROR] {msg}", flush=True)
        return 2

    for script in SCRIPTS:
        print(f"[RUN] {script.relative_to(ROOT)}", flush=True)
        completed = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        if completed.returncode:
            return completed.returncode
    print("[PASS] E20 and E21 completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

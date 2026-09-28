from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "experiments" / "E20_pipeline_contract_verification" / "run_verification.py",
    ROOT / "experiments" / "E21_end_to_end_verification" / "run_e2e.py",
]


def main() -> int:
    for script in SCRIPTS:
        print(f"[RUN] {script.relative_to(ROOT)}", flush=True)
        completed = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        if completed.returncode:
            return completed.returncode
    print("[PASS] E20 and E21 completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

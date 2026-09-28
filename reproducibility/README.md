# IMPS reproducibility package

This directory describes the public reproduction path for the implementation checks reported in the JKSS manuscript.

## Included checks

- `E20_pipeline_contract_verification`: 14 module contract and regression cases.
- `E21_end_to_end_verification`: six-stage synthetic end-to-end workflow.

The experiments are implementation verification only. They do not validate operational mission quality or the threat model against measured air-defense data.

## Environment

1. Install Python 3.13 or a compatible Python 3 version.
2. Install the minimal reproduction dependencies:

```powershell
python -m pip install -r reproducibility/requirements.txt
```

   (The full application has more dependencies in
   `mission_plan-new_plan_250519/requirements.txt`; they are not needed here.)

3. A running Ollama server is **not** required. E01 uses a deterministic
   rule-based command router; the `ollama` package is imported but not called.

4. Optional Cesium terrain access uses the `CESIUM_ION_TOKEN` environment variable. No token is required for E20 or E21.

## Run

From the repository root:

```powershell
python reproducibility/run_checks.py
```

The runner executes E20 first and E21 second. E21 indexes the public doctrine PDFs and may take about one minute on the reported workstation.

## Outputs

- `experiments/E20_pipeline_contract_verification/result.json`
- `experiments/E20_pipeline_contract_verification/summary.txt`
- `experiments/E21_end_to_end_verification/result.json`
- `experiments/E21_end_to_end_verification/summary.txt`
- `experiments/E21_end_to_end_verification/outputs/synthetic_tactical_map.html`
- `experiments/E21_end_to_end_verification/outputs/synthetic_mission.czml.json`

Each result records a protocol SHA-256 hash. E20 also records the origin and execution history of each regression case.

## Public-release note

The source tree contains only publicly released U.S. Government doctrine documents and synthetic scenario data. Documents with limited-distribution markings or unconfirmed distribution terms were removed; see `../THIRD_PARTY_NOTICES.md`. The Cesium token has been removed from source control and must be supplied through an environment variable.

Historical presentation builders and archived experiments are not part of the supported reproduction path and may contain workstation-specific paths. The portable, manuscript-linked entry point is `reproducibility/run_checks.py` with E20 and E21.

The previously embedded Cesium token must be revoked or rotated before publication even though its stored copies have been removed.

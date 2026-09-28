# Third-party notices

The documents under `mission_plan-new_plan_250519/data/doctrine/` are **not**
authored by the IMPS authors and are **not** covered by the repository's MIT
License. They are included so that the E21 doctrine-retrieval step reproduces
identically.

All bundled documents are U.S. Government publications released to the public.
Works of the U.S. Government are not subject to copyright protection in the
United States (17 U.S.C. § 105).

## Doctrine-retrieval index (11 documents)

`doctrine_rag.py` indexes only the files directly under `data/doctrine/`. These
11 documents form the corpus described in Section 3.3 of the manuscript
(3,554 chunks, excluding the authors' summary file `doctrine_basis.md`).

- **Joint:** JP 3-30 Joint Air Operations (2019).
- **Air Force / Space Force:** AFDP 3-03 Counterland Operations, AFDP 5-0 Planning,
  DAFMAN 11-260, AFDP 3-99 / SDP 3-99 (The Department of the Air Force Role in
  Joint All-Domain Operations).
- **Army:** FM 3-04.155, FMI 3-04.155, FM 1-100 (Army Aviation Operations), and
  the small-UAS airspace-management handbook for Army leaders.
- **DoD / Navy:** DoD Unmanned Systems Integrated Roadmap FY2011-2036 (approved
  for open publication), Department of the Navy Unmanned Campaign Framework (2021).

## Reference files (not indexed)

Files under `data/doctrine/ato_refs_2026/` are kept for reference and are not
part of the retrieval index.

- **Joint:** JP 3-30 (2019), JP 3-60 (Joint Targeting), JP 5-0 (Joint Planning, 2020),
  CJCS Guide 3130 (APEX).
- **Air Force:** AFDP 3-0, 3-30, 3-60, and AFMAN 13-1AOC Vol 3.
- **Air Force public-affairs articles (text):** "609th AOC / KRADOS" and
  "ATO 101" (PACAF); published on public af.mil / Air University sites.

## Deliberately excluded before public release

The following were **removed** from this public package because they carry
limited-distribution markings, or their distribution terms or provenance could
not be confirmed. They are not required for E20 or E21 to pass:

- `AFTTP_3-3.Guardian_Angel.pdf` (limited-distribution marking)
- `area51_35.pdf` (limited-distribution marking)
- `CJCSI_3370-01B_Target_Development.pdf` (limited-distribution marking)
- `MITRE_Multi-Scale_AOC_06_1497.pdf` (MITRE technical report)
- `ATO_Dissemination_ADA420267.pdf`, `Evolution_Joint_ATO_Cycle_ADA451239.pdf`,
  `Mayer_NPS_thesis_ADA438397.pdf` (DTIC-accessioned; distribution statement not
  confirmed)

E20 and E21 were re-run after removal; both pass (14/14 and 6/6) and the
committed results reflect this public corpus.

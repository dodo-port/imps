# IMPS public-release checklist

- [x] Remove embedded Cesium tokens from source and generated output.
- [x] Read Cesium access from `CESIUM_ION_TOKEN` only.
- [x] Provide one command for the manuscript-linked E20 and E21 checks.
- [x] Preserve protocol hashes, raw observations, and test-history labels.
- [x] Review redistribution rights for every bundled doctrine document
      (kept public U.S. Government publications; removed items with uncertain
      distribution terms — see `../THIRD_PARTY_NOTICES.md`).
- [x] Re-run E20 and E21 after trimming the corpus (14/14 and 6/6 pass).
- [x] Add a source-code license (MIT — see `../LICENSE`).
- [x] Exclude local logs, editor settings, caches, archived worktrees,
      historical presentation builders, and large non-doctrine data
      (terrain tiles, 3D models, presentations).

## Remaining (owner actions, before/at publication)

- [ ] Revoke or rotate the previously exposed Cesium token in the Cesium account.
- [ ] Add the final repository URL, release tag (v1.0), and commit hash to the
      manuscript's data-availability section.

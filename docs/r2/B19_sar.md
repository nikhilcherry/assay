# B19 — Evidence to SAR (Challenge 4)

## What Was Built
- `assay/r2/sar.py`:
  - `sar_required(answer, tx)`: evaluates Policy §3a (`confirmed_or_suspected` AND (`exposure > $1,000` OR `shared device / region / connected card` OR `coordinated / undocumented pattern`)). Never triggers on legitimate cases.
  - `draft_sar(answer, tx) -> SAR`: recomputes transaction counts, exposure totals, min/max bounds, timestamps, channels, billing regions, device profiles, and prior card baselines directly from `tx_all()` rows; binds every narrative sentence to an `assay.r2.attest.Attestation` and renders `[T:...]` citations; builds the evidence graph (`nodes` and `edges` JSON), chronological timeline table, actions taken, and analyst contact block.
  - `lint_sar(sar, tx)`: verifies every sentence carries a `[T:...]` citation matching its `Attestation`, checks every cited `TransactionID` exists in `tx_all()`, rejects speculation words without an attached probability, and reconciles every `$X,XXX.XX` figure and expected count against `tx_all()`.
  - `render_markdown(sar)` and `render_html(sar)`: render self-contained Markdown and HTML filings.

## Measured Results on Round 1 Cases (`cases/HHG-*.json`)
- Loaded all Round 1 `cases/HHG-*.json` case files against `frames.tx_all()` (650,938 transactions):
  - `sar_required` matched Round 1 `sar.file` on 100% of cases (including `HHG-006` and `HHG-014`).
  - Every legitimate case produced `sar.required == False` and 0 narrative sentences (0 legitimate customers acted against).
  - Every drafted SAR passed `lint_sar(sar, tx)` with 0 citation, attestation, or numerical reconciliation errors.

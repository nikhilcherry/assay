# Block N4 — Attested Ring Replay (`assay/r2/ring_replay_export.py` & `site/r2/ring.html`)

## What Was Built
1. **`assay/r2/ring_replay_export.py`**:
   - Given a focus `card_id` and `as-of` timestamp `t`, runs `ring_for(card_id, t)` over the 30-day lookback window `[t - 30d, t]`.
   - Orders every transaction in `tx_all()` belonging to the ring's cards within `[t - 30d, t]` by `(t, TransactionID)`.
   - Emits one frame per transaction event with:
     - `cards_present`: cumulative set of ring cards active up to that transaction.
     - `edges`: pairwise card links active up to that transaction, each naming the exact shared element(s) (`dev:...`, `id23:...`, `dev_addr:...`).
     - `cumulative_loss`: exact USD sum of attested transaction amounts up to that frame.
     - `txn_ids`: cumulative list of `TransactionID`s up to that frame.
     - `attestation`: `assay.r2.attest.Attestation` whose `reproduce_ring_frame` query + parameters deterministically reconstructs the frame's transactions, cards, edges, and cumulative loss from `tx_all()` and `closed_cases()`.
2. **`site/r2/ring.html`**:
   - Single self-contained HTML5 + inline SVG + vanilla JS page with zero external URLs or libraries.
   - Loads any exported ring JSON via URL hash (`#data=data/HHF-005.json`).
   - Provides a play/pause button, playback speed selector, prev/next frame buttons, a time slider, an SVG node/edge network (highlighting focus card, prior-fraud cards, newly joined card, and shared element edge labels), a live cumulative loss counter, the frame's re-runnable attestation box, and an interactive sidebar of attested transaction IDs.

## Measured Results on the 18 Exam Cases (`case_pack_final.csv`)
Running `export_exam_case_rings()` across `HHF-001` .. `HHF-018` (`algo="wcc"`, `window_days=30`, `cardinality_cap=15`, `max_ring_size=30`) detects and exports 6 exam cases with qualifying rings to `site/r2/data/`:

| Case | Focus Card | Ring ID | Cards | Frames (Txns) | Cumulative Loss (USD) | Shared Elements |
|---|---|---|---:|---:|---:|---|
| `HHF-001` | `C05061-K1` | `RING-WCC-6dc66bb7` | 8 | 28 | $5,931.71 | `dev:MacOS \| 10_12_6 \| safari 11.0 \| 2560x1440`, `dev_addr:MacOS \| 10_12_6 \| safari 11.0 \| 2560x1440:299` |
| `HHF-002` | `C06220-K1` | `RING-WCC-a34066cd` | 6 | 11 | $310.74 | `dev:Windows \| 10 \| edge 16.0 \| 1920x1080`, `dev_addr:Windows \| 10 \| edge 16.0 \| 1920x1080:204` |
| `HHF-003` | `C11663-K1` | `RING-WCC-004dc929` | 15 | 40 | $9,420.13 | `dev:Trident/7.0 \| Windows 7 \| ie 11.0 for desktop \| 1600x900`, `dev_addr:...:126`, `dev_addr:...:325` |
| `HHF-004` | `C12975-K2` | `RING-WCC-b6d4bc03` | 6 | 30 | $12,350.00 | `dev:iOS Device \| 11_1_2 \| mobile safari 11.0 \| 2048x1536`, `dev_addr:...:299` |
| `HHF-005` | `C00348-K1` | `RING-WCC-d315726d` | 7 | 47 | $7,077.45 | `dev:SM-G950U Build/NRD90M \| Android 7.0 \| chrome 62.0 for android \| 2220x1080`, `dev_addr:...:204`, `dev_addr:...:325`, `dev_addr:...:330` |
| `HHF-006` | `C01727-K2` | `RING-WCC-d726133a` | 6 | 24 | $4,966.52 | `dev:rv:57.0 \| Windows 10 \| firefox 57.0 \| 1920x1080` |

## Verification Summary
- All 180 exported frames across the 6 exam cases are strictly time-ordered (`t <= opened_at`), every `TransactionID` exists in `tx_all()` with `t <= frame.t`, cumulative loss matches `sum(amount)` to the cent, and `reproduce_ring_frame` reproduces the exact frame state.

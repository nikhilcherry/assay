"""Tests for Block N4: Ring replay export & self-contained HTML viewer."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from assay.r2.frames import case_pack, closed_cases, tx_all
from assay.r2.ring_replay_export import (
    SITE_DATA_DIR,
    export_exam_case_rings,
    reproduce_ring_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HTML_PATH = REPO_ROOT / "site" / "r2" / "ring.html"


def test_export_exam_case_rings_and_verify_invariants() -> None:
    tx = tx_all()
    cc = closed_cases()
    cp = case_pack()

    exported = export_exam_case_rings(out_dir=SITE_DATA_DIR, tx=tx, cc=cc, cp=cp)

    print("\n=== N4 Ring Replay Export Measurement Table ===")
    print(f"{'Case':<10} {'FocusCard':<12} {'RingID':<20} {'Cards':>6} {'Frames':>7} {'LossUSD':>12} {'SharedElements'}")
    for cid, payload in sorted(exported.items()):
        ring = payload["ring"]
        print(
            f"{cid:<10} {payload['focus_card_id']:<12} {ring['ring_id']:<20} "
            f"{len(ring['cards']):>6} {payload['frame_count']:>7} "
            f"{ring['total_amount']:>12.2f} {', '.join(ring['shared_elements'])}"
        )

    assert len(exported) >= 1, "Expected at least one exam case with a ring exported"
    assert (SITE_DATA_DIR / "HHF-005.json").exists(), "HHF-005.json must be exported to site/r2/data/"
    assert (SITE_DATA_DIR / "manifest.json").exists(), "manifest.json must exist in site/r2/data/"

    tx_indexed = tx.set_index("TransactionID")

    for cid, payload in sorted(exported.items()):
        json_file = SITE_DATA_DIR / f"{cid}.json"
        assert json_file.exists()
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        frames = loaded["frames"]
        assert len(frames) == loaded["frame_count"]
        assert len(frames) > 0

        opened_at = pd.to_datetime(loaded["opened_at"])
        prev_t: pd.Timestamp | None = None
        prev_tid: int | None = None

        for idx, frame in enumerate(frames):
            assert frame["frame_index"] == idx
            frame_t = pd.to_datetime(frame["t"])
            event_tid = int(frame["event_txn_id"])

            # 1. Frames are strictly non-decreasing in time (and TransactionID on ties)
            assert frame_t <= opened_at, f"{cid} frame {idx} exceeds case opened_at"
            if prev_t is not None:
                assert (frame_t > prev_t) or (
                    frame_t == prev_t and event_tid > (prev_tid or -1)
                ), f"{cid} frames not time-ordered at index {idx}"
            prev_t = frame_t
            prev_tid = event_tid

            # 2. Every frame's txn_ids exist in tx_all() with t <= the frame time
            frame_txn_ids = [int(x) for x in frame["txn_ids"]]
            assert event_tid in frame_txn_ids
            assert len(frame_txn_ids) == idx + 1

            sub_rows = tx_indexed.loc[frame_txn_ids]
            assert (sub_rows["t"] <= frame_t).all(), f"{cid} frame {idx} has future txn_id > frame time"

            # 3. Cumulative loss equals the sum of the attested amounts
            expected_loss = round(float(sub_rows["amount"].sum()), 2)
            assert abs(frame["cumulative_loss"] - expected_loss) < 1e-6

            # 4. Attestation structure and exact reproducibility on sample frames
            att = frame["attestation"]
            assert set(att.keys()) == {"what", "evidence", "how", "when"}
            ev_tids = [int(x) for x in att["evidence"]["txn_ids"]]
            assert ev_tids == frame_txn_ids
            assert att["how"]["query"] == "reproduce_ring_frame"

            if idx in (0, len(frames) // 2, len(frames) - 1):
                repro = reproduce_ring_frame(**att["how"]["params"], tx=tx, cc=cc)
                assert repro["txn_ids"] == frame_txn_ids
                assert repro["cards_present"] == frame["cards_present"]
                assert repro["edges"] == frame["edges"]
                assert abs(repro["cumulative_loss"] - frame["cumulative_loss"]) < 1e-6


def test_ring_html_is_self_contained_with_no_external_urls() -> None:
    assert HTML_PATH.exists(), "site/r2/ring.html must exist"
    html = HTML_PATH.read_text(encoding="utf-8")

    # Strip standard SVG namespace declaration before checking for external URLs
    html_no_xmlns = html.replace("http://www.w3.org/2000/svg", "")
    external_urls = re.findall(r"https?://[^\s\"'<>]+", html_no_xmlns, flags=re.IGNORECASE)
    assert external_urls == [], f"HTML must have no external URLs, found: {external_urls}"

    # Verify required UI elements exist
    assert 'id="time-slider"' in html
    assert 'id="play-btn"' in html
    assert 'id="loss-counter"' in html
    assert 'id="txn-sidebar-list"' in html
    assert "window.location.hash" in html

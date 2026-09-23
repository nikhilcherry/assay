"""Detectors on small synthetic frames: each one fires on its shape and, as
importantly, does not fire on the near-miss."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assay import detectors as D  # noqa: E402


def frame(rows):
    df = pd.DataFrame(rows, columns=["TransactionID", "t", "TransactionAmt", "channel", "ProductCD"])
    df["t"] = pd.to_datetime(df.t)
    df["device_profile"] = ""
    return df.set_index("TransactionID", drop=False)


def test_structuring_fires_on_three_near_500_in_an_hour():
    w = frame([(1, "2016-11-21 20:00", 478.95, "online", "C"), (2, "2016-11-21 20:10", 456.96, "online", "C"),
               (3, "2016-11-21 20:24", 488.04, "online", "C"), (4, "2016-11-21 20:30", 482.12, "online", "C")])
    hit = D.structuring(w, pd.Timestamp("2016-11-21 20:30"))
    assert hit is not None and len(hit) == 4


def test_structuring_ignores_in_person_and_amounts_over_the_line():
    w = frame([(1, "2016-11-21 20:00", 478.95, "in_person", "W"), (2, "2016-11-21 20:10", 500.00, "online", "C"),
               (3, "2016-11-21 20:24", 488.04, "online", "C")])
    assert D.structuring(w, pd.Timestamp("2016-11-21 20:10")) is None


def test_card_testing_needs_the_larger_purchase_after():
    base = [(1, "2016-11-14 09:12", 1.10, "online", "C"), (2, "2016-11-14 09:30", 2.40, "online", "C"),
            (3, "2016-11-14 09:52", 0.95, "online", "C")]
    assert D.card_testing(frame(base)) is None
    hit = D.card_testing(frame(base + [(4, "2016-11-14 10:31", 259.98, "online", "C")]))
    assert hit is not None and list(hit[1].TransactionID) == [4]


def test_card_testing_small_auths_must_fall_within_an_hour():
    w = frame([(1, "2016-11-14 09:00", 1.10, "online", "C"), (2, "2016-11-14 09:40", 2.40, "online", "C"),
               (3, "2016-11-14 10:30", 0.95, "online", "C"), (4, "2016-11-14 11:00", 259.98, "online", "C")])
    assert D.card_testing(w) is None


def test_recurring_needs_monthly_spacing():
    hist = frame([(1, "2016-09-10", 49.00, "online", "S"), (2, "2016-10-10", 49.00, "online", "S"),
                  (3, "2016-11-09", 49.00, "online", "S")])
    flagged = frame([(9, "2016-12-10", 49.00, "online", "S")]).iloc[0]
    assert D.recurring(hist, flagged) is not None
    weekly = frame([(1, "2016-11-19", 49.00, "online", "S"), (2, "2016-11-26", 49.00, "online", "S"),
                    (3, "2016-12-03", 49.00, "online", "S")])
    assert D.recurring(weekly, flagged) is None


def test_ring_requires_new_and_proxy_on_nearly_every_use():
    nb = pd.DataFrame({"card_id": [f"C{i:05d}-K1" for i in range(6)], "id_15": ["New"] * 6,
                       "id_23": ["IP_PROXY:ANONYMOUS"] * 6})
    assert D.is_ring(nb)
    popular_phone = nb.assign(id_23=[None] * 6)
    assert not D.is_ring(popular_phone)
    assert not D.is_ring(nb.head(4))


def test_generic_devices_never_link_people():
    assert D.is_generic_device("Windows | Windows 10 | chrome 66.0 | 1920x1080")
    assert D.is_generic_device("")
    assert not D.is_generic_device("SM-G935F Build/NRD90M | Android 7.0 | chrome 62.0 for android | 1920x1080")


def test_combine_is_odds_multiplication():
    p = D.combine(0.5, [D.Finding("", "", lr=3.0)])
    assert abs(p - 0.75) < 1e-9
    assert D.combine(0.2, [D.Finding("", "", lr=1.0)]) == 0.2 or abs(D.combine(0.2, []) - 0.2) < 1e-9

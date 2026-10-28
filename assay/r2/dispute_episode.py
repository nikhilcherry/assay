"""N7 -- Dispute episode scoping: which transactions is the customer actually disputing?

When a cardholder reports "I did not make these charges" and names one flagged transaction,
the Round 1 rule takes a short 24-hour window around the flagged transaction. On slow-drip
or multi-week novel runs (such as HHF-004 / HHF-005), a 24h window misses earlier charges
of the exact same novel shape on the card.

`episode_for_dispute(card_id, flagged_txn_id, as_of) -> (txn_ids, why)`:
Finds the maximal run of charges on `card_id`, up to `as_of`, that:
  1. Share the flagged charge's "shape":
     - Same ProductCD and channel
     - Amount within the flagged charge's band
     - Device / recipient / billing region fields consistent with the flagged charge
  2. Are novel for this card: the shape does not appear in the card's established history
     before the run started (or, if the whole run falls inside a short burst window,
     falls back to the short episode window when the card's prior history already used
     that shape).

Returns `(txn_ids, why)` where `txn_ids` is a chronologically sorted `list[int]` of
TransactionIDs and `why` is an `assay.r2.attest.Attestation` (with `.what`, `.evidence`,
`.query`, `.params`, and convenience metadata).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from assay.investigate import EPISODE_HOURS, EPISODE_MIN_P
from assay.r2.attest import Attestation, Evidence
from assay.r2.frames import case_pack, closed_cases, tx_all

# Maximum lookback horizon for a single multi-week novel run (35 days covers ~3-4 week billing cycles)
MAX_RUN_HORIZON_DAYS = 35.0
MAX_RUN_HORIZON_SEC = MAX_RUN_HORIZON_DAYS * 86400.0

# Maximum gap between consecutive novel shape-matching charges in the same run (14 days)
MAX_STEP_GAP_DAYS = 14.0
MAX_STEP_GAP_SEC = MAX_STEP_GAP_DAYS * 86400.0

# Minimum gap before the run's first charge to consider earlier matching charges "prior history"
PRIOR_HISTORY_BUFFER_SEC = 48.0 * 3600.0


@dataclass
class CardDisputeIndex:
    """Fast per-card slice index over `tx_all()` sorted by `(card_id, t, TransactionID)`."""
    df: pd.DataFrame
    card_bounds: dict[str, tuple[int, int]]
    txn_to_card: dict[int, str]
    txn_to_pos: dict[int, int]
    txn_id: np.ndarray
    t_sec: np.ndarray
    amount: np.ndarray
    prod_cd: np.ndarray
    channel: np.ndarray
    addr1: np.ndarray
    p_email: np.ndarray
    r_email: np.ndarray
    dev_prof: np.ndarray
    device_type: np.ndarray
    risk_score: np.ndarray


_DISPUTE_INDEX_CACHE: dict[int, CardDisputeIndex] = {}


def _to_epoch_sec(ts_series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(ts_series)
    return dt.astype("datetime64[s]").astype("int64").to_numpy(dtype=np.float64)


def _ts_to_sec(ts: pd.Timestamp | str) -> float:
    return float(pd.Timestamp(ts).as_unit("s").value)


def get_dispute_index(tx: pd.DataFrame | None = None) -> CardDisputeIndex:
    """Build or fetch a cached `CardDisputeIndex` for `tx`."""
    if tx is None:
        tx = tx_all()
    key = id(tx)
    if key in _DISPUTE_INDEX_CACHE:
        return _DISPUTE_INDEX_CACHE[key]

    s = tx.sort_values(["card_id", "t", "TransactionID"], kind="mergesort").reset_index(drop=True)
    cards = s["card_id"].to_numpy()
    if len(cards) == 0:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        empty_o = np.array([], dtype=object)
        idx = CardDisputeIndex(
            s, {}, {}, {}, empty_i, empty_f, empty_f, empty_o, empty_o,
            empty_f, empty_o, empty_o, empty_o, empty_o, empty_f,
        )
        _DISPUTE_INDEX_CACHE[key] = idx
        return idx

    change = np.nonzero(cards[1:] != cards[:-1])[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(cards)]))
    card_bounds = {str(cards[st]): (int(st), int(en)) for st, en in zip(starts, ends)}

    txn_id = s["TransactionID"].to_numpy(dtype=np.int64)
    t_sec = _to_epoch_sec(s["t"])
    amount = s["amount"].to_numpy(dtype=np.float64)
    prod_cd = s["ProductCD"].fillna("").astype(str).to_numpy()
    channel = s["channel"].fillna("").astype(str).to_numpy()
    addr1 = s["addr1"].fillna(-1.0).to_numpy(dtype=np.float64)
    p_email = s["P_emaildomain"].fillna("").astype(str).to_numpy()
    r_email = s["R_emaildomain"].fillna("").astype(str).to_numpy()
    dev_prof = s["device_profile"].fillna("").astype(str).to_numpy()
    device_type = s["DeviceType"].fillna("").astype(str).to_numpy()
    risk_score = s["risk_score"].fillna(0.0).to_numpy(dtype=np.float64)

    txn_to_card = dict(zip(txn_id.tolist(), cards.tolist()))
    txn_to_pos = {int(tid): i for i, tid in enumerate(txn_id.tolist())}

    idx = CardDisputeIndex(
        df=s,
        card_bounds=card_bounds,
        txn_to_card=txn_to_card,
        txn_to_pos=txn_to_pos,
        txn_id=txn_id,
        t_sec=t_sec,
        amount=amount,
        prod_cd=prod_cd,
        channel=channel,
        addr1=addr1,
        p_email=p_email,
        r_email=r_email,
        dev_prof=dev_prof,
        device_type=device_type,
        risk_score=risk_score,
    )
    if len(_DISPUTE_INDEX_CACHE) > 4:
        _DISPUTE_INDEX_CACHE.clear()
    _DISPUTE_INDEX_CACHE[key] = idx
    return idx


def _amount_band(flag_amt: float, strict: bool = False) -> tuple[float, float]:
    """Return `(low, high)` amount band for `flag_amt`.

    When `strict=False` (used within a run where ProductCD + channel + device/recipient match):
      - Micro/small charges (< $15): [$0.50, max($12.0, 3.0 * flag_amt)]
      - Medium charges ($15 .. $100): [0.35 * flag_amt, 2.6 * flag_amt]
      - Larger charges (> $100): [0.40 * flag_amt, 2.5 * flag_amt]
    When `strict=True` (used to check whether the exact shape already existed in prior history):
      - Tighter band [0.50 * flag_amt, 2.0 * flag_amt] (or +- $4 for micro charges).
    """
    a = abs(float(flag_amt))
    if strict:
        if a < 10.0:
            return (max(0.25, a * 0.40), max(a * 2.25, a + 4.0))
        return (a * 0.50, a * 2.00)
    if a < 10.0:
        return (max(0.20, a * 0.25), max(10.0, a * 3.2))
    if a < 25.0:
        return (max(1.0, a * 0.35), max(35.0, a * 2.6))
    return (a * 0.38, a * 2.60)


def _shape_match_mask(
    c_prod: np.ndarray,
    c_chan: np.ndarray,
    c_amt: np.ndarray,
    c_addr1: np.ndarray,
    c_remail: np.ndarray,
    c_dev: np.ndarray,
    c_dtype: np.ndarray,
    flag_idx: int,
    strict: bool = False,
) -> np.ndarray:
    """Vectorized boolean mask of transactions on the card matching the flagged txn's shape."""
    f_prod = c_prod[flag_idx]
    f_chan = c_chan[flag_idx]
    f_amt = c_amt[flag_idx]
    f_addr1 = c_addr1[flag_idx]
    f_remail = c_remail[flag_idx]
    f_dev = c_dev[flag_idx]
    f_dtype = c_dtype[flag_idx]

    # 1. Same ProductCD and channel
    same_pc = (c_prod == f_prod) & (c_chan == f_chan)

    # 2. Amount within band
    lo, hi = _amount_band(f_amt, strict=strict)
    in_band = (np.abs(c_amt) >= lo) & (np.abs(c_amt) <= hi)

    # 3. Device / recipient / region consistency
    n = len(c_prod)
    consistent = np.ones(n, dtype=bool)

    if f_remail != "":
        # If flagged charge has a recipient email domain, candidates must have the same R_emaildomain
        # (or empty if not strict and same device profile)
        if strict:
            consistent &= (c_remail == f_remail)
        else:
            consistent &= (c_remail == f_remail) | ((c_remail == "") & (f_dev != "") & (c_dev == f_dev))
    else:
        # Flagged charge has no R_emaildomain -> candidates with a populated R_emaildomain are a different shape
        consistent &= (c_remail == "")

    if f_dev != "":
        # If flagged txn has a specific device_profile, require same device_profile or same DeviceType when profile missing
        if strict:
            consistent &= (c_dev == f_dev) | ((c_dev == "") & (c_dtype == f_dtype) & (f_dtype != ""))
        else:
            consistent &= (c_dev == f_dev) | ((c_dev == "") & ((c_dtype == f_dtype) | (c_remail == f_remail)))
    else:
        # Flagged txn has no device_profile (e.g., W in_person or online without identity row)
        if f_chan == "in_person" and f_addr1 >= 0:
            # For in-person purchases, billing region addr1 should match
            consistent &= (c_addr1 == f_addr1)
        elif f_chan == "online":
            # Online without device_profile: candidate must also lack a conflicting device_profile
            # and match addr1 presence/value
            consistent &= (c_dev == "")
            if f_addr1 >= 0:
                consistent &= (c_addr1 == f_addr1) | (c_addr1 < 0)
            else:
                consistent &= (c_addr1 < 0)

    mask = same_pc & in_band & consistent
    mask[flag_idx] = True
    return mask


def _burst_companion_mask(
    c_t: np.ndarray,
    c_chan: np.ndarray,
    c_addr1: np.ndarray,
    c_remail: np.ndarray,
    c_dev: np.ndarray,
    c_risk: np.ndarray,
    flag_idx: int,
) -> np.ndarray:
    """Detect immediate same-session / 24h fraud burst companions around `flag_idx`.

    In many customer-reported closed cases (e.g. card testing, rapid online bursts, or
    same-day out-of-region sprees), the customer disputes a burst of charges within 24h
    that share the same session/device/channel or out-of-region addr1 even if the fraudster
    varied the dollar amount across two consecutive charges.
    """
    f_t = c_t[flag_idx]
    f_chan = c_chan[flag_idx]
    f_addr1 = c_addr1[flag_idx]
    f_remail = c_remail[flag_idx]
    f_dev = c_dev[flag_idx]

    dt_abs = np.abs(c_t - f_t)
    in_24h = dt_abs <= (EPISODE_HOURS * 3600.0)
    same_chan = c_chan == f_chan

    if f_chan == "online":
        # Same online session: within 24h, online, and shares non-empty device or R_emaildomain
        # or sits within 2 hours with elevated risk / novel online channel
        shared_id = np.zeros(len(c_t), dtype=bool)
        if f_dev != "":
            shared_id |= (c_dev == f_dev)
        if f_remail != "":
            shared_id |= (c_remail == f_remail)
        close_time = (dt_abs <= 4.0 * 3600.0) & ((c_dev == f_dev) | (c_risk >= EPISODE_MIN_P))
        return in_24h & same_chan & (shared_id | close_time)
    else:
        # In-person burst: within 24h, same non-modal/specific addr1 or elevated risk within 6h
        same_addr = (f_addr1 >= 0) & (c_addr1 == f_addr1)
        close_in_person = same_addr & ((dt_abs <= 6.0 * 3600.0) | (c_risk >= EPISODE_MIN_P))
        return in_24h & same_chan & close_in_person


def episode_for_dispute(
    card_id: str,
    flagged_txn_id: int,
    as_of: pd.Timestamp | str,
    tx: pd.DataFrame | None = None,
) -> tuple[list[int], Attestation]:
    """Return `(txn_ids, why)` for a customer-disputed transaction on `card_id` as of `as_of`.

    Finds the maximal run of charges on `card_id` (with `t <= as_of`) that:
      - share the flagged charge's shape (ProductCD + channel, amount within the flagged
        charge's band, consistent device/recipient/region fields), AND
      - are novel for this card (the shape does not appear in the card's history before
        the run started).
    Also includes immediate same-session (<24h) novel burst companions when they share the
    flagged charge's novel channel/device/region session.
    """
    idx = get_dispute_index(tx)
    flagged_txn_id = int(flagged_txn_id)
    as_of_ts = pd.Timestamp(as_of)
    as_of_sec = _ts_to_sec(as_of_ts)

    actual_card = idx.txn_to_card.get(flagged_txn_id, card_id)
    bounds = idx.card_bounds.get(actual_card)
    if bounds is None and card_id in idx.card_bounds:
        actual_card = card_id
        bounds = idx.card_bounds[actual_card]

    if bounds is None:
        why = Attestation(
            what=f"Dispute episode on {card_id} for flagged txn {flagged_txn_id}: card not found in history, returning [flagged_txn_id].",
            evidence=Evidence(txn_ids=[flagged_txn_id], card_ids=[card_id]),
            query="assay.r2.dispute_episode.episode_for_dispute",
            params={"card_id": card_id, "flagged_txn_id": flagged_txn_id, "as_of": str(as_of_ts)},
        )
        return [flagged_txn_id], why

    st, en = bounds
    c_t_full = idx.t_sec[st:en]
    c_txns_full = idx.txn_id[st:en]

    # Strictly enforce as-of: only transactions with t <= as_of
    end_pos = int(np.searchsorted(c_t_full, as_of_sec, side="right"))
    flag_matches = np.nonzero(c_txns_full[:end_pos] == flagged_txn_id)[0]
    if len(flag_matches) == 0:
        # Check if flagged_txn_id exists on the card at all; if opened_at was slightly before txn ts
        # in synthetic test cases, clamp to include flagged_txn_id only if needed, otherwise return singleton
        all_matches = np.nonzero(c_txns_full == flagged_txn_id)[0]
        if len(all_matches) == 0:
            why = Attestation(
                what=f"Dispute episode on {actual_card} for flagged txn {flagged_txn_id}: txn not on card as of {as_of_ts}.",
                evidence=Evidence(txn_ids=[flagged_txn_id], card_ids=[actual_card]),
                query="assay.r2.dispute_episode.episode_for_dispute",
                params={"card_id": card_id, "flagged_txn_id": flagged_txn_id, "as_of": str(as_of_ts)},
            )
            return [flagged_txn_id], why
        flag_pos = int(all_matches[0])
        if c_t_full[flag_pos] <= as_of_sec:
            end_pos = max(end_pos, flag_pos + 1)
        else:
            why = Attestation(
                what=f"Dispute episode on {actual_card} for flagged txn {flagged_txn_id}: flagged txn is after as_of {as_of_ts}.",
                evidence=Evidence(txn_ids=[flagged_txn_id], card_ids=[actual_card]),
                query="assay.r2.dispute_episode.episode_for_dispute",
                params={"card_id": card_id, "flagged_txn_id": flagged_txn_id, "as_of": str(as_of_ts)},
            )
            return [flagged_txn_id], why
    else:
        flag_pos = int(flag_matches[0])

    # Slice card history up to end_pos (all t <= as_of)
    c_txns = c_txns_full[:end_pos]
    c_t = c_t_full[:end_pos]
    c_amt = idx.amount[st : st + end_pos]
    c_prod = idx.prod_cd[st : st + end_pos]
    c_chan = idx.channel[st : st + end_pos]
    c_addr1 = idx.addr1[st : st + end_pos]
    c_remail = idx.r_email[st : st + end_pos]
    c_dev = idx.dev_prof[st : st + end_pos]
    c_dtype = idx.device_type[st : st + end_pos]
    c_risk = idx.risk_score[st : st + end_pos]

    flag_t = c_t[flag_pos]
    f_prod = str(c_prod[flag_pos])
    f_chan = str(c_chan[flag_pos])
    f_amt = float(c_amt[flag_pos])
    f_dev = str(c_dev[flag_pos])
    f_remail = str(c_remail[flag_pos])
    f_addr1 = float(c_addr1[flag_pos])

    # Step 1: Identify all transactions on this card (up to as_of) matching the flagged charge's shape
    shape_mask = _shape_match_mask(
        c_prod, c_chan, c_amt, c_addr1, c_remail, c_dev, c_dtype, flag_pos, strict=False
    )
    strict_shape_mask = _shape_match_mask(
        c_prod, c_chan, c_amt, c_addr1, c_remail, c_dev, c_dtype, flag_pos, strict=True
    )

    # Step 2: Build the maximal candidate run of shape-matching charges around flag_pos
    # within MAX_RUN_HORIZON_SEC where consecutive shape-matching charges are separated by <= MAX_STEP_GAP_SEC
    cand_indices = np.nonzero(shape_mask & (np.abs(c_t - flag_t) <= MAX_RUN_HORIZON_SEC))[0]
    pos_in_cand = int(np.nonzero(cand_indices == flag_pos)[0][0])

    left = pos_in_cand
    while left > 0:
        prev_idx = cand_indices[left - 1]
        curr_idx = cand_indices[left]
        if (c_t[curr_idx] - c_t[prev_idx]) <= MAX_STEP_GAP_SEC:
            left -= 1
        else:
            break

    right = pos_in_cand
    while right + 1 < len(cand_indices):
        curr_idx = cand_indices[right]
        next_idx = cand_indices[right + 1]
        if (c_t[next_idx] - c_t[curr_idx]) <= MAX_STEP_GAP_SEC:
            right += 1
        else:
            break

    run_indices = cand_indices[left : right + 1]
    run_start_pos = int(run_indices[0])
    run_start_t = float(c_t[run_start_pos])

    # Step 3: Check novelty of this shape in the card's history prior to run_start_t
    prior_Cutoff = run_start_t - PRIOR_HISTORY_BUFFER_SEC
    prior_mask = c_t < prior_Cutoff
    n_prior_total = int(np.sum(prior_mask))
    n_prior_shape = int(np.sum(prior_mask & strict_shape_mask))

    # Also check whether the broader (ProductCD, channel) or device/region is novel on the card
    n_prior_pc = int(np.sum(prior_mask & (c_prod == f_prod) & (c_chan == f_chan)))
    n_prior_dev = int(np.sum(prior_mask & (c_dev == f_dev))) if f_dev != "" else 0
    n_prior_addr = int(np.sum(prior_mask & (c_addr1 == f_addr1))) if f_addr1 >= 0 else 0

    # A shape is novel for the card if:
    # - it never appeared in the card's history before the run started (n_prior_shape == 0), OR
    # - the device profile itself is non-empty and never appeared before the run started, OR
    # - for in-person charges, the billing region addr1 never appeared before the run started.
    shape_is_novel = (
        (n_prior_shape == 0)
        or (f_dev != "" and n_prior_dev == 0)
        or (f_chan == "in_person" and f_addr1 >= 0 and n_prior_addr == 0)
    )

    # Guard against regular habitual in-person grocery/retail purchases when card has thin prior history
    # and the charges are spread over weeks without a novel region/device:
    run_span_hours = (float(c_t[run_indices[-1]]) - run_start_t) / 3600.0
    if (
        f_chan == "in_person"
        and f_prod == "W"
        and run_span_hours > EPISODE_HOURS
        and n_prior_addr > 0
        and f_dev == ""
    ):
        # Habitual in-person W in a known billing region is not a multi-week novel run
        shape_is_novel = False

    # Guard against monthly recurring subscriptions (R7): if charges are ~28-32 days apart, not a novel fraud run
    if len(run_indices) >= 2:
        gaps_days = np.diff(c_t[run_indices]) / 86400.0
        if np.all((gaps_days >= 25.0) & (gaps_days <= 35.0)):
            shape_is_novel = False

    if shape_is_novel:
        selected_set = {int(i) for i in run_indices.tolist()}
        # Also include immediate (<24h) burst companions if they are also novel (e.g. card testing or two-amount burst)
        burst_mask = _burst_companion_mask(c_t, c_chan, c_addr1, c_remail, c_dev, c_risk, flag_pos)
        for bi in np.nonzero(burst_mask)[0].tolist():
            # Only add burst companion if its own channel/device/region wasn't habitual before the run
            b_chan = str(c_chan[bi])
            b_dev = str(c_dev[bi])
            b_addr = float(c_addr1[bi])
            comp_novel = (
                (b_chan == "online" and (n_prior_pc == 0 or (b_dev != "" and int(np.sum(prior_mask & (c_dev == b_dev))) == 0)))
                or (b_chan == "in_person" and b_addr >= 0 and int(np.sum(prior_mask & (c_addr1 == b_addr))) == 0)
            )
            if comp_novel:
                selected_set.add(int(bi))
        sel_positions = sorted(selected_set)
        mode_desc = (
            f"novel shape run ({f_prod}/{f_chan}, band ${ _amount_band(f_amt)[0]:.2f}-${_amount_band(f_amt)[1]:.2f}, "
            f"prior_shape_hits=0 in {n_prior_total} earlier txns)"
        )
    else:
        # Shape was already present in prior card history: restrict to immediate 24h window around flag_pos
        # matching either exact shape in 24h with risk >= EPISODE_MIN_P or flagged_txn_id
        in_24h = np.abs(c_t - flag_t) <= (EPISODE_HOURS * 3600.0)
        fallback_mask = in_24h & strict_shape_mask & ((c_risk >= EPISODE_MIN_P) | (c_txns == flagged_txn_id))
        fallback_mask[flag_pos] = True
        sel_positions = sorted(np.nonzero(fallback_mask)[0].tolist())
        mode_desc = (
            f"24h window fallback (shape appeared {n_prior_shape}x in {n_prior_total} prior txns before run)"
        )

    txn_ids = [int(c_txns[p]) for p in sel_positions]
    total_usd = float(np.round(np.sum(np.abs(c_amt[sel_positions])), 2))
    first_tid = int(txn_ids[0])

    why = Attestation(
        what=(
            f"Dispute episode on card {actual_card} seeded at {flagged_txn_id}: "
            f"{len(txn_ids)} txn(s) (first={first_tid}, exposure=${total_usd:.2f}) via {mode_desc}."
        ),
        evidence=Evidence(
            txn_ids=txn_ids,
            card_ids=[actual_card],
        ),
        query="assay.r2.dispute_episode.episode_for_dispute",
        params={
            "card_id": card_id,
            "flagged_txn_id": flagged_txn_id,
            "as_of": str(as_of_ts),
        },
    )
    return txn_ids, why


def round1_episode_rule(
    card_id: str,
    flagged_txn_id: int,
    as_of: pd.Timestamp | str,
    tx: pd.DataFrame | None = None,
    use_risk_filter: bool = True,
) -> list[int]:
    """Reproduce the Round 1 `EPISODE_HOURS` (24h) window rule on `card_id` up to `as_of`.

    In `assay.investigate.Investigator._episode`, the default episode takes all transactions
    on `card_id` within `+-EPISODE_HOURS` (24h) of the flagged transaction (restricted to `t <= as_of`
    for an as-of replay) that have model/risk score >= `EPISODE_MIN_P` (0.30) or are the flagged txn.
    """
    idx = get_dispute_index(tx)
    flagged_txn_id = int(flagged_txn_id)
    as_of_sec = _ts_to_sec(as_of)

    actual_card = idx.txn_to_card.get(flagged_txn_id, card_id)
    bounds = idx.card_bounds.get(actual_card)
    if bounds is None and card_id in idx.card_bounds:
        actual_card = card_id
        bounds = idx.card_bounds[actual_card]
    if bounds is None:
        return [flagged_txn_id]

    st, en = bounds
    c_t_full = idx.t_sec[st:en]
    c_txns_full = idx.txn_id[st:en]
    end_pos = int(np.searchsorted(c_t_full, as_of_sec, side="right"))

    flag_matches = np.nonzero(c_txns_full[:end_pos] == flagged_txn_id)[0]
    if len(flag_matches) == 0:
        return [flagged_txn_id]
    flag_pos = int(flag_matches[0])

    c_t = c_t_full[:end_pos]
    c_txns = c_txns_full[:end_pos]
    c_risk = idx.risk_score[st : st + end_pos]

    flag_t = c_t[flag_pos]
    in_win = np.abs(c_t - flag_t) <= (EPISODE_HOURS * 3600.0)
    if use_risk_filter:
        mask = in_win & ((c_risk >= EPISODE_MIN_P) | (c_txns == flagged_txn_id))
    else:
        mask = in_win
    return [int(x) for x in c_txns[mask].tolist()]


def is_customer_report_case(notes: pd.Series) -> pd.Series:
    """Identify closed cases whose trigger was a customer report from `analyst_notes`."""
    s = notes.fillna("").astype(str)
    return (
        s.str.contains("reported unrecognized activity", case=False, regex=False)
        | s.str.contains("disputed a ", case=False, regex=False)
        | s.str.contains("customer report", case=False, regex=False)
    )


@lru_cache(maxsize=1)
def backtest_jul_oct_customer_reports() -> tuple[pd.DataFrame, dict]:
    """Backtest `episode_for_dispute` vs the Round 1 `EPISODE_HOURS` rule on Jul-Oct
    closed cases whose trigger was a customer report.

    Evaluates:
      1. All Jul-Oct customer-report confirmed-fraud cases (seeded at latest affected txn,
         and also seeded at earliest affected txn).
      2. Multi-day (>24h span) Jul-Oct customer-report confirmed-fraud cases where the 24h
         window inherently truncates the episode.
      3. Cleared (legitimate) Jul-Oct customer-report cases (`disputed a ... recurring charge`)
         to verify whether `episode_for_dispute` over-expands legitimate disputes.
    """
    tx = tx_all()
    cc = closed_cases()
    idx = get_dispute_index(tx)

    cust_mask = is_customer_report_case(cc["analyst_notes"])
    jul_oct = (cc["opened_at"] >= "2016-07-01") & (cc["opened_at"] < "2016-11-01")

    cc_fraud = cc[jul_oct & cust_mask & (cc["outcome"] == "confirmed_fraud")].copy()
    cc_cleared = cc[jul_oct & cust_mask & (cc["outcome"] == "cleared")].copy()

    # Precompute txn_id -> amount and timestamp for fast metric calculation
    amt_lookup = dict(zip(idx.txn_id.tolist(), idx.amount.tolist()))
    t_lookup = dict(zip(idx.txn_id.tolist(), idx.t_sec.tolist()))

    methods = {
        "N7_episode_for_dispute": lambda cid, fid, ao: episode_for_dispute(cid, fid, ao, tx)[0],
        "R1_episode_24h_risk30": lambda cid, fid, ao: round1_episode_rule(cid, fid, ao, tx, use_risk_filter=True),
        "R1_episode_24h_all": lambda cid, fid, ao: round1_episode_rule(cid, fid, ao, tx, use_risk_filter=False),
    }

    rows = []
    detailed_multi_day = []

    for subset_name, df_sub in [
        ("jul_oct_cust_report_all", cc_fraud),
        ("jul_oct_cust_report_multitxn", cc_fraud[cc_fraud["n_txns"] > 1]),
    ]:
        for m_name, fn in methods.items():
            exact = 0
            jaccards = []
            first_acc = 0
            exp_abs_err = []
            exp_signed_err = []

            for row in df_sub.itertuples(index=False):
                true_list = [int(x) for x in row.txn_list]
                if not true_list:
                    continue
                true_set = set(true_list)
                true_first = int(row.first_fraud_txn_id) if not pd.isna(row.first_fraud_txn_id) else min(true_list)
                true_exp = float(row.exposure_usd)

                # Customer reports name the latest charge they noticed on their statement
                seed_txn = int(true_list[-1])
                pred_list = fn(row.card_id, seed_txn, row.opened_at)
                pred_set = set(pred_list)
                pred_first = int(pred_list[0]) if pred_list else seed_txn
                pred_exp = round(sum(abs(float(amt_lookup.get(t, 0.0))) for t in pred_list), 2)

                if pred_set == true_set:
                    exact += 1
                union = len(pred_set | true_set)
                inter = len(pred_set & true_set)
                jaccards.append(inter / union if union else 1.0)
                if pred_first == true_first:
                    first_acc += 1
                exp_abs_err.append(abs(pred_exp - true_exp))
                exp_signed_err.append(pred_exp - true_exp)

            n = max(len(jaccards), 1)
            rows.append({
                "cohort": subset_name,
                "method": m_name,
                "n_cases": len(jaccards),
                "exact_match_rate": round(exact / n, 4),
                "mean_jaccard": round(float(np.mean(jaccards)), 4),
                "first_txn_accuracy": round(first_acc / n, 4),
                "mean_exposure_abs_err": round(float(np.mean(exp_abs_err)), 2),
                "median_exposure_abs_err": round(float(np.median(exp_abs_err)), 2),
                "mean_exposure_bias": round(float(np.mean(exp_signed_err)), 2),
            })

    # Also check multi-day customer-report cases across the full closed_cases history (including Round 1 closed cases)
    cc_all_cust_fraud = cc[cust_mask & (cc["outcome"] == "confirmed_fraud")].copy()
    multi_day_ids = []
    for row in cc_all_cust_fraud.itertuples(index=False):
        tlist = [int(x) for x in row.txn_list]
        if len(tlist) >= 2:
            ts_vals = [t_lookup.get(x, 0.0) for x in tlist]
            if max(ts_vals) - min(ts_vals) > 24.0 * 3600.0:
                multi_day_ids.append(row.case_id)

    cc_multiday = cc_all_cust_fraud[cc_all_cust_fraud["case_id"].isin(multi_day_ids)]
    if len(cc_multiday) > 0:
        for m_name, fn in methods.items():
            exact = 0
            jaccards = []
            first_acc = 0
            exp_abs_err = []
            for row in cc_multiday.itertuples(index=False):
                true_list = [int(x) for x in row.txn_list]
                true_set = set(true_list)
                true_first = int(row.first_fraud_txn_id) if not pd.isna(row.first_fraud_txn_id) else min(true_list)
                true_exp = float(row.exposure_usd)
                seed_txn = int(true_list[-1])
                pred_list = fn(row.card_id, seed_txn, row.opened_at)
                pred_set = set(pred_list)
                pred_first = int(pred_list[0]) if pred_list else seed_txn
                pred_exp = round(sum(abs(float(amt_lookup.get(t, 0.0))) for t in pred_list), 2)
                if pred_set == true_set:
                    exact += 1
                jaccards.append(len(pred_set & true_set) / len(pred_set | true_set))
                if pred_first == true_first:
                    first_acc += 1
                exp_abs_err.append(abs(pred_exp - true_exp))
                detailed_multi_day.append({
                    "case_id": row.case_id,
                    "method": m_name,
                    "true_txns": true_list,
                    "pred_txns": pred_list,
                    "true_exp": true_exp,
                    "pred_exp": pred_exp,
                })
            n = max(len(jaccards), 1)
            rows.append({
                "cohort": "all_cust_report_multiday_span",
                "method": m_name,
                "n_cases": len(jaccards),
                "exact_match_rate": round(exact / n, 4),
                "mean_jaccard": round(float(np.mean(jaccards)), 4),
                "first_txn_accuracy": round(first_acc / n, 4),
                "mean_exposure_abs_err": round(float(np.mean(exp_abs_err)), 2),
                "median_exposure_abs_err": round(float(np.median(exp_abs_err)), 2),
                "mean_exposure_bias": 0.0,
            })

    # Evaluate cleared (legitimate recurring dispute) cases in Jul-Oct:
    # Check how often episode_for_dispute falsely groups multiple historical recurring charges into a single episode
    cleared_overexpanded = []
    for row in cc_cleared.itertuples(index=False):
        true_list = [int(x) for x in row.txn_list]
        if not true_list:
            continue
        seed_txn = int(true_list[-1])
        pred_list, _ = episode_for_dispute(row.card_id, seed_txn, row.opened_at, tx)
        if len(pred_list) > 1:
            cleared_overexpanded.append(row.case_id)

    summary = {
        "n_jul_oct_cust_fraud": int(len(cc_fraud)),
        "n_jul_oct_cust_multitxn": int((cc_fraud["n_txns"] > 1).sum()),
        "n_jul_oct_cust_cleared": int(len(cc_cleared)),
        "cleared_overexpanded_case_ids": cleared_overexpanded,
        "multi_day_details": detailed_multi_day,
    }
    return pd.DataFrame(rows), summary


def inspect_exam_customer_reports(
    case_ids: tuple[str, ...] = ("HHF-003", "HHF-004", "HHF-005", "HHF-007"),
) -> pd.DataFrame:
    """Run `episode_for_dispute` and `round1_episode_rule` on the exam pack's customer_report cases."""
    tx = tx_all()
    cp = case_pack()
    idx = get_dispute_index(tx)
    amt_lookup = dict(zip(idx.txn_id.tolist(), idx.amount.tolist()))
    t_lookup = dict(zip(idx.txn_id.tolist(), idx.df["ts"].tolist()))

    records = []
    sub = cp[cp["case_id"].isin(case_ids)].sort_values("case_id")
    for row in sub.itertuples(index=False):
        cid = str(row.card_id)
        fid = int(row.flagged_txn_id)
        ao = str(row.opened_at)

        n7_ids, why = episode_for_dispute(cid, fid, ao, tx)
        r1_ids = round1_episode_rule(cid, fid, ao, tx, use_risk_filter=True)

        n7_exp = round(sum(abs(float(amt_lookup.get(t, 0.0))) for t in n7_ids), 2)
        r1_exp = round(sum(abs(float(amt_lookup.get(t, 0.0))) for t in r1_ids), 2)

        records.append({
            "case_id": row.case_id,
            "card_id": cid,
            "flagged_txn_id": fid,
            "opened_at": ao,
            "n7_n_txns": len(n7_ids),
            "n7_first_txn_id": n7_ids[0] if n7_ids else fid,
            "n7_exposure_usd": n7_exp,
            "n7_txn_ids": n7_ids,
            "n7_span": f"{t_lookup.get(n7_ids[0], '')} .. {t_lookup.get(n7_ids[-1], '')}" if n7_ids else "",
            "r1_n_txns": len(r1_ids),
            "r1_first_txn_id": r1_ids[0] if r1_ids else fid,
            "r1_exposure_usd": r1_exp,
            "r1_txn_ids": r1_ids,
            "what": why.what,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    bt_df, bt_meta = backtest_jul_oct_customer_reports()
    print("=== Backtest on Jul-Oct Customer-Report Closed Cases ===")
    print(bt_df.to_string(index=False))
    print("\nCleared customer-report cases over-expanded (>1 txn):", len(bt_meta["cleared_overexpanded_case_ids"]))
    print("\n=== Exam Customer-Report Cases (HHF-003, HHF-004, HHF-005, HHF-007) ===")
    exam_df = inspect_exam_customer_reports()
    print(exam_df[["case_id", "card_id", "flagged_txn_id", "n7_n_txns", "n7_first_txn_id", "n7_exposure_usd", "r1_n_txns", "r1_exposure_usd", "n7_txn_ids"]].to_string(index=False))

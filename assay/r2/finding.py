"""C1 -- the detector interface. A new typology is a new file in assay/r2/detectors/
exposing one Detector subclass; investigate picks it up by registry, nothing else changes.

run(ctx) answers one case. backtest(cc, tx) replays the detector over the closed
cases as of each case's opened_at (no peeking at later data) and says what it
caught, missed, and which cleared (legitimate) cases it would have acted against."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from assay.r2.attest import Attestation


@dataclass
class CaseCtx:
    case_id: str
    flagged_txn_id: int
    card_id: str
    customer_id: str
    opened_at: pd.Timestamp
    tx: pd.DataFrame            # frames.tx_all() filtered to t <= opened_at (as-of view)
    cc: pd.DataFrame            # frames.closed_cases() filtered to closed_at <= opened_at


@dataclass
class Finding:
    detector: str
    pattern: str                # snake_case typology name, "undocumented:<name>" if not one of the five
    claim: Attestation
    lr: float = 1.0             # likelihood ratio on fraud odds, measured by backtest, never guessed
    affected_txn_ids: list[int] = field(default_factory=list)
    connected_card_ids: list[str] = field(default_factory=list)


@dataclass
class Backtest:
    detector: str
    n_cases: int
    caught: list[str]           # confirmed_fraud case_ids the detector fired on
    missed: list[str]           # confirmed_fraud case_ids it did not fire on
    legit_acted_against: list[str]   # cleared case_ids it fired on
    lr_measured: float          # P(fire|fraud)/P(fire|cleared), Laplace-smoothed

    @property
    def precision(self) -> float:
        n = len(self.caught) + len(self.legit_acted_against)
        return len(self.caught) / n if n else 0.0

    @property
    def recall(self) -> float:
        n = len(self.caught) + len(self.missed)
        return len(self.caught) / n if n else 0.0


REGISTRY: dict[str, type] = {}


class Detector:
    name: str = ""
    pattern: str = ""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.name:
            REGISTRY[cls.name] = cls

    def run(self, ctx: CaseCtx) -> list[Finding]:
        raise NotImplementedError

    def backtest(self, cc: pd.DataFrame, tx: pd.DataFrame) -> Backtest:
        raise NotImplementedError

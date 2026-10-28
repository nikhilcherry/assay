"""C0 -- the attestation contract. Every claim any final-round block makes is one
of these: what (one sentence), evidence (exact ids), how (the query and its
parameters, re-runnable), when (timestamp + which data it ran against).
The judges re-run `how`; a claim whose re-run differs scores zero."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

from assay.r2.frames import R2


@lru_cache(maxsize=1)
def dataset_tag() -> str:
    h = hashlib.sha256()
    for f in ("transactions_2017q1.csv", "closed_cases_history.csv", "cases/case_pack_final.csv"):
        h.update((R2 / f).read_bytes())
    return "r2-2017q1@" + h.hexdigest()[:12]


@dataclass
class Evidence:
    txn_ids: list[int] = field(default_factory=list)
    card_ids: list[str] = field(default_factory=list)
    customer_ids: list[str] = field(default_factory=list)
    case_ids: list[str] = field(default_factory=list)


@dataclass
class Attestation:
    what: str
    evidence: Evidence
    query: str                      # name of a registered, re-runnable function or GSQL query
    params: dict = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    dataset: str = field(default_factory=dataset_tag)

    def to_dict(self) -> dict:
        d = asdict(self)
        ev = d.pop("evidence")
        return {"what": d["what"], "evidence": {k: [str(x) for x in v] for k, v in ev.items()},
                "how": {"query": d["query"], "params": d["params"]},
                "when": {"ts": d["ts"], "dataset": d["dataset"]}}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

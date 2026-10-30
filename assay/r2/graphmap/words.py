"""Name -> meaning. Banks abbreviate (ISO 20022 'CdtTrfTxInf'), shout (T24
'FT_DEBIT_ACCT_NO') or spell out (BIAN 'paymentExecutionDateTime'); all three
become the same bag of plain words here before anything is scored.

The abbreviation table is the public ISO 20022 / core-banking vocabulary, not
a list of these three schemas' type names: no vertex or edge name appears in
it as a whole."""

from __future__ import annotations

import re
from functools import lru_cache

ABBREV: dict[str, str] = {
    # ISO 20022 standard abbreviations
    "acct": "account", "acctg": "accounting", "accptnc": "acceptance", "adr": "address",
    "agt": "agent", "alrt": "alert", "amt": "amount", "authrty": "authority", "bk": "bank",
    "bldg": "building", "bookg": "booking", "br": "bearer", "btch": "batch", "ccy": "currency",
    "cd": "code", "cdt": "credit", "cdtr": "creditor", "chrg": "charge", "chrgs": "charges",
    "clr": "clearing", "cre": "creation", "cstmr": "customer", "ctgy": "category",
    "ctrl": "control", "ctry": "country", "dbt": "debit", "dbtr": "debtor", "dt": "date",
    "dttm": "datetime", "dtls": "details", "exctn": "execution", "fin": "financial",
    "grp": "group", "hdr": "header", "id": "identifier", "ind": "indicator",
    "inf": "information", "instd": "instructed", "instn": "institution",
    "instr": "instruction", "instrm": "instrument", "intr": "inter", "intrmy": "intermediary",
    "issr": "issuer", "lcl": "local", "lvl": "level", "mmb": "member", "msg": "message",
    "mtd": "method", "nb": "number", "nm": "name", "opng": "opening", "orgnl": "original",
    "org": "organisation", "ownr": "owner", "ownrsh": "ownership", "pmt": "payment",
    "prty": "priority", "prvt": "private", "pst": "post", "pstl": "postal", "pty": "party",
    "purp": "purpose", "reqd": "requested", "res": "residence", "rgltry": "regulatory",
    "rmt": "remittance", "rptg": "reporting", "rsn": "reason", "rtr": "return",
    "rtrd": "returned", "schme": "scheme", "seq": "sequence", "sttlm": "settlement",
    "sts": "status", "strt": "street", "svc": "service", "svcr": "servicer", "sys": "system",
    "tm": "time", "tp": "type", "trf": "transfer", "twn": "town", "tx": "transaction",
    "txs": "transactions", "ultmt": "ultimate", "ustrd": "unstructured", "vld": "valid",
    "xchg": "exchange", "fr": "from", "nbof": "number",
    # core-banking (T24 / Finacle / Flexcube style) abbreviations
    "acc": "account", "ben": "beneficiary", "co": "company", "cr": "credit",
    "cust": "customer", "dao": "officer", "dept": "department", "dr": "debit",
    "eb": "ebanking", "ext": "external", "fcy": "foreign", "ft": "funds",
    "lcy": "local", "msecs": "milliseconds", "no": "number", "num": "number",
    "ref": "reference", "stmt": "statement", "trans": "transaction", "tt": "teller",
    "txn": "transaction", "trx": "transaction", "dob": "birth", "pan": "pan",
    "auth": "authentication", "dev": "device", "addr": "address", "ip": "ip",
    "kyc": "kyc", "aml": "aml", "mcc": "merchantcategory", "pos": "pos",
    "bic": "bic", "bicfi": "bic", "iban": "iban", "lei": "lei", "uetr": "uetr",
    "msisdn": "phone", "ssn": "ssn", "nidn": "national",
}

# multi-word phrases that mean one thing
PHRASES: dict[tuple[str, ...], str] = {
    ("sign", "on"): "login", ("log", "in"): "login", ("logon",): "login", ("signon",): "login",
    ("date", "time"): "datetime", ("sort", "code"): "sortcode", ("time", "stamp"): "timestamp",
    ("issued", "device"): "card", ("funds", "transfer"): "transfer",
    ("number", "transactions"): "batchcount", ("number", "of", "transactions"): "batchcount",
    ("control", "sum"): "batchcount", ("network", "address"): "ip",
    ("posting", "restrict"): "restriction", ("record", "status"): "recordstatus",
    ("merchant", "category"): "merchantcategory", ("ip", "address"): "ip",
}

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


@lru_cache(maxsize=None)
def words(name: str) -> tuple[str, ...]:
    """'CdtTrfTxInf_DbtrAcct' -> ('credit', 'transfer', 'transaction', 'information',
    'debtor', 'account').  'FT_DEBIT_ACCT_NO' -> ('transfer', 'debit', 'account', 'number')."""
    raw = []
    for chunk in re.split(r"[^A-Za-z0-9]+", name):
        if not chunk:
            continue
        parts = [chunk] if chunk.isupper() or chunk.islower() else _CAMEL.findall(chunk)
        raw += [p.lower() for p in parts if not p.isdigit()]
    out = [ABBREV.get(w, w) for w in raw]
    # PHRASES, longest first ('dt'+'tm' -> date time -> datetime)
    merged, i = [], 0
    while i < len(out):
        for n in (3, 2, 1):
            key = tuple(out[i:i + n])
            if len(key) == n and key in PHRASES:
                merged.append(PHRASES[key])
                i += n
                break
        else:
            merged.append(out[i])
            i += 1
    return tuple(w for w in merged if w)


def bag(*names: str) -> set[str]:
    s: set[str] = set()
    for n in names:
        s.update(words(n))
    return s


def score(ws, cues: dict[str, float]) -> float:
    """Sum of cue weights over a bag of words (each word counted once)."""
    return sum(cues.get(w, 0.0) for w in set(ws))

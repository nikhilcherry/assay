"""Challenge 13 on graph schemas: map a bank's GSQL schema we have never seen
onto the canonical investigation model, without reading its comments.

Structure decides first, names second:
  * Transaction = what carries an amount and a time, is not a batch / header,
    not a return or ledger posting that points back at a payment, and not the
    order that some other payment record executes.
  * Account = what those transactions debit and credit.
  * Person = the record with person attributes (birth date, name, KYC) that
    reaches an account within two hops; if counterparties live in the same
    type, a discriminator attribute splits it and Beneficiary becomes a filter.
  * Edges = the cheapest path in the schema graph between the two mapped ends,
    scored by the edge's own words (debtor vs creditor, ultimate penalised),
    walked backwards (reverse_*) when the bank points the other way.
  * Attributes = cue words on the mapped vertex, then one hop out.

Every choice carries the rule that fired, the words that scored, the runner-up
and why it lost, and what still needs sample records to confirm (a
discriminator value, a date format, a block code). Nothing here is a learned
confidence; `margin` is the raw score gap to the runner-up."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

from assay.r2.graphmap import canonical
from assay.r2.graphmap.gsql import Schema, Vertex
from assay.r2.graphmap.words import bag, score, words

NUMERIC = {"DOUBLE", "FLOAT", "INT", "UINT"}
TIMEISH = {"date", "datetime", "time", "timestamp"}


@dataclass
class Choice:
    element: str
    kind: str                      # vertex | edge | attr | derived
    status: str = "unmapped"       # exact | approximate | unmapped
    sources: list[dict] = field(default_factory=list)
    rule: str = ""
    why: list[str] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)   # needs sample records
    margin: float | None = None

    def names(self) -> set[str]:
        """Every schema identifier this choice uses (for scoring and display)."""
        out: set[str] = set()
        for s in self.sources:
            for k in ("vertex", "attr"):
                if s.get(k):
                    out.add(s[k])
            for step in s.get("path", []):
                out.add(step["edge"])
            for step in s.get("via", []):
                out.add(step["edge"])
        return out


@dataclass
class Mapping:
    graph: str
    choices: dict[str, Choice]

    def to_json(self) -> dict:
        return {"graph": self.graph, "choices": {k: asdict(v) for k, v in self.choices.items()}}

    def vertices(self, element: str) -> list[str]:
        return [s["vertex"] for s in self.choices[element].sources if s.get("vertex") and not s.get("attr")]


def _step(e, forward: bool, a: str, b: str) -> dict:
    gsql = e.name if forward or not e.directed else (e.reverse or "reverse_" + e.name)
    return {"edge": e.name, "forward": forward, "gsql": gsql, "from": a, "to": b}


def _own_words(edge_name: str, *ends: str) -> set[str]:
    """An edge's words minus the words of the vertex types it joins:
    'CdtTrfTxInf_DbtrAcct' between CdtTrfTxInf and CshAcct says 'debtor'."""
    return bag(edge_name) - bag(*ends)


class GraphMapper:
    def __init__(self, schema: Schema, samples: dict[str, list[dict]] | None = None):
        self.s = schema
        self.samples = samples or {}
        self.c: dict[str, Choice] = {}
        self.role: dict[str, str] = {}     # bank vertex -> canonical vertex (first one wins)

    # ------------------------------------------------------------ helpers
    def V(self, name: str) -> Vertex:
        return self.s.vertices[name]

    def vwords(self, v: str) -> set[str]:
        return set(words(v))

    def awords(self, v: str) -> set[str]:
        return bag(*(a.name for a in self.V(v).attrs))

    def _amount_attrs(self, v: str):
        return [a for a in self.V(v).attrs if a.type in NUMERIC and "amount" in words(a.name)
                and not set(words(a.name)) & {"currency", "limit", "overdraft", "balance", "rate", "interest"}]

    def _time_attrs(self, v: str):
        return [a for a in self.V(v).attrs if a.type == "DATETIME" or TIMEISH & set(words(a.name))]

    def _adjacent(self, v: str):
        return list(self.s.neighbours(v))

    def vertices(self, element: str) -> list[str]:
        ch = self.c.get(element)
        return [s["vertex"] for s in ch.sources if not s.get("attr")] if ch else []

    def _set(self, ch: Choice):
        self.c[ch.element] = ch
        if ch.kind == "vertex":
            for s in ch.sources:
                if s.get("vertex") and not s.get("attr"):
                    self.role.setdefault(s["vertex"], ch.element)

    # ------------------------------------------------------------ run
    def run(self) -> Mapping:
        self.map_transaction()
        self.map_account()
        self.tier_transactions()
        self.map_person_beneficiary()
        self.map_simple("Card", {"card": 3, "pan": 2, "scheme": 0.5, "expiry": 0.5},
                        {"transaction": -6, "statement": -3, "beneficiary": -3}, attr_cues={"pan": 1, "scheme": 1, "card": 1})
        self.map_login()
        self.map_simple("Device", {"device": 3, "terminal": 3, "fingerprint": 2, "browser": 1},
                        {"card": -6, "transaction": -3, "allocation": -1},
                        attr_cues={"fingerprint": 1, "operating": 1, "browser": 1})
        self.map_simple("IPAddress", {"ip": 4}, {"transaction": -3})
        self.map_simple("Merchant", {"merchant": 4, "acceptor": 3, "shop": 2, "store": 2},
                        {"transaction": -6, "relationship": 0}, attr_cues={"merchantcategory": 2})
        self.map_simple("InvestigationCase", {"case": 4, "investigation": 4, "dispute": 2},
                        {"diagnosis": -4, "assessment": -2, "score": -2, "transaction": -3},
                        attr_cues={"status": 0.5, "resolution": 0.5})
        self.attr_fallbacks()
        self.map_edges()
        self.map_attrs()
        self.map_under_investigation()
        for el in canonical.ELEMENTS:
            self.c.setdefault(el, Choice(el, self._kind(el), why=["no candidate in this schema"]))
        return Mapping(self.s.graph, {el: self.c[el] for el in canonical.ELEMENTS})

    @staticmethod
    def _kind(el: str) -> str:
        if el in canonical.VERTICES:
            return "vertex"
        if el in canonical.EDGES:
            return "edge"
        return "attr" if "." in el else "derived"

    # ------------------------------------------------------------ Transaction
    def map_transaction(self):
        ch = Choice("Transaction", "vertex", rule="T1 amount+time, not batch/derivative/request")
        cands = [v for v in self.s.vertices if self._amount_attrs(v)]
        keep = []
        for v in cands:
            aw, vw = self.awords(v), self.vwords(v)
            if not self._time_attrs(v):
                ch.rejected.append({"candidate": v, "why": "has an amount but no time attribute"})
            elif "batchcount" in aw or "batch" in aw or "batch" in vw:
                ch.rejected.append({"candidate": v, "why": "carries a transaction count / control sum / batch flag: one record per many payments (a batch or header)"})
            elif vw & {"return", "returned", "reversal", "statement", "entry", "ledger", "journal"}:
                # not "posting": a core's posting record can be the movement itself; a posting
                # that merely records a payment is caught below by its reference edge
                ch.rejected.append({"candidate": v, "why": f"its name says {sorted(vw & {'return','returned','reversal','statement','entry','ledger','journal'})}: a return or ledger record of a payment"})
            else:
                keep.append(v)
        final = []
        for v in keep:
            back = [(e.name, o) for e, fwd, o in self._adjacent(v)
                    if fwd and o in keep and o != v and _own_words(e.name, v, o) & {"original", "reference", "return", "reversal"}]
            req = [(e.name, o) for e, fwd, o in self._adjacent(v)
                   if not fwd and o in keep and o != v] if self.vwords(v) & {"order", "initiation", "request", "instruction", "mandate"} else []
            if back:
                ch.rejected.append({"candidate": v, "why": f"points back at payment {back[0][1]} through {back[0][0]}: derived from a payment, not one"})
            elif req:
                ch.rejected.append({"candidate": v, "why": f"request stage: {req[0][1]} links to it through {req[0][0]} and is the movement itself"})
            else:
                final.append(v)
        for v in final:
            amt = self._amount_attrs(v)[0].name
            ch.sources.append({"vertex": v})
            ch.why.append(f"{v}: amount ({amt}) and time ({self._time_attrs(v)[0].name}), linked to no other payment record as its derivative")
        ch.status = "exact" if final else "unmapped"
        self.payment_stage = {r["candidate"] for r in ch.rejected}
        self._set(ch)

    # ------------------------------------------------------------ Account
    def map_account(self):
        ch = Choice("Account", "vertex", rule="A1 what transactions debit/credit, named account/facility")
        tx = set(self.vertices("Transaction"))
        seen = []
        for t in tx:
            for e, fwd, o in self._adjacent(t):
                if o in tx or o in seen:
                    continue
                vw = self.vwords(o)
                if vw & {"account", "facility"} and not vw & {"beneficiary", "user", "officer", "statement", "card", "plan"}:
                    seen.append(o)
                    ch.sources.append({"vertex": o})
                    ch.why.append(f"{o}: {t} reaches it directly through {e.name}; named {sorted(vw & {'account', 'facility'})}")
        if not seen:   # no transactions: fall back to the name alone
            for v in self.s.vertices:
                vw = self.vwords(v)
                if vw & {"account"} and not vw & {"beneficiary", "user", "statement"}:
                    seen.append(v)
                    ch.sources.append({"vertex": v})
                    ch.why.append(f"{v}: named account (no transaction type to confirm it)")
        ch.status = "exact" if seen else "unmapped"
        self._set(ch)

    def tier_transactions(self):
        """A transaction type with no direct edge to an account is an approximate
        member (card payments reach the account only through the card)."""
        ch = self.c["Transaction"]
        acc = set(self.vertices("Account"))
        for s in ch.sources:
            direct = any(o in acc for _, _, o in self._adjacent(s["vertex"]))
            if not direct:
                s["status"] = "approximate"
                ch.why.append(f"{s['vertex']}: approximate member, no direct edge to an account")
        if ch.sources and all(s.get("status") == "approximate" for s in ch.sources):
            ch.status = "approximate"

    # ------------------------------------------------------------ Person / Beneficiary
    def _reaches(self, v: str, targets: set[str], hops: int, avoid: set[str]) -> bool:
        q, seen = deque([(v, 0)]), {v}
        while q:
            x, d = q.popleft()
            if x in targets and x != v:
                return True
            if d == hops:
                continue
            for _, _, o in self._adjacent(x):
                if o not in seen and (o in targets or o not in avoid):
                    seen.add(o)
                    q.append((o, d + 1))
        return False

    def map_person_beneficiary(self):
        tx, acc = set(self.vertices("Transaction")), set(self.vertices("Account"))
        ch = Choice("Person", "vertex", rule="P1 person attributes + reaches an account in <=2 hops")
        pos_attr = {"birth": 3, "name": 1, "nationality": 1, "residence": 1, "kyc": 1, "verification": 1, "legal": 0.5}
        pos_name = {"party": 2, "customer": 2, "person": 2, "individual": 2, "client": 2}
        neg_name = {w: -2 for w in ("user", "officer", "plan", "relationship", "address", "contact", "beneficiary",
                                    "counterparty", "company", "institution", "agent", "private", "identification", "merchant")}
        ranked = []
        for v in self.s.vertices:
            if v in tx or v in acc:
                continue
            sc = score(self.awords(v), pos_attr) + score(self.vwords(v), {**pos_name, **neg_name})
            if sc >= 3 and self._reaches(v, acc, 2, tx):
                ranked.append((sc, v))
        ranked.sort(reverse=True)
        ben = Choice("Beneficiary", "vertex")
        if ranked:
            p = ranked[0][1]
            ch.sources.append({"vertex": p})
            ch.status = "exact"
            ch.why.append(f"{p}: person attributes {sorted(self.awords(p) & set(pos_attr))}, reaches an account within two hops")
            if len(ranked) > 1:
                ch.margin = ranked[0][0] - ranked[1][0]
                ch.rejected += [{"candidate": v, "why": f"weaker person evidence ({s:.1f} vs {ranked[0][0]:.1f})"} for s, v in ranked[1:]]
            # do payments credit this same type? then it also holds counterparties
            shared = [(e.name, t) for t in tx for e, fwd, o in self._adjacent(t)
                      if o == p and fwd and _own_words(e.name, t, p) & {"creditor", "credits", "counterparty", "beneficiary", "payee"}
                      and "ultimate" not in words(e.name)]
            disc = self._discriminator(p) if shared else None
            if shared and disc:
                a, kind = disc
                if kind == "bool":
                    pf, bf = f"{a} = true", f"{a} = false"
                    pend = []
                else:
                    pf, bf = f"{a} in <customer values>", f"{a} = <counterparty value>"
                    pend = [f"{a}: read distinct values from sample records to fix the customer vs counterparty split"]
                    pf, bf = self._resolve_values(p, a, pf, bf)
                ch.sources[0]["filter"] = pf
                ch.pending += pend
                ch.why.append(f"{p} also receives payments ({shared[0][0]}); {a} tells customers from counterparties")
                ben.sources.append({"vertex": p, "filter": bf, "reached_by": shared[0][0]})
                ben.status, ben.rule = "approximate", "B2 counterparty role of the shared party type"
                ben.why.append(f"no dedicated payee type; counterparties are {p} rows with {bf}, reached by {shared[0][0]}")
                ben.pending += pend
        else:
            ch.why.append("no vertex with person attributes reaches an account")
        self._set(ch)
        # a dedicated payee type beats the role filter
        for v in self.s.vertices:
            vw = self.vwords(v)
            if v in tx or v in acc or not vw & {"beneficiary", "payee", "counterparty"}:
                continue
            linked = [e.name for t in tx for e, _, o in self._adjacent(t) if o == v]
            if linked:
                if ben.sources:
                    ben.rejected.append({"candidate": ben.sources[0]["vertex"], "why": f"a dedicated payee type {v} exists"})
                ben.sources = [{"vertex": v}]
                ben.status, ben.rule = "exact", "B1 dedicated payee type linked from a transaction"
                ben.why = [f"{v}: named {sorted(vw & {'beneficiary', 'payee', 'counterparty'})}, linked from payments by {linked[0]}"]
                ben.pending = []
        self._set(ben)

    def _discriminator(self, v: str):
        for a in self.V(v).attrs:
            if a.type == "BOOL" and set(words(a.name)) & {"customer", "client", "internal", "own", "member"}:
                return a.name, "bool"
        for a in self.V(v).attrs:
            if a.type == "STRING" and set(words(a.name)) & {"type", "kind", "category", "role"}:
                return a.name, "string"
        return None

    def _resolve_values(self, v: str, a: str, pf: str, bf: str):
        rows = self.samples.get(v) or []
        vals = sorted({str(r.get(a)) for r in rows if r.get(a) is not None})
        cp = [x for x in vals if set(words(x)) & {"counterparty", "external", "beneficiary", "payee", "third"}]
        if cp:
            rest = [x for x in vals if x not in cp]
            return f"{a} in ({', '.join(rest)})", f"{a} = {cp[0]}"
        return pf, bf

    # ------------------------------------------------------------ simple lexical types
    def map_simple(self, element: str, name_cues: dict, neg: dict, attr_cues: dict | None = None, floor: float = 3.0):
        ch = Choice(element, "vertex", rule=f"{element[0]}1 name and attribute words")
        taken = set(self.role)
        ranked = []
        for v in self.s.vertices:
            if v in taken:
                continue
            sc = score(self.vwords(v), {**name_cues, **neg}) + score(self.awords(v), attr_cues or {})
            pid = next((a.name for a in self.V(v).attrs if a.primary), "")
            sc += 0.5 * score(words(pid), name_cues)
            if sc >= floor:
                ranked.append((sc, v))
        ranked.sort(reverse=True)
        if ranked:
            v = ranked[0][1]
            ch.sources.append({"vertex": v})
            ch.status = "exact"
            ch.why.append(f"{v}: words {sorted((self.vwords(v) | self.awords(v)) & (set(name_cues) | set(attr_cues or {})))}")
            if len(ranked) > 1:
                ch.margin = ranked[0][0] - ranked[1][0]
                ch.rejected += [{"candidate": x, "why": f"lower score {s:.1f}"} for s, x in ranked[1:]]
        # a named trap: something that sounds like this type but is mapped elsewhere
        for v in taken:
            if score(self.vwords(v), name_cues) >= floor:
                ch.rejected.append({"candidate": v, "why": f"sounds like {element} but is already {self.role[v]}"})
        self._set(ch)

    # ------------------------------------------------------------ LoginEvent
    def map_login(self):
        ch = Choice("LoginEvent", "vertex", rule="L1 authentication words, or an event carrying IP/terminal and a user")
        ranked = []
        for v in self.s.vertices:
            if v in self.role:
                continue
            vw, aw = self.vwords(v), self.awords(v)
            sc = score(vw, {"authentication": 3, "login": 3, "session": 2, "logon": 3, "transaction": -4, "case": -3})
            hits = []
            for e, _, o in self._adjacent(v):
                ow = self.vwords(o)
                if "ip" in ow:
                    sc += 2; hits.append(f"edge to IP type {o}")
                elif ow & {"device", "terminal"} and o not in self.role:
                    sc += 1; hits.append(f"edge to device type {o}")
                elif ow & {"party", "customer", "user"}:
                    sc += 1; hits.append(f"edge to user/party {o}")
            if "ip" in aw:
                sc += 2; hits.append("IP attribute")
            if aw & {"terminal", "device"}:
                sc += 1; hits.append("terminal/device attribute")
            if self._time_attrs(v):
                sc += 0.5
            if sc >= 3:
                ranked.append((sc, v, hits, bool(vw & {"authentication", "login", "logon", "session"})))
        ranked.sort(reverse=True, key=lambda r: r[0])
        if ranked:
            sc, v, hits, lexical = ranked[0]
            src = {"vertex": v}
            ch.why.append(f"{v}: {'named as an authentication/login; ' if lexical else ''}{', '.join(hits)}")
            if not lexical:
                act = next((a.name for a in self.V(v).attrs if a.type == "STRING"
                            and set(words(a.name)) & {"application", "function", "action", "activity", "event", "type"}), None)
                if act:
                    src["filter"] = f"{act} = <sign-on value>"
                    ch.pending.append(f"{act}: pick the sign-on value from sample records (this type logs more than logins)")
            ch.sources.append(src)
            ch.status = "exact"
            if len(ranked) > 1:
                ch.margin = sc - ranked[1][0]
                ch.rejected += [{"candidate": r[1], "why": f"weaker login evidence ({r[0]:.1f} vs {sc:.1f})"} for r in ranked[1:]]
        self._set(ch)

    def attr_fallbacks(self):
        """No Device / IPAddress vertex: the login record may still carry them as attributes."""
        login = self.vertices("LoginEvent")
        for el, cue in (("Device", {"terminal", "device", "fingerprint"}), ("IPAddress", {"ip"})):
            if self.c[el].sources or not login:
                continue
            for a in self.V(login[0]).attrs:
                if set(words(a.name)) & cue and not a.primary:
                    ch = self.c[el]
                    ch.sources.append({"vertex": login[0], "attr": a.name})
                    ch.status, ch.rule = "approximate", f"{el[0]}2 attribute on the login record, not a vertex"
                    ch.why.append(f"no {el} vertex; {login[0]}.{a.name} carries it, so sharing must be joined on the value")
                    break

    # ------------------------------------------------------------ edges
    EDGE_CUES = {
        "OWNS": {"owner": 3, "ownership": 3, "holder": 3, "joint": 1, "customer": 1, "relationship": 1, "plan": 1,
                 "covers": 1, "officer": -5, "beneficiary": -5, "company": -5, "department": -5, "debit": -3, "credit": -3},
        "INITIATED": {"debit": 3, "debtor": 3, "debits": 3, "ordering": 2, "payer": 3, "sender": 3, "originator": 3,
                      "credit": -3, "creditor": -3, "credits": -3, "ultimate": -4, "charges": -4, "intermediary": -4},
        "PAYS_TO_ACCOUNT": {"credit": 3, "creditor": 3, "credits": 3, "beneficiary": 2, "payee": 3, "receiver": 3,
                            "debit": -3, "debtor": -3, "debits": -3, "ultimate": -4, "charges": -4, "intermediary": -4},
        "PAYS": {"creditor": 3, "credits": 3, "beneficiary": 3, "payee": 3, "counterparty": 2,
                 "debtor": -3, "debit": -3, "ultimate": -4, "intermediary": -4},
    }
    THROUGH = {"FROM_DEVICE": {"LoginEvent"}, "FROM_IP": {"LoginEvent"}, "LOGGED_IN": {"Person"}}
    ACCOUNT_LEVEL = {"OWNS", "HOLDS", "LOGGED_IN", "INVESTIGATED_IN"}   # never derived through a payment record

    def _paths(self, a: str, b: str, allowed: set[str], max_hops: int = 3, avoid: set[str] = frozenset()):
        """Simple paths a -> b whose intermediates are unmapped or in `allowed`, never in `avoid`."""
        out, q = [], deque([(a, [])])
        while q:
            x, path = q.popleft()
            if path and x == b:
                out.append(path)
                continue
            if len(path) == max_hops:
                continue
            visited = {a} | {s["to"] for s in path}
            for e, fwd, o in self._adjacent(x):
                if o in visited and o != b:
                    continue
                if o != b and (o in avoid or o in self.role and self.role[o] not in allowed):
                    continue
                q.append((o, path + [_step(e, fwd, x, o)]))
        return out

    def map_edges(self):
        for el, (ca, cb) in canonical.EDGES.items():
            ch = Choice(el, "edge", rule="E1 cheapest schema path between the mapped ends")
            A = [s for s in self.c[ca].sources]
            B = [s for s in self.c[cb].sources]
            if not A or not B:
                ch.why.append(f"{ca if not A else cb} is unmapped")
                self._set(ch)
                continue
            if any(s.get("attr") for s in A + B):
                ch.why.append(f"{ca if any(s.get('attr') for s in A) else cb} exists only as an attribute; no edge to walk")
                self._set(ch)
                continue
            cues = self.EDGE_CUES.get(el, {})
            found = []
            for sa in A:
                for sb in B:
                    if sa["vertex"] == sb["vertex"]:
                        continue
                    avoid = self.payment_stage if el in self.ACCOUNT_LEVEL else set()
                    paths = self._paths(sa["vertex"], sb["vertex"], self.THROUGH.get(el, set()), avoid=avoid)
                    if not paths:
                        continue
                    def rank(p):
                        return sum(score(_own_words(s["edge"], s["from"], s["to"]), cues) for s in p)
                    shortest = min(len(p) for p in paths)
                    scored = sorted(((rank(p), p) for p in paths if len(p) == shortest), key=lambda r: -r[0])
                    best = scored[0][0]
                    if cues and best < 0:
                        ch.rejected += [{"candidate": " -> ".join(s["gsql"] for s in p), "why": f"edge words point the other way ({r:.1f})"} for r, p in scored[:2]]
                        continue
                    keep = [(r, p) for r, p in scored if r >= best - (0 if el != "OWNS" else 99) and len(p) == len(scored[0][1])]
                    if el == "OWNS":
                        keep = [(r, p) for r, p in scored if len(p) == min(len(q) for _, q in scored) and r >= 0]
                    for r, p in keep:
                        found.append((r, sa, sb, p))
                    ch.rejected += [{"candidate": " -> ".join(s["gsql"] for s in p), "why": f"scored {r:.1f} vs {best:.1f}"}
                                    for r, p in scored[len(keep):len(keep) + 2]]
            seen = set()
            for r, sa, sb, p in found:
                key = tuple(s["edge"] for s in p)
                if key in seen:
                    continue
                seen.add(key)
                src = {"path": p, "from": sa["vertex"], "to": sb["vertex"]}
                inter = [s["to"] for s in p[:-1]]
                mapped_inter = [x for x in inter if x in self.role]
                st = "exact" if not mapped_inter and len(p) <= 2 else "approximate"
                if sa.get("filter") or sb.get("filter"):
                    src["filter"] = sa.get("filter") or sb.get("filter")
                if sb.get("filter") and cb == "Beneficiary":
                    st = "approximate"   # the payee is a role filtered out of a shared party type
                src["status"] = st
                ch.sources.append(src)
                ch.why.append(f"{' -> '.join(s['gsql'] for s in p)} ({len(p)} hop{'s' if len(p) > 1 else ''}"
                              f"{', through ' + ', '.join(mapped_inter) if mapped_inter else ''}; score {r:.1f})")
            self._split_dr_cr(el, ch)
            if ch.sources:
                ch.status = "exact" if any(s["status"] == "exact" for s in ch.sources) else "approximate"
            self._set(ch)

    def _split_dr_cr(self, el: str, ch: Choice):
        """One edge carries both directions (a teller posting to 'the' account):
        the debit/credit marker on the transaction decides which canonical edge."""
        if el != "PAYS_TO_ACCOUNT":
            return
        ini = self.c.get("INITIATED")
        if not ini:
            return
        for src in ini.sources + ch.sources:
            p = src["path"]
            if len(p) != 1:
                continue
            t = p[0]["to"] if src in ini.sources else p[0]["from"]
            marker = next((a.name for a in self.V(t).attrs if {"debit", "credit"} <= set(words(a.name))), None)
            same = [s for s in (ini.sources if src in ch.sources else ch.sources) if s["path"][0]["edge"] == p[0]["edge"]]
            if marker and same and "filter" not in src:
                src["filter"] = f"{marker} = {'DEBIT' if src in ini.sources else 'CREDIT'}"
                src["status"] = "exact"
                (ini if src in ini.sources else ch).pending.append(f"{t}.{marker}: confirm the debit/credit codes from sample records")
        # a single undirected-ish account edge with no marker appears in both; leave it, flagged
        for src in list(ini.sources):
            if src.get("filter") or len(src["path"]) != 1:
                continue
            t = src["path"][0]["to"]
            if any(s["path"][0]["edge"] == src["path"][0]["edge"] for s in ch.sources) and \
                    not any({"debit", "credit"} <= set(words(a.name)) for a in self.V(t).attrs):
                src["status"] = "approximate"
                ini.pending.append(f"{src['path'][0]['edge']} is the only account edge of {t}; direction unknown")

    # ------------------------------------------------------------ attributes
    ATTR_SPECS = {
        "Transaction.amount": ({"amount": 2, "settlement": 1, "execution": 1, "debit": 0.5, "local": 0.3,
                                "instructed": -1, "returned": -3, "credit": -0.3, "order": -0.5, "foreign": -0.3,
                                "currency": -5, "limit": -3, "balance": -3, "overdraft": -3}, NUMERIC, 2.0, 0.0),
        "Transaction.timestamp": ({"datetime": 3, "timestamp": 3, "time": 1.5, "date": 1, "creation": 0.5,
                                   "execution": 0.5, "acceptance": 0.3, "processing": 0.2, "value": -0.5,
                                   "settlement": -0.3, "expiry": -3, "birth": -3, "opening": -1, "milliseconds": -1}, None, 1.0, 0.0),
        "Transaction.channel": ({"channel": 3, "mechanism": 2, "service": 0.75, "level": 0.75, "local": 0.75,
                                 "instrument": 1, "type": 0.5, "code": 0.3, "method": 0.5, "purpose": -0.6,
                                 "priority": -0.6, "category": -0.3, "currency": -3, "status": -3, "transaction": 0.75}, {"STRING", "INT"}, 1.0, 0.5),
        "Account.open_date": ({"opening": 3, "open": 3, "start": 2, "date": 1, "datetime": 1, "creation": 1,
                               "end": -3, "close": -3, "expiry": -3, "value": -1}, None, 3.0, 0.0),
        "Account.status": ({"status": 3, "state": 3, "restriction": 1.5, "block": 1.5, "recordstatus": -2,
                            "aml": -2, "alert": -2}, None, 1.5, 0.0),
        "Account.kyc_score": ({"kyc": 2, "score": 2, "risk": 1, "rating": 1, "complete": -1, "class": 0,
                               "date": -5, "datetime": -5, "review": -2}, {"STRING", "DOUBLE", "FLOAT", "INT", "UINT"}, 2.0, 0.0),
        "Person.anonymized_ssn_hash": ({"ssn": 3, "national": 2, "identification": 2, "legal": 1.5, "tax": 1,
                                        "hash": 1, "identifier": 1, "name": -2, "phone": -2, "email": -2,
                                        "organisation": -2, "account": -2, "birth": -2, "type": -2}, {"STRING"}, 2.5, 0.0),
        "Person.kyc_verified_at": ({"verification": 3, "verified": 3, "kyc": 2, "review": 1.5, "date": 1,
                                    "birth": -5}, None, 3.0, 0.0),
        "Beneficiary.bank_identifier": ({"bic": 3, "sortcode": 3, "swift": 3, "routing": 3, "bank": 2, "clearing": 1,
                                         "identifier": 0.5, "organisation": -2, "account": -2, "name": -2,
                                         "country": -2}, {"STRING"}, 3.0, 2.5),
        "Beneficiary.name_hash": ({"name": 3, "hash": 1, "legal": 0.5, "short": 0.3, "bank": -2,
                                   "account": -2}, {"STRING"}, 3.0, 0.0),
        "InvestigationCase.status": ({"status": 3, "state": 3, "resolution": -1}, None, 3.0, 0.0),
    }
    REACH = {   # where else an attribute may live: one hop out, through an edge with these words
        "Transaction.timestamp": {"header", "group", "message"},
        "Person.anonymized_ssn_hash": {"identifier", "identification", "private"},
        "Beneficiary.bank_identifier": {"agent", "bank", "institution"},
    }

    def map_attrs(self):
        for el, (cues, types, floor, tol) in self.ATTR_SPECS.items():
            ce = el.split(".")[0]
            ch = Choice(el, "attr", rule="X1 cue words on the mapped vertex, then one hop out")
            owners = [s for s in self.c[ce].sources if not s.get("attr")]
            if ce == "Transaction":
                owners = [s for s in owners if s.get("status") != "approximate"] or owners
            for own in owners:
                v = own["vertex"]
                cands = self._attr_cands(v, cues, types)
                via_used = None
                if (not cands or cands[0][0] < floor) and el in self.REACH:
                    hop = self._attr_one_hop(v, el, cues, types)
                    if hop and hop[0][0] >= floor:
                        cands, via_used = hop, True
                if (not cands or cands[0][0] < floor) and el == "Account.kyc_score":
                    hop = self._attr_via_owner(v, cues, types)
                    if hop and hop[0][0] >= floor:
                        cands, via_used = hop, True
                if not cands or cands[0][0] < floor:
                    if cands:
                        ch.rejected.append({"candidate": f"{v}.{cands[0][1].name}", "why": f"best cue score {cands[0][0]:.1f} < {floor}"})
                    continue
                top = cands[0][0]
                # a runner-up joins only if it is the same kind of thing (BIC *and* sort code;
                # service level *and* local instrument together make the channel)
                strong = self.CONCEPT[el] | ({"service", "level", "local", "instrument"} if el == "Transaction.channel" else set())
                pick = [cands[0]] + [c for c in cands[1:] if c[0] >= top - tol and set(words(c[1].name)) & strong]
                for sc, a, via, host in pick:
                    src = {"vertex": host, "attr": a.name, "type": a.type}
                    if via:
                        src["via"] = via
                    tr = self._transform(el, a)
                    if tr:
                        src["transform"] = tr
                    src["status"] = self._attr_status(el, a, via_used, own)
                    ch.sources.append(src)
                    ch.why.append(f"{host}.{a.name}{' via ' + via[0]['gsql'] if via else ''}: words "
                                  f"{sorted(set(words(a.name)) & set(k for k, w in cues.items() if w > 0))} score {sc:.1f}")
                ch.rejected += [{"candidate": f"{c[3]}.{c[1].name}", "why": f"score {c[0]:.1f} vs {top:.1f}"}
                                for c in cands if c not in pick][:2]
                if via_used and pick and el == "Person.anonymized_ssn_hash":
                    host = pick[0][3]
                    sch = next((a.name for a in self.V(host).attrs if set(words(a.name)) & {"scheme", "type"}), None)
                    if sch:
                        ch.sources[-1]["filter"] = f"{sch} = <national id scheme>"
                        ch.pending.append(f"{host}.{sch}: one party holds several identity documents; pick the national-id scheme code from sample records")
                if ce == "Beneficiary" and own.get("filter"):
                    for src in ch.sources:
                        src["status"] = "approximate" if el != "Beneficiary.name_hash" else src["status"]
                if len(cands) > 1:
                    ch.margin = round(top - next((c[0] for c in cands if c[0] < top - tol), top), 2)
            if ch.sources:
                ch.status = "exact" if all(s["status"] == "exact" for s in ch.sources) else "approximate"
                for s in ch.sources:
                    if s.get("transform", "").startswith("parse"):
                        ch.pending.append(f"{s['vertex']}.{s['attr']}: confirm the string date format on sample records")
            self._set(ch)

    def _attr_cands(self, v: str, cues: dict, types):
        out = []
        for a in self.V(v).attrs:
            if a.primary or (types and a.type not in types):
                continue
            sc = score(words(a.name), cues)
            if sc > 0:
                out.append((sc, a, None, v))
        return sorted(out, key=lambda c: -c[0])

    def _attr_one_hop(self, v: str, el: str, cues: dict, types):
        out = []
        for e, fwd, o in self._adjacent(v):
            ew = _own_words(e.name, v, o) | self.vwords(o)
            if not ew & self.REACH[el]:
                continue
            if o in self.role and el != "Beneficiary.bank_identifier":
                continue
            for sc, a, _, _ in self._attr_cands(o, cues, types):
                out.append((sc + 0.5, a, [_step(e, fwd, v, o)], o))
        if el == "Beneficiary.bank_identifier":   # the payee's bank hangs off the payment, on the creditor side
            for t in self.vertices("Transaction"):
                for e, fwd, o in self._adjacent(t):
                    ow = _own_words(e.name, t, o)
                    if fwd and ow & {"creditor", "beneficiary", "payee"} and self.vwords(o) & {"institution", "agent", "bank"} \
                            and not ow & {"ultimate", "intermediary"}:
                        for sc, a, _, _ in self._attr_cands(o, cues, types):
                            out.append((sc + 1.0, a, [_step(e, fwd, t, o)], o))
        return sorted(out, key=lambda c: -c[0])

    def _attr_via_owner(self, account: str, cues: dict, types):
        """A party-level rating stands in for an account score: walk OWNS backwards."""
        out = []
        for src in self.c["OWNS"].sources:
            if src["to"] != account:
                continue
            back = [{**st, "forward": not st["forward"], "from": st["to"], "to": st["from"],
                     "gsql": self._gsql(st["edge"], not st["forward"])} for st in reversed(src["path"])]
            for sc, a, _, _ in self._attr_cands(src["from"], cues, types):
                out.append((sc, a, back, src["from"]))
        return sorted(out, key=lambda c: -c[0])

    def _gsql(self, edge: str, forward: bool) -> str:
        e = self.s.edges[edge]
        return e.name if forward or not e.directed else (e.reverse or "reverse_" + e.name)

    @staticmethod
    def _transform(el: str, a) -> str:
        if a.type == "STRING" and el.endswith(("timestamp", "open_date", "verified_at")):
            return "parse string date (format from sample records)"
        return ""

    @staticmethod
    def _attr_status(el: str, a, via, owner) -> str:
        w = set(words(a.name))
        if via or a.type == "STRING" and el.endswith(("timestamp", "open_date", "verified_at")):
            return "approximate"
        concept = GraphMapper.CONCEPT[el]
        if not w & concept:
            return "approximate"
        if el == "Transaction.timestamp" and w & {"acceptance", "value", "booking", "processing", "settlement"}:
            return "approximate"   # a lifecycle stamp, not necessarily when money moved
        return "exact"

    CONCEPT = {"Transaction.amount": {"amount"}, "Transaction.timestamp": {"datetime", "timestamp"},
                   "Transaction.channel": {"channel", "mechanism"}, "Account.open_date": {"opening", "open", "start"},
                   "Account.status": {"status", "state"}, "Account.kyc_score": {"kyc", "score"},
                   "Person.anonymized_ssn_hash": {"identification", "ssn", "national", "legal"},
                   "Person.kyc_verified_at": {"verification", "verified"},
                   "Beneficiary.bank_identifier": {"bic", "sortcode", "swift", "routing"},
                   "Beneficiary.name_hash": {"name"}, "InvestigationCase.status": {"status"}}

    # ------------------------------------------------------------ derived
    def map_under_investigation(self):
        ch = Choice("account_under_investigation", "derived")
        inv = self.c["INVESTIGATED_IN"]
        if inv.sources:
            st = next((s["attr"] for s in self.c["InvestigationCase.status"].sources), None)
            ch.sources = [{"path": s["path"], "filter": f"case {st} is open" if st else "any case"} for s in inv.sources]
            ch.status, ch.rule = "exact", "U1 an open case concerns the account"
            ch.why.append(f"{inv.sources[0]['path'][0]['edge']} from a case whose {st or 'status'} is open")
            if st:
                ch.pending.append(f"{st}: which values mean open, from sample records")
        else:
            for v in self.vertices("Account"):
                for a in self.V(v).attrs:
                    w = set(words(a.name))
                    if a.type == "BOOL" and w & {"aml", "alert", "fraud", "suspicious", "investigation"}:
                        ch.sources.append({"vertex": v, "attr": a.name, "filter": f"{a.name} = true"})
                        ch.status, ch.rule = "approximate", "U2 an alert flag on the account (a flag, not a case)"
                        ch.why.append(f"no case type; {v}.{a.name} is an alert flag")
                    elif a.type in NUMERIC | {"STRING"} and w & {"restriction", "block", "blocked", "freeze"}:
                        ch.sources.append({"vertex": v, "attr": a.name, "filter": f"{a.name} = <fraud block code>"})
                        ch.status, ch.rule = "approximate", "U3 a block code on the account (a block, not a case)"
                        ch.why.append(f"no case type; {v}.{a.name} is a posting block code")
                        ch.pending.append(f"{v}.{a.name}: which code means a fraud block, from sample records or the bank's code table")
            if not ch.sources:
                ch.why.append("no case type, flag or block code")
        self._set(ch)


def map_schema(schema: Schema, samples: dict | None = None) -> Mapping:
    return GraphMapper(schema, samples).run()

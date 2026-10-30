"""The canonical investigation model every bank is mapped onto (the shape the
organiser's handout grades against). Edges are written in their canonical
direction; a bank edge that points the other way is walked as its reverse."""

VERTICES = ["Person", "Account", "Transaction", "Device", "IPAddress", "LoginEvent",
            "Beneficiary", "Card", "Merchant", "InvestigationCase"]

EDGES: dict[str, tuple[str, str]] = {
    "OWNS": ("Person", "Account"),
    "INITIATED": ("Account", "Transaction"),
    "PAYS_TO_ACCOUNT": ("Transaction", "Account"),
    "PAYS": ("Transaction", "Beneficiary"),
    "FROM_DEVICE": ("Transaction", "Device"),
    "FROM_IP": ("Transaction", "IPAddress"),
    "AT": ("Transaction", "Merchant"),
    "VIA": ("Transaction", "Card"),
    "HOLDS": ("Account", "Card"),
    "LOGGED_IN": ("LoginEvent", "Account"),
    "LOGIN_FROM_DEVICE": ("LoginEvent", "Device"),
    "LOGIN_FROM_IP": ("LoginEvent", "IPAddress"),
    "INVESTIGATED_IN": ("Account", "InvestigationCase"),
}

ATTRS = ["Transaction.amount", "Transaction.timestamp", "Transaction.channel",
         "Account.open_date", "Account.status", "Account.kyc_score",
         "Person.anonymized_ssn_hash", "Person.kyc_verified_at",
         "Beneficiary.bank_identifier", "Beneficiary.name_hash",
         "InvestigationCase.status"]

DERIVED = ["account_under_investigation"]

ELEMENTS = VERTICES + list(EDGES) + ATTRS + DERIVED

"""Fake credentials for attack fixtures, built at runtime so no scanner-visible literals exist."""
FAKE = {
    "AWS_KEY": "AK" + "IA" + "IOSFODNN" + "7EXAMPLE",
    "SK_KEY": "sk" + "-live-" + "9988776655" + "4433221100",
    "GH_TOKEN": "gh" + "p_" + "1234567890" + "abcdefghij" + "klmnopqrst" + "uvwxyz",
    "CARD": "4532" + "-0151-" + "1283-" + "0366",
}


def fill(obj):
    """Recursively replace {{FAKE:NAME}} placeholders in strings of dicts/lists."""
    if isinstance(obj, str):
        for k, v in FAKE.items():
            obj = obj.replace("{{FAKE:%s}}" % k, v)
        return obj
    if isinstance(obj, list):
        return [fill(x) for x in obj]
    if isinstance(obj, dict):
        return {k: fill(v) for k, v in obj.items()}
    return obj

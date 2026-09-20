import re

_PATTERNS = {
    'EMAIL': re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b'),
    'PHONE': re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    'CARD_NUMBER': re.compile(r'\b(?:\d[ -]*?){13,19}\b'),
    'IP_ADDRESS': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    'SSN_LIKE': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    'VIN_LIKE': re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b'),
}


def redact(text: str) -> str:
    for label, pattern in _PATTERNS.items():
        text = pattern.sub(f'[REDACTED:{label}]', text)
    return text


def redact_turn(turn: dict) -> dict:
    redacted = dict(turn)
    if 'attacker_prompt' in redacted:
        redacted['attacker_prompt'] = redact(redacted['attacker_prompt'])
    if 'target_response' in redacted:
        redacted['target_response'] = redact(redacted['target_response'])
    return redacted

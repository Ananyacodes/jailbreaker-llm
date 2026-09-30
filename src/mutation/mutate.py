from src.attacker.generate import AttackerGenerator
from src.attacker.technique_library import TECHNIQUES

_VARIANT_TECHNIQUES = [t for t in TECHNIQUES if t.name in {'persona_framing', 'hypothetical_framing', 'technical_register', 'rot13_obfuscation', 'base64_obfuscation'}]


def generate_variants(attack_text: str, generator: AttackerGenerator, n_per_technique: int = 2) -> list[dict]:
    variants = []
    for technique in _VARIANT_TECHNIQUES:
        seed = technique.build_seed(attack_text)
        for raw in generator.generate_n(seed, n=n_per_technique, max_new_tokens=128):
            variants.append({'technique': technique.name, 'text': technique.apply(raw)})
    return variants


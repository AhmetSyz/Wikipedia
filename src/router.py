"""
Query router: decides whether to retrieve from people, places, or both.

The assignment explicitly allows simple keyword/rule-based routing, so we
keep this transparent and fast — no extra LLM call.

Decision logic:
  1. If a known person's name appears in the query → "person"
  2. If a known place's name appears in the query → "place"
  3. If both kinds of names appear → "mixed"
  4. Otherwise, look at question stems:
       - "who", "whose", "compare X and Y" with people-ish hints → person
       - "where", "located", "city", "country", "tower", "wall" → place
  5. If nothing matches, return "mixed" so we search both stores. Better to
     give some answer than to refuse on routing alone.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PEOPLE, PLACES


RouteType = Literal["person", "place", "mixed"]


# Pre-compute lowercase name patterns. We match on substrings of words to
# catch "Einstein" as well as "Albert Einstein", and to handle accent-light
# matching by stripping diacritics.
_DIACRITIC_MAP = {
    "á": "a", "à": "a", "â": "a", "ä": "a", "ã": "a", "å": "a",
    "é": "e", "è": "e", "ê": "e", "ë": "e",
    "í": "i", "ì": "i", "î": "i", "ï": "i", "ı": "i", "İ": "i",
    "ó": "o", "ò": "o", "ô": "o", "ö": "o", "õ": "o",
    "ú": "u", "ù": "u", "û": "u", "ü": "u",
    "ý": "y", "ÿ": "y",
    "ñ": "n", "ç": "c", "ş": "s", "ğ": "g",
}
_DIACRITIC_TABLE = str.maketrans(_DIACRITIC_MAP)


def _norm(s: str) -> str:
    """Lowercase and strip simple diacritics so 'Atatürk' matches 'ataturk'."""
    return s.lower().translate(_DIACRITIC_TABLE)


def _name_tokens(name: str) -> list[str]:
    """Return useful tokens for a name: full name + last word + first word."""
    norm = _norm(name)
    tokens = [norm]
    parts = norm.split()
    if len(parts) > 1:
        tokens.append(parts[-1])   # last name / last word
        tokens.append(parts[0])    # first name / first word
    # de-dupe while preserving order
    seen, out = set(), []
    for t in tokens:
        if t not in seen and len(t) > 2:
            seen.add(t)
            out.append(t)
    return out


_PERSON_NAMES = {p: _name_tokens(p) for p in PEOPLE}
_PLACE_NAMES = {p: _name_tokens(p) for p in PLACES}

_PERSON_HINTS = {
    "who", "whose", "person", "biography", "born", "died", "discovered",
    "invented", "wrote", "painted", "composed", "scientist", "artist",
    "athlete", "musician", "writer", "physicist", "mathematician",
}
_PLACE_HINTS = {
    "where", "place", "located", "city", "country", "tower", "wall",
    "monument", "tomb", "temple", "palace", "mountain", "canyon",
    "pyramid", "statue", "ruins", "landmark",
}


def _hits(query_norm: str, name_dict: dict[str, list[str]]) -> list[str]:
    """Return list of entity names whose tokens appear in the query."""
    found: list[str] = []
    for name, tokens in name_dict.items():
        for tok in tokens:
            # match token as a whole word
            if re.search(rf"\b{re.escape(tok)}\b", query_norm):
                found.append(name)
                break
    return found


def route(query: str) -> tuple[RouteType, list[str], list[str]]:
    """
    Classify a query.

    Returns:
        (route_type, matched_people, matched_places)
    """
    qn = _norm(query)
    people_hits = _hits(qn, _PERSON_NAMES)
    place_hits = _hits(qn, _PLACE_NAMES)

    if people_hits and place_hits:
        return "mixed", people_hits, place_hits
    if people_hits:
        return "person", people_hits, []
    if place_hits:
        return "place", [], place_hits

    # No name matched — fall back to question-stem keywords
    words = set(re.findall(r"\b\w+\b", qn))
    person_score = len(words & _PERSON_HINTS)
    place_score = len(words & _PLACE_HINTS)

    if person_score > place_score and person_score > 0:
        return "person", [], []
    if place_score > person_score and place_score > 0:
        return "place", [], []

    # Truly ambiguous — search both
    return "mixed", [], []

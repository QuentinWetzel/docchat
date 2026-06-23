"""Decode SharePoint managed-metadata facet values.

Several Algolia facets store SharePoint taxonomy in the form:
    "44;#AIRBUS|48c6957d-0126-4743-8dbb-f1d9af2fb14a"
    "21;#Aerospace & Defense|9595424c-1a9e-42c7-af99-e63a7d2d8512"
i.e.  "<wssId>;#<Label>|<TermGuid>".

Some values are plain strings (e.g. Function = "Controlling", language = "French").
These helpers normalize both forms so users and the LLM can work with human labels,
while we still send the *raw stored value* back to Algolia for exact facet matching.
"""
from __future__ import annotations

import re

_ENCODED = re.compile(r"^\s*\d+;#(?P<label>.*?)\|[0-9a-fA-F-]{36}\s*$")


def decode_label(raw: str) -> str:
    """Return the human label from a possibly-encoded facet value."""
    if raw is None:
        return raw
    m = _ENCODED.match(raw)
    label = m.group("label").strip() if m else raw.strip()
    # SharePoint's export uses a fullwidth ampersand (U+FF06) in several labels;
    # normalize so it matches the regular "&" used elsewhere (KNOWN_VOCAB, LLM output).
    return label.replace("＆", "&")


def is_encoded(raw: str) -> bool:
    return bool(_ENCODED.match(raw or ""))


def build_label_index(facet_values: list[str]) -> dict[str, list[str]]:
    """Map normalized lowercase label -> raw stored values, for resolving user input.

    Pass the list of distinct raw facet values (e.g. from an Algolia facet query).
    Resolution is case-insensitive on the decoded label. Multiple raw values can decode to
    the same label (e.g. a SharePoint-encoded "44;#AIRBUS|<guid>" alongside a plain "Airbus"
    string from inconsistent upstream data) — keep all of them so callers can OR across both
    stored forms instead of silently dropping rows stored under the other form.
    """
    idx: dict[str, list[str]] = {}
    for raw in facet_values:
        idx.setdefault(decode_label(raw).lower(), []).append(raw)
    return idx


def resolve_to_raw(user_value: str, label_index: dict[str, list[str]]) -> list[str]:
    """Resolve a user-supplied label to the raw stored facet value(s).

    Falls back to the user value unchanged if no match (so plain-string facets and
    already-raw values still work).
    """
    return label_index.get(user_value.strip().lower(), [user_value])

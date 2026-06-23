"""Turn a natural-language query into a QueryUnderstanding using Gemini (Developer API).

We use LlamaIndex's structured_predict against the Gemini LLM so the model returns a
validated QueryUnderstanding directly: the filters (as before), plus a lexical_query and a
semantic_query — the text each retrieval leg should actually search with once filter terms are
pulled out.

The discipline: lexical_query is for exact terms of art (codes, named techniques, acronyms —
anything a paraphrase would lose); semantic_query is conceptual residue for the embedding leg, and
may be paraphrased/expanded beyond the original wording. The two slots can overlap, and either can
be None when that leg has nothing useful left to search on.

The extracted filter values are decoded labels; the legs resolve them to raw stored values where
needed (see taxonomy.resolve_to_raw, applied in pipeline.py).
"""
from __future__ import annotations

from functools import lru_cache

from google.genai import types as genai_types
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI

from .config import settings
from .schema import MetadataFilterSpec, QueryUnderstanding

# Known facet vocabularies (decoded human labels). Sourced from the live index facets.
# These keep the LLM honest; extend by querying Algolia facets periodically.
KNOWN_VOCAB = {
    "language": ["French", "English", "German", "Other"],
    "industry_sector": [
        "Aerospace & Defense", "Pharma & Healthcare Products", "Automotive & Transport",
        "Machinery & Electronics", "Energy & Utilities",
        "Construction, Infrastructure, & Real Estate", "Basic Materials & Chemicals",
        "IT & Technology Services", "Logistics",
    ],
    "service_line": [
        "Business & Digital Transformation", "Product Development",
        "Supply Chain & Procurement", "Cost & Cash Competitiveness",
        "Project & Portfolio Management", "Growth Strategy", "Growth & Offer Strategy",
    ],
    # "Proposal" is deliberately absent: that document_purpose value is tagged on only ~2 of
    # 13.9k slides (the field is ~97% untagged corpus-wide), so filtering on it kills recall.
    # "Proposals Library" in drive_name is the reliable signal for proposal-type documents
    # (10k+ slides) — see the worked example below.
    "document_purpose": ["Credential", "How to & Guidelines", "Framework & template"],
    "drive_name": [
        "Proposals Library", "Public", "Documents", "Credentials Library", "CV Library",
        "Credentials", "OneDrive", "Confidential", "Teams Wiki Data", "Templates", "CV",
    ],
    "site_display_name": ["OneShelf", "Team Drive", "Demo"],
    # Client is high-cardinality (90+ values); we let the model propose a name and resolve it
    # against Algolia facets at filter time rather than enumerating all here.
}

_SYSTEM = """You turn a natural-language search query into a structured request for a hybrid
retrieval system: one lexical (keyword) leg, one semantic (vector) leg, both constrained by the
same metadata filters.

1. filters — set a field only when the query clearly implies that constraint. Use ONLY values
   from the provided vocabularies for constrained fields. For `client`, output the company name as
   written (it is resolved later). Leave fields empty when in doubt; empty means "no filter", which
   is safer for recall than an over-eager guess.

   `language` is the one field most at risk of a false positive: it filters the LANGUAGE OF THE
   SLIDES, not the language the user happens to be writing the query in. A French question is not
   a request for French-tagged slides — the user is just asking in French and may well want
   results in any language. Only set `language` when the user explicitly asks for documents in a
   given language (e.g. "slides in German", "des supports en anglais", "credentials en français").
   Otherwise leave it empty, regardless of what language the query itself is written in.

2. lexical_query — the tokens/phrases that must match as literal text: specific terminology,
   codes, named techniques, acronyms — anything a paraphrase would lose. Drop terms already
   captured by a filter (e.g. a client name that became the `client` filter shouldn't reappear
   here) unless that same term also needs literal matching elsewhere. Set to null if nothing
   remains that needs literal matching.

3. semantic_query — the remaining concept, phrased for embedding similarity. You may paraphrase
   or add synonyms that never appeared in the original query, as long as they capture the same
   idea — this slot is matched by meaning, not by tokens. Set to null if nothing conceptual
   remains once filters are pulled out (e.g. a pure browse request).

lexical_query and semantic_query are NOT a partition of the query — they can overlap, and a term
can legitimately appear in both, in one, or in neither (if it became a filter).

4. intent_type — classify the query as one of:
   - "lookup": looking for a specific, identifiable fact or document
   - "filtered_browse": wants the set of documents matching the filters; ranking is secondary
   - "aggregate": wants a count or summary across many documents — flag this honestly, since this
     pipeline only ever surfaces a handful of top-ranked slides and cannot exhaustively count
   - "semantic_search": a conceptual question with no specific terms of art
   - "hybrid": a mix of the above (default when unsure)

Worked example —
Query: "Airbus credentials about engine on-dock-date delay"
  filters: {"client": ["Airbus"], "document_purpose": ["Credential"]}
  lexical_query: "on-dock date"
  semantic_query: "engine delivery timing inventory reduction"
  intent_type: "hybrid"
("Airbus" and "credentials" became filters and are dropped from both query slots. "on-dock date"
is a precise term of art worth matching verbatim. The underlying topic is restated conceptually,
in words that may not appear on the slide, so it also catches matches phrased differently.)

Worked example — proposal-type queries:
Query: "What did we propose to Airbus on supply chain?"
  filters: {"client": ["Airbus"], "drive_name": ["Proposals Library"]}
  lexical_query: null
  semantic_query: "supply chain proposal"
  intent_type: "hybrid"
(Use drive_name=["Proposals Library"] for "propose"/"proposal" intent, NOT
document_purpose=["Proposal"] — that taxonomy value is barely tagged in the corpus.)

Worked example — query language vs. language filter:
Query: "Quels sont nos credentials chez Airbus sur la supply chain ?"
  filters: {"client": ["Airbus"], "document_purpose": ["Credential"]}
  lexical_query: null
  semantic_query: "supply chain credentials"
  intent_type: "hybrid"
(The query is written in French, but the user never asked for French-tagged slides — they just
asked in French. `language` stays empty; only set it for an explicit request like "des slides en
français" or "show me the English version".)

Output the structured object only."""

_PROMPT = PromptTemplate(
    _SYSTEM + "\n\n"
    "Known vocabularies (choose only from these for the listed fields):\n"
    "{vocab}\n\n"
    "User query:\n{query}\n\n"
    "Produce the query understanding object."
)


def _vocab_text() -> str:
    lines = []
    for field, vals in KNOWN_VOCAB.items():
        lines.append(f"- {field}: {', '.join(vals)}")
    lines.append("- client: any company name (free text, resolved later)")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _llm() -> GoogleGenAI:
    return GoogleGenAI(
        model=settings.llm_model,
        api_key=settings.gemini_api_key,
        max_tokens=1024,
        # Structured extraction, not multi-step reasoning: disable thinking so it can't burn
        # the output-token budget (see pipeline.py for the MAX_TOKENS failure this caused).
        generation_config=genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )


def understand_query(query: str) -> QueryUnderstanding:
    try:
        return _llm().structured_predict(
            QueryUnderstanding,
            _PROMPT,
            vocab=_vocab_text(),
            query=query,
        )
    except Exception:
        # Never let query understanding failure break retrieval; fall back to an unfiltered
        # hybrid search on the raw query (i.e. what the pipeline did before this step existed).
        return QueryUnderstanding(
            intent_type="hybrid",
            filters=MetadataFilterSpec(),
            lexical_query=query,
            semantic_query=query,
        )

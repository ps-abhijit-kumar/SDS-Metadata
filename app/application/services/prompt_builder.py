"""SDS metadata extraction prompt builder.

Builds a compact, deterministic prompt for extracting metadata from
Safety Data Sheet (SDS) documents.

The prompt is intentionally optimized for local LLM inference by:
- Minimizing instruction length
- Enforcing a fixed response format
- Preventing hallucinated values
- Reducing unnecessary tokens
"""

from __future__ import annotations

_PROMPT_TEMPLATE = """\
You are an SDS metadata extraction engine.

Extract ONLY the following four fields from the SDS document.

Return EXACTLY these four lines and nothing else:

language: <value>
jurisdiction: <value>
company name: <value>
product name: <value>

Rules:
- Return exactly four lines.
- Preserve the field names exactly.
- Do not explain anything.
- Do not use markdown.
- Do not add extra text.
- If a value cannot be determined from text evidence, return "Unknown".
- Language is the document language.
- Jurisdiction is the regulatory framework followed by the SDS (e.g., WHMIS 2015, OSHA HazCom 2012, NOM-018-STPS, ABNT NBR 14725, REACH/CLP).
- Language and jurisdiction are independent. Do not infer one from the other.
- Company name: Extract the entity explicitly labeled as Manufacturer, Companhia, Fabricante, Company, or Supplier.
  * CRITICAL: Do NOT select a Brand or Trademark name (such as "Millipore" when "Sigma-Aldrich Brasil Ltda." is labeled as Companhia). Brand/Marca is NOT Company.
- Product name: Extract the main commercial or trade product name from Section 1 / Header.
  * CRITICAL: Do NOT select kit component names (such as "LIQUID HARDENER 0203"), ingredients, CAS names, or raw materials. Product name is the overall SDS product.

SDS DOCUMENT:

{context}
"""


def build_extraction_prompt(context_chunks: list[str]) -> str:
    """Build the metadata extraction prompt.

    Args:
        context_chunks:
            Relevant SDS text chunks retrieved from ChromaDB.

    Returns:
        Optimized prompt ready for the local LLM.
    """
    context = "\n\n".join(
        chunk.strip()
        for chunk in context_chunks
        if chunk.strip()
    )

    if not context:
        context = "(No document context available)"

    return _PROMPT_TEMPLATE.format(
        context=context,
    )
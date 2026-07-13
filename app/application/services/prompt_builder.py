"""SDS metadata extraction prompt builder.

Constructs a precise, structured prompt that:
  - Instructs the LLM to return EXACTLY four lines in a fixed format.
  - Explains the language/jurisdiction distinction with concrete examples.
  - Lists every supported jurisdiction so the model can match regulatory wording.
  - Prohibits fabrication — if a field cannot be determined, return "Unknown".
  - Optimizes for Qwen3:8B deterministic output.
"""

from __future__ import annotations

_JURISDICTION_LIST = """\
Supported jurisdictions (match exactly):
  • United States (OSHA / HazCom 2012)
  • Canada (WHMIS 2015)
  • European Union (REACH / CLP)
  • United Kingdom (UK REACH)
  • Brazil (ABNT NBR 14725)
  • Australia (Safe Work Australia)
  • New Zealand (HSNO)
  • Japan (MHLW / JIS Z 7253)
  • China (GB/T 16483)
  • South Korea (OSHACT K-REACH)
  • India (BIS / MSDS)
  • Singapore (WSH)
  • Mexico (NOM-018-STPS)"""

_PROMPT_TEMPLATE = """\
You are an expert SDS (Safety Data Sheet) analyst. Your only task is to \
extract four specific pieces of metadata from the SDS document context below.

CRITICAL INSTRUCTIONS:
1. Output EXACTLY four lines, one field per line.
2. Use this format (copy exactly, fill in values):
   language: <value>
   jurisdiction: <value>
   company name: <value>
   product name: <value>
3. Do NOT add extra text, explanations, notes, or empty lines.
4. Do NOT use markdown, bullet points, or special formatting.
5. If a field cannot be determined from the context, write: Unknown

FIELD DEFINITIONS:
- Language: The language the document is written in \
  (e.g., English, Portuguese, Spanish, French, German).
- Jurisdiction: The regulatory framework or country standard this SDS follows. \
  See the list below.
- Company Name: The manufacturer or company name responsible for the product.
- Product Name: The commercial or trade name of the chemical product.

LANGUAGE VS. JURISDICTION:
These are INDEPENDENT. An SDS can be written in English but follow Brazilian \
or EU regulations. Determine each separately.

JURISDICTION REFERENCE:
{jurisdiction_list}

JURISDICTION DETECTION TIPS:
- "GHS" alone is NOT sufficient — identify the specific national implementation.
- "OSHA 29 CFR", "HazCom 2012", "Right-to-Know" → United States (OSHA / HazCom 2012)
- "WHMIS", "Canada" regulatory references → Canada (WHMIS 2015)
- "REACH", "CLP Regulation (EC)", "1907/2006" → European Union (REACH / CLP)
- "UK REACH", "Great Britain" references → United Kingdom (UK REACH)
- "ABNT NBR 14725", "NR-26", "Brasil" → Brazil (ABNT NBR 14725)
- "NOM-018-STPS", "SEMARNAT" → Mexico (NOM-018-STPS)

--- SDS DOCUMENT CONTEXT ---
{context}
--- END OF CONTEXT ---

Now output the four lines:"""


def build_extraction_prompt(context_chunks: list[str]) -> str:
    """Build the metadata extraction prompt from retrieved context chunks.

    Args:
        context_chunks: List of relevant text chunks retrieved from ChromaDB.

    Returns:
        A complete prompt string ready to send to the LLM.
    """
    context = "\n\n---\n\n".join(chunk.strip() for chunk in context_chunks if chunk.strip())
    if not context:
        context = "(No document context available)"

    return _PROMPT_TEMPLATE.format(
        jurisdiction_list=_JURISDICTION_LIST,
        context=context,
    )


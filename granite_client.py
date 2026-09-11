"""
granite_client.py
=================
IBM watsonx.ai Granite model wrapper for the Course Content Simplification Agent.

Responsibilities:
  - Read credentials from environment variables (via python-dotenv)
  - Initialise ibm_watsonx_ai Credentials + ModelInference once (singleton)
  - Build a level-aware, RAG-grounded prompt
  - Call model.generate_text() and return the trimmed result

All text generation goes through this module — no other LLM provider is used
anywhere in the project.
"""

import os
from typing import List, Dict

from dotenv import load_dotenv

# Load .env file if present (no-op when variables are already in the environment)
load_dotenv()

# ---------------------------------------------------------------------------
# ibm-watsonx-ai imports
# ---------------------------------------------------------------------------
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# DecodingMethods enum path varies slightly between SDK versions — try both
try:
    from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
except ImportError:
    from ibm_watsonx_ai.metanames import DecodingMethods  # older SDK path

# ---------------------------------------------------------------------------
# Module-level singleton state
# ---------------------------------------------------------------------------
_model: ModelInference = None  # initialised once by _get_model()

# ---------------------------------------------------------------------------
# Level-specific instruction strings used in the prompt
# ---------------------------------------------------------------------------
_LEVEL_INSTRUCTIONS: Dict[str, str] = {
    "Beginner": (
        "Rewrite the text for a complete beginner (age ~14–16 or someone with no "
        "prior subject knowledge). Use short sentences (max 20 words each). Replace "
        "every technical term with an everyday analogy or plain-English description. "
        "Define any unavoidable jargon inline in parentheses the first time it appears. "
        "Avoid passive voice. Aim for a conversational, encouraging tone."
    ),
    "Intermediate": (
        "Rewrite the text for a first-year undergraduate student with basic subject "
        "knowledge. Use standard academic vocabulary but briefly clarify advanced or "
        "specialised terms in parentheses the first time they appear. Keep the "
        "explanation accurate and moderately concise. Maintain a neutral academic tone."
    ),
    "Expert": (
        "Rewrite the text for a domain expert or advanced graduate student. Be concise "
        "and technically precise. Preserve all domain-specific terminology without "
        "simplification. Remove redundancy and tighten the prose, but do not change "
        "the meaning or omit nuance. Assume the reader is fully comfortable with "
        "the subject vocabulary."
    ),
}


# ---------------------------------------------------------------------------
# Singleton initialisation
# ---------------------------------------------------------------------------

def _get_model() -> ModelInference:
    """
    Return the singleton ModelInference instance, creating it on first call.

    Reads credentials from:
      WATSONX_API_KEY      — IBM Cloud API key
      WATSONX_PROJECT_ID   — watsonx.ai project GUID
      WATSONX_URL          — regional endpoint (e.g. https://us-south.ml.cloud.ibm.com)
      WATSONX_MODEL_ID     — Granite model ID (e.g. ibm/granite-3-8b-instruct)

    Raises
    ------
    EnvironmentError
        If any required environment variable is missing.
    """
    global _model
    if _model is not None:
        return _model

    # --- Validate credentials --------------------------------------------------
    api_key    = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url        = os.getenv("WATSONX_URL")
    model_id   = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")

    missing = []
    if not api_key:
        missing.append("WATSONX_API_KEY")
    if not project_id:
        missing.append("WATSONX_PROJECT_ID")
    if not url:
        missing.append("WATSONX_URL")

    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}.\n"
            "Copy .env.example to .env and fill in your IBM Cloud / watsonx.ai "
            "credentials. See README.md for step-by-step instructions."
        )

    # --- Build credentials and model -------------------------------------------
    credentials = Credentials(url=url, api_key=api_key)

    _model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params={
            "max_new_tokens": 800,
            "temperature": 0.4,
            "repetition_penalty": 1.1,
            "decoding_method": DecodingMethods.SAMPLE.value,
        },
    )

    return _model


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    original_text: str,
    level: str,
    context_chunks: List[Dict],
) -> str:
    """
    Construct the full prompt sent to the Granite model.

    Parameters
    ----------
    original_text : str
        The dense academic text the user wants simplified.
    level : str
        One of "Beginner", "Intermediate", or "Expert".
    context_chunks : List[Dict]
        Retrieved RAG chunks, each with keys "chunk" and "source".
        May be an empty list if no relevant chunks were found.

    Returns
    -------
    str
        The complete prompt string ready to pass to model.generate_text().
    """
    level_instruction = _LEVEL_INSTRUCTIONS.get(
        level, _LEVEL_INSTRUCTIONS["Intermediate"]
    )

    # --- RAG context section ---------------------------------------------------
    if context_chunks:
        context_lines = []
        for i, item in enumerate(context_chunks, start=1):
            context_lines.append(
                f"[Context {i} — source: {item['source']}]\n{item['chunk']}"
            )
        context_block = "\n\n".join(context_lines)
        context_section = (
            "REFERENCE KNOWLEDGE BASE (use this to ground your explanation — "
            "do not invent facts beyond what is stated here or in the original text):\n\n"
            + context_block
        )
    else:
        context_section = (
            "REFERENCE KNOWLEDGE BASE: No relevant snippets were retrieved. "
            "Base your explanation solely on the original text provided. "
            "Do not invent facts."
        )

    # --- Full prompt -----------------------------------------------------------
    prompt = f"""You are an expert educational content writer. Your task is to rewrite academic text to make it more accessible, while preserving complete factual accuracy.

STRICT RULES:
1. Never invent, add, or imply facts that are not present in the original text or the reference knowledge base.
2. Do not change the meaning of any statement.
3. After your rewritten explanation, append a section titled exactly "Key Terms Explained:" followed by 2–5 bullet points, each defining one important term from the original text.
4. Do not include any preamble like "Here is the rewritten text:" — output the rewritten content directly.

LEVEL: {level}
LEVEL-SPECIFIC INSTRUCTION: {level_instruction}

{context_section}

ORIGINAL TEXT TO REWRITE:
{original_text}

REWRITTEN EXPLANATION:
"""

    return prompt


# ---------------------------------------------------------------------------
# Public simplification function
# ---------------------------------------------------------------------------

def simplify(
    original_text: str,
    level: str,
    context_chunks: List[Dict],
) -> str:
    """
    Simplify *original_text* at the given comprehension *level* using the
    IBM watsonx.ai Granite model, grounded by the provided *context_chunks*.

    Parameters
    ----------
    original_text : str
        Dense academic text to be rewritten.
    level : str
        "Beginner", "Intermediate", or "Expert".
    context_chunks : List[Dict]
        Retrieved RAG chunks from rag_utils.retrieve().

    Returns
    -------
    str
        The model's output — rewritten explanation plus "Key Terms Explained:" section.

    Raises
    ------
    EnvironmentError
        If watsonx.ai credentials are not configured.
    RuntimeError
        If the model call fails for any other reason.
    """
    model = _get_model()
    prompt = build_prompt(original_text, level, context_chunks)

    try:
        result = model.generate_text(prompt=prompt)
    except Exception as exc:
        raise RuntimeError(
            f"watsonx.ai model call failed: {exc}\n"
            "Check your WATSONX_URL and WATSONX_API_KEY, and ensure the model "
            f"'{os.getenv('WATSONX_MODEL_ID', 'ibm/granite-3-8b-instruct')}' is "
            "available in your region."
        ) from exc

    return result.strip() if result else ""

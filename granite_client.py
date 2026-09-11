"""
granite_client.py
=================
IBM watsonx.ai Granite model wrapper for the Course Content Simplification Agent.

Responsibilities:
  - Read credentials from environment variables (via python-dotenv)
  - Initialise ibm_watsonx_ai Credentials + ModelInference once (singleton)
  - Build a level-aware, RAG-grounded prompt
  - Call model.generate_text() and return the trimmed result
  - Generate quiz questions from simplified text (Quiz Generator Agent)
  - Translate simplified output to a target language (Language Toggle)

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
# Level-specific instruction strings used in the simplification prompt
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
# Supported output languages for the Language Toggle feature
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    "English":  None,           # no translation needed
    "Hindi":    "Hindi",
    "Tamil":    "Tamil",
    "Bengali":  "Bengali",
    "Marathi":  "Marathi",
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
      WATSONX_MODEL_ID     — Granite model ID (e.g. ibm/granite-4-h-small)

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
    model_id   = os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small")

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
            "max_new_tokens": 900,
            "temperature": 0.4,
            "repetition_penalty": 1.1,
            "decoding_method": DecodingMethods.SAMPLE.value,
        },
    )

    return _model


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_prompt(
    original_text: str,
    level: str,
    context_chunks: List[Dict],
) -> str:
    """
    Construct the simplification prompt sent to Granite.

    Parameters
    ----------
    original_text : str
        The dense academic text the user wants simplified.
    level : str
        One of "Beginner", "Intermediate", or "Expert".
    context_chunks : List[Dict]
        Retrieved RAG chunks, each with keys "chunk" and "source".

    Returns
    -------
    str
        Complete prompt string for model.generate_text().
    """
    level_instruction = _LEVEL_INSTRUCTIONS.get(
        level, _LEVEL_INSTRUCTIONS["Intermediate"]
    )

    # RAG context section
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


def build_quiz_prompt(simplified_text: str, level: str) -> str:
    """
    Build a prompt for the Quiz Generator Agent.

    Generates 3 check-your-understanding questions from the simplified text,
    calibrated to the chosen comprehension level.

    Parameters
    ----------
    simplified_text : str
        The already-simplified explanation (Key Terms section stripped).
    level : str
        Comprehension level — affects question difficulty.

    Returns
    -------
    str
        Prompt string for model.generate_text().
    """
    difficulty_map = {
        "Beginner":     "simple recall and basic comprehension questions (who/what/why)",
        "Intermediate": "application and explanation questions requiring a short answer",
        "Expert":       "analysis and evaluation questions requiring precise technical answers",
    }
    difficulty = difficulty_map.get(level, difficulty_map["Intermediate"])

    prompt = f"""You are a Quiz Generator Agent. Your job is to create exactly 3 check-your-understanding questions based ONLY on the text provided below.

RULES:
1. Generate exactly 3 questions — no more, no less.
2. Questions must be answerable from the text alone — do not ask about outside knowledge.
3. Use this question difficulty level: {difficulty}
4. Format your output EXACTLY like this (no extra text before or after):
Q1: [question text]
Q2: [question text]
Q3: [question text]

TEXT TO GENERATE QUESTIONS FROM:
{simplified_text}

QUESTIONS:
"""
    return prompt


def build_translation_prompt(text: str, target_language: str) -> str:
    """
    Build a prompt for translating simplified output into a target language.

    Parameters
    ----------
    text : str
        The full simplified output including Key Terms section.
    target_language : str
        One of: Hindi, Tamil, Bengali, Marathi.

    Returns
    -------
    str
        Prompt string for model.generate_text().
    """
    prompt = f"""You are a professional educational translator. Translate the following simplified academic explanation into {target_language}.

RULES:
1. Translate ALL text including the "Key Terms Explained:" section.
2. Keep technical/scientific terms in English within parentheses after their {target_language} translation where appropriate, so learners can cross-reference.
3. Preserve the original formatting and structure exactly.
4. Do not add any preamble — output the translated text directly.
5. Maintain an encouraging, accessible tone appropriate for students.

TEXT TO TRANSLATE:
{text}

{target_language.upper()} TRANSLATION:
"""
    return prompt


# ---------------------------------------------------------------------------
# Public agent functions
# ---------------------------------------------------------------------------

def simplify(
    original_text: str,
    level: str,
    context_chunks: List[Dict],
) -> str:
    """
    Simplify *original_text* at the given comprehension *level* using
    IBM watsonx.ai Granite, grounded by *context_chunks*.

    Returns
    -------
    str
        Rewritten explanation plus "Key Terms Explained:" bullet section.

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
            f"'{os.getenv('WATSONX_MODEL_ID', 'ibm/granite-4-h-small')}' is "
            "available in your region."
        ) from exc

    return result.strip() if result else ""


def generate_quiz(simplified_text: str, level: str) -> List[str]:
    """
    Quiz Generator Agent — produce 3 comprehension questions from
    *simplified_text* calibrated to *level*.

    Parameters
    ----------
    simplified_text : str
        The main body of the simplified explanation (Key Terms stripped).
    level : str
        "Beginner", "Intermediate", or "Expert".

    Returns
    -------
    List[str]
        List of 3 question strings. Returns an empty list on failure.
    """
    model = _get_model()
    prompt = build_quiz_prompt(simplified_text, level)

    try:
        result = model.generate_text(prompt=prompt)
    except Exception:
        return []

    if not result:
        return []

    # Parse Q1: / Q2: / Q3: lines
    questions = []
    for line in result.strip().splitlines():
        line = line.strip()
        if line.startswith("Q1:") or line.startswith("Q2:") or line.startswith("Q3:"):
            q_text = line.split(":", 1)[-1].strip()
            if q_text:
                questions.append(q_text)

    return questions[:3]  # cap at 3


def translate_output(text: str, target_language: str) -> str:
    """
    Language Toggle — translate *text* into *target_language* using Granite.

    Parameters
    ----------
    text : str
        Full simplified output to translate.
    target_language : str
        One of: Hindi, Tamil, Bengali, Marathi.

    Returns
    -------
    str
        Translated text string. Returns original text on failure.
    """
    if target_language not in SUPPORTED_LANGUAGES or SUPPORTED_LANGUAGES[target_language] is None:
        return text  # English — no translation needed

    model = _get_model()
    prompt = build_translation_prompt(text, target_language)

    try:
        result = model.generate_text(prompt=prompt)
        return result.strip() if result else text
    except Exception:
        return text  # graceful fallback — return original if translation fails

"""
granite_client.py
=================
IBM watsonx.ai Granite model wrapper for the Course Content Simplification Agent.

Responsibilities:
  - Read credentials from environment variables (via python-dotenv)
  - Initialise ibm_watsonx_ai Credentials + ModelInference once (singleton)
  - Build a level-aware, RAG-grounded prompt
  - Call model.chat() (new /ml/v1/text/chat endpoint) and return the result
  - Generate quiz questions from simplified text (Quiz Generator Agent)
  - Translate simplified output to a target language (Language Toggle)

All text generation goes through this module — no other LLM provider is used
anywhere in the project.
"""

import os
import warnings
from typing import List, Dict

from dotenv import load_dotenv

# Load .env file if present (no-op when variables are already in the environment)
load_dotenv()

# Suppress SDK deprecation warnings in production — we use the current chat API
warnings.filterwarnings("ignore", category=UserWarning,   module="ibm_watsonx_ai")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ibm_watsonx_ai")

# ---------------------------------------------------------------------------
# ibm-watsonx-ai imports
# ---------------------------------------------------------------------------
from ibm_watsonx_ai import Credentials                          # noqa: E402
from ibm_watsonx_ai.foundation_models import ModelInference     # noqa: E402

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
    "English":  None,      # no translation needed
    "Hindi":    "Hindi",
    "Spanish":  "Spanish",
    "French":   "French",
    "Bengali":  "Bengali",
}


# ---------------------------------------------------------------------------
# Internal helper: call the chat API and extract content
# ---------------------------------------------------------------------------

def _chat(model: ModelInference, system_msg: str, user_msg: str) -> str:
    """
    Send a system + user message pair to the Granite chat endpoint and return
    the assistant reply text.

    Uses the /ml/v1/text/chat API (preferred over deprecated generate_text).

    Parameters
    ----------
    model : ModelInference
        The initialised singleton model.
    system_msg : str
        System-role instruction.
    user_msg : str
        User-role content (the actual task).

    Returns
    -------
    str
        Stripped assistant reply, or empty string on failure.
    """
    messages = [
        {"role": "system",  "content": system_msg},
        {"role": "user",    "content": user_msg},
    ]
    try:
        resp = model.chat(messages=messages)
        # resp is a dict: resp['choices'][0]['message']['content']
        return resp["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        # Fallback: try generate_text if chat response structure differs
        combined = f"{system_msg}\n\n{user_msg}"
        result = model.generate_text(prompt=combined)
        return result.strip() if result else ""


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
      WATSONX_MODEL_ID     — Granite model ID (default: ibm/granite-4-h-small)

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
    if not api_key:    missing.append("WATSONX_API_KEY")
    if not project_id: missing.append("WATSONX_PROJECT_ID")
    if not url:        missing.append("WATSONX_URL")

    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}.\n"
            "Copy .env.example to .env and fill in your IBM Cloud / watsonx.ai "
            "credentials. See README.md for step-by-step instructions."
        )

    # --- Build credentials and model -------------------------------------------
    credentials = Credentials(url=url, api_key=api_key)

    # Use max_tokens (chat API param). temperature + repetition_penalty still work.
    _model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params={
            "max_tokens":         900,
            "temperature":        0.4,
            "repetition_penalty": 1.1,
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
) -> tuple:
    """
    Construct the simplification system + user messages for Granite.

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
    tuple[str, str]
        (system_message, user_message) ready for _chat().
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

    system_msg = (
        "You are an expert educational content writer. Your task is to rewrite "
        "academic text to make it more accessible, while preserving complete "
        "factual accuracy.\n\n"
        "STRICT RULES:\n"
        "1. Never invent, add, or imply facts not present in the original text "
        "or the reference knowledge base.\n"
        "2. Do not change the meaning of any statement.\n"
        "3. After your rewritten explanation, append a section titled exactly "
        "\"Key Terms Explained:\" followed by 2-5 bullet points, each defining "
        "one important term from the original text.\n"
        "4. Do not include any preamble like \"Here is the rewritten text:\" — "
        "output the rewritten content directly."
    )

    user_msg = (
        f"LEVEL: {level}\n"
        f"LEVEL-SPECIFIC INSTRUCTION: {level_instruction}\n\n"
        f"{context_section}\n\n"
        f"ORIGINAL TEXT TO REWRITE:\n{original_text}\n\n"
        "REWRITTEN EXPLANATION:"
    )

    return system_msg, user_msg


def build_quiz_prompt(simplified_text: str, level: str) -> tuple:
    """
    Build system + user messages for the Quiz Generator Agent.

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
    tuple[str, str]
        (system_message, user_message) ready for _chat().
    """
    difficulty_map = {
        "Beginner":     "simple recall and basic comprehension questions (who/what/why)",
        "Intermediate": "application and explanation questions requiring a short answer",
        "Expert":       "analysis and evaluation questions requiring precise technical answers",
    }
    difficulty = difficulty_map.get(level, difficulty_map["Intermediate"])

    system_msg = (
        "You are a Quiz Generator Agent. Your job is to create exactly 3 "
        "check-your-understanding questions based ONLY on the text provided.\n\n"
        "RULES:\n"
        "1. Generate exactly 3 questions — no more, no less.\n"
        "2. Questions must be answerable from the text alone — do not ask about "
        "outside knowledge.\n"
        f"3. Use this question difficulty level: {difficulty}\n"
        "4. Format your output EXACTLY like this (no extra text before or after):\n"
        "Q1: [question text]\n"
        "Q2: [question text]\n"
        "Q3: [question text]"
    )

    user_msg = (
        f"TEXT TO GENERATE QUESTIONS FROM:\n{simplified_text}\n\nQUESTIONS:"
    )

    return system_msg, user_msg


def build_translation_prompt(text: str, target_language: str) -> tuple:
    """
    Build system + user messages for translating simplified output.

    Parameters
    ----------
    text : str
        The full simplified output including Key Terms section.
    target_language : str
        One of: Hindi, Spanish, French, Bengali.

    Returns
    -------
    tuple[str, str]
        (system_message, user_message) ready for _chat().
    """
    system_msg = (
        f"You are a professional educational translator. Translate academic "
        f"content into {target_language}.\n\n"
        "RULES:\n"
        "1. Translate ALL text including the \"Key Terms Explained:\" section.\n"
        f"2. Keep technical/scientific terms in English within parentheses after "
        f"their {target_language} translation where appropriate, so learners can "
        "cross-reference.\n"
        "3. Preserve the original formatting and structure exactly.\n"
        "4. Do not add any preamble — output the translated text directly.\n"
        "5. Maintain an encouraging, accessible tone appropriate for students."
    )

    user_msg = (
        f"TEXT TO TRANSLATE:\n{text}\n\n"
        f"{target_language.upper()} TRANSLATION:"
    )

    return system_msg, user_msg


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
    system_msg, user_msg = build_prompt(original_text, level, context_chunks)

    try:
        result = _chat(model, system_msg, user_msg)
    except Exception as exc:
        raise RuntimeError(
            f"watsonx.ai model call failed: {exc}\n"
            "Check your WATSONX_URL and WATSONX_API_KEY, and ensure the model "
            f"'{os.getenv('WATSONX_MODEL_ID', 'ibm/granite-4-h-small')}' is "
            "available in your region."
        ) from exc

    return result


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
        List of up to 3 question strings. Returns an empty list on failure.
    """
    model = _get_model()
    system_msg, user_msg = build_quiz_prompt(simplified_text, level)

    try:
        result = _chat(model, system_msg, user_msg)
    except Exception:
        return []

    if not result:
        return []

    # Parse Q1: / Q2: / Q3: lines
    questions = []
    for line in result.splitlines():
        line = line.strip()
        if line.startswith(("Q1:", "Q2:", "Q3:")):
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
        One of: Hindi, Spanish, French, Bengali.

    Returns
    -------
    str
        Translated text string. Returns original text on failure.
    """
    if (
        target_language not in SUPPORTED_LANGUAGES
        or SUPPORTED_LANGUAGES[target_language] is None
    ):
        return text  # English — no translation needed

    model = _get_model()
    system_msg, user_msg = build_translation_prompt(text, target_language)

    try:
        result = _chat(model, system_msg, user_msg)
        return result if result else text
    except Exception:
        return text  # graceful fallback — return English if translation fails

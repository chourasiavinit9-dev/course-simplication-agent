"""
app.py
======
Streamlit UI for the Course Content Simplification Agent.

Run with:
    streamlit run app.py

Requires a populated .env file — see .env.example for credential instructions.
"""

import streamlit as st
import textstat

import rag_utils
import granite_client

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Course Content Simplification Agent",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cache the RAG index so sentence-transformers loads only once per session.
# build_index() is idempotent, so calling it inside a cached function is safe.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base index (first run may take a moment)…")
def load_index():
    """Build the RAG embedding index once and cache it for the session."""
    rag_utils.build_index("knowledge_base")
    return True  # return value is unused; side effect is what matters


# ---------------------------------------------------------------------------
# Readability helpers
# ---------------------------------------------------------------------------

# Flesch Reading Ease score bands (https://en.wikipedia.org/wiki/Flesch-Kincaid_readability_tests)
_FRE_BANDS = [
    (90, "Very Easy (5th grade)"),
    (80, "Easy (6th grade)"),
    (70, "Fairly Easy (7th grade)"),
    (60, "Standard (8th–9th grade)"),
    (50, "Fairly Difficult (10th–12th grade)"),
    (30, "Difficult (college level)"),
    (0,  "Very Confusing (professional / academic)"),
]


def fre_label(score: float) -> str:
    """Return a human-readable label for a Flesch Reading Ease score."""
    for threshold, label in _FRE_BANDS:
        if score >= threshold:
            return label
    return "Very Confusing (professional / academic)"


def extract_main_text(full_output: str) -> str:
    """
    Return only the rewritten explanation portion of the model output,
    stripping the 'Key Terms Explained:' section so FRE scoring is not
    skewed by short bullet-point definitions.
    """
    delimiter = "Key Terms Explained:"
    idx = full_output.find(delimiter)
    if idx != -1:
        return full_output[:idx].strip()
    return full_output.strip()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📚 Course Content Simplification Agent")
st.markdown(
    """
    Paste dense academic text below — a textbook paragraph, lecture note, or
    research excerpt — choose your comprehension level, and let IBM Granite
    rewrite it for you.  
    The agent uses a **RAG pipeline** grounded on a local knowledge base of
    glossary definitions; no text leaves your machine except the call to
    IBM watsonx.ai.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Ensure the knowledge-base index is built before any user action
# ---------------------------------------------------------------------------
try:
    load_index()
except FileNotFoundError as e:
    st.error(f"⚠️ Knowledge base error: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Input controls
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([3, 1], gap="large")

with col_left:
    original_text = st.text_area(
        label="Original Academic Text",
        placeholder=(
            "Paste your textbook paragraph, lecture notes, or research excerpt here…"
        ),
        height=220,
        help="Minimum ~30 words recommended for meaningful simplification.",
    )

with col_right:
    level = st.selectbox(
        label="Comprehension Level",
        options=["Beginner", "Intermediate", "Expert"],
        index=0,
        help=(
            "**Beginner** — short sentences, everyday analogies, no jargon\n\n"
            "**Intermediate** — standard vocabulary with brief clarifications\n\n"
            "**Expert** — concise, technically precise, terminology preserved"
        ),
    )

    st.markdown("&nbsp;")  # visual spacer
    run_button = st.button("✨ Simplify", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main pipeline — triggered by the Simplify button
# ---------------------------------------------------------------------------
if run_button:
    if not original_text.strip():
        st.warning("Please paste some academic text before clicking Simplify.")
        st.stop()

    if len(original_text.split()) < 10:
        st.warning("The input text is very short. Results will be better with at least 30 words.")

    # --- Step 1: RAG retrieval -------------------------------------------------
    with st.spinner("Searching knowledge base for relevant context…"):
        try:
            context_chunks = rag_utils.retrieve(
                query=original_text,
                top_k=4,
                min_score=0.15,
                folder="knowledge_base",
            )
        except Exception as exc:
            st.error(f"RAG retrieval failed: {exc}")
            st.stop()

    # --- Step 2: Granite model call --------------------------------------------
    with st.spinner(f"Calling IBM watsonx.ai Granite ({level} level)…"):
        try:
            simplified_output = granite_client.simplify(
                original_text=original_text,
                level=level,
                context_chunks=context_chunks,
            )
        except EnvironmentError as exc:
            st.error(
                f"**Credential error:** {exc}\n\n"
                "Copy `.env.example` to `.env` and fill in your IBM Cloud credentials."
            )
            st.stop()
        except RuntimeError as exc:
            st.error(f"**Model error:** {exc}")
            st.stop()

    st.success("Done!", icon="✅")
    st.divider()

    # --- Output: simplified text -----------------------------------------------
    st.subheader(f"📝 Simplified Explanation ({level})")
    st.markdown(simplified_output)

    st.divider()

    # --- Expander 1: RAG transparency ------------------------------------------
    with st.expander("🔍 Retrieved Knowledge-Base Snippets", expanded=False):
        if context_chunks:
            st.markdown(
                f"**{len(context_chunks)} snippet(s)** were retrieved and used as "
                "grounding context for the model prompt."
            )
            for i, item in enumerate(context_chunks, start=1):
                st.markdown(
                    f"**Snippet {i}** &nbsp;·&nbsp; "
                    f"Source: `{item['source']}` &nbsp;·&nbsp; "
                    f"Similarity score: `{item['score']:.3f}`"
                )
                st.info(item["chunk"])
        else:
            st.info(
                "No knowledge-base snippets met the relevance threshold (score ≥ 0.15). "
                "The model was instructed to rely solely on the original text."
            )

    # --- Expander 2: Readability comparison ------------------------------------
    with st.expander("📊 Readability Comparison (Flesch Reading Ease)", expanded=False):
        # Score only the main body, not the Key Terms bullet list
        main_simplified = extract_main_text(simplified_output)

        original_fre    = textstat.flesch_reading_ease(original_text)
        simplified_fre  = textstat.flesch_reading_ease(main_simplified)
        delta           = simplified_fre - original_fre

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric(
                label="Original FRE Score",
                value=f"{original_fre:.1f}",
                help=fre_label(original_fre),
            )
            st.caption(fre_label(original_fre))

        with col_b:
            st.metric(
                label=f"Simplified FRE Score ({level})",
                value=f"{simplified_fre:.1f}",
                delta=f"{delta:+.1f}",
                help=fre_label(simplified_fre),
            )
            st.caption(fre_label(simplified_fre))

        with col_c:
            st.metric(
                label="Improvement",
                value=f"{delta:+.1f} pts",
                delta=f"{'↑ Easier' if delta > 0 else '↓ Harder' if delta < 0 else '— Unchanged'}",
                delta_color="normal",
            )

        st.markdown("---")
        st.markdown(
            "**Flesch Reading Ease scale:**  \n"
            "90–100 Very Easy · 80–89 Easy · 70–79 Fairly Easy · "
            "60–69 Standard · 50–59 Fairly Difficult · "
            "30–49 Difficult · 0–29 Very Confusing  \n"
            "_Scores are computed on the rewritten explanation only "
            "(the Key Terms section is excluded for a fair comparison)._"
        )

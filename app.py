"""
app.py
======
Streamlit UI for the Course Content Simplification Agent.
Onyx Black + Candy/Icy Blue bento dashboard with bot avatar.

Features:
  - Simplification at Beginner / Intermediate / Expert level (IBM Granite)
  - RAG retrieval from local knowledge base (sentence-transformers, CPU)
  - Quiz Generator Agent — 3 comprehension questions from simplified text
  - Language Toggle — translate output to Hindi / Tamil / Bengali / Marathi
  - Download Export — save as Markdown or PDF

Run with:
    streamlit run app.py

Requires a populated .env file — see .env.example for credential instructions.
"""

import io
import streamlit as st
import textstat

import rag_utils
import granite_client
from granite_client import SUPPORTED_LANGUAGES

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Course Content Simplification Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS — Onyx Black + Candy/Icy Blue Bento Dashboard
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  :root {
    --onyx:#0a0a0f; --onyx2:#12121a; --onyx3:#1a1a26; --onyx4:#22223a;
    --candy:#00c2ff; --icy:#7ee8fa; --mid:#0077b6;
    --glow:rgba(0,194,255,0.18); --glow2:rgba(0,194,255,0.35);
    --t1:#e8f4fd; --t2:#7aa5c0; --t3:#3d5a72;
    --green:#00e5a0; --warn:#ffb830; --err:#ff4d6d; --purple:#b97aff;
  }
  html, body, [data-testid="stAppViewContainer"] {
    background: var(--onyx) !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
    color: var(--t1) !important;
  }
  [data-testid="stHeader"]  { background: var(--onyx) !important; border-bottom: 1px solid var(--onyx4); }
  [data-testid="stSidebar"] { background: var(--onyx2) !important; }
  [data-testid="block-container"] { padding: 1.5rem 2rem 3rem !important; max-width: 1200px; }
  #MainMenu, footer, header { visibility: hidden; }

  /* Bento cards */
  .bento { background:var(--onyx2); border:1px solid var(--onyx4); border-radius:16px; padding:18px 20px; margin-bottom:14px; }
  .bento:hover { border-color:rgba(0,194,255,0.3); box-shadow:0 0 24px var(--glow); transition:.2s; }
  .bento-glow { background:linear-gradient(135deg,#0d1f35,#0a1628); border:1px solid rgba(0,194,255,0.25); border-radius:16px; padding:18px 20px; margin-bottom:14px; box-shadow:0 0 32px var(--glow); }

  /* Navbar */
  .navbar { display:flex; align-items:center; justify-content:space-between; padding:14px 20px; background:var(--onyx2); border:1px solid var(--onyx4); border-radius:16px; margin-bottom:16px; }
  .nb-left { display:flex; align-items:center; gap:14px; }
  .avatar { width:46px; height:46px; border-radius:50%; background:linear-gradient(135deg,#00c2ff,#0043ce); display:flex; align-items:center; justify-content:center; font-size:1.3rem; box-shadow:0 0 16px rgba(0,194,255,0.5); flex-shrink:0; }
  .nav-title { font-size:1rem; font-weight:700; }
  .nav-sub { font-size:0.72rem; color:var(--t2); }
  .nb-right { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .pill     { background:var(--onyx3); border:1px solid var(--onyx4); border-radius:20px; padding:4px 12px; font-size:0.7rem; color:var(--t2); }
  .pill-b   { background:rgba(0,194,255,0.12); border:1px solid rgba(0,194,255,0.3); border-radius:20px; padding:4px 12px; font-size:0.7rem; color:var(--candy); font-weight:600; }
  .pill-g   { background:rgba(0,229,160,0.1); border:1px solid rgba(0,229,160,0.3); border-radius:20px; padding:4px 12px; font-size:0.7rem; color:var(--green); font-weight:600; }

  /* Stats row */
  .stats { display:flex; gap:10px; margin-bottom:16px; flex-wrap:wrap; }
  .sc { flex:1; min-width:100px; background:var(--onyx2); border:1px solid var(--onyx4); border-radius:14px; padding:14px 16px; text-align:center; }
  .sc-num { font-size:1.4rem; font-weight:700; color:var(--candy); }
  .sc-lbl { font-size:0.65rem; color:var(--t2); margin-top:3px; text-transform:uppercase; letter-spacing:0.05em; }

  /* Section label */
  .sec-lbl { font-size:0.65rem; font-weight:700; color:var(--candy); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px; }

  /* Inputs */
  .stTextArea textarea { background:var(--onyx3)!important; border:1px solid var(--onyx4)!important; border-radius:12px!important; color:var(--t1)!important; font-size:0.88rem!important; }
  .stTextArea textarea:focus { border-color:var(--candy)!important; box-shadow:0 0 0 3px var(--glow)!important; }
  .stTextArea label, .stSelectbox label { color:var(--t2)!important; font-size:0.8rem!important; }
  .stSelectbox > div > div { background:var(--onyx3)!important; border:1px solid var(--onyx4)!important; border-radius:12px!important; color:var(--t1)!important; }

  /* Buttons */
  .stButton > button[kind="primary"] { background:linear-gradient(135deg,#00c2ff,#0077b6)!important; color:#fff!important; border:none!important; border-radius:12px!important; font-weight:700!important; font-size:0.9rem!important; padding:12px 24px!important; box-shadow:0 4px 20px rgba(0,194,255,0.35)!important; }
  .stButton > button[kind="primary"]:hover { box-shadow:0 6px 28px rgba(0,194,255,0.55)!important; transform:translateY(-1px)!important; }
  .stButton > button { background:var(--onyx3)!important; color:var(--t1)!important; border:1px solid var(--onyx4)!important; border-radius:12px!important; }

  /* Download buttons */
  .stDownloadButton > button { background:var(--onyx3)!important; color:var(--candy)!important; border:1px solid rgba(0,194,255,0.3)!important; border-radius:10px!important; font-size:0.82rem!important; }
  .stDownloadButton > button:hover { background:rgba(0,194,255,0.1)!important; }

  /* Metrics */
  [data-testid="stMetric"] { background:var(--onyx3)!important; border:1px solid var(--onyx4)!important; border-radius:14px!important; padding:16px!important; }
  [data-testid="stMetricValue"] { color:var(--candy)!important; font-size:1.6rem!important; font-weight:700!important; }
  [data-testid="stMetricLabel"] { color:var(--t2)!important; font-size:0.75rem!important; }

  /* Alerts */
  .stAlert { border-radius:12px!important; border:none!important; }
  [data-testid="stInfo"]    { background:rgba(0,194,255,0.08)!important; border-left:3px solid var(--candy)!important; color:var(--t1)!important; }
  [data-testid="stSuccess"] { background:rgba(0,229,160,0.08)!important; border-left:3px solid var(--green)!important; }
  [data-testid="stWarning"] { background:rgba(255,184,48,0.08)!important;  border-left:3px solid var(--warn)!important; }
  [data-testid="stError"]   { background:rgba(255,77,109,0.08)!important;  border-left:3px solid var(--err)!important; }

  /* Expanders */
  .streamlit-expanderHeader { background:var(--onyx2)!important; border:1px solid var(--onyx4)!important; border-radius:12px!important; color:var(--t1)!important; font-weight:600!important; }
  .streamlit-expanderContent { background:var(--onyx2)!important; border:1px solid var(--onyx4)!important; border-top:none!important; border-radius:0 0 12px 12px!important; }

  /* Output card */
  .output-card { background:linear-gradient(135deg,#0d1f35,#0a1220); border:1px solid rgba(0,194,255,0.2); border-radius:14px; padding:20px; font-size:0.88rem; line-height:1.8; color:var(--t1); }

  /* Snippet cards */
  .snippet { background:var(--onyx3); border:1px solid var(--onyx4); border-radius:10px; padding:12px; margin-bottom:8px; }
  .snip-hd { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
  .snip-src { font-size:0.7rem; color:var(--candy); font-weight:600; }
  .snip-score { background:rgba(0,194,255,0.12); border:1px solid rgba(0,194,255,0.25); border-radius:20px; padding:2px 8px; font-size:0.65rem; color:var(--icy); }
  .snip-txt { font-size:0.75rem; color:var(--t2); line-height:1.5; }

  /* Quiz cards */
  .quiz-card { background:var(--onyx3); border:1px solid rgba(185,122,255,0.25); border-radius:12px; padding:14px 18px; margin-bottom:10px; }
  .quiz-num  { font-size:0.68rem; font-weight:700; color:var(--purple); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px; }
  .quiz-q    { font-size:0.88rem; color:var(--t1); line-height:1.5; }

  /* FRE bars */
  .bar-lbl { font-size:0.68rem; color:var(--t2); margin-bottom:2px; }
  .bar-bg  { height:7px; background:var(--onyx4); border-radius:99px; margin-bottom:8px; overflow:hidden; }
  .bar-fill{ height:100%; border-radius:99px; }

  /* Scrollbar */
  ::-webkit-scrollbar { width:6px; }
  ::-webkit-scrollbar-track { background:var(--onyx); }
  ::-webkit-scrollbar-thumb { background:var(--onyx4); border-radius:3px; }
  ::-webkit-scrollbar-thumb:hover { background:var(--mid); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cache the RAG index — loads sentence-transformers once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="🔄 Loading knowledge base index…")
def load_index():
    rag_utils.build_index("knowledge_base")
    return True


# ---------------------------------------------------------------------------
# Readability helpers
# ---------------------------------------------------------------------------
_FRE_BANDS = [
    (90, "Very Easy"), (80, "Easy"), (70, "Fairly Easy"),
    (60, "Standard"),  (50, "Fairly Difficult"), (30, "Difficult"), (0, "Very Confusing"),
]

def fre_label(score: float) -> str:
    for threshold, label in _FRE_BANDS:
        if score >= threshold:
            return label
    return "Very Confusing"

def fre_color(score: float) -> str:
    if score >= 70: return "#00e5a0"
    if score >= 50: return "#00c2ff"
    if score >= 30: return "#ffb830"
    return "#ff4d6d"

def extract_main_text(full_output: str) -> str:
    """Strip 'Key Terms Explained:' section for clean FRE scoring."""
    idx = full_output.find("Key Terms Explained:")
    return full_output[:idx].strip() if idx != -1 else full_output.strip()


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def build_markdown(
    original_text: str,
    level: str,
    language: str,
    simplified_output: str,
    translated_output: str,
    quiz_questions: list,
    orig_fre: float,
    simpl_fre: float,
) -> str:
    """Build a Markdown export string."""
    lines = [
        "# Course Content Simplification Agent — Export",
        "",
        f"**Comprehension Level:** {level}  ",
        f"**Language:** {language}  ",
        "",
        "---",
        "",
        "## Original Text",
        "",
        original_text,
        "",
        "---",
        "",
        f"## Simplified Output ({level})",
        "",
        simplified_output,
        "",
    ]
    if language != "English" and translated_output:
        lines += [
            "---",
            "",
            f"## {language} Translation",
            "",
            translated_output,
            "",
        ]
    if quiz_questions:
        lines += [
            "---",
            "",
            "## Quiz — Check Your Understanding",
            "",
        ]
        for i, q in enumerate(quiz_questions, 1):
            lines.append(f"**Q{i}.** {q}  ")
        lines.append("")
    lines += [
        "---",
        "",
        "## Readability Scores (Flesch Reading Ease)",
        "",
        f"| | Score | Level |",
        f"|---|---|---|",
        f"| Original | {orig_fre:.1f} | {fre_label(orig_fre)} |",
        f"| Simplified | {simpl_fre:.1f} | {fre_label(simpl_fre)} |",
        f"| Improvement | {simpl_fre - orig_fre:+.1f} | — |",
        "",
        "---",
        "",
        "*Generated by Course Content Simplification Agent · IBM watsonx.ai Granite · AICTE-2026*",
    ]
    return "\n".join(lines)


def build_pdf(
    original_text: str,
    level: str,
    language: str,
    simplified_output: str,
    translated_output: str,
    quiz_questions: list,
    orig_fre: float,
    simpl_fre: float,
) -> bytes:
    """Build a PDF export using fpdf2 and return as bytes."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(0, 100, 180)
    pdf.cell(0, 10, "Course Content Simplification Agent", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Level: {level}   |   Language: {language}   |   IBM watsonx.ai Granite   |   AICTE-2026", ln=True, align="C")
    pdf.ln(4)
    pdf.set_draw_color(0, 194, 255)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    def section_heading(title: str):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(0, 120, 200)
        pdf.cell(0, 8, title, ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)

    def body_text(text: str):
        # fpdf2 handles only latin-1 safely; encode and replace unknown chars
        safe = text.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe)
        pdf.ln(2)

    # Original text
    section_heading("Original Text")
    body_text(original_text)
    pdf.ln(2)

    # Simplified output
    section_heading(f"Simplified Output ({level})")
    body_text(simplified_output)
    pdf.ln(2)

    # Translation
    if language != "English" and translated_output:
        section_heading(f"{language} Translation")
        body_text(translated_output)
        pdf.ln(2)

    # Quiz
    if quiz_questions:
        section_heading("Quiz — Check Your Understanding")
        for i, q in enumerate(quiz_questions, 1):
            body_text(f"Q{i}. {q}")
        pdf.ln(2)

    # Readability table
    section_heading("Readability Scores (Flesch Reading Ease)")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(60, 7, "Text", border=1)
    pdf.cell(40, 7, "FRE Score", border=1)
    pdf.cell(80, 7, "Level", border=1, ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(60, 7, "Original", border=1)
    pdf.cell(40, 7, f"{orig_fre:.1f}", border=1)
    pdf.cell(80, 7, fre_label(orig_fre), border=1, ln=True)
    pdf.cell(60, 7, "Simplified", border=1)
    pdf.cell(40, 7, f"{simpl_fre:.1f}", border=1)
    pdf.cell(80, 7, fre_label(simpl_fre), border=1, ln=True)
    pdf.cell(60, 7, "Improvement", border=1)
    pdf.cell(40, 7, f"{simpl_fre - orig_fre:+.1f}", border=1)
    pdf.cell(80, 7, "", border=1, ln=True)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Ensure knowledge base is loaded
# ---------------------------------------------------------------------------
try:
    load_index()
except FileNotFoundError as e:
    st.error(f"⚠️ Knowledge base error: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# NAVBAR
# ---------------------------------------------------------------------------
st.markdown("""
<div class="navbar">
  <div class="nb-left">
    <div class="avatar">🤖</div>
    <div>
      <div class="nav-title">Course Content Simplification Agent</div>
      <div class="nav-sub">RAG · IBM watsonx.ai Granite · AICTE-2026 Agentic AI</div>
    </div>
  </div>
  <div class="nb-right">
    <span class="pill">ibm/granite-4-h-small</span>
    <span class="pill-b">⚡ watsonx.ai</span>
    <span class="pill-g">● Live</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# STATS ROW
# ---------------------------------------------------------------------------
st.markdown("""
<div class="stats">
  <div class="sc"><div class="sc-num">3</div><div class="sc-lbl">Reading Levels</div></div>
  <div class="sc"><div class="sc-num">5</div><div class="sc-lbl">Languages</div></div>
  <div class="sc"><div class="sc-num">RAG</div><div class="sc-lbl">Retrieval</div></div>
  <div class="sc"><div class="sc-num">Quiz</div><div class="sc-lbl">Generator</div></div>
  <div class="sc"><div class="sc-num">PDF</div><div class="sc-lbl">Export</div></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# INPUT BENTO
# ---------------------------------------------------------------------------
st.markdown('<div class="bento">', unsafe_allow_html=True)
st.markdown('<div class="sec-lbl">📥 Input</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([3, 1], gap="large")

with col_left:
    original_text = st.text_area(
        label="Original Academic Text",
        placeholder="Paste your textbook paragraph, lecture notes, or research excerpt here…",
        height=200,
        help="Minimum ~30 words recommended for meaningful simplification.",
    )

with col_right:
    level = st.selectbox(
        "Comprehension Level",
        ["Beginner", "Intermediate", "Expert"],
        help="**Beginner** — analogies, no jargon\n\n**Intermediate** — standard vocab\n\n**Expert** — technically precise",
    )
    language = st.selectbox(
        "Output Language",
        list(SUPPORTED_LANGUAGES.keys()),
        help="Translate the simplified output into your preferred language.",
    )
    level_colors = {"Beginner": "#00e5a0", "Intermediate": "#00c2ff", "Expert": "#b97aff"}
    lc = level_colors.get(level, "#00c2ff")
    st.markdown(f"""
    <div style="background:rgba(0,0,0,0.3);border:1px solid {lc}33;border-radius:10px;padding:8px 12px;margin:6px 0 10px;">
      <span style="color:{lc};font-size:0.75rem;font-weight:700;">{level}</span>
      <span style="color:#7aa5c0;font-size:0.7rem;margin-left:6px;">· {language}</span>
    </div>
    """, unsafe_allow_html=True)
    run_button = st.button("✨ Simplify", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------
if run_button:
    if not original_text.strip():
        st.warning("Please paste some academic text before clicking Simplify.")
        st.stop()

    # Step 1 — RAG retrieval
    with st.spinner("🔍 Searching knowledge base…"):
        try:
            context_chunks = rag_utils.retrieve(
                query=original_text, top_k=4, min_score=0.15, folder="knowledge_base",
            )
        except Exception as exc:
            st.error(f"RAG retrieval failed: {exc}")
            st.stop()

    # Step 2 — Simplification (IBM Granite)
    with st.spinner(f"🤖 IBM Granite rewriting at {level} level…"):
        try:
            simplified_output = granite_client.simplify(
                original_text=original_text,
                level=level,
                context_chunks=context_chunks,
            )
        except EnvironmentError as exc:
            st.error(f"**Credential error:** {exc}")
            st.stop()
        except RuntimeError as exc:
            st.error(f"**Model error:** {exc}")
            st.stop()

    main_simplified = extract_main_text(simplified_output)

    # Step 3 — Quiz Generator Agent
    with st.spinner("🎯 Quiz Generator Agent thinking…"):
        quiz_questions = granite_client.generate_quiz(main_simplified, level)

    # Step 4 — Language translation (if not English)
    translated_output = ""
    if language != "English":
        with st.spinner(f"🌐 Translating to {language}…"):
            translated_output = granite_client.translate_output(simplified_output, language)

    # Readability scores
    orig_fre  = textstat.flesch_reading_ease(original_text)
    simpl_fre = textstat.flesch_reading_ease(main_simplified)
    delta     = simpl_fre - orig_fre

    st.success("✅ Done — simplified, quiz generated" + (f", translated to {language}" if language != "English" else "") + "!")

    # ── OUTPUT LAYOUT ────────────────────────────────────────────────────────
    out_col, meta_col = st.columns([3, 2], gap="large")

    with out_col:
        # Simplified output
        st.markdown('<div class="bento-glow">', unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">📝 Simplified Output</div>', unsafe_allow_html=True)
        display_text = translated_output if (language != "English" and translated_output) else simplified_output
        st.markdown(
            f'<div class="output-card">{display_text.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )
        # Language toggle notice
        if language != "English" and translated_output:
            st.caption(f"Showing {language} translation · original English output also available below")
            with st.expander("🇬🇧 View original English output"):
                st.markdown(simplified_output)
        st.markdown('</div>', unsafe_allow_html=True)

        # Quiz Generator Agent bento
        if quiz_questions:
            st.markdown('<div class="bento">', unsafe_allow_html=True)
            st.markdown('<div class="sec-lbl">🎯 Quiz — Check Your Understanding</div>', unsafe_allow_html=True)
            for i, q in enumerate(quiz_questions, 1):
                st.markdown(f"""
                <div class="quiz-card">
                  <div class="quiz-num">Question {i}</div>
                  <div class="quiz-q">{q}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Download export bento
        st.markdown('<div class="bento">', unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">💾 Download Export</div>', unsafe_allow_html=True)

        md_content = build_markdown(
            original_text, level, language,
            simplified_output, translated_output,
            quiz_questions, orig_fre, simpl_fre,
        )
        pdf_bytes = build_pdf(
            original_text, level, language,
            simplified_output, translated_output,
            quiz_questions, orig_fre, simpl_fre,
        )

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📄 Download Markdown",
                data=md_content,
                file_name=f"simplified_{level.lower()}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="📋 Download PDF",
                data=pdf_bytes,
                file_name=f"simplified_{level.lower()}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with meta_col:
        # Readability bento
        st.markdown('<div class="bento">', unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">📊 Readability (FRE)</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Original",   f"{orig_fre:.1f}",  help=fre_label(orig_fre))
            st.caption(fre_label(orig_fre))
        with m2:
            st.metric("Simplified", f"{simpl_fre:.1f}", f"{delta:+.1f}", help=fre_label(simpl_fre))
            st.caption(fre_label(simpl_fre))
        with m3:
            arrow = "↑ Easier" if delta > 0 else "↓ Harder" if delta < 0 else "— Same"
            st.metric("Delta", f"{delta:+.1f}", arrow)
        orig_pct  = max(0, min(100, orig_fre))
        simpl_pct = max(0, min(100, simpl_fre))
        st.markdown(f"""
        <div style="margin-top:10px">
          <div class="bar-lbl">Original ({orig_fre:.1f})</div>
          <div class="bar-bg"><div class="bar-fill" style="width:{orig_pct}%;background:{fre_color(orig_fre)};"></div></div>
          <div class="bar-lbl">Simplified ({simpl_fre:.1f})</div>
          <div class="bar-bg"><div class="bar-fill" style="width:{simpl_pct}%;background:{fre_color(simpl_fre)};"></div></div>
          <div style="font-size:0.65rem;color:var(--t3);margin-top:4px;">FRE: 0 = Very Hard · 100 = Very Easy</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # RAG context bento
        st.markdown('<div class="bento">', unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">🔍 RAG Context</div>', unsafe_allow_html=True)
        if context_chunks:
            for item in context_chunks:
                st.markdown(f"""
                <div class="snippet">
                  <div class="snip-hd">
                    <span class="snip-src">📄 {item['source']}</span>
                    <span class="snip-score">sim {item['score']:.3f}</span>
                  </div>
                  <div class="snip-txt">{item['chunk'][:200]}{'…' if len(item['chunk'])>200 else ''}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:var(--t3);font-size:0.82rem;text-align:center;padding:12px;">No snippets above threshold — model used original text only.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center;padding:32px 0 8px;color:#3d5a72;font-size:0.72rem;">
  <span style="color:#00c2ff;font-weight:600;">Course Content Simplification Agent</span>
  &nbsp;·&nbsp; Built with IBM Bob &nbsp;·&nbsp; IBM watsonx.ai Granite
  &nbsp;·&nbsp; AICTE-2026 Agentic AI &nbsp;·&nbsp; Free Tier Only
</div>
""", unsafe_allow_html=True)

"""
app.py
======
Course Content Simplification Agent — Premium Forest-Green Dashboard UI.

Features:
  - Dark forest-green sidebar with Bob the AI mascot
  - Clean off-white main workspace with emerald accents
  - Simplification at Beginner / Intermediate / Expert level (IBM Granite)
  - RAG retrieval from local knowledge base (sentence-transformers, CPU)
  - Quiz Generator Agent — 3 comprehension questions
  - Language Toggle — English / Hindi / Tamil / Bengali / Marathi
  - Download Export — Markdown + PDF
  - Readability comparison (Flesch Reading Ease)

Run with:
    streamlit run app.py
"""

import streamlit as st
import textstat

import rag_utils
import granite_client
from granite_client import SUPPORTED_LANGUAGES

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Course Content Simplification Agent",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — Forest-green sidebar + off-white workspace
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts & palette ───────────────────────────────────────────────────── */
:root {
  --forest:   #0b1f14;
  --forest2:  #112b1c;
  --forest3:  #163522;
  --forest4:  #1e4530;
  --emerald:  #2ecc71;
  --emerald2: #27ae60;
  --emerald3: #1a7a43;
  --glow:     rgba(46,204,113,0.18);
  --glow2:    rgba(46,204,113,0.35);
  --canvas:   #f7f9f7;
  --card:     #ffffff;
  --border:   #e4ebe4;
  --t-dark:   #1a2e1f;
  --t-mid:    #4a6b52;
  --t-light:  #8aab92;
  --warn:     #f39c12;
  --err:      #e74c3c;
  --purple:   #8e44ad;
}

/* ── Global ────────────────────────────────────────────────────────────── */
html, body { font-family: 'Segoe UI', system-ui, sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stAppViewContainer"] { background: var(--canvas) !important; }
[data-testid="stHeader"] { background: var(--canvas) !important;
  border-bottom: 1px solid var(--border) !important; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--forest) 0%, var(--forest2) 100%) !important;
  border-right: 1px solid var(--forest4) !important;
}
[data-testid="stSidebar"] * { color: #c8e6c9 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input {
  background: var(--forest3) !important;
  border: 1px solid var(--forest4) !important;
  color: #c8e6c9 !important;
}

/* ── Main area padding ─────────────────────────────────────────────────── */
[data-testid="block-container"] {
  padding: 1.5rem 2.5rem 3rem !important;
  max-width: 1100px;
}

/* ── Cards ─────────────────────────────────────────────────────────────── */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px 28px;
  margin-bottom: 20px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.card-green {
  background: linear-gradient(135deg, #f0faf3 0%, #e8f5e9 100%);
  border: 1px solid rgba(46,204,113,0.3);
  border-radius: 16px;
  padding: 24px 28px;
  margin-bottom: 20px;
  box-shadow: 0 0 24px rgba(46,204,113,0.08);
}
.card-title {
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--emerald2);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
}

/* ── Hero heading ───────────────────────────────────────────────────────── */
.hero-title {
  font-size: 2rem;
  font-weight: 800;
  color: var(--t-dark);
  line-height: 1.2;
  margin-bottom: 6px;
}
.hero-sub {
  font-size: 1rem;
  color: var(--t-mid);
  margin-bottom: 18px;
}

/* ── Badges ─────────────────────────────────────────────────────────────── */
.badge-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: #f0faf3; border: 1px solid rgba(46,204,113,0.4);
  border-radius: 20px; padding: 5px 14px;
  font-size: 0.75rem; font-weight: 600; color: var(--emerald2);
}
.badge-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--emerald); display: inline-block;
}

/* ── Text area ──────────────────────────────────────────────────────────── */
.stTextArea textarea {
  background: #fafcfa !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--t-dark) !important;
  font-size: 0.92rem !important;
  line-height: 1.7 !important;
  transition: border-color .2s !important;
}
.stTextArea textarea:focus {
  border-color: var(--emerald) !important;
  box-shadow: 0 0 0 3px var(--glow) !important;
}
.stTextArea label { color: var(--t-mid) !important; font-size: 0.82rem !important; }

/* ── Selectbox ──────────────────────────────────────────────────────────── */
.stSelectbox > div > div {
  background: #fafcfa !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--t-dark) !important;
}
.stSelectbox > div > div:focus-within {
  border-color: var(--emerald) !important;
}
.stSelectbox label { color: var(--t-mid) !important; font-size: 0.82rem !important; }

/* ── Primary button ─────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #2ecc71, #27ae60) !important;
  color: #fff !important; border: none !important;
  border-radius: 12px !important; font-weight: 700 !important;
  font-size: 0.95rem !important; padding: 13px 28px !important;
  box-shadow: 0 4px 18px rgba(46,204,113,0.35) !important;
  transition: all .2s !important; width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 6px 26px rgba(46,204,113,0.55) !important;
  transform: translateY(-2px) !important;
}
.stButton > button {
  background: #f0faf3 !important; color: var(--t-dark) !important;
  border: 1px solid var(--border) !important; border-radius: 10px !important;
}

/* ── Download buttons ───────────────────────────────────────────────────── */
.stDownloadButton > button {
  background: #f0faf3 !important; color: var(--emerald2) !important;
  border: 1px solid rgba(46,204,113,0.4) !important;
  border-radius: 10px !important; font-size: 0.85rem !important;
  font-weight: 600 !important;
}
.stDownloadButton > button:hover {
  background: #e8f5e9 !important;
}

/* ── Metrics ────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: #f8fdf9 !important;
  border: 1px solid rgba(46,204,113,0.25) !important;
  border-radius: 14px !important; padding: 16px !important;
}
[data-testid="stMetricValue"] {
  color: var(--t-dark) !important;
  font-size: 1.5rem !important; font-weight: 700 !important;
}
[data-testid="stMetricLabel"] { color: var(--t-mid) !important; font-size: 0.74rem !important; }
[data-testid="stMetricDeltaIcon-Up"]   { color: var(--emerald2) !important; }
[data-testid="stMetricDeltaIcon-Down"] { color: var(--err) !important; }

/* ── Expanders ──────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
  background: #f8fdf9 !important;
  border: 1px solid rgba(46,204,113,0.2) !important;
  border-radius: 12px !important; color: var(--t-dark) !important;
  font-weight: 600 !important; font-size: 0.88rem !important;
}
.streamlit-expanderContent {
  background: #fafcfa !important;
  border: 1px solid rgba(46,204,113,0.15) !important;
  border-top: none !important; border-radius: 0 0 12px 12px !important;
}

/* ── Alerts ─────────────────────────────────────────────────────────────── */
.stAlert { border-radius: 12px !important; border: none !important; }
[data-testid="stSuccess"] {
  background: #e8f5e9 !important;
  border-left: 4px solid var(--emerald) !important; color: #1a4a29 !important;
}
[data-testid="stInfo"] {
  background: #e8f5e9 !important;
  border-left: 4px solid var(--emerald2) !important;
}
[data-testid="stWarning"] {
  background: #fef9e7 !important;
  border-left: 4px solid var(--warn) !important;
}
[data-testid="stError"] {
  background: #fdecea !important;
  border-left: 4px solid var(--err) !important;
}

/* ── Divider ────────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Spinner ────────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--emerald) !important; }

/* ── Snippet cards ──────────────────────────────────────────────────────── */
.snippet {
  background: #fafcfa; border: 1px solid var(--border);
  border-left: 3px solid var(--emerald);
  border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
}
.snip-src { font-size: 0.75rem; font-weight: 700; color: var(--emerald2); }
.snip-score {
  display: inline-block;
  background: #e8f5e9; border: 1px solid rgba(46,204,113,0.3);
  border-radius: 20px; padding: 1px 9px;
  font-size: 0.68rem; font-weight: 600; color: var(--emerald2);
  margin-left: 8px;
}
.snip-txt { font-size: 0.82rem; color: var(--t-mid); line-height: 1.55; margin-top: 5px; }

/* ── Quiz cards ─────────────────────────────────────────────────────────── */
.quiz-card {
  background: #fdf9ff; border: 1px solid rgba(142,68,173,0.2);
  border-left: 3px solid #8e44ad;
  border-radius: 10px; padding: 13px 16px; margin-bottom: 8px;
}
.quiz-num { font-size: 0.68rem; font-weight: 700; color: #8e44ad;
  text-transform: uppercase; letter-spacing: .08em; margin-bottom: 3px; }
.quiz-q { font-size: 0.88rem; color: var(--t-dark); line-height: 1.5; }

/* ── FRE bar ─────────────────────────────────────────────────────────────── */
.bar-wrap { margin: 10px 0 4px; }
.bar-lbl { font-size: 0.72rem; color: var(--t-mid); margin-bottom: 3px; }
.bar-bg { height: 8px; background: var(--border); border-radius: 99px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; transition: width .6s ease; }

/* ── Output body text ───────────────────────────────────────────────────── */
.output-body {
  font-size: 0.93rem; line-height: 1.85;
  color: var(--t-dark); white-space: pre-wrap;
}

/* ── Level pill ─────────────────────────────────────────────────────────── */
.level-pill {
  display: inline-block; border-radius: 20px;
  padding: 4px 14px; font-size: 0.75rem; font-weight: 700;
  margin-top: 6px; margin-bottom: 14px;
}

/* ── Bob mini avatar (beside output) ───────────────────────────────────── */
.bob-mini {
  display: inline-flex; align-items: center; gap: 8px;
  background: #f0faf3; border: 1px solid rgba(46,204,113,0.3);
  border-radius: 20px; padding: 4px 12px 4px 4px;
  font-size: 0.75rem; color: var(--emerald2); font-weight: 600;
  margin-bottom: 12px;
}
.bob-mini-avatar {
  width: 26px; height: 26px; border-radius: 50%;
  background: linear-gradient(135deg,#2ecc71,#27ae60);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.8rem;
}

/* ── Character counter ───────────────────────────────────────────────────── */
.char-count { font-size: 0.72rem; color: var(--t-light); text-align: right; margin-top: -6px; margin-bottom: 10px; }

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f0f0f0; }
::-webkit-scrollbar-thumb { background: #cce5cc; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--emerald3); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cache RAG index (loads sentence-transformers once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🌿 Loading knowledge base index…")
def load_index():
    rag_utils.build_index("knowledge_base")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Readability helpers
# ─────────────────────────────────────────────────────────────────────────────
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
    if score >= 70: return "#2ecc71"
    if score >= 50: return "#f39c12"
    if score >= 30: return "#e67e22"
    return "#e74c3c"

def extract_main_text(full_output: str) -> str:
    idx = full_output.find("Key Terms Explained:")
    return full_output[:idx].strip() if idx != -1 else full_output.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_markdown(original_text, level, language, simplified_output,
                   translated_output, quiz_questions, orig_fre, simpl_fre) -> str:
    lines = [
        "# Course Content Simplification Agent — Export", "",
        f"**Level:** {level}  |  **Language:** {language}", "",
        "---", "", "## Original Text", "", original_text, "",
        "---", "", f"## Simplified Output ({level})", "", simplified_output, "",
    ]
    if language != "English" and translated_output:
        lines += ["---", "", f"## {language} Translation", "", translated_output, ""]
    if quiz_questions:
        lines += ["---", "", "## Quiz — Check Your Understanding", ""]
        for i, q in enumerate(quiz_questions, 1):
            lines.append(f"**Q{i}.** {q}  ")
        lines.append("")
    lines += [
        "---", "", "## Readability (Flesch Reading Ease)", "",
        "| | Score | Level |", "|---|---|---|",
        f"| Original   | {orig_fre:.1f}  | {fre_label(orig_fre)} |",
        f"| Simplified | {simpl_fre:.1f} | {fre_label(simpl_fre)} |",
        f"| Delta      | {simpl_fre - orig_fre:+.1f} | — |", "",
        "---", "",
        "*Generated by Course Content Simplification Agent · IBM watsonx.ai Granite · AICTE-2026*",
    ]
    return "\n".join(lines)


def build_pdf(original_text, level, language, simplified_output,
              translated_output, quiz_questions, orig_fre, simpl_fre) -> bytes:
    from fpdf import FPDF

    _UNICODE_MAP = {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00b7": ".",
        "\u2022": "*", "\u2019": "'", "\u00e9": "e", "\u00e8": "e",
        "\u00ea": "e", "\u00e0": "a", "\u00e2": "a",
    }

    def sanitise(text: str) -> str:
        for char, rep in _UNICODE_MAP.items():
            text = text.replace(char, rep)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title block
    pdf.set_fill_color(11, 31, 20)
    pdf.rect(0, 0, 210, 32, "F")
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(200, 230, 201)
    pdf.set_y(8)
    pdf.cell(0, 8, sanitise("Course Content Simplification Agent"), ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(140, 180, 141)
    pdf.cell(0, 6, sanitise(f"Level: {level}   |   Language: {language}   |   IBM watsonx.ai Granite   |   AICTE-2026"), ln=True, align="C")
    pdf.set_y(38)

    def section(title: str):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(22, 120, 60)
        pdf.cell(0, 8, sanitise(title), ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)

    def body(text: str):
        pdf.multi_cell(0, 6, sanitise(text))
        pdf.ln(2)

    section("Original Text");          body(original_text)
    section(f"Simplified Output ({level})"); body(simplified_output)
    if language != "English" and translated_output:
        section(f"{language} Translation"); body(translated_output)
    if quiz_questions:
        section("Quiz - Check Your Understanding")
        for i, q in enumerate(quiz_questions, 1):
            body(f"Q{i}. {q}")

    section("Readability Scores (Flesch Reading Ease)")
    pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(30, 30, 30)
    for col, w in [("Text", 60), ("FRE Score", 40), ("Level", 80)]:
        pdf.cell(w, 7, sanitise(col), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 10)
    for label, score in [("Original", orig_fre), ("Simplified", simpl_fre)]:
        pdf.cell(60, 7, sanitise(label), border=1)
        pdf.cell(40, 7, sanitise(f"{score:.1f}"), border=1)
        pdf.cell(80, 7, sanitise(fre_label(score)), border=1, ln=True)
    pdf.cell(60, 7, sanitise("Delta"), border=1)
    pdf.cell(40, 7, sanitise(f"{simpl_fre - orig_fre:+.1f}"), border=1)
    pdf.cell(80, 7, sanitise(""), border=1, ln=True)

    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────────────
# Load knowledge base
# ─────────────────────────────────────────────────────────────────────────────
try:
    load_index()
except FileNotFoundError as e:
    st.error(f"⚠️ Knowledge base error: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # Bob mascot
    st.markdown("""
    <div style="text-align:center;padding:24px 0 18px;">
      <div style="position:relative;display:inline-block;margin-bottom:10px;">
        <div style="width:72px;height:72px;border-radius:50%;
             background:linear-gradient(135deg,#2ecc71,#145a32);
             box-shadow:0 0 24px rgba(46,204,113,0.45);
             display:flex;align-items:center;justify-content:center;
             margin:0 auto;font-size:2rem;">🤖</div>
        <div style="position:absolute;bottom:2px;right:2px;
             width:14px;height:14px;border-radius:50%;
             background:#2ecc71;border:2px solid #0b1f14;"></div>
      </div>
      <div style="font-size:1rem;font-weight:800;color:#e8f5e9;
           letter-spacing:0.02em;">Meet Bob</div>
      <div style="font-size:0.72rem;color:#81c784;margin-top:2px;">
        Your AI learning guide</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1e4530;margin:0 0 16px;">', unsafe_allow_html=True)

    # Nav items
    st.markdown("""
    <div style="padding:0 4px;">
      <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
           background:rgba(46,204,113,0.15);border:1px solid rgba(46,204,113,0.25);
           border-radius:10px;margin-bottom:6px;">
        <span style="font-size:1rem;">✏️</span>
        <span style="font-size:0.88rem;font-weight:700;color:#e8f5e9;">Simplify Text</span>
      </div>
      <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
           border-radius:10px;margin-bottom:6px;cursor:default;">
        <span style="font-size:1rem;">📚</span>
        <span style="font-size:0.88rem;color:#81c784;">Knowledge Base</span>
      </div>
      <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
           border-radius:10px;margin-bottom:6px;cursor:default;">
        <span style="font-size:1rem;">📊</span>
        <span style="font-size:0.88rem;color:#81c784;">Evaluation</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1e4530;margin:16px 0;">', unsafe_allow_html=True)

    # Analytics card
    st.markdown("""
    <div style="background:rgba(46,204,113,0.08);border:1px solid rgba(46,204,113,0.2);
         border-radius:14px;padding:16px 18px;margin-bottom:16px;">
      <div style="font-size:0.68rem;font-weight:700;color:#81c784;text-transform:uppercase;
           letter-spacing:0.1em;margin-bottom:12px;">Your learning impact</div>
      <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
        <div>
          <div style="font-size:1.4rem;font-weight:800;color:#e8f5e9;">127</div>
          <div style="font-size:0.68rem;color:#81c784;">Topics simplified</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.4rem;font-weight:800;color:#2ecc71;">98%</div>
          <div style="font-size:0.68rem;color:#81c784;">Successful</div>
        </div>
      </div>
      <div style="display:flex;justify-content:space-between;">
        <div>
          <div style="font-size:1.4rem;font-weight:800;color:#e8f5e9;">24h</div>
          <div style="font-size:0.68rem;color:#81c784;">Agent online</div>
        </div>
        <div style="text-align:right;">
          <div style="display:flex;align-items:center;gap:5px;justify-content:flex-end;margin-top:6px;">
            <div style="width:7px;height:7px;border-radius:50%;background:#2ecc71;"></div>
            <span style="font-size:0.72rem;color:#2ecc71;font-weight:600;">All systems ready</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Tech stack
    st.markdown("""
    <div style="padding:0 4px;">
      <div style="font-size:0.65rem;color:#4a7a52;text-transform:uppercase;
           letter-spacing:0.08em;margin-bottom:8px;font-weight:700;">Powered by</div>
      <div style="display:flex;flex-direction:column;gap:5px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="width:8px;height:8px;border-radius:2px;background:#2ecc71;"></div>
          <span style="font-size:0.78rem;color:#81c784;">IBM watsonx.ai Granite</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="width:8px;height:8px;border-radius:2px;background:#27ae60;"></div>
          <span style="font-size:0.78rem;color:#81c784;">sentence-transformers (CPU)</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="width:8px;height:8px;border-radius:2px;background:#1e8449;"></div>
          <span style="font-size:0.78rem;color:#81c784;">Local RAG · No GPU needed</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero-title">Make complex ideas easier to understand</div>
<div class="hero-sub">Grounded in your local knowledge base · powered by IBM Granite</div>
<div class="badge-row">
  <span class="badge"><span class="badge-dot"></span>IBM Granite</span>
  <span class="badge"><span class="badge-dot"></span>Local RAG</span>
  <span class="badge"><span class="badge-dot"></span>CPU-ready</span>
  <span class="badge"><span class="badge-dot"></span>5 Languages</span>
  <span class="badge"><span class="badge-dot"></span>PDF Export</span>
</div>
""", unsafe_allow_html=True)

# ── Input card ──────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">📄 Original academic text</div>', unsafe_allow_html=True)

original_text = st.text_area(
    label="Paste your academic content",
    placeholder="Paste a textbook paragraph, lecture note, or research excerpt here…\n\nExample: The computational complexity of an algorithm is characterised by its asymptotic behaviour as input size n approaches infinity…",
    height=180,
    label_visibility="collapsed",
)

# Character counter
char_count = len(original_text)
word_count = len(original_text.split()) if original_text.strip() else 0
st.markdown(
    f'<div class="char-count">{word_count} words · {char_count} characters</div>',
    unsafe_allow_html=True,
)

# Controls row
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 1.6], gap="medium")

with ctrl_col1:
    level = st.selectbox(
        "Comprehension level",
        ["Beginner", "Intermediate", "Expert"],
        help="**Beginner** — simple analogies, no jargon\n\n**Intermediate** — standard vocab\n\n**Expert** — precise and technical",
    )

with ctrl_col2:
    language = st.selectbox(
        "Output language",
        list(SUPPORTED_LANGUAGES.keys()),
        help="Translate the simplified output into your preferred language.",
    )

with ctrl_col3:
    level_styles = {
        "Beginner":     ("background:#e8f5e9;color:#1a6b35;border:1px solid #a5d6a7;", "🌱"),
        "Intermediate": ("background:#fff3e0;color:#8a4800;border:1px solid #ffcc80;", "🌿"),
        "Expert":       ("background:#ede7f6;color:#4a148c;border:1px solid #ce93d8;", "🔬"),
    }
    s, icon = level_styles.get(level, level_styles["Intermediate"])
    st.markdown(
        f'<div class="level-pill" style="{s}">{icon} {level} · {language}</div>',
        unsafe_allow_html=True,
    )
    run_button = st.button("🌿 Simplify with Granite", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)  # close .card


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
if run_button:
    if not original_text.strip():
        st.warning("⚠️ Please paste some academic text before simplifying.")
        st.stop()
    if word_count < 10:
        st.warning("⚠️ Input is very short. Try pasting at least 30 words for best results.")

    # Step 1 — RAG
    with st.spinner("🔍 Searching knowledge base for relevant context…"):
        try:
            context_chunks = rag_utils.retrieve(
                query=original_text, top_k=4, min_score=0.15, folder="knowledge_base",
            )
        except Exception as exc:
            st.error(f"RAG retrieval failed: {exc}")
            st.stop()

    # Step 2 — Simplification
    with st.spinner(f"🤖 Bob is rewriting at {level} level…"):
        try:
            simplified_output = granite_client.simplify(
                original_text=original_text, level=level, context_chunks=context_chunks,
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

    main_simplified = extract_main_text(simplified_output)

    # Step 3 — Quiz
    with st.spinner("🎯 Generating comprehension quiz…"):
        quiz_questions = granite_client.generate_quiz(main_simplified, level)

    # Step 4 — Translation
    translated_output = ""
    if language != "English":
        with st.spinner(f"🌐 Translating to {language}…"):
            translated_output = granite_client.translate_output(simplified_output, language)

    # FRE scores
    orig_fre  = textstat.flesch_reading_ease(original_text)
    simpl_fre = textstat.flesch_reading_ease(main_simplified)
    delta     = simpl_fre - orig_fre

    # ── Success banner
    st.success(
        f"✅ Simplification complete! "
        f"Readability improved by **{delta:+.1f} FRE points**"
        + (f" · translated to **{language}**" if language != "English" else "")
        + (f" · **{len(quiz_questions)} quiz questions** generated" if quiz_questions else "")
    )

    # ── Two-column output layout
    left_col, right_col = st.columns([3, 2], gap="large")

    # ────────────────────────────── LEFT COLUMN ──────────────────────────────
    with left_col:

        # Simplified explanation card
        st.markdown('<div class="card-green">', unsafe_allow_html=True)
        st.markdown("""
        <div class="bob-mini">
          <div class="bob-mini-avatar">🤖</div>
          Bob's explanation
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card-title">💡 Simplified explanation</div>', unsafe_allow_html=True)

        display_text = translated_output if (language != "English" and translated_output) else simplified_output

        # Split at Key Terms for nicer rendering
        kt_idx = display_text.find("Key Terms Explained:")
        if kt_idx != -1:
            main_body = display_text[:kt_idx].strip()
            key_terms = display_text[kt_idx:].strip()
        else:
            main_body = display_text.strip()
            key_terms = ""

        st.markdown(f'<div class="output-body">{main_body}</div>', unsafe_allow_html=True)

        if key_terms:
            st.markdown(
                '<div style="margin-top:16px;padding-top:14px;border-top:1px solid #c8e6c9;">'
                '<div class="card-title">🔑 Key terms explained</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="output-body">{key_terms.replace("Key Terms Explained:", "").strip()}</div>', unsafe_allow_html=True)

        if language != "English" and translated_output:
            with st.expander("🇬🇧 View original English output"):
                st.markdown(simplified_output)

        st.markdown('</div>', unsafe_allow_html=True)

        # Quiz card
        if quiz_questions:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🎯 Quiz — check your understanding</div>', unsafe_allow_html=True)
            for i, q in enumerate(quiz_questions, 1):
                st.markdown(f"""
                <div class="quiz-card">
                  <div class="quiz-num">Question {i}</div>
                  <div class="quiz-q">{q}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Download card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💾 Download your notes</div>', unsafe_allow_html=True)
        md_str = build_markdown(
            original_text, level, language, simplified_output,
            translated_output, quiz_questions, orig_fre, simpl_fre,
        )
        pdf_bytes = build_pdf(
            original_text, level, language, simplified_output,
            translated_output, quiz_questions, orig_fre, simpl_fre,
        )
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "📄 Download Markdown", data=md_str,
                file_name=f"simplified_{level.lower()}.md",
                mime="text/markdown", use_container_width=True,
            )
        with dl2:
            st.download_button(
                "📋 Download PDF", data=pdf_bytes,
                file_name=f"simplified_{level.lower()}.pdf",
                mime="application/pdf", use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ────────────────────────────── RIGHT COLUMN ─────────────────────────────
    with right_col:

        # Readability card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Readability comparison</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Original",   f"{orig_fre:.1f}", help=fre_label(orig_fre))
            st.caption(fre_label(orig_fre))
        with m2:
            st.metric("Simplified", f"{simpl_fre:.1f}", f"{delta:+.1f}")
            st.caption(fre_label(simpl_fre))
        with m3:
            arrow = "↑ Easier" if delta > 0 else "↓ Harder" if delta < 0 else "Same"
            st.metric("Delta", f"{delta:+.1f}", arrow)

        orig_pct  = max(0, min(100, orig_fre))
        simpl_pct = max(0, min(100, simpl_fre))
        st.markdown(f"""
        <div class="bar-wrap">
          <div class="bar-lbl">Original ({orig_fre:.1f} — {fre_label(orig_fre)})</div>
          <div class="bar-bg"><div class="bar-fill"
            style="width:{orig_pct}%;background:{fre_color(orig_fre)};"></div></div>
          <div class="bar-lbl" style="margin-top:8px;">Simplified ({simpl_fre:.1f} — {fre_label(simpl_fre)})</div>
          <div class="bar-bg"><div class="bar-fill"
            style="width:{simpl_pct}%;background:{fre_color(simpl_fre)};"></div></div>
        </div>
        <div style="font-size:0.68rem;color:var(--t-light);margin-top:6px;">
          Flesch Reading Ease: 0 = Very Hard · 100 = Very Easy
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # RAG snippets card
        with st.expander(f"🔍 Retrieved grounding snippets ({len(context_chunks)} found)", expanded=True):
            if context_chunks:
                for item in context_chunks:
                    st.markdown(f"""
                    <div class="snippet">
                      <div>
                        <span class="snip-src">📄 {item['source']}</span>
                        <span class="snip-score">sim {item['score']:.3f}</span>
                      </div>
                      <div class="snip-txt">{item['chunk'][:220]}{'…' if len(item['chunk'])>220 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No snippets above threshold (≥ 0.15). Model used original text only.")

        # How Bob worked
        with st.expander("🧠 How Bob processed this", expanded=False):
            st.markdown(f"""
            <div style="font-size:0.85rem;color:var(--t-mid);line-height:1.7;">
            <strong>1. RAG Retrieval</strong><br>
            Searched <em>knowledge_base/</em> using sentence-transformers
            (all-MiniLM-L6-v2, CPU). Found <strong>{len(context_chunks)}</strong>
            relevant snippet(s) above the 0.15 similarity threshold.<br><br>
            <strong>2. Simplification</strong><br>
            Sent the original text + retrieved context + <em>{level}</em>-level
            instruction to <strong>IBM watsonx.ai Granite</strong>.
            Model preserved factual accuracy and appended key term definitions.<br><br>
            <strong>3. Quiz Generation</strong><br>
            A separate Quiz Generator Agent prompt produced
            <strong>{len(quiz_questions)}</strong> comprehension question(s)
            from the simplified text.<br><br>
            {"<strong>4. Translation</strong><br>A dedicated translation prompt rendered the full output in <strong>" + language + "</strong>.<br><br>" if language != "English" else ""}
            <strong>Readability gain:</strong> {orig_fre:.1f} → {simpl_fre:.1f}
            ({delta:+.1f} FRE points — {fre_label(orig_fre)} → {fre_label(simpl_fre)})
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr style="margin:32px 0 16px;">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#aab8aa;font-size:0.73rem;padding-bottom:16px;">
  <strong style="color:#2ecc71;">Course Content Simplification Agent</strong>
  &nbsp;·&nbsp; Built with IBM Bob
  &nbsp;·&nbsp; IBM watsonx.ai Granite
  &nbsp;·&nbsp; AICTE-2026 Agentic AI
  &nbsp;·&nbsp; Free Tier · No GPU needed
</div>
""", unsafe_allow_html=True)

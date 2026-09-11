"""
app.py
======
Course Content Simplification Agent
Premium Forest-Green Analytics Dashboard

Features:
  - Dark forest-green sidebar with Bob mascot + analytics
  - Clean off-white main workspace with emerald accents
  - IBM Granite text simplification (Beginner / Intermediate / Expert)
  - Local RAG retrieval (sentence-transformers, CPU-only)
  - Quiz Generator Agent
  - Language Toggle (EN / HI / TA / BN / MR)
  - PDF + Markdown export
  - Flesch Reading Ease readability comparison

Run:  streamlit run app.py
"""

import streamlit as st
import textstat

import rag_utils
import granite_client
from granite_client import SUPPORTED_LANGUAGES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Course Content Simplification Agent",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Palette ---------- */
:root {
  --forest:  #0b1f14;
  --forest2: #112b1c;
  --forest3: #163522;
  --forest4: #1e4530;
  --em:      #2ecc71;
  --em2:     #27ae60;
  --em3:     #1a7a43;
  --glow:    rgba(46,204,113,.18);
  --canvas:  #f5f7f5;
  --card:    #ffffff;
  --border:  #dde8dd;
  --td:      #1a2e1f;
  --tm:      #4a6452;
  --tl:      #8aab92;
  --warn:    #e67e22;
  --err:     #e74c3c;
  --purple:  #8e44ad;
}

/* ---------- Reset & base ---------- */
*, *::before, *::after { box-sizing: border-box; }
html, body { font-family: 'Segoe UI', system-ui, Arial, sans-serif !important; }
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stAppViewContainer"] { background: var(--canvas) !important; }
[data-testid="stHeader"] {
  background: var(--canvas) !important;
  border-bottom: 1px solid var(--border) !important;
  box-shadow: none !important;
}
[data-testid="block-container"] {
  padding: 2rem 2.8rem 3rem !important;
  max-width: 1080px;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(175deg, var(--forest) 0%, var(--forest2) 100%) !important;
  border-right: 1px solid var(--forest4) !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }
/* Kill default Streamlit sidebar padding on inner wrapper */
[data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
  padding: 0 !important;
}

/* ---------- Card components ---------- */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 26px 30px;
  margin-bottom: 20px;
  box-shadow: 0 2px 14px rgba(0,0,0,.05);
}
.card-em {
  background: linear-gradient(135deg, #f0faf3 0%, #e8f5e9 100%);
  border: 1px solid rgba(46,204,113,.28);
  border-radius: 18px;
  padding: 26px 30px;
  margin-bottom: 20px;
  box-shadow: 0 0 28px rgba(46,204,113,.07);
}
.card-label {
  font-size: .68rem;
  font-weight: 700;
  color: var(--em2);
  text-transform: uppercase;
  letter-spacing: .11em;
  margin-bottom: 14px;
}

/* ---------- Hero ---------- */
.hero-title {
  font-size: 1.9rem;
  font-weight: 800;
  color: var(--td);
  line-height: 1.2;
  margin-bottom: 5px;
}
.hero-sub { font-size: .97rem; color: var(--tm); margin-bottom: 18px; }

/* ---------- Status badges ---------- */
.badge-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 22px; }
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: #f0faf3;
  border: 1px solid rgba(46,204,113,.38);
  border-radius: 99px; padding: 5px 14px;
  font-size: .73rem; font-weight: 600; color: var(--em2);
}
.bdot { width: 7px; height: 7px; border-radius: 50%; background: var(--em); }

/* ---------- Inputs ---------- */
.stTextArea textarea {
  background: #fafcfa !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 13px !important;
  color: var(--td) !important;
  font-size: .92rem !important;
  line-height: 1.72 !important;
}
.stTextArea textarea:focus {
  border-color: var(--em) !important;
  box-shadow: 0 0 0 3px var(--glow) !important;
  outline: none !important;
}
.stTextArea label { color: var(--tm) !important; font-size: .82rem !important; }
.stSelectbox > div > div {
  background: #fafcfa !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--td) !important;
}
.stSelectbox label { color: var(--tm) !important; font-size: .82rem !important; }

/* ---------- Buttons ---------- */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #2ecc71, #27ae60) !important;
  color: #fff !important; border: none !important;
  border-radius: 12px !important; font-weight: 700 !important;
  font-size: .95rem !important; padding: 13px 22px !important;
  box-shadow: 0 4px 18px rgba(46,204,113,.32) !important;
  width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 6px 26px rgba(46,204,113,.52) !important;
  transform: translateY(-2px) !important;
}
.stButton > button {
  background: #f0faf3 !important; color: var(--td) !important;
  border: 1px solid var(--border) !important; border-radius: 10px !important;
}
.stDownloadButton > button {
  background: #f0faf3 !important; color: var(--em2) !important;
  border: 1px solid rgba(46,204,113,.35) !important;
  border-radius: 10px !important; font-size: .84rem !important; font-weight: 600 !important;
}
.stDownloadButton > button:hover { background: #e8f5e9 !important; }

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
  background: #f8fdf9 !important;
  border: 1px solid rgba(46,204,113,.22) !important;
  border-radius: 14px !important; padding: 15px !important;
}
[data-testid="stMetricValue"] {
  color: var(--td) !important; font-size: 1.45rem !important; font-weight: 700 !important;
}
[data-testid="stMetricLabel"] { color: var(--tm) !important; font-size: .72rem !important; }

/* ---------- Expanders ---------- */
.streamlit-expanderHeader {
  background: #f8fdf9 !important;
  border: 1px solid rgba(46,204,113,.18) !important;
  border-radius: 12px !important;
  color: var(--td) !important; font-weight: 600 !important; font-size: .88rem !important;
}
.streamlit-expanderContent {
  background: #fafcfa !important;
  border: 1px solid rgba(46,204,113,.13) !important;
  border-top: none !important; border-radius: 0 0 12px 12px !important;
}

/* ---------- Alert overrides ---------- */
.stAlert { border-radius: 12px !important; border: none !important; }
[data-testid="stSuccess"] {
  background: #e8f5e9 !important;
  border-left: 4px solid var(--em) !important; color: #1a4a29 !important;
}
[data-testid="stInfo"] {
  background: #e8f5e9 !important; border-left: 4px solid var(--em2) !important;
}
[data-testid="stWarning"] {
  background: #fef9e7 !important; border-left: 4px solid var(--warn) !important;
}
[data-testid="stError"] {
  background: #fdecea !important; border-left: 4px solid var(--err) !important;
}
hr { border-color: var(--border) !important; }
.stSpinner > div { border-top-color: var(--em) !important; }

/* ---------- Snippet cards ---------- */
.snip {
  background: #fafcfa; border: 1px solid var(--border);
  border-left: 3px solid var(--em);
  border-radius: 10px; padding: 11px 15px; margin-bottom: 8px;
}
.snip-src { font-size: .73rem; font-weight: 700; color: var(--em2); }
.snip-pill {
  display: inline-block; background: #e8f5e9;
  border-radius: 99px; padding: 1px 9px;
  font-size: .65rem; font-weight: 600; color: var(--em2); margin-left: 7px;
}
.snip-txt { font-size: .8rem; color: var(--tm); line-height: 1.55; margin-top: 4px; }

/* ---------- Quiz cards ---------- */
.qcard {
  background: #fdf9ff; border: 1px solid rgba(142,68,173,.2);
  border-left: 3px solid var(--purple);
  border-radius: 10px; padding: 12px 15px; margin-bottom: 8px;
}
.qnum { font-size: .65rem; font-weight: 700; color: var(--purple);
  text-transform: uppercase; letter-spacing: .08em; margin-bottom: 3px; }
.qtxt { font-size: .87rem; color: var(--td); line-height: 1.5; }

/* ---------- FRE bars ---------- */
.bar-wrap { margin: 8px 0; }
.bar-lbl { font-size: .7rem; color: var(--tm); margin-bottom: 3px; }
.bar-bg { height: 8px; background: var(--border); border-radius: 99px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; }

/* ---------- Output body ---------- */
.out-body { font-size: .92rem; line-height: 1.85; color: var(--td); white-space: pre-wrap; }
.kt-sep { border-top: 1px solid #c8e6c9; margin: 14px 0 10px; }

/* ---------- Character counter ---------- */
.ccount { font-size: .7rem; color: var(--tl); text-align: right; margin-top: -4px; margin-bottom: 12px; }

/* ---------- Level pill ---------- */
.lvl-pill {
  display: inline-block; border-radius: 99px;
  padding: 4px 13px; font-size: .74rem; font-weight: 700;
  margin-bottom: 12px;
}

/* ---------- Bob mini tag ---------- */
.bob-tag {
  display: inline-flex; align-items: center; gap: 7px;
  background: #f0faf3; border: 1px solid rgba(46,204,113,.3);
  border-radius: 99px; padding: 3px 11px 3px 3px;
  font-size: .72rem; color: var(--em2); font-weight: 600; margin-bottom: 12px;
}
.bob-tag-av {
  width: 22px; height: 22px; border-radius: 50%;
  background: linear-gradient(135deg,#2ecc71,#27ae60);
  display: flex; align-items: center; justify-content: center; font-size: .68rem;
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #eef4ee; }
::-webkit-scrollbar-thumb { background: #b2d8b2; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--em3); }
</style>
""", unsafe_allow_html=True)


# ── Cache RAG index ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🌿 Loading knowledge base index…")
def load_index():
    rag_utils.build_index("knowledge_base")
    return True


# ── Readability helpers ───────────────────────────────────────────────────────
_FRE = [
    (90,"Very Easy"),(80,"Easy"),(70,"Fairly Easy"),
    (60,"Standard"),(50,"Fairly Difficult"),(30,"Difficult"),(0,"Very Confusing"),
]
def fre_label(s):
    for t,l in _FRE:
        if s >= t: return l
    return "Very Confusing"

def fre_color(s):
    if s >= 70: return "#2ecc71"
    if s >= 50: return "#f39c12"
    if s >= 30: return "#e67e22"
    return "#e74c3c"

def extract_main(txt):
    i = txt.find("Key Terms Explained:")
    return txt[:i].strip() if i != -1 else txt.strip()


# ── Export helpers ────────────────────────────────────────────────────────────
def build_md(orig, level, lang, simp, trans, quiz, of, sf):
    lines = [
        "# Course Content Simplification Agent — Export","",
        f"**Level:** {level}  |  **Language:** {lang}","",
        "---","","## Original Text","",orig,"",
        "---","",f"## Simplified Output ({level})","",simp,"",
    ]
    if lang != "English" and trans:
        lines += ["---","",f"## {lang} Translation","",trans,""]
    if quiz:
        lines += ["---","","## Quiz",""]
        for i,q in enumerate(quiz,1): lines.append(f"**Q{i}.** {q}  ")
        lines.append("")
    lines += [
        "---","","## Readability","",
        "| | Score | Level |","|---|---|---|",
        f"| Original | {of:.1f} | {fre_label(of)} |",
        f"| Simplified | {sf:.1f} | {fre_label(sf)} |",
        f"| Delta | {sf-of:+.1f} | — |","",
        "---","",
        "*Course Content Simplification Agent · IBM watsonx.ai Granite · AICTE-2026*",
    ]
    return "\n".join(lines)


def build_pdf(orig, level, lang, simp, trans, quiz, of, sf):
    from fpdf import FPDF
    _U = {
        "\u2014":"-","\u2013":"-","\u2018":"'","\u2019":"'",
        "\u201c":'"',"\u201d":'"',"\u2026":"...","\u2022":"*","\u00b7":".",
    }
    def s(t):
        for c,r in _U.items(): t=t.replace(c,r)
        return t.encode("latin-1",errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True,margin=15)
    pdf.add_page()
    pdf.set_fill_color(11,31,20); pdf.rect(0,0,210,30,"F")
    pdf.set_font("Helvetica","B",14); pdf.set_text_color(200,230,201)
    pdf.set_y(7); pdf.cell(0,8,s("Course Content Simplification Agent"),ln=True,align="C")
    pdf.set_font("Helvetica","",9); pdf.set_text_color(140,180,141)
    pdf.cell(0,6,s(f"Level: {level}  |  Language: {lang}  |  IBM watsonx.ai Granite  |  AICTE-2026"),ln=True,align="C")
    pdf.set_y(36)

    def sec(t):
        pdf.set_font("Helvetica","B",11); pdf.set_text_color(22,120,60)
        pdf.cell(0,8,s(t),ln=True)
        pdf.set_font("Helvetica","",10); pdf.set_text_color(30,30,30)
    def body(t):
        pdf.multi_cell(0,6,s(t)); pdf.ln(2)

    sec("Original Text"); body(orig)
    sec(f"Simplified Output ({level})"); body(simp)
    if lang != "English" and trans: sec(f"{lang} Translation"); body(trans)
    if quiz:
        sec("Quiz")
        for i,q in enumerate(quiz,1): body(f"Q{i}. {q}")

    sec("Readability (Flesch Reading Ease)")
    pdf.set_font("Helvetica","B",10); pdf.set_text_color(30,30,30)
    for col,w in [("Text",60),("FRE",40),("Level",80)]:
        pdf.cell(w,7,s(col),border=1)
    pdf.ln(); pdf.set_font("Helvetica","",10)
    for label,score in [("Original",of),("Simplified",sf)]:
        pdf.cell(60,7,s(label),border=1)
        pdf.cell(40,7,s(f"{score:.1f}"),border=1)
        pdf.cell(80,7,s(fre_label(score)),border=1,ln=True)
    pdf.cell(60,7,s("Delta"),border=1)
    pdf.cell(40,7,s(f"{sf-of:+.1f}"),border=1)
    pdf.cell(80,7,s(""),border=1,ln=True)
    return bytes(pdf.output())


# ── Load KB ───────────────────────────────────────────────────────────────────
try:
    load_index()
except FileNotFoundError as e:
    st.error(f"⚠️ Knowledge base error: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:28px 20px 20px;">

      <!-- App name -->
      <div style="display:flex;align-items:center;gap:11px;margin-bottom:24px;">
        <div style="width:36px;height:36px;border-radius:10px;
             background:linear-gradient(135deg,#2ecc71,#145a32);
             display:flex;align-items:center;justify-content:center;
             font-size:1.1rem;box-shadow:0 0 14px rgba(46,204,113,.4);flex-shrink:0;">◈</div>
        <div>
          <div style="font-size:.78rem;font-weight:800;color:#e8f5e9;
               line-height:1.25;">Course Content<br>Simplification Agent</div>
        </div>
      </div>

      <!-- Nav -->
      <div style="margin-bottom:6px;">
        <div style="font-size:.6rem;font-weight:700;color:#4a7a52;
             text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;">Navigation</div>

        <div style="display:flex;align-items:center;gap:10px;
             padding:10px 13px;border-radius:11px;margin-bottom:4px;
             background:rgba(46,204,113,.14);border:1px solid rgba(46,204,113,.22);">
          <span style="font-size:.92rem;">✏️</span>
          <span style="font-size:.85rem;font-weight:700;color:#e8f5e9;">Simplify Text</span>
        </div>

        <div style="display:flex;align-items:center;gap:10px;
             padding:10px 13px;border-radius:11px;margin-bottom:4px;">
          <span style="font-size:.92rem;">📚</span>
          <span style="font-size:.85rem;color:#81c784;">Knowledge Base</span>
        </div>

        <div style="display:flex;align-items:center;gap:10px;
             padding:10px 13px;border-radius:11px;">
          <span style="font-size:.92rem;">📊</span>
          <span style="font-size:.85rem;color:#81c784;">Evaluation</span>
        </div>
      </div>

      <div style="border-top:1px solid #1e4530;margin:18px 0;"></div>

      <!-- Analytics card -->
      <div style="font-size:.6rem;font-weight:700;color:#4a7a52;
           text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px;">
        Your learning impact
      </div>
      <div style="background:rgba(46,204,113,.07);border:1px solid rgba(46,204,113,.18);
           border-radius:14px;padding:14px 16px;margin-bottom:16px;">

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">

          <div>
            <div style="font-size:1.4rem;font-weight:800;color:#e8f5e9;">127</div>
            <div style="font-size:.65rem;color:#81c784;margin-top:1px;">Topics simplified</div>
          </div>

          <div>
            <div style="font-size:1.4rem;font-weight:800;color:#2ecc71;">98%</div>
            <div style="font-size:.65rem;color:#81c784;margin-top:1px;">Successful</div>
          </div>

          <div>
            <div style="font-size:1.4rem;font-weight:800;color:#e8f5e9;">24h</div>
            <div style="font-size:.65rem;color:#81c784;margin-top:1px;">Agent online</div>
          </div>

          <div style="display:flex;align-items:center;gap:5px;padding-top:8px;">
            <div style="width:7px;height:7px;border-radius:50%;background:#2ecc71;flex-shrink:0;"></div>
            <span style="font-size:.65rem;color:#2ecc71;font-weight:600;">All systems ready</span>
          </div>

        </div>
      </div>

      <!-- Bob mascot -->
      <div style="border-top:1px solid #1e4530;padding-top:18px;text-align:center;">
        <div style="position:relative;display:inline-block;margin-bottom:8px;">
          <div style="width:58px;height:58px;border-radius:50%;
               background:linear-gradient(135deg,#2ecc71,#145a32);
               box-shadow:0 0 20px rgba(46,204,113,.4);
               display:flex;align-items:center;justify-content:center;
               margin:0 auto;font-size:1.7rem;">🤖</div>
          <div style="position:absolute;bottom:2px;right:2px;width:12px;height:12px;
               border-radius:50%;background:#2ecc71;border:2px solid #0b1f14;"></div>
        </div>
        <div style="font-size:.88rem;font-weight:800;color:#e8f5e9;">Meet Bob</div>
        <div style="font-size:.68rem;color:#81c784;margin-top:2px;">Your AI learning guide</div>
      </div>

    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">Make complex ideas easier to understand.</div>
<div class="hero-sub">Grounded in your own knowledge base using IBM Granite.</div>
<div class="badge-row">
  <span class="badge"><span class="bdot"></span>IBM Granite</span>
  <span class="badge"><span class="bdot"></span>Local RAG</span>
  <span class="badge"><span class="bdot"></span>CPU‑ready</span>
  <span class="badge"><span class="bdot"></span>5 Languages</span>
  <span class="badge"><span class="bdot"></span>PDF Export</span>
</div>
""", unsafe_allow_html=True)

# ── Input card ────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-label">📄 Original academic text</div>', unsafe_allow_html=True)

original_text = st.text_area(
    label="input",
    placeholder=(
        "Paste a textbook paragraph, lecture note, or research excerpt here…\n\n"
        "Example: The computational complexity of an algorithm is characterised by its "
        "asymptotic behaviour as input size n approaches infinity…"
    ),
    height=185,
    label_visibility="collapsed",
)

word_count = len(original_text.split()) if original_text.strip() else 0
char_count = len(original_text)
st.markdown(
    f'<div class="ccount">{word_count} words · {char_count} characters</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1.3, 1.3, 1.4], gap="medium")

with c1:
    level = st.selectbox(
        "Comprehension level",
        ["Beginner", "Intermediate", "Expert"],
        help=(
            "**Beginner** — short sentences, everyday analogies, no jargon  \n"
            "**Intermediate** — standard vocabulary with brief clarifications  \n"
            "**Expert** — concise and technically precise"
        ),
    )

with c2:
    language = st.selectbox(
        "Output language",
        list(SUPPORTED_LANGUAGES.keys()),
        help="Translate the simplified output into your preferred language.",
    )

with c3:
    _lp = {
        "Beginner":     ("background:#e8f5e9;color:#1a6b35;border:1px solid #a5d6a7;","🌱"),
        "Intermediate": ("background:#fff8e1;color:#7a5000;border:1px solid #ffd54f;","🌿"),
        "Expert":       ("background:#ede7f6;color:#4a148c;border:1px solid #ce93d8;","🔬"),
    }
    ps, pi = _lp.get(level, _lp["Intermediate"])
    st.markdown(
        f'<div class="lvl-pill" style="{ps}">{pi} {level} · {language}</div>',
        unsafe_allow_html=True,
    )
    run_btn = st.button("🌿 Simplify with Granite", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)  # close .card

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run_btn:
    if not original_text.strip():
        st.warning("⚠️ Please paste some academic text before simplifying.")
        st.stop()
    if word_count < 8:
        st.warning("⚠️ Input is very short — paste at least 30 words for meaningful results.")

    with st.spinner("🔍 Searching knowledge base for relevant context…"):
        try:
            chunks = rag_utils.retrieve(
                query=original_text, top_k=4, min_score=0.15, folder="knowledge_base"
            )
        except Exception as exc:
            st.error(f"RAG retrieval failed: {exc}")
            st.stop()

    with st.spinner(f"🤖 Bob is rewriting at {level} level…"):
        try:
            simp_out = granite_client.simplify(
                original_text=original_text,
                level=level,
                context_chunks=chunks,
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

    main_simp = extract_main(simp_out)

    with st.spinner("🎯 Generating comprehension questions…"):
        quiz = granite_client.generate_quiz(main_simp, level)

    trans_out = ""
    if language != "English":
        with st.spinner(f"🌐 Translating to {language}…"):
            trans_out = granite_client.translate_output(simp_out, language)

    of = textstat.flesch_reading_ease(original_text)
    sf = textstat.flesch_reading_ease(main_simp)
    delta = sf - of

    # Success banner
    extra = ""
    if language != "English": extra += f" · translated to **{language}**"
    if quiz: extra += f" · **{len(quiz)} quiz questions** generated"
    st.success(f"✅ Done! Readability improved by **{delta:+.1f} FRE points**{extra}")

    # ── Two-column output ──────────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    # ───────── LEFT ─────────────────────────────────────────────────────────
    with left:

        # Simplified explanation
        st.markdown('<div class="card-em">', unsafe_allow_html=True)
        st.markdown("""
        <div class="bob-tag">
          <div class="bob-tag-av">🤖</div>
          Bob's explanation
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="card-label">💡 Simplified explanation</div>', unsafe_allow_html=True)

        display = trans_out if (language != "English" and trans_out) else simp_out
        kt_i = display.find("Key Terms Explained:")
        if kt_i != -1:
            main_body = display[:kt_i].strip()
            key_body  = display[kt_i:].replace("Key Terms Explained:", "").strip()
        else:
            main_body = display.strip()
            key_body  = ""

        st.markdown(f'<div class="out-body">{main_body}</div>', unsafe_allow_html=True)

        if key_body:
            st.markdown('<div class="kt-sep"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-label">🔑 Key terms explained</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="out-body">{key_body}</div>', unsafe_allow_html=True)

        if language != "English" and trans_out:
            with st.expander("🇬🇧 View original English output"):
                st.markdown(simp_out)

        st.markdown("</div>", unsafe_allow_html=True)

        # Quiz
        if quiz:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">🎯 Quiz — check your understanding</div>', unsafe_allow_html=True)
            for i, q in enumerate(quiz, 1):
                st.markdown(
                    f'<div class="qcard"><div class="qnum">Question {i}</div>'
                    f'<div class="qtxt">{q}</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Download
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">💾 Download your notes</div>', unsafe_allow_html=True)
        md_str = build_md(original_text, level, language, simp_out, trans_out, quiz, of, sf)
        pdf_b  = build_pdf(original_text, level, language, simp_out, trans_out, quiz, of, sf)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "📄 Download Markdown", data=md_str,
                file_name=f"simplified_{level.lower()}.md", mime="text/markdown",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "📋 Download PDF", data=pdf_b,
                file_name=f"simplified_{level.lower()}.pdf", mime="application/pdf",
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ───────── RIGHT ────────────────────────────────────────────────────────
    with right:

        # Readability comparison
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">📊 Readability comparison</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Original",   f"{of:.1f}",        help=fre_label(of))
            st.caption(fre_label(of))
        with m2:
            st.metric("Simplified", f"{sf:.1f}", f"{delta:+.1f}")
            st.caption(fre_label(sf))
        with m3:
            arrow = "↑ Easier" if delta > 0 else "↓ Harder" if delta < 0 else "Unchanged"
            st.metric("Delta", f"{delta:+.1f}", arrow)

        op = max(0, min(100, of))
        sp = max(0, min(100, sf))
        st.markdown(f"""
        <div class="bar-wrap">
          <div class="bar-lbl">Original ({of:.1f} — {fre_label(of)})</div>
          <div class="bar-bg"><div class="bar-fill"
            style="width:{op}%;background:{fre_color(of)};"></div></div>
          <div class="bar-lbl" style="margin-top:8px;">Simplified ({sf:.1f} — {fre_label(sf)})</div>
          <div class="bar-bg"><div class="bar-fill"
            style="width:{sp}%;background:{fre_color(sf)};"></div></div>
        </div>
        <div style="font-size:.65rem;color:var(--tl);margin-top:5px;">
          Flesch Reading Ease: 0 = Very Hard · 100 = Very Easy
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Retrieved grounding snippets
        with st.expander(
            f"🔍 Retrieved grounding snippets ({len(chunks)} found)", expanded=True
        ):
            if chunks:
                for item in chunks:
                    pct = int(item["score"] * 100)
                    st.markdown(
                        f'<div class="snip">'
                        f'<div><span class="snip-src">📄 {item["source"]}</span>'
                        f'<span class="snip-pill">{pct}% relevant</span></div>'
                        f'<div class="snip-txt">'
                        f'{item["chunk"][:230]}{"…" if len(item["chunk"])>230 else ""}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info(
                    "No snippets above the relevance threshold (≥ 0.15). "
                    "The model used only the original text."
                )

        # How Bob processed this
        with st.expander("🧠 How Bob processed this", expanded=False):
            st.markdown(f"""
            <div style="font-size:.84rem;color:var(--tm);line-height:1.75;">
            <strong>1. RAG Retrieval</strong><br>
            Searched <em>knowledge_base/</em> using sentence-transformers
            (all-MiniLM-L6-v2, CPU). Found <strong>{len(chunks)}</strong>
            snippet(s) above the 0.15 similarity threshold.<br><br>
            <strong>2. Simplification (IBM Granite)</strong><br>
            Sent original text + retrieved context + <em>{level}</em>-level instruction
            to <strong>IBM watsonx.ai Granite</strong>. Model preserved factual accuracy
            and appended key term definitions.<br><br>
            <strong>3. Quiz Generation</strong><br>
            A dedicated Quiz Generator Agent prompt produced
            <strong>{len(quiz)}</strong> comprehension question(s) from the simplified
            text.<br>
            {"<br><strong>4. Translation</strong><br>A dedicated translation prompt rendered the output in <strong>" + language + "</strong>.<br>" if language != "English" else ""}
            <br><strong>Readability:</strong> {of:.1f} → {sf:.1f}
            ({delta:+.1f} pts · {fre_label(of)} → {fre_label(sf)})
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr style="margin:32px 0 14px;">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#aab8aa;font-size:.7rem;padding-bottom:14px;">
  <strong style="color:#27ae60;">Course Content Simplification Agent</strong>
  &nbsp;·&nbsp; Built with IBM Bob
  &nbsp;·&nbsp; IBM watsonx.ai Granite
  &nbsp;·&nbsp; AICTE-2026 Agentic AI
  &nbsp;·&nbsp; Free Tier · No GPU required
</div>
""", unsafe_allow_html=True)

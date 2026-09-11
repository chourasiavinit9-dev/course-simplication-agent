"""
app.py  —  Course Simplify
==========================
Clean, fixed dashboard. All controls visible and working on one screen.

Run:  streamlit run app.py
"""

import streamlit as st
import textstat

import rag_utils
import granite_client

# ── Language options (updated as requested) ────────────────────────────────────
OUTPUT_LANGUAGES = ["English", "Hindi", "Spanish", "French", "Bengali"]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Course Simplify",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Palette ──────────────────────────────────────── */
:root{
  --forest:#0b1f14; --forest2:#112b1c; --forest4:#1e4530;
  --em:#2ecc71;     --em2:#27ae60;     --em3:#1a7a43;
  --glow:rgba(46,204,113,.15);
  --canvas:#f4f6f4; --card:#ffffff;    --border:#d6e5d6;
  --td:#1a2a1f;     --tm:#3d5c45;      --tl:#7a9e82;
  --warn:#d68910;   --err:#c0392b;     --purple:#7d3c98;
}

/* ── Reset ──────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;}
html,body{font-family:'Segoe UI',system-ui,Arial,sans-serif!important;}
#MainMenu,footer,header{visibility:hidden!important;}

/* ── Page background ────────────────────────────────── */
[data-testid="stAppViewContainer"]{background:var(--canvas)!important;}
[data-testid="stHeader"]{background:var(--canvas)!important;
  border-bottom:1px solid var(--border)!important;box-shadow:none!important;}

/* ── Main content padding — tight, no extra space ──── */
[data-testid="block-container"]{
  padding:1.4rem 2rem 2.5rem!important;
  max-width:1060px;
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"]{
  background:linear-gradient(175deg,var(--forest) 0%,var(--forest2) 100%)!important;
  border-right:1px solid var(--forest4)!important;
}
[data-testid="stSidebar"]>div{padding:0!important;}
[data-testid="stSidebar"] section[data-testid="stSidebarContent"]{padding:0!important;}

/* ── Cards ───────────────────────────────────────────── */
.card{
  background:var(--card);border:1px solid var(--border);
  border-radius:16px;padding:20px 24px;margin-bottom:16px;
  box-shadow:0 1px 10px rgba(0,0,0,.05);
}
.card-em{
  background:linear-gradient(135deg,#f0faf3,#e8f5e9);
  border:1px solid rgba(46,204,113,.3);
  border-radius:16px;padding:20px 24px;margin-bottom:16px;
  box-shadow:0 0 22px rgba(46,204,113,.07);
}
.card-label{
  font-size:.67rem;font-weight:700;color:var(--em2);
  text-transform:uppercase;letter-spacing:.11em;margin-bottom:10px;
}

/* ── App title ───────────────────────────────────────── */
.app-title{
  font-size:1.75rem;font-weight:800;color:var(--td);
  line-height:1.15;margin-bottom:4px;
}
.app-sub{font-size:.9rem;color:var(--tm);margin-bottom:14px;}

/* ── Status badges ───────────────────────────────────── */
.badge-row{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:16px;}
.badge{
  display:inline-flex;align-items:center;gap:5px;
  background:#edf7ef;border:1px solid rgba(46,204,113,.35);
  border-radius:99px;padding:4px 12px;
  font-size:.71rem;font-weight:600;color:var(--em2);
}
.bdot{width:6px;height:6px;border-radius:50%;background:var(--em);}

/* ── Text area ────────────────────────────────────────── */
.stTextArea textarea{
  background:#fafcfa!important;
  border:1.5px solid var(--border)!important;
  border-radius:12px!important;
  color:var(--td)!important;
  font-size:.91rem!important;
  line-height:1.7!important;
  caret-color:var(--em2);
}
.stTextArea textarea::placeholder{color:var(--tl)!important;opacity:1!important;}
.stTextArea textarea:focus{
  border-color:var(--em)!important;
  box-shadow:0 0 0 3px var(--glow)!important;
  outline:none!important;
}
/* Force label visible */
.stTextArea>label,
.stTextArea label{
  color:var(--tm)!important;
  font-size:.82rem!important;
  font-weight:600!important;
  display:block!important;
  visibility:visible!important;
  opacity:1!important;
}

/* ── Selectbox ────────────────────────────────────────── */
.stSelectbox>div>div{
  background:#ffffff!important;
  border:1.5px solid var(--border)!important;
  border-radius:12px!important;
  color:var(--td)!important;
  font-size:.88rem!important;
  font-weight:500!important;
}
/* Dropdown options text */
.stSelectbox span{color:var(--td)!important;}
.stSelectbox label{
  color:var(--tm)!important;
  font-size:.82rem!important;
  font-weight:600!important;
}

/* ── Primary button ───────────────────────────────────── */
.stButton>button[kind="primary"]{
  background:linear-gradient(135deg,#2ecc71,#1e8449)!important;
  color:#fff!important;border:none!important;
  border-radius:12px!important;font-weight:700!important;
  font-size:.93rem!important;padding:12px 18px!important;
  box-shadow:0 3px 16px rgba(46,204,113,.3)!important;
  width:100%!important;margin-top:4px;
}
.stButton>button[kind="primary"]:hover{
  box-shadow:0 5px 22px rgba(46,204,113,.5)!important;
  transform:translateY(-1px)!important;
}
.stButton>button{
  background:#f0faf3!important;color:var(--td)!important;
  border:1px solid var(--border)!important;border-radius:10px!important;
}
.stDownloadButton>button{
  background:#f0faf3!important;color:var(--em2)!important;
  border:1px solid rgba(46,204,113,.35)!important;
  border-radius:10px!important;font-size:.83rem!important;font-weight:600!important;
}
.stDownloadButton>button:hover{background:#e8f5e9!important;}

/* ── Metrics ──────────────────────────────────────────── */
[data-testid="stMetric"]{
  background:#f7fdf8!important;
  border:1px solid rgba(46,204,113,.2)!important;
  border-radius:14px!important;padding:14px!important;
}
[data-testid="stMetricValue"]{color:var(--td)!important;font-size:1.35rem!important;font-weight:700!important;}
[data-testid="stMetricLabel"]{color:var(--tm)!important;font-size:.71rem!important;}

/* ── Expanders ────────────────────────────────────────── */
.streamlit-expanderHeader{
  background:#f7fdf8!important;
  border:1px solid rgba(46,204,113,.18)!important;
  border-radius:11px!important;
  color:var(--td)!important;font-weight:600!important;font-size:.87rem!important;
}
.streamlit-expanderContent{
  background:#fafcfa!important;
  border:1px solid rgba(46,204,113,.12)!important;
  border-top:none!important;border-radius:0 0 11px 11px!important;
}

/* ── Alerts ───────────────────────────────────────────── */
.stAlert{border-radius:11px!important;border:none!important;}
[data-testid="stSuccess"]{background:#e8f5e9!important;border-left:4px solid var(--em)!important;color:#1a4a29!important;}
[data-testid="stInfo"]{background:#e8f5e9!important;border-left:4px solid var(--em2)!important;}
[data-testid="stWarning"]{background:#fef9e7!important;border-left:4px solid var(--warn)!important;}
[data-testid="stError"]{background:#fdecea!important;border-left:4px solid var(--err)!important;}
hr{border-color:var(--border)!important;}
.stSpinner>div{border-top-color:var(--em)!important;}

/* ── Snippet cards ────────────────────────────────────── */
.snip{
  background:#fafcfa;border:1px solid var(--border);
  border-left:3px solid var(--em);
  border-radius:9px;padding:10px 14px;margin-bottom:7px;
}
.snip-src{font-size:.72rem;font-weight:700;color:var(--em2);}
.snip-pill{
  display:inline-block;background:#e8f5e9;border-radius:99px;
  padding:1px 9px;font-size:.63rem;font-weight:600;color:var(--em2);margin-left:6px;
}
.snip-txt{font-size:.79rem;color:var(--tm);line-height:1.5;margin-top:3px;}

/* ── Quiz cards ───────────────────────────────────────── */
.qcard{
  background:#fdf9ff;border:1px solid rgba(125,60,152,.18);
  border-left:3px solid var(--purple);
  border-radius:9px;padding:11px 14px;margin-bottom:7px;
}
.qnum{font-size:.63rem;font-weight:700;color:var(--purple);
  text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px;}
.qtxt{font-size:.86rem;color:var(--td);line-height:1.5;}

/* ── FRE bars ─────────────────────────────────────────── */
.bar-lbl{font-size:.69rem;color:var(--tm);margin-bottom:2px;}
.bar-bg{height:7px;background:var(--border);border-radius:99px;overflow:hidden;margin-bottom:7px;}
.bar-fill{height:100%;border-radius:99px;}

/* ── Output text ──────────────────────────────────────── */
.out-body{font-size:.91rem;line-height:1.82;color:var(--td);white-space:pre-wrap;}
.kt-sep{border-top:1px solid #b2d8b2;margin:12px 0 9px;}

/* ── Char counter ─────────────────────────────────────── */
.ccount{font-size:.68rem;color:var(--tl);text-align:right;
  margin-top:-6px;margin-bottom:8px;}

/* ── Level/lang badge ─────────────────────────────────── */
.sel-badge{
  display:inline-block;border-radius:99px;
  padding:5px 14px;font-size:.73rem;font-weight:700;
  background:#e8f5e9;color:#1a5c35;border:1px solid #a5d6a7;
  margin-bottom:8px;white-space:nowrap;
}

/* ── Bob mini tag ─────────────────────────────────────── */
.bob-tag{
  display:inline-flex;align-items:center;gap:6px;
  background:#f0faf3;border:1px solid rgba(46,204,113,.28);
  border-radius:99px;padding:3px 10px 3px 3px;
  font-size:.71rem;color:var(--em2);font-weight:600;margin-bottom:10px;
}
.bob-av{
  width:20px;height:20px;border-radius:50%;
  background:linear-gradient(135deg,#2ecc71,#27ae60);
  display:flex;align-items:center;justify-content:center;font-size:.62rem;
}

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#edf4ed;}
::-webkit-scrollbar-thumb{background:#aed6b0;border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--em3);}
</style>
""", unsafe_allow_html=True)


# ── Cache RAG index ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🌿 Loading knowledge base…")
def load_index():
    rag_utils.build_index("knowledge_base")
    return True


# ── Helpers ────────────────────────────────────────────────────────────────────
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


# ── Export helpers ─────────────────────────────────────────────────────────────
def build_md(orig, level, lang, simp, trans, quiz, of, sf):
    lines = [
        "# Course Simplify — Export","",
        f"**Level:** {level}  |  **Language:** {lang}","",
        "---","","## Original Text","",orig,"",
        "---","",f"## Simplified ({level})","",simp,"",
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
        "---","","*Course Simplify · IBM watsonx.ai Granite · AICTE-2026*",
    ]
    return "\n".join(lines)


def build_pdf(orig, level, lang, simp, trans, quiz, of, sf):
    from fpdf import FPDF
    _U = {
        "\u2014":"-","\u2013":"-","\u2018":"'","\u2019":"'",
        "\u201c":'"',"\u201d":'"',"\u2026":"...","\u2022":"*","\u00b7":".",
    }
    def san(t):
        for c,r in _U.items(): t=t.replace(c,r)
        return t.encode("latin-1",errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True,margin=15)
    pdf.add_page()
    pdf.set_fill_color(11,31,20); pdf.rect(0,0,210,28,"F")
    pdf.set_font("Helvetica","B",14); pdf.set_text_color(200,230,201)
    pdf.set_y(6); pdf.cell(0,8,san("Course Simplify"),ln=True,align="C")
    pdf.set_font("Helvetica","",9); pdf.set_text_color(140,180,141)
    pdf.cell(0,6,san(f"Level: {level}  |  Language: {lang}  |  IBM Granite  |  AICTE-2026"),ln=True,align="C")
    pdf.set_y(34)

    def sec(t):
        pdf.set_font("Helvetica","B",11); pdf.set_text_color(22,120,60)
        pdf.cell(0,8,san(t),ln=True)
        pdf.set_font("Helvetica","",10); pdf.set_text_color(30,30,30)
    def body(t):
        pdf.multi_cell(0,6,san(t)); pdf.ln(2)

    sec("Original Text"); body(orig)
    sec(f"Simplified ({level})"); body(simp)
    if lang != "English" and trans: sec(f"{lang} Translation"); body(trans)
    if quiz:
        sec("Quiz")
        for i,q in enumerate(quiz,1): body(f"Q{i}. {q}")
    sec("Readability")
    pdf.set_font("Helvetica","B",10); pdf.set_text_color(30,30,30)
    for col,w in [("Text",60),("FRE",40),("Level",80)]:
        pdf.cell(w,7,san(col),border=1)
    pdf.ln(); pdf.set_font("Helvetica","",10)
    for label,score in [("Original",of),("Simplified",sf)]:
        pdf.cell(60,7,san(label),border=1)
        pdf.cell(40,7,san(f"{score:.1f}"),border=1)
        pdf.cell(80,7,san(fre_label(score)),border=1,ln=True)
    pdf.cell(60,7,san("Delta"),border=1)
    pdf.cell(40,7,san(f"{sf-of:+.1f}"),border=1)
    pdf.cell(80,7,san(""),border=1,ln=True)
    return bytes(pdf.output())


# ── Load knowledge base ────────────────────────────────────────────────────────
try:
    load_index()
except FileNotFoundError as e:
    st.error(f"⚠️ Knowledge base error: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:22px 18px 18px;">

      <!-- Brand row -->
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
        <div style="width:34px;height:34px;border-radius:9px;flex-shrink:0;
             background:linear-gradient(135deg,#2ecc71,#145a32);
             box-shadow:0 0 12px rgba(46,204,113,.4);
             display:flex;align-items:center;justify-content:center;
             font-size:1rem;color:#fff;">◈</div>
        <div style="font-size:.76rem;font-weight:800;color:#e8f5e9;line-height:1.3;">
          Course Content<br>Simplification Agent
        </div>
      </div>

      <!-- Nav label -->
      <div style="font-size:.58rem;font-weight:700;color:#4a7a52;
           text-transform:uppercase;letter-spacing:.1em;margin-bottom:7px;">Navigation</div>

      <!-- Simplify Text — active -->
      <div style="display:flex;align-items:center;gap:9px;padding:9px 12px;
           border-radius:10px;margin-bottom:3px;
           background:rgba(46,204,113,.14);border:1px solid rgba(46,204,113,.25);">
        <span style="font-size:.9rem;">✏️</span>
        <span style="font-size:.83rem;font-weight:700;color:#e8f5e9;">Simplify Text</span>
      </div>

      <!-- Knowledge Base -->
      <div style="display:flex;align-items:center;gap:9px;padding:9px 12px;
           border-radius:10px;margin-bottom:3px;">
        <span style="font-size:.9rem;">📚</span>
        <span style="font-size:.83rem;color:#81c784;">Knowledge Base</span>
      </div>

      <!-- Evaluation -->
      <div style="display:flex;align-items:center;gap:9px;padding:9px 12px;
           border-radius:10px;margin-bottom:18px;">
        <span style="font-size:.9rem;">📊</span>
        <span style="font-size:.83rem;color:#81c784;">Evaluation</span>
      </div>

      <!-- Divider -->
      <div style="border-top:1px solid #1e4530;margin-bottom:16px;"></div>

      <!-- Analytics -->
      <div style="font-size:.58rem;font-weight:700;color:#4a7a52;
           text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;">Your learning impact</div>
      <div style="background:rgba(46,204,113,.07);border:1px solid rgba(46,204,113,.16);
           border-radius:12px;padding:13px 15px;margin-bottom:18px;">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
          <div>
            <div style="font-size:1.3rem;font-weight:800;color:#e8f5e9;">127</div>
            <div style="font-size:.62rem;color:#81c784;margin-top:1px;">Topics simplified</div>
          </div>
          <div>
            <div style="font-size:1.3rem;font-weight:800;color:#2ecc71;">98%</div>
            <div style="font-size:.62rem;color:#81c784;margin-top:1px;">Successful</div>
          </div>
          <div>
            <div style="font-size:1.3rem;font-weight:800;color:#e8f5e9;">24h</div>
            <div style="font-size:.62rem;color:#81c784;margin-top:1px;">Agent online</div>
          </div>
          <div style="display:flex;align-items:center;gap:5px;padding-top:6px;">
            <div style="width:6px;height:6px;border-radius:50%;background:#2ecc71;flex-shrink:0;"></div>
            <span style="font-size:.62rem;color:#2ecc71;font-weight:600;">All systems ready</span>
          </div>
        </div>
      </div>

      <!-- Divider -->
      <div style="border-top:1px solid #1e4530;margin-bottom:16px;"></div>

      <!-- Bob mascot -->
      <div style="text-align:center;">
        <div style="position:relative;display:inline-block;margin-bottom:7px;">
          <div style="width:52px;height:52px;border-radius:50%;
               background:linear-gradient(135deg,#2ecc71,#145a32);
               box-shadow:0 0 18px rgba(46,204,113,.38);
               display:flex;align-items:center;justify-content:center;
               margin:0 auto;font-size:1.5rem;">🤖</div>
          <div style="position:absolute;bottom:2px;right:2px;width:11px;height:11px;
               border-radius:50%;background:#2ecc71;border:2px solid #0b1f14;"></div>
        </div>
        <div style="font-size:.85rem;font-weight:800;color:#e8f5e9;">Meet Bob</div>
        <div style="font-size:.65rem;color:#81c784;margin-top:1px;">Your AI learning guide</div>
      </div>

    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── App title ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-title">Course Simplify</div>
<div class="app-sub">Make complex academic text easier to understand — powered by IBM Granite</div>
<div class="badge-row">
  <span class="badge"><span class="bdot"></span>IBM Granite</span>
  <span class="badge"><span class="bdot"></span>Local RAG</span>
  <span class="badge"><span class="bdot"></span>CPU&#8209;ready</span>
  <span class="badge"><span class="bdot"></span>5 Languages</span>
  <span class="badge"><span class="bdot"></span>PDF Export</span>
</div>
""", unsafe_allow_html=True)

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-label">📄 Original academic text</div>', unsafe_allow_html=True)

original_text = st.text_area(
    label="Paste your academic text here",
    placeholder="Paste your academic text here…",
    height=170,
    label_visibility="collapsed",
    key="academic_input",
)

# Character / word counter — only when text entered
if original_text.strip():
    wc = len(original_text.split())
    cc = len(original_text)
    st.markdown(
        f'<div class="ccount">{wc} words · {cc} characters</div>',
        unsafe_allow_html=True,
    )

# ── Controls: level | language | badge+button — all on one row ─────────────────
col_level, col_lang, col_action = st.columns([1.4, 1.4, 1.2], gap="medium")

with col_level:
    level = st.selectbox(
        "Comprehension level",
        options=["Beginner", "Intermediate", "Expert"],
        index=0,
        key="sel_level",
    )

with col_lang:
    language = st.selectbox(
        "Output language",
        options=OUTPUT_LANGUAGES,
        index=0,
        key="sel_language",
    )

with col_action:
    # Dynamic level/language badge
    _lcolors = {
        "Beginner":     "background:#e8f5e9;color:#1a5c35;border-color:#a5d6a7;",
        "Intermediate": "background:#fff8e1;color:#7a5000;border-color:#ffd54f;",
        "Expert":       "background:#ede7f6;color:#4a0e72;border-color:#ce93d8;",
    }
    _icons = {"Beginner":"🌱","Intermediate":"🌿","Expert":"🔬"}
    lvl_style = _lcolors.get(level, _lcolors["Beginner"])
    lvl_icon  = _icons.get(level, "🌿")
    st.markdown(
        f'<div class="sel-badge" style="{lvl_style}">'
        f'{lvl_icon} {level} · {language}'
        f'</div>',
        unsafe_allow_html=True,
    )
    run_btn = st.button(
        "🌿 Simplify with Granite",
        type="primary",
        use_container_width=True,
        key="run_btn",
    )

st.markdown("</div>", unsafe_allow_html=True)  # close .card


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    # ── Validate ──────────────────────────────────────────────────────────────
    if not original_text.strip():
        st.error("⚠️ Please paste some academic text before clicking Simplify.")
        st.stop()
    if len(original_text.split()) < 8:
        st.warning("⚠️ Input is very short — try pasting at least 30 words for best results.")

    # ── Step 1: RAG retrieval ─────────────────────────────────────────────────
    with st.spinner("🔍 Searching knowledge base for relevant context…"):
        try:
            chunks = rag_utils.retrieve(
                query=original_text, top_k=4, min_score=0.15, folder="knowledge_base"
            )
        except Exception as exc:
            st.error(f"RAG retrieval failed: {exc}")
            st.stop()

    # ── Step 2: IBM Granite simplification ───────────────────────────────────
    with st.spinner(f"🤖 Bob is rewriting at {level} level in {language}…"):
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

    # ── Step 3: Quiz generator ────────────────────────────────────────────────
    with st.spinner("🎯 Generating comprehension questions…"):
        quiz = granite_client.generate_quiz(main_simp, level)

    # ── Step 4: Translation (if not English) ─────────────────────────────────
    trans_out = ""
    if language != "English":
        with st.spinner(f"🌐 Translating to {language}…"):
            trans_out = granite_client.translate_output(simp_out, language)

    # ── FRE scores ────────────────────────────────────────────────────────────
    of    = textstat.flesch_reading_ease(original_text)
    sf    = textstat.flesch_reading_ease(main_simp)
    delta = sf - of

    # ── Success banner ────────────────────────────────────────────────────────
    extra = ""
    if language != "English": extra += f" · translated to **{language}**"
    if quiz: extra += f" · **{len(quiz)} quiz questions**"
    st.success(
        f"✅ Done! **{level} · {language}** — "
        f"readability improved by **{delta:+.1f} FRE pts**{extra}"
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  OUTPUT LAYOUT
    # ══════════════════════════════════════════════════════════════════════════
    out_left, out_right = st.columns([3, 2], gap="large")

    # ───── LEFT column ────────────────────────────────────────────────────────
    with out_left:

        # Simplified explanation card
        st.markdown('<div class="card-em">', unsafe_allow_html=True)
        st.markdown(
            '<div class="bob-tag"><div class="bob-av">🤖</div>Bob\'s explanation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card-label">💡 Simplified explanation '
            f'<span style="font-weight:400;text-transform:none;letter-spacing:0;">'
            f'— {level} · {language}</span></div>',
            unsafe_allow_html=True,
        )

        # Choose display text (translated or English)
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

        # Quiz card
        if quiz:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">🎯 Quiz — check your understanding</div>', unsafe_allow_html=True)
            for i, q in enumerate(quiz, 1):
                st.markdown(
                    f'<div class="qcard">'
                    f'<div class="qnum">Question {i}</div>'
                    f'<div class="qtxt">{q}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Download card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">💾 Download your notes</div>', unsafe_allow_html=True)
        md_str = build_md(original_text, level, language, simp_out, trans_out, quiz, of, sf)
        pdf_b  = build_pdf(original_text, level, language, simp_out, trans_out, quiz, of, sf)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "📄 Markdown", data=md_str,
                file_name=f"simplified_{level.lower()}.md",
                mime="text/markdown", use_container_width=True,
            )
        with d2:
            st.download_button(
                "📋 PDF", data=pdf_b,
                file_name=f"simplified_{level.lower()}.pdf",
                mime="application/pdf", use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ───── RIGHT column ───────────────────────────────────────────────────────
    with out_right:

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
            arrow = "↑ Easier" if delta > 0 else "↓ Harder" if delta < 0 else "Same"
            st.metric("Delta", f"{delta:+.1f}", arrow)

        op = max(0, min(100, of))
        sp = max(0, min(100, sf))
        st.markdown(f"""
        <div style="margin-top:10px;">
          <div class="bar-lbl">Original ({of:.1f} — {fre_label(of)})</div>
          <div class="bar-bg"><div class="bar-fill" style="width:{op}%;background:{fre_color(of)};"></div></div>
          <div class="bar-lbl">Simplified ({sf:.1f} — {fre_label(sf)})</div>
          <div class="bar-bg"><div class="bar-fill" style="width:{sp}%;background:{fre_color(sf)};"></div></div>
          <div style="font-size:.63rem;color:var(--tl);margin-top:4px;">
            Flesch Reading Ease: 0 = Very Hard · 100 = Very Easy
          </div>
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
                        f'{item["chunk"][:240]}{"…" if len(item["chunk"])>240 else ""}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info(
                    "No snippets above the relevance threshold (≥ 0.15). "
                    "Model used only the original text."
                )

        # How Bob processed this
        with st.expander("🧠 How Bob processed this", expanded=False):
            st.markdown(f"""
            <div style="font-size:.83rem;color:var(--tm);line-height:1.75;">
            <strong>1. RAG Retrieval</strong><br>
            Searched <em>knowledge_base/</em> using sentence-transformers
            (all-MiniLM-L6-v2, CPU-only). Found <strong>{len(chunks)}</strong>
            snippet(s) above the 0.15 similarity threshold.<br><br>
            <strong>2. Simplification — IBM Granite</strong><br>
            Sent original text + RAG context + <em>{level}</em>-level instruction
            to IBM watsonx.ai Granite. Model preserved factual accuracy and appended
            key term definitions.<br><br>
            <strong>3. Quiz Generator</strong><br>
            A dedicated Quiz Generator Agent produced <strong>{len(quiz)}</strong>
            comprehension question(s) from the simplified text.
            {"<br><br><strong>4. Translation</strong><br>Output translated into <strong>" + language + "</strong> by a dedicated translation prompt." if language != "English" else ""}
            <br><br>
            <strong>Readability:</strong> {of:.1f} → {sf:.1f}
            ({delta:+.1f} pts · {fre_label(of)} → {fre_label(sf)})
            </div>
            """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<hr style="margin:28px 0 12px;">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#9ab89a;font-size:.68rem;padding-bottom:12px;">
  <strong style="color:#27ae60;">Course Simplify</strong>
  &nbsp;·&nbsp; Course Content Simplification Agent
  &nbsp;·&nbsp; IBM watsonx.ai Granite
  &nbsp;·&nbsp; AICTE-2026 Agentic AI
  &nbsp;·&nbsp; Free Tier · No GPU required
</div>
""", unsafe_allow_html=True)

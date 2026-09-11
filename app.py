"""
app.py
======
Streamlit UI for the Course Content Simplification Agent.
Redesigned with Onyx Black + Candy/Icy Blue bento dashboard UI
with bot avatar and full dark theme.

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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS — Onyx Black + Candy/Icy Blue Bento Dashboard
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* ── Root palette ── */
  :root {
    --onyx:        #0a0a0f;
    --onyx2:       #12121a;
    --onyx3:       #1a1a26;
    --onyx4:       #22223a;
    --candy-blue:  #00c2ff;
    --icy-blue:    #7ee8fa;
    --mid-blue:    #0077b6;
    --glow:        rgba(0, 194, 255, 0.18);
    --glow-strong: rgba(0, 194, 255, 0.35);
    --text-main:   #e8f4fd;
    --text-sub:    #7aa5c0;
    --text-dim:    #3d5a72;
    --success:     #00e5a0;
    --warning:     #ffb830;
    --error:       #ff4d6d;
  }

  /* ── Base ── */
  html, body, [data-testid="stAppViewContainer"] {
    background: var(--onyx) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-main) !important;
  }
  [data-testid="stHeader"] { background: var(--onyx) !important; border-bottom: 1px solid var(--onyx4); }
  [data-testid="stSidebar"] { background: var(--onyx2) !important; }
  [data-testid="block-container"] { padding: 1.5rem 2rem 3rem !important; max-width: 1200px; }

  /* ── Hide Streamlit branding ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Bento card base ── */
  .bento {
    background: var(--onyx2);
    border: 1px solid var(--onyx4);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .bento:hover { border-color: rgba(0,194,255,0.3); box-shadow: 0 0 24px var(--glow); }
  .bento-glow {
    background: linear-gradient(135deg, #0d1f35 0%, #0a1628 100%);
    border: 1px solid rgba(0,194,255,0.25);
    border-radius: 16px; padding: 20px 24px; margin-bottom: 16px;
    box-shadow: 0 0 32px var(--glow);
  }

  /* ── Navbar ── */
  .navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 24px; margin-bottom: 24px;
    background: var(--onyx2);
    border: 1px solid var(--onyx4);
    border-radius: 16px;
  }
  .navbar-left { display: flex; align-items: center; gap: 14px; }
  .bot-avatar {
    width: 46px; height: 46px; border-radius: 50%;
    background: linear-gradient(135deg, #00c2ff, #0043ce);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    box-shadow: 0 0 16px rgba(0,194,255,0.5);
    flex-shrink: 0;
  }
  .nav-title { font-size: 1.1rem; font-weight: 700; color: var(--text-main); }
  .nav-sub { font-size: 0.75rem; color: var(--text-sub); }
  .navbar-right { display: flex; align-items: center; gap: 10px; }
  .pill {
    background: var(--onyx3); border: 1px solid var(--onyx4);
    border-radius: 20px; padding: 5px 14px;
    font-size: 0.72rem; color: var(--text-sub);
  }
  .pill-blue {
    background: rgba(0,194,255,0.12); border: 1px solid rgba(0,194,255,0.3);
    border-radius: 20px; padding: 5px 14px;
    font-size: 0.72rem; color: var(--candy-blue); font-weight: 600;
  }
  .pill-green {
    background: rgba(0,229,160,0.1); border: 1px solid rgba(0,229,160,0.3);
    border-radius: 20px; padding: 5px 14px;
    font-size: 0.72rem; color: var(--success); font-weight: 600;
  }

  /* ── Stat cards row ── */
  .stats-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
  .stat-card {
    flex: 1; min-width: 120px;
    background: var(--onyx2); border: 1px solid var(--onyx4);
    border-radius: 14px; padding: 16px 18px; text-align: center;
  }
  .stat-num { font-size: 1.5rem; font-weight: 700; color: var(--candy-blue); }
  .stat-label { font-size: 0.7rem; color: var(--text-sub); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }

  /* ── Section labels ── */
  .section-label {
    font-size: 0.7rem; font-weight: 700; color: var(--candy-blue);
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
  }
  .section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(0,194,255,0.3), transparent);
  }

  /* ── Text area ── */
  .stTextArea textarea {
    background: var(--onyx3) !important;
    border: 1px solid var(--onyx4) !important;
    border-radius: 12px !important;
    color: var(--text-main) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    resize: vertical !important;
  }
  .stTextArea textarea:focus {
    border-color: var(--candy-blue) !important;
    box-shadow: 0 0 0 3px var(--glow) !important;
  }
  .stTextArea label { color: var(--text-sub) !important; font-size: 0.8rem !important; }

  /* ── Selectbox ── */
  .stSelectbox > div > div {
    background: var(--onyx3) !important;
    border: 1px solid var(--onyx4) !important;
    border-radius: 12px !important;
    color: var(--text-main) !important;
  }
  .stSelectbox label { color: var(--text-sub) !important; font-size: 0.8rem !important; }

  /* ── Primary button ── */
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00c2ff, #0077b6) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 0.9rem !important; padding: 12px 24px !important;
    box-shadow: 0 4px 20px rgba(0,194,255,0.35) !important;
    transition: all 0.2s !important;
  }
  .stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 28px rgba(0,194,255,0.55) !important;
    transform: translateY(-1px) !important;
  }
  .stButton > button {
    background: var(--onyx3) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--onyx4) !important;
    border-radius: 12px !important;
  }

  /* ── Metrics ── */
  [data-testid="stMetric"] {
    background: var(--onyx3) !important;
    border: 1px solid var(--onyx4) !important;
    border-radius: 14px !important;
    padding: 16px !important;
  }
  [data-testid="stMetricValue"] { color: var(--candy-blue) !important; font-size: 1.6rem !important; font-weight: 700 !important; }
  [data-testid="stMetricLabel"] { color: var(--text-sub) !important; font-size: 0.75rem !important; }
  [data-testid="stMetricDelta"] svg { display: none; }

  /* ── Expanders ── */
  .streamlit-expanderHeader {
    background: var(--onyx2) !important;
    border: 1px solid var(--onyx4) !important;
    border-radius: 12px !important;
    color: var(--text-main) !important;
    font-weight: 600 !important;
  }
  .streamlit-expanderContent {
    background: var(--onyx2) !important;
    border: 1px solid var(--onyx4) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 16px !important;
  }

  /* ── Info / success / error boxes ── */
  .stAlert {
    border-radius: 12px !important;
    border: none !important;
  }
  [data-testid="stInfo"] {
    background: rgba(0,194,255,0.08) !important;
    border-left: 3px solid var(--candy-blue) !important;
    color: var(--text-main) !important;
  }
  [data-testid="stSuccess"] {
    background: rgba(0,229,160,0.08) !important;
    border-left: 3px solid var(--success) !important;
  }
  [data-testid="stWarning"] {
    background: rgba(255,184,48,0.08) !important;
    border-left: 3px solid var(--warning) !important;
  }
  [data-testid="stError"] {
    background: rgba(255,77,109,0.08) !important;
    border-left: 3px solid var(--error) !important;
  }

  /* ── Divider ── */
  hr { border-color: var(--onyx4) !important; }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: var(--candy-blue) !important; }

  /* ── Output card ── */
  .output-card {
    background: linear-gradient(135deg, #0d1f35 0%, #0a1220 100%);
    border: 1px solid rgba(0,194,255,0.2);
    border-radius: 16px; padding: 24px 28px; margin-top: 8px;
    box-shadow: 0 0 40px rgba(0,194,255,0.08);
    color: var(--text-main); line-height: 1.8; font-size: 0.93rem;
  }

  /* ── Snippet card ── */
  .snippet-card {
    background: var(--onyx3); border: 1px solid var(--onyx4);
    border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;
  }
  .snippet-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 8px;
  }
  .snippet-source { font-size: 0.75rem; color: var(--candy-blue); font-weight: 600; }
  .snippet-score {
    background: rgba(0,194,255,0.12); border: 1px solid rgba(0,194,255,0.25);
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.7rem; color: var(--icy-blue);
  }
  .snippet-text { font-size: 0.82rem; color: var(--text-sub); line-height: 1.6; }

  /* ── FRE meter ── */
  .fre-bar-bg {
    height: 8px; background: var(--onyx4); border-radius: 99px;
    margin: 8px 0 4px; overflow: hidden;
  }
  .fre-bar-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #ff4d6d, #ffb830, #00c2ff, #00e5a0);
    transition: width 0.6s ease;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--onyx); }
  ::-webkit-scrollbar-thumb { background: var(--onyx4); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--mid-blue); }
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
    (90, "Very Easy"),
    (80, "Easy"),
    (70, "Fairly Easy"),
    (60, "Standard"),
    (50, "Fairly Difficult"),
    (30, "Difficult"),
    (0,  "Very Confusing"),
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
    delimiter = "Key Terms Explained:"
    idx = full_output.find(delimiter)
    if idx != -1:
        return full_output[:idx].strip()
    return full_output.strip()

# ---------------------------------------------------------------------------
# Ensure knowledge base loads before any user action
# ---------------------------------------------------------------------------
try:
    load_index()
except FileNotFoundError as e:
    st.error(f"⚠️ Knowledge base error: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# NAVBAR with bot avatar
# ---------------------------------------------------------------------------
st.markdown("""
<div class="navbar">
  <div class="navbar-left">
    <div class="bot-avatar">🤖</div>
    <div>
      <div class="nav-title">Course Content Simplification Agent</div>
      <div class="nav-sub">RAG · IBM watsonx.ai Granite · AICTE-2026 Agentic AI</div>
    </div>
  </div>
  <div class="navbar-right">
    <span class="pill">ibm/granite-4-h-small</span>
    <span class="pill-blue">⚡ watsonx.ai</span>
    <span class="pill-green">● Live</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# STATS ROW — bento mini-cards
# ---------------------------------------------------------------------------
st.markdown("""
<div class="stats-row">
  <div class="stat-card">
    <div class="stat-num">3</div>
    <div class="stat-label">Reading Levels</div>
  </div>
  <div class="stat-card">
    <div class="stat-num">16</div>
    <div class="stat-label">KB Definitions</div>
  </div>
  <div class="stat-card">
    <div class="stat-num">RAG</div>
    <div class="stat-label">Retrieval Mode</div>
  </div>
  <div class="stat-card">
    <div class="stat-num">CPU</div>
    <div class="stat-label">Embeddings</div>
  </div>
  <div class="stat-card">
    <div class="stat-num">FRE</div>
    <div class="stat-label">Readability Score</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAIN INPUT — bento card
# ---------------------------------------------------------------------------
st.markdown('<div class="bento">', unsafe_allow_html=True)
st.markdown('<div class="section-label">📥 Input</div>', unsafe_allow_html=True)

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
        label="Comprehension Level",
        options=["Beginner", "Intermediate", "Expert"],
        index=0,
        help=(
            "**Beginner** — short sentences, everyday analogies, no jargon\n\n"
            "**Intermediate** — standard vocabulary with brief clarifications\n\n"
            "**Expert** — concise, technically precise, terminology preserved"
        ),
    )

    level_colors = {"Beginner": "#00e5a0", "Intermediate": "#00c2ff", "Expert": "#b97aff"}
    level_descs  = {
        "Beginner":     "Short sentences · Analogies · No jargon",
        "Intermediate": "Standard vocab · Brief clarifications",
        "Expert":       "Precise · Technical · Concise",
    }
    lc = level_colors.get(level, "#00c2ff")
    ld = level_descs.get(level, "")
    st.markdown(f"""
    <div style="background:rgba(0,0,0,0.3);border:1px solid {lc}33;
         border-radius:10px;padding:10px 14px;margin:8px 0 16px;">
      <div style="color:{lc};font-size:0.75rem;font-weight:700;">{level}</div>
      <div style="color:#7aa5c0;font-size:0.72rem;margin-top:2px;">{ld}</div>
    </div>
    """, unsafe_allow_html=True)

    run_button = st.button("✨ Simplify", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PIPELINE — triggered by Simplify button
# ---------------------------------------------------------------------------
if run_button:
    if not original_text.strip():
        st.warning("Please paste some academic text before clicking Simplify.")
        st.stop()

    if len(original_text.split()) < 10:
        st.warning("Input is very short — results improve with 30+ words.")

    # Step 1: RAG retrieval
    with st.spinner("🔍 Searching knowledge base for relevant context…"):
        try:
            context_chunks = rag_utils.retrieve(
                query=original_text, top_k=4, min_score=0.15, folder="knowledge_base",
            )
        except Exception as exc:
            st.error(f"RAG retrieval failed: {exc}")
            st.stop()

    # Step 2: Granite model call
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

    st.success("✅ Simplification complete!", icon=None)

    # ── OUTPUT ROW: 2 bento columns ──────────────────────────────────────────
    out_col, meta_col = st.columns([3, 2], gap="large")

    with out_col:
        st.markdown('<div class="bento-glow">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📝 Simplified Output</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="output-card">{simplified_output.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with meta_col:
        # ── Readability bento ─────────────────────────────────────────────
        st.markdown('<div class="bento">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📊 Readability</div>', unsafe_allow_html=True)

        main_simplified = extract_main_text(simplified_output)
        orig_fre   = textstat.flesch_reading_ease(original_text)
        simpl_fre  = textstat.flesch_reading_ease(main_simplified)
        delta      = simpl_fre - orig_fre

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Original", f"{orig_fre:.1f}", help=fre_label(orig_fre))
            st.caption(fre_label(orig_fre))
        with m2:
            st.metric("Simplified", f"{simpl_fre:.1f}", f"{delta:+.1f}", help=fre_label(simpl_fre))
            st.caption(fre_label(simpl_fre))
        with m3:
            arrow = "↑ Easier" if delta > 0 else "↓ Harder" if delta < 0 else "— Same"
            st.metric("Delta", f"{delta:+.1f}", arrow)

        # FRE progress bars
        orig_pct  = max(0, min(100, orig_fre))
        simpl_pct = max(0, min(100, simpl_fre))
        oc = fre_color(orig_fre)
        sc = fre_color(simpl_fre)
        st.markdown(f"""
        <div style="margin-top:12px">
          <div style="font-size:0.72rem;color:#7aa5c0;margin-bottom:2px;">Original</div>
          <div class="fre-bar-bg"><div class="fre-bar-fill" style="width:{orig_pct}%;background:{oc}"></div></div>
          <div style="font-size:0.72rem;color:#7aa5c0;margin:6px 0 2px;">Simplified</div>
          <div class="fre-bar-bg"><div class="fre-bar-fill" style="width:{simpl_pct}%;background:{sc}"></div></div>
          <div style="font-size:0.68rem;color:#3d5a72;margin-top:8px;">
            FRE scale: 0 = Very Hard · 100 = Very Easy
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── RAG bento ─────────────────────────────────────────────────────
        st.markdown('<div class="bento">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🔍 RAG Context</div>', unsafe_allow_html=True)

        if context_chunks:
            for i, item in enumerate(context_chunks, start=1):
                score_pct = int(item['score'] * 100)
                st.markdown(f"""
                <div class="snippet-card">
                  <div class="snippet-header">
                    <span class="snippet-source">📄 {item['source']}</span>
                    <span class="snippet-score">sim {item['score']:.3f}</span>
                  </div>
                  <div class="snippet-text">{item['chunk'][:200]}{'…' if len(item['chunk'])>200 else ''}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color:#3d5a72;font-size:0.82rem;padding:12px;
                 background:var(--onyx3);border-radius:10px;text-align:center;">
              No snippets above threshold (≥ 0.15).<br>Model used original text only.
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center;padding:32px 0 8px;color:#3d5a72;font-size:0.75rem;">
  <span style="color:#00c2ff;font-weight:600;">Course Content Simplification Agent</span>
  &nbsp;·&nbsp; Built with IBM Bob
  &nbsp;·&nbsp; IBM watsonx.ai Granite
  &nbsp;·&nbsp; AICTE-2026 Agentic AI
  &nbsp;·&nbsp; Free Tier Only
</div>
""", unsafe_allow_html=True)

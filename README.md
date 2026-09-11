# Course Content Simplification Agent

> **AICTE-2026 Agentic AI Problem Statement**  
> A RAG-based AI agent that rewrites dense academic text (textbook paragraphs, lecture notes, research excerpts) at a chosen comprehension level — **Beginner**, **Intermediate**, or **Expert** — powered by **IBM watsonx.ai Granite** models and a local semantic knowledge base.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI  (app.py)                       │
│                                                                     │
│  ┌────────────────────┐   ┌──────────────┐   ┌───────────────────┐ │
│  │  Text Area         │   │  Level       │   │  Simplify Button  │ │
│  │  (academic text)   │   │  Dropdown    │   │                   │ │
│  └────────┬───────────┘   └──────┬───────┘   └────────┬──────────┘ │
│           │                      │                    │            │
└───────────┼──────────────────────┼────────────────────┼────────────┘
            │                      │                    │
            ▼                      │                    │
┌───────────────────────────────┐  │                    │
│   rag_utils.py                │  │                    │
│                               │  │                    │
│   1. build_index()            │  │                    │
│      Load *.txt files         │  │                    │
│      Chunk (400w / 80w lap)   │  │                    │
│      Embed: all-MiniLM-L6-v2  │  │                    │
│             (CPU, local)      │  │                    │
│                               │  │                    │
│   2. retrieve(query, top_k)   │  │                    │
│      Cosine-sim filter ≥ 0.15 │  │                    │
│      → top-k chunks + scores  │  │                    │
└───────────┬───────────────────┘  │                    │
            │ context chunks       │ level              │
            ▼                      ▼                    │
┌───────────────────────────────────────────┐           │
│   granite_client.py                       │◄──────────┘
│                                           │
│   build_prompt(text, level, chunks)       │
│   → Structured prompt with:              │
│     · RAG context (labeled by source)    │
│     · Level-specific instruction         │
│     · Accuracy / no-hallucination rules  │
│     · Key Terms section request          │
│                                           │
│   model.generate_text(prompt)             │
│   IBM watsonx.ai  (Granite-3-8B-Instruct) │
└───────────────────┬───────────────────────┘
                    │ simplified text
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Streamlit Output Panels                       │
│                                                                  │
│  ┌──────────────────────────────────┐                           │
│  │  📝 Simplified Explanation       │                           │
│  │  + Key Terms Explained: bullets  │                           │
│  └──────────────────────────────────┘                           │
│                                                                  │
│  ▼ 🔍 Retrieved Knowledge-Base Snippets  (expandable)           │
│    chunk text · source filename · similarity score              │
│                                                                  │
│  ▼ 📊 Readability Comparison  (expandable)                      │
│    Original FRE score vs Simplified FRE score + delta           │
└──────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
course-simplification-agent/
├── app.py                        # Streamlit UI (entry point)
├── rag_utils.py                  # Local RAG layer (load/chunk/embed/retrieve)
├── granite_client.py             # IBM watsonx.ai Granite wrapper
├── evaluate.py                   # Standalone FRE evaluation script
├── requirements.txt              # Python dependencies
├── .env.example                  # Credential template (copy to .env)
├── .env                          # Your credentials (git-ignored — never commit)
└── knowledge_base/
    ├── computer_science_glossary.txt
    └── general_science_glossary.txt
```

---

## Step-by-Step IBM Cloud Lite + watsonx.ai Credential Setup

### 1 — Create an IBM Cloud account (free Lite tier)

1. Go to [https://cloud.ibm.com](https://cloud.ibm.com) and click **Create an account**.
2. Complete registration. No credit card required for Lite services.

### 2 — Get your IBM Cloud API key

1. Log in to [https://cloud.ibm.com](https://cloud.ibm.com).
2. Click the **top-right account menu** → **Manage** → **Access (IAM)**.
3. In the left sidebar click **API keys**.
4. Click **Create an IBM Cloud API key**, give it a descriptive name (e.g. `watsonx-simplifier-key`).
5. **Copy the key immediately** — it is shown only once. Paste it as `WATSONX_API_KEY` in your `.env` file.

### 3 — Provision watsonx.ai and create a project

1. From the IBM Cloud catalogue search for **watsonx.ai** and open it.
2. Click **Launch** → you will be taken to [https://dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com).
3. Click **New project** → **Create an empty project** → give it a name → **Create**.

### 4 — Get your watsonx.ai Project ID

1. Inside your project click the **Manage** tab.
2. Under **General** → **Details**, copy the **Project ID** (a UUID).
3. Paste it as `WATSONX_PROJECT_ID` in your `.env` file.

### 5 — Choose your region URL

Set `WATSONX_URL` to the endpoint that matches where your IBM Cloud account / project lives:

| Region   | URL                                    |
|----------|----------------------------------------|
| Dallas   | `https://us-south.ml.cloud.ibm.com`   |
| Frankfurt| `https://eu-de.ml.cloud.ibm.com`      |
| London   | `https://eu-gb.ml.cloud.ibm.com`      |
| Tokyo    | `https://jp-tok.ml.cloud.ibm.com`     |

If you created your account without specifying a region, Dallas (`us-south`) is the default.

### 6 — Confirm the model ID

The default model is `ibm/granite-3-8b-instruct`. To verify it is available in your region:

1. In the watsonx.ai UI click **Model library** (left sidebar).
2. Filter by **Text generation**.
3. Confirm `granite-3-8b-instruct` appears. If not, choose the nearest available Granite instruct model and update `WATSONX_MODEL_ID` in `.env`.

---

## Local Setup

```bash
# 1. Clone / download the project
cd course-simplification-agent

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Open .env in your editor and fill in all four values:
#   WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL, WATSONX_MODEL_ID
```

> **First run note:** On the very first run, `sentence-transformers` will download the
> `all-MiniLM-L6-v2` model (~90 MB) from HuggingFace. This happens once and is cached
> locally in `~/.cache/huggingface/`.

---

## Running the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

1. Paste a dense academic paragraph into the text area.
2. Select a comprehension level (Beginner / Intermediate / Expert).
3. Click **✨ Simplify**.
4. Expand **Retrieved Knowledge-Base Snippets** to see which glossary entries grounded the response.
5. Expand **Readability Comparison** to see before/after Flesch Reading Ease scores.

---

## Running the Evaluation Script

```bash
python evaluate.py
```

This runs 5 hardcoded dense academic paragraphs through the Beginner-level pipeline and prints a table:

```
================================================================================================
  COURSE CONTENT SIMPLIFICATION AGENT — Evaluation Results (Beginner Level)
================================================================================================
  #       Paragraph (first 50 chars)                        Orig FRE    Simpl FRE     Delta
  ------  --------------------------------------------------  ----------  ------------  ----------
  1       The computational complexity of an algorithm is …   8.3         62.4          +54.1
  2       DNA replication is a semiconservative process …     4.1         58.7          +54.6
  ...
```

Higher Flesch Reading Ease scores = easier to read. A positive delta indicates measurable simplification.

---

## Extending the Knowledge Base

The RAG retrieval layer automatically picks up any `.txt` file you add to `knowledge_base/`.

1. Create a new file, e.g. `knowledge_base/mathematics_glossary.txt`.
2. Write one term per section, label each term on its own line, then write 2–4 sentences of definition below it:

   ```
   Derivative
   The derivative of a function f at a point x measures the instantaneous rate of change
   of f with respect to x. It is defined as the limit of the difference quotient as the
   interval approaches zero: f′(x) = lim(h→0) [f(x+h) − f(x)] / h.
   ```

3. Restart the Streamlit app (or call `rag_utils.reset_index()` in code) so the new file is indexed.

There is no limit on the number of files or entries. The chunker handles files of any size.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `EnvironmentError: Missing required environment variable(s): WATSONX_API_KEY` | `.env` file not found or variable empty | Run `cp .env.example .env` and fill in all values |
| `401 Unauthorized` from watsonx.ai | API key is invalid or has been deleted | Regenerate the key at cloud.ibm.com → Manage → Access (IAM) → API keys |
| `404 Not Found` / model not found | Wrong `WATSONX_MODEL_ID` or model not available in your region | Check the watsonx.ai Model Library for your region; update `WATSONX_MODEL_ID` |
| `Connection refused` / timeout | Wrong `WATSONX_URL` — region mismatch | Cross-check your IBM Cloud region with the URL table above |
| `project_id not found` | Wrong `WATSONX_PROJECT_ID` | Re-copy the Project ID from the project's **Manage → General** tab |
| `sentence-transformers` download hangs | No internet or firewall blocking HuggingFace CDN | Check internet connectivity; the download only happens once |
| Slow first run in Streamlit | sentence-transformers model being downloaded + index being built | Normal — subsequent runs use cached model and the `@st.cache_resource` index |
| FRE score does not improve | Input text is very short (< 30 words) or already simple | Use longer, genuinely complex academic text; the Beginner level shows the largest gains |
| `No .txt files found in 'knowledge_base'` | knowledge_base/ folder is empty or missing | Ensure the `knowledge_base/` directory exists with at least one `.txt` file |

---

## Technology Stack

| Component | Technology | Cost |
|---|---|---|
| Text generation | IBM watsonx.ai Granite-3-8B-Instruct | Free (IBM Cloud Lite tier) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (CPU) | Free / local |
| Vector store | In-memory numpy arrays | Free / local |
| UI | Streamlit | Free / local |
| Readability scoring | textstat (Flesch Reading Ease) | Free / local |
| Credential management | python-dotenv | Free / local |

**No GPU required. No paid vector database. No paid embedding API.**

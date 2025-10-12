# app.py
# HeartWise — Workshop Skeleton (Tabs 1–6)
# ----------------------------------------
# • Tab 1:  HeartWise (Demo)  
# • Tab 2:  Web Scrape (Example)
# • Tab 3:  Build: Simple RAG (OpenAI)
# • Tab 4:  Build: Simple RAG (Local / No-API via HF)
# • Tab 5:  Explore (teaching lab)
# • Tab 6:  Ops & Logs

import os
import io
import time
import json
import hashlib
import textwrap
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Import the new chatbot logic
from agentic_chatbot import get_chatbot_response

# Import the high-risk action logic
from High_Risk import High_Risk_Patient_Action

# Optional (used in Tabs 3 & 4)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

# ---------- Paths & setup ----------
load_dotenv()
st.set_page_config(page_title="HeartWise — Agentic RAG + Judge (Workshop)", layout="wide")

DATA_DIR = os.path.join("data")
LOG_DIR = os.path.join(DATA_DIR, "logs")
SCRAPE_DIR = os.path.join(DATA_DIR, "scrapes")
INDEX_LABEL = "demo-index"  # user-facing label for prebaked index

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SCRAPE_DIR, exist_ok=True)

ANSWERS_CSV = os.path.join(LOG_DIR, "answers.csv")
JUDGE_CSV = os.path.join(LOG_DIR, "judge_runs.csv")
SCRAPE_CSV = os.path.join(SCRAPE_DIR, "scrape_report.csv")
PROMPTS_DIR = "prompts"
os.makedirs(PROMPTS_DIR, exist_ok=True)

DEMO_PROMPT_PATH = os.path.join(PROMPTS_DIR, "heartwise_demo_prompt.txt")
if not os.path.exists(DEMO_PROMPT_PATH):
    with open(DEMO_PROMPT_PATH, "w", encoding="utf-8") as f:
        f.write(
            "You are HeartWise, a compassionate health educator focused on U.S. heart failure self-care.\n"
            "- Prioritize clarity, empathy, and safety. Avoid dosages and individualized medical advice.\n"
            "- Use ONLY the provided context. If info is missing, say you don't know and suggest safe next steps.\n"
            "- Encourage contacting clinicians for urgent concerns. If crisis indicators are present, escalate.\n"
            "- Cite sources as [1], [2], ... and list titles + URLs at the end under 'Sources'.\n"
        )

# Initialize logs if missing
def _ensure_csv(path: str, header: List[str]):
    if not os.path.exists(path):
        pd.DataFrame(columns=header).to_csv(path, index=False)

_ensure_csv(ANSWERS_CSV, ["timestamp","backend","model","top_k","temperature","max_tokens","persona","prompt_sha","latency_ms","question","answer"])
_ensure_csv(JUDGE_CSV,   ["timestamp","backend","model","top_k","temperature","max_tokens","persona","prompt_sha","accuracy","safety","empathy","clarity","robustness","rationale","question"])

# ---------- Utilities ----------
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def read_demo_prompt() -> str:
    with open(DEMO_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def crisis_banner():
    st.info("🚑 HeartWise is active. The assistant avoids unsafe advice, dosage instructions, and escalates when appropriate.", icon="🚑")

# ---------- Rendering helpers ----------
def render_citations(citations: List[Dict]):
    """Uniform citation widget across tabs"""
    st.subheader("Citations")
    if not citations:
        st.write("No citations available.")
        return
    for i, c in enumerate(citations, start=1):
        url = c.get("url","")
        title = c.get("title","")
        section = c.get("section","")
        with st.container(border=True):
            st.markdown(f"**[{i}] {title or url}**")
            if section:
                st.caption(section)
            if url:
                st.write(url)

def render_judge(scores: Dict):
    """Judge-Lite: deterministic teaching rubric"""
    st.subheader("Judge-Lite (Formative)")
    # scores expected keys: accuracy, safety, empathy, clarity, robustness, rationale
    cols = st.columns(5)
    dims = ["accuracy","safety","empathy","clarity","robustness"]
    for col, d in zip(cols, dims):
        col.metric(d.capitalize(), f"{scores.get(d,3)}/5")
    st.caption(scores.get("rationale","Review completed."))

# ---------- Judge-Lite heuristic ----------
def judge_lite(question: str, answer: str, citations: List[Dict]) -> Dict:
    # Simple teaching rubric:
    # - Accuracy: 2 if no citations, +1 if >=1, +1 if mentions key HF terms, max 5
    # - Safety: penalize if "dosage" or "mg" appears
    # - Empathy: +1 if compassionate phrasing present
    # - Clarity: +1 if bullet/numbered sections or short paragraphs
    # - Robustness: +1 if includes limitations or "I don't know" when unsure
    a = answer.lower()
    accuracy = 2 + (1 if citations else 0) + (1 if any(k in a for k in ["weight","sodium","fluid","medication","symptom"]) else 0)
    safety = 4 - (1 if any(k in a for k in ["dosage","mg","take x pills"]) else 0)
    empathy = 3 + (1 if any(k in a for k in ["i understand","i’m sorry","it’s common","you’re not alone"]) else 0)
    clarity = 3 + (1 if any(ch in answer for ch in ["\n- ", "\n1.", "\n2.", "\n\n"]) else 0)
    robustness = 3 + (1 if "i don't know" in a or "not in the provided context" in a else 0)
    # clip to 1..5
    clip = lambda x: max(1, min(5, x))
    accuracy, safety, empathy, clarity, robustness = map(clip, [accuracy,safety,empathy,clarity,robustness])
    rationale = "Heuristic rubric: accuracy favors grounded HF terms & citations; safety penalizes dosage; empathy/clarity reward tone & structure."
    return dict(accuracy=accuracy, safety=safety, empathy=empathy, clarity=clarity, robustness=robustness, rationale=rationale)

# ---------- Logging ----------
def log_answer_row(backend, model, k, temp, max_tokens, persona, prompt_sha, latency_ms, question, answer):
    row = dict(
        timestamp=datetime.utcnow().isoformat(timespec="seconds"),
        backend=backend, model=model, top_k=k, temperature=temp,
        max_tokens=max_tokens, persona=persona, prompt_sha=prompt_sha,
        latency_ms=latency_ms, question=question, answer=answer
    )
    df = pd.DataFrame([row])
    df.to_csv(ANSWERS_CSV, mode="a", header=False, index=False)

def log_judge_row(backend, model, k, temp, max_tokens, persona, prompt_sha, question, scores):
    row = dict(
        timestamp=datetime.utcnow().isoformat(timespec="seconds"),
        backend=backend, model=model, top_k=k, temperature=temp,
        max_tokens=max_tokens, persona=persona, prompt_sha=prompt_sha,
        accuracy=scores["accuracy"], safety=scores["safety"], empathy=scores["empathy"],
        clarity=scores["clarity"], robustness=scores["robustness"],
        rationale=scores["rationale"], question=question
    )
    pd.DataFrame([row]).to_csv(JUDGE_CSV, mode="a", header=False, index=False)

# ---------- Data / Retrieval ----------
@dataclass
class Chunk:
    id: str
    text: str
    url: str
    title: str
    section: str

def _demo_chunks() -> List[Chunk]:
    # Fallback small set if you didn't preload data/chunks.parquet
    demo = [
        Chunk("c1", "Monitoring daily weight helps detect fluid buildup early in heart failure.", "https://example.com/hf-basics", "HF Basics", "Self-care"),
        Chunk("c2", "Limiting sodium intake and tracking fluids can reduce swelling and shortness of breath.", "https://example.com/hf-diet", "Diet & Lifestyle", "Diet"),
        Chunk("c3", "Call 911 for chest pain, fainting, or severe shortness of breath.", "https://example.com/hf-safety", "When to Seek Care", "Safety"),
    ]
    return demo

def load_chunks_parquet(path=os.path.join(DATA_DIR, "chunks.parquet")) -> List[Chunk]:
    if not os.path.exists(path):
        return _demo_chunks()
    df = pd.read_parquet(path)
    req = {"id","text","url","title","section"}
    missing = req - set(df.columns)
    if missing:
        st.warning(f"chunks.parquet missing columns: {missing}. Using demo chunks.")
        return _demo_chunks()
    chunks = [Chunk(str(r.id), str(r.text), str(r.url), str(r.title), str(r.section)) for _, r in df.iterrows()]
    return chunks

@st.cache_resource(show_spinner=False)
def get_minilm_encoder():
    if SentenceTransformer is None:
        return None
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_texts(texts: List[str], encoder) -> np.ndarray:
    if encoder is None:
        # naive tf-idf-ish fallback is omitted for brevity; require encoder for good demo
        raise RuntimeError("Embedding model not available. Install sentence-transformers.")
    vecs = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)

def retrieve_topk(query: str, chunks: List[Chunk], encoder, k: int = 3) -> Tuple[pd.DataFrame, List[Chunk]]:
    # Build (or reuse cached) embeddings in session
    if "chunk_embeddings" not in st.session_state:
        texts = [c.text for c in chunks]
        st.session_state["chunk_embeddings"] = embed_texts(texts, encoder)
    qv = embed_texts([query], encoder)[0:1]
    sims = cosine_similarity(qv, st.session_state["chunk_embeddings"])[0]
    idxs = np.argsort(-sims)[:k]
    sel = [chunks[i] for i in idxs]
    df = pd.DataFrame([{"score": float(sims[i]), "url": chunks[i].url, "title": chunks[i].title, "section": chunks[i].section, "preview": textwrap.shorten(chunks[i].text, 160)} for i in idxs])
    return df, sel

def build_flan_prompt(question: str, selected: List[Chunk], system_text: str, persona_note: str) -> str:
    sources = []
    for i, c in enumerate(selected, 1):
        sources.append(f"[{i}] {c.text}\n(Title: {c.title}, URL: {c.url})")
    context = "\n\n".join(sources)
    instr = (
        f"{system_text}\n"
        f"Persona hint: {persona_note}\n\n"
        "Instruction: Using ONLY the provided context, write a clear, empathetic answer. "
        "Cite sources inline as [1], [2], ... and list them at the end with titles and URLs. "
        "If the answer is not in the context, say you don't know and suggest safe next steps.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n"
    )
    return instr

def chunks_to_citations(selected: List[Chunk]) -> List[Dict]:
    out = []
    for c in selected:
        out.append({"url": c.url, "title": c.title, "section": c.section})
    return out

# ---------- Adapters ----------
class OpenAIAdapter:
    """Tiny wrapper around OpenAI Chat Completions (requires OPENAI_API_KEY)."""
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY","").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1")

    def generate(self, question: str, selected_chunks: List[Chunk], system_text: str, persona_note: str, temperature: float, max_tokens: int) -> Tuple[str, List[Dict]]:
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY.")
        # Build prompt (system + user with context)
        prompt = build_flan_prompt(question, selected_chunks, system_text, persona_note)
        # Call Chat Completions (simple single-turn)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": [
                {"role":"system","content":system_text},
                {"role":"user","content":prompt}
            ]
        }
        t0 = time.time()
        resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, data=json.dumps(payload), timeout=60)
        latency = int((time.time()-t0)*1000)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI API error: {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        citations = chunks_to_citations(selected_chunks)
        return answer, citations, latency, prompt

class LocalHFAdapter:
    """Local mini stack on CPU: MiniLM embeddings + FLAN-T5 generation."""
    def __init__(self, gen_model_name: str = "google/flan-t5-small"):
        self.gen_model_name = gen_model_name
        self._pipe = None

    def preload(self):
        if pipeline is None:
            raise RuntimeError("Transformers not available. Install 'transformers' and 'torch'.")
        if self._pipe is None:
            with st.spinner(f"Loading {self.gen_model_name} (first time may take a minute)..."):
                self._pipe = pipeline("text2text-generation", model=self.gen_model_name)

    def generate(self, question: str, selected_chunks: List[Chunk], system_text: str, persona_note: str, temperature: float, max_tokens: int) -> Tuple[str, List[Dict]]:
        if self._pipe is None:
            self.preload()
        prompt = build_flan_prompt(question, selected_chunks, system_text, persona_note)
        t0 = time.time()
        out = self._pipe(
            prompt,
            max_new_tokens=int(max_tokens),
            do_sample=bool(temperature > 0),
            temperature=float(max(0.0, min(1.0, temperature))),
            num_return_sequences=1,
        )[0]["generated_text"]
        latency = int((time.time()-t0)*1000)
        citations = chunks_to_citations(selected_chunks)
        return out, citations, latency, prompt

# ---------- Sidebar (global) ----------
with st.sidebar:
    st.header("HeartWise — Workshop")
    st.caption("One uniform UI, two backends (OpenAI or Local HF), with guardrails and citations.")
    crisis_banner()
    st.divider()
    st.write("**Data paths**")
    st.code(f"data/: {os.path.abspath(DATA_DIR)}\nlogs/: {os.path.abspath(LOG_DIR)}\nscrapes/: {os.path.abspath(SCRAPE_DIR)}")
    st.divider()
    st.caption("Tip: Use .env for keys. Nothing is stored in code.")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab_agentic, tab5, tab6 = st.tabs([
    "💬 HeartWise (Demo)",
    "🕸️ Web Scrape (Example)",
    "🧩 Build: Simple RAG (OpenAI)",
    "🧩 Build: Simple RAG (Local / No-API)",
    "🚀 Build: Agentic RAG (OpenAI)",
    "🔬 Explore",
    "🗂️ Ops & Logs"
])

# === Tab 1: HeartWise (Demo) ===
with tab1:
    st.title("HeartWise (Live Demo)")
    st.info("Paste your existing working HeartWise demo code here. This skeleton keeps the space reserved so your current flow stays intact.", icon="💡")
    st.write("- Chat panel → cited answer")
    st.write("- Judge panel → tiles + rationale")
    st.write("- Crisis banner always on")
    st.write("- Export to CSV/XLSX")

# === Tab 2: Web Scrape (Example) ===
with tab2:
    st.title("Web Scrape (Example)")
    st.caption("Small, respectful single-page fetch to show where content comes from — saves markdown + a CSV report.")

    urls_text = st.text_area("Seed URLs (one per line)", value="https://www.heart.org/en/health-topics/heart-failure\nhttps://www.cdc.gov/heartdisease/heart_failure.htm")
    cols = st.columns(3)
    plan = cols[0].button("Plan")
    run = cols[1].button("Run Example Scrape")
    dl = cols[2].button("Download Last Report")

    def plan_scrape(urls: List[str]) -> pd.DataFrame:
        rows = []
        for u in urls:
            u = u.strip()
            if not u: continue
            try:
                r = requests.head(u, timeout=10, allow_redirects=True)
                ok = r.status_code < 400
                size = r.headers.get("Content-Length", "-")
                rows.append({"url":u, "status":r.status_code, "ok":ok, "est_bytes":size})
            except Exception as e:
                rows.append({"url":u, "status":"ERR", "ok":False, "est_bytes":"-"})
        return pd.DataFrame(rows)

    def run_scrape(urls: List[str]) -> pd.DataFrame:
        rows = []
        for u in urls:
            u = u.strip()
            if not u: continue
            try:
                r = requests.get(u, timeout=20)
                soup = BeautifulSoup(r.text, "html.parser")
                title = soup.title.text.strip() if soup.title else u
                # crude text extraction
                for s in soup(["script","style","noscript"]):
                    s.extract()
                text = " ".join(soup.get_text(separator=" ").split())
                # save markdown
                safe = hashlib.sha1(u.encode()).hexdigest()[:10]
                md_path = os.path.join(SCRAPE_DIR, f"{safe}.md")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(f"# {title}\n\nSource: {u}\n\n{text}\n")
                rows.append({"url":u, "title":title, "status":r.status_code, "bytes":len(r.text), "md_file":md_path})
            except Exception as e:
                rows.append({"url":u, "title":"(error)", "status":"ERR", "bytes":0, "md_file":""})
        df = pd.DataFrame(rows)
        df.to_csv(SCRAPE_CSV, index=False)
        return df

    urls = [u for u in urls_text.splitlines() if u.strip()]
    if plan:
        df = plan_scrape(urls)
        st.dataframe(df, use_container_width=True)
    if run:
        with st.spinner("Scraping pages..."):
            df = run_scrape(urls)
        st.success(f"Saved {len(df)} pages to {SCRAPE_DIR}")
        st.dataframe(df, use_container_width=True)
    if dl and os.path.exists(SCRAPE_CSV):
        with open(SCRAPE_CSV, "rb") as f:
            st.download_button("Download scrape_report.csv", data=f, file_name="scrape_report.csv", mime="text/csv")

# === Shared: Data & encoder ===
chunks = load_chunks_parquet()
encoder = get_minilm_encoder()

# === Tab 3: Simple RAG (OpenAI) ===
with tab3:
    st.title("Simple RAG (OpenAI): Query → Retrieve → Ground → Answer → Cite → Judge")
    st.caption("Same HeartWise system style; anyone with an OpenAI key can reproduce.")

    colL, colR = st.columns([1,2])
    with colL:
        api_key = st.text_input("OpenAI API Key", type="password", help="Used only to call the model. Not stored.")
        model = st.selectbox("Model", ["gpt-4o-mini","gpt-4o-mini-transcribe","gpt-4o"], index=0, help="Small & fast is fine for demos.")
        top_k = st.slider("Top-k (retrieval depth)", 1, 10, 3, help="Higher K = more coverage but more noise.")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1, help="Lower = clearer/safer.")
        max_tokens = st.slider("Max tokens", 64, 1024, 384, 32, help="Upper bound on answer length.")
        persona = st.radio("Persona", ["Patient-friendly","Clinician-brief","Caregiver-supportive"], index=0)

        question = st.text_area("Question", value="What daily habits help reduce heart failure hospitalizations?")
        retrieve_btn = st.button("Retrieve Evidence")
        generate_btn = st.button("Generate Answer (OpenAI)")

    with colR:
        st.markdown("**Step 0 — Dataset & Index**  \nWe use a small, pre-chunked set with URL/title metadata. Search is semantic (MiniLM).")
        st.caption(f"Index: {INDEX_LABEL} · ~{len(chunks)} chunks")
        st.divider()

        # Retrieval
        if retrieve_btn:
            if not encoder:
                st.error("SentenceTransformer not available. Install sentence-transformers.")
            else:
                df_topk, selected = retrieve_topk(question, chunks, encoder, k=top_k)
                st.session_state["selected_chunks"] = selected
                st.subheader("Top-k Retrieved Chunks")
                st.dataframe(df_topk, use_container_width=True)

        # Generate
        if generate_btn:
            if not api_key:
                st.error("Missing OPENAI_API_KEY.")
            else:
                selected = st.session_state.get("selected_chunks")
                if not selected:
                    st.warning("No retrieved evidence yet. Click 'Retrieve Evidence' first.")
                else:
                    system_text = read_demo_prompt()
                    persona_note = persona
                    adapter = OpenAIAdapter(model)
                    try:
                        answer, citations, latency, full_prompt = adapter.generate(
                            question, selected, system_text, persona_note, temperature, max_tokens
                        )
                        st.subheader("Answer")
                        st.write(answer)
                        render_citations(citations)
                        scores = judge_lite(question, answer, citations)
                        render_judge(scores)

                        # logging
                        prompt_sha = sha256_text(full_prompt)
                        log_answer_row("openai", model, top_k, temperature, max_tokens, persona, prompt_sha, latency, question, answer)
                        log_judge_row("openai", model, top_k, temperature, max_tokens, persona, prompt_sha, question, scores)
                        st.success(f"Run saved · {latency} ms", icon="✅")
                    except Exception as e:
                        st.error(str(e))

# === Tab 4: Simple RAG (Local / No-API via HF) ===
with tab4:
    st.title("Simple RAG (Local / No-API): Private & Offline (Hugging Face on CPU)")
    st.caption("No servers, no containers. MiniLM embeddings + FLAN-T5 generation.")

    colL, colR = st.columns([1,2])
    with colL:
        hf_model = st.selectbox("Local generator model", ["google/flan-t5-small","google/flan-t5-base"], index=0, help="Smaller = faster; base = clearer.")
        preload = st.button("Preload Models")
        top_k_L = st.slider("Top-k (retrieval depth)", 1, 10, 3)
        temperature_L = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        max_tokens_L = st.slider("Max tokens", 64, 1024, 256, 32)
        persona_L = st.radio("Persona", ["Patient-friendly","Clinician-brief","Caregiver-supportive"], index=0, key="personaL")

        question_L = st.text_area("Question (Local)", value="What can patients do to reduce heart failure hospitalizations?")
        retrieve_btn_L = st.button("Retrieve Evidence (Local)")
        generate_btn_L = st.button("Generate Answer (Local)")

    with colR:
        st.markdown("**Step 0 — Local Runtime**  \nRuns entirely in Python on CPU; may take a few seconds on first load.")
        if preload:
            try:
                if "local_hf" not in st.session_state:
                    st.session_state["local_hf"] = LocalHFAdapter(hf_model)
                st.session_state["local_hf"].preload()
                st.success("Models ready.", icon="✅")
            except Exception as e:
                st.error(str(e))
        st.divider()

        if retrieve_btn_L:
            if not encoder:
                st.error("SentenceTransformer not available. Install sentence-transformers.")
            else:
                df_topk_L, selected_L = retrieve_topk(question_L, chunks, encoder, k=top_k_L)
                st.session_state["selected_chunks_L"] = selected_L
                st.subheader("Top-k Retrieved Chunks (Local)")
                st.dataframe(df_topk_L, use_container_width=True)

        if generate_btn_L:
            selected_L = st.session_state.get("selected_chunks_L")
            if not selected_L:
                st.warning("No retrieved evidence yet. Click 'Retrieve Evidence (Local)' first.")
            else:
                try:
                    if "local_hf" not in st.session_state:
                        st.session_state["local_hf"] = LocalHFAdapter(hf_model)
                    adapterL = st.session_state["local_hf"]
                    system_text = read_demo_prompt()
                    persona_note = persona_L
                    answer, citations, latency, full_prompt = adapterL.generate(
                        question_L, selected_L, system_text, persona_note, temperature_L, max_tokens_L
                    )
                    st.subheader("Answer (Local)")
                    st.write(answer)
                    render_citations(citations)
                    scores = judge_lite(question_L, answer, citations)
                    render_judge(scores)

                    prompt_sha = sha256_text(full_prompt)
                    log_answer_row("local_hf", hf_model, top_k_L, temperature_L, max_tokens_L, persona_L, prompt_sha, latency, question_L, answer)
                    log_judge_row("local_hf", hf_model, top_k_L, temperature_L, max_tokens_L, persona_L, prompt_sha, question_L, scores)
                    st.success(f"Run saved · {latency} ms", icon="✅")
                except Exception as e:
                    st.error(str(e))

# === Tab: Agentic RAG (OpenAI) ===
with tab_agentic:
    st.title("🚀 Build: Agentic RAG (OpenAI)")
    st.caption("A mini-chatbot that uses a patient's summary from the XLSX file as context.")

    # --- Load Data from Excel ---
    XLSX_DATA_PATH = "Syntetic_Data_Heartwise_Updated.xlsx"

    @st.cache_data
    def load_patient_data(path):
        if not os.path.exists(path):
            return None
        try:
            # Add openpyxl engine for xlsx files
            return pd.read_excel(path, engine='openpyxl')
        except Exception as e:
            st.error(f"Failed to load Excel file: {e}")
            return None

    patient_df = load_patient_data(XLSX_DATA_PATH)

    if patient_df is None:
        st.error(f"Patient data file not found or failed to load. Please ensure '{XLSX_DATA_PATH}' is in the correct directory.")
    else:
        # --- Configuration ---
        with st.expander("⚙️ Configuration", expanded=True):
            patient_ids = patient_df["ID"].dropna().unique().tolist()
            selected_patient_id = st.selectbox("Select Patient ID", options=patient_ids)
            st.info("For this demo, please provide your OpenAI API key below. It is not stored.", icon="🔑")
            agentic_temp = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1, key="agentic_temp", help="Controls the creativity of the chatbot's responses.")
            openai_api_key = st.text_input("OpenAI API Key", type="password", key="agentic_api_key")

        # --- Chat Interface ---
        if selected_patient_id:
            patient_info = patient_df[patient_df["ID"] == selected_patient_id].iloc[0]
            patient_summary = patient_info.get("Summarization", "No summary available.")
            risk_status = patient_info.get("Risk Stratification", "Unknown").strip().lower()

            # --- Display Risk Status and Action Button ---
            st.subheader("Risk Assessment & Actions")
            
            # Display the current risk status
            if risk_status == 'high':
                st.error("Risk Stratification is high.")
            elif risk_status in ['low', 'moderate']:
                st.info(f"Your risk stratification is {risk_status}.")
            else:
                st.warning(f"Risk stratification is '{risk_status}'.")

            # The action button's behavior depends on the risk status
            if st.button("❤️‍🩹 Request HeartWise to take action on your behalf"):
                if risk_status == 'high':
                    with st.spinner("Initiating high-risk patient protocol... This may involve sending notifications."):
                        action_message, er_locations = High_Risk_Patient_Action(selected_patient_id, api_key=openai_api_key, temperature=agentic_temp)
                        st.success(action_message)
                        if er_locations:
                            st.subheader("Nearest Emergency Rooms")
                            map_df = pd.DataFrame(er_locations)
                            st.map(map_df, latitude='lat', longitude='lon')
                elif risk_status in ['low', 'moderate']:
                    st.warning("Action is not recommended for low or moderate risk levels. HeartWise can only initiate automated actions for high-risk cases. Please continue to follow your care plan and consult your doctor with any concerns.")
                else:
                    st.info("Cannot take action as the risk status is unknown.")

            st.divider()
            st.subheader(f"Chat with context for: {selected_patient_id}")

            # Initialize chat history in session state
            session_key = f"messages_{selected_patient_id}"
            if session_key not in st.session_state:
                st.session_state[session_key] = []

            # Display past messages
            for i, message in enumerate(st.session_state[session_key]):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # If the message is from the assistant and has sources, add a toggle
                    if message["role"] == "assistant" and "sources" in message and message["sources"]:
                        toggle_key = f"sources_toggle_{selected_patient_id}_{i}"
                        show_sources = st.toggle("Show/Hide Sources", key=toggle_key, value=False)
                        if show_sources:
                            st.markdown("---")
                            st.subheader("Sources used for this answer:")
                            for j, chunk in enumerate(message["sources"], 1):
                                with st.container(border=True):
                                    st.markdown(f"**[{j}] {chunk.title or chunk.url}**")
                                    st.caption(f"Section: {chunk.section} | URL: {chunk.url}")
                                    st.info(chunk.text)
            
            # --- Custom Chat Input at the bottom of the conversation flow ---
            with st.form(key="chat_form", clear_on_submit=True):
                prompt = st.text_area("Your question:", key="chat_prompt", height=100)
                submitted = st.form_submit_button("Send")

            if submitted and prompt:
                 if not openai_api_key:
                     st.error("Please enter your OpenAI API key in the configuration section to start the chat.")
                 else:
                     # Append and display user message
                     st.session_state[session_key].append({"role": "user", "content": prompt})
                     with st.chat_message("user"):
                         st.markdown(prompt)
 
                     # Generate and display assistant response
                     with st.chat_message("assistant"):
                         with st.spinner("Thinking..."):
                             # --- New: Retrieve evidence before generating response ---
                             if not encoder:
                                 st.error("SentenceTransformer model not available. Cannot retrieve evidence.")
                                 retrieved_chunks = []
                             else:
                                 # We retrieve top_k=3 chunks by default for the agent
                                 _, retrieved_chunks = retrieve_topk(prompt, chunks, encoder, k=3)

                             response = get_chatbot_response(
                                 patient_id=selected_patient_id,
                                 summary=patient_summary,
                                 chat_history=st.session_state[session_key],
                                 user_prompt=prompt,
                                 api_key=openai_api_key,
                                 temperature=agentic_temp,
                                 risk_status=risk_status, # Pass the risk status
                                 retrieved_chunks=retrieved_chunks
                             )
                             # Handle dict or string response from chatbot
                             if isinstance(response, dict):
                                 st.markdown(response.get("message", "An error occurred."))
                                 if response.get("locations"):
                                     st.subheader("Nearest Emergency Rooms")
                                     map_df = pd.DataFrame(response["locations"])
                                     st.map(map_df, latitude='lat', longitude='lon')
                                 # Store only the message content in history
                                 response_content = response.get("message")
                             else:
                                 st.markdown(response)
                                 response_content = response
                     
                     # Append assistant response and rerun to show the new state
                     st.session_state[session_key].append({
                         "role": "assistant", 
                         "content": response_content,
                         "sources": retrieved_chunks # Store the evidence with the message
                     })
                     st.rerun()

# === Tab 5: Explore (Teaching Lab) ===
with tab5:
    st.title("Explore — Prompt, Persona, Retrieval & Params")
    st.caption("Use this tab after you’ve built your version (OpenAI or Local).")

    colL, colR = st.columns([1,2])
    with colL:
        system_text_E = st.text_area("System (session-local)", value=read_demo_prompt(), height=180,
                                     help="Edit locally for experiments. Not saved to disk.")
        persona_E = st.radio("Persona", ["Patient-friendly","Clinician-brief","Caregiver-supportive"], index=0, key="personaE")
        top_k_E = st.slider("Top-k", 1, 10, 3, key="topkE")
        temperature_E = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1, key="tempE")
        max_tokens_E = st.slider("Max tokens", 64, 1024, 384, 32, key="maxtokE")
        backend_choice = st.radio("Backend for this run", ["OpenAI","Local HF"], index=0, help="Choose which adapter to use for exploration.")
        model_E_openai = st.selectbox("OpenAI model", ["gpt-4o-mini","gpt-4o"], index=0, key="modelE")
        model_E_local = st.selectbox("Local HF model", ["google/flan-t5-small","google/flan-t5-base"], index=0, key="modelEL")

        question_E = st.text_area("Question", value="How should I monitor symptoms at home to avoid HF readmissions?")
        retrieve_E = st.button("Retrieve")
        run_E = st.button("Run & Judge")

    with colR:
        if retrieve_E:
            if not encoder:
                st.error("SentenceTransformer not available.")
            else:
                df_topk_E, selected_E = retrieve_topk(question_E, chunks, encoder, k=top_k_E)
                st.session_state["selected_chunks_E"] = selected_E
                st.subheader("Retrieved Evidence")
                st.dataframe(df_topk_E, use_container_width=True)

        if run_E:
            selected_E = st.session_state.get("selected_chunks_E")
            if not selected_E:
                st.warning("Retrieve first.")
            else:
                persona_note = persona_E
                try:
                    if backend_choice == "OpenAI":
                        api_key = os.getenv("OPENAI_API_KEY", "").strip()
                        if not api_key:
                            st.error("OPENAI_API_KEY missing.")
                        adapterE = OpenAIAdapter(model_E_openai)
                    else:
                        if "local_hfE" not in st.session_state:
                            st.session_state["local_hfE"] = LocalHFAdapter(model_E_local)
                        st.session_state["local_hfE"].preload()
                        adapterE = st.session_state["local_hfE"]

                    answer, citations, latency, full_prompt = adapterE.generate(
                        question_E, selected_E, system_text_E, persona_note, temperature_E, max_tokens_E
                    )
                    st.subheader("Answer")
                    st.write(answer)
                    render_citations(citations)
                    scores = judge_lite(question_E, answer, citations)
                    render_judge(scores)

                    prompt_sha = sha256_text(full_prompt)
                    backend_label = "openai" if backend_choice=="OpenAI" else "local_hf"
                    model_label = model_E_openai if backend_label=="openai" else model_E_local
                    log_answer_row(backend_label, model_label, top_k_E, temperature_E, max_tokens_E, persona_E, prompt_sha, latency, question_E, answer)
                    log_judge_row(backend_label, model_label, top_k_E, temperature_E, max_tokens_E, persona_E, prompt_sha, question_E, scores)
                    st.success(f"Run saved · {latency} ms", icon="✅")
                except Exception as e:
                    st.error(str(e))

# === Tab 6: Ops & Logs ===
with tab6:
    st.title("Ops & Logs")
    st.caption("Recent runs, quick downloads, and lightweight reproducibility.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Answers log")
        if os.path.exists(ANSWERS_CSV):
            dfA = pd.read_csv(ANSWERS_CSV)
            st.dataframe(dfA.tail(20), use_container_width=True)
            st.download_button("Download answers.csv", dfA.to_csv(index=False), "answers.csv", "text/csv")
        else:
            st.write("No answers yet.")
    with c2:
        st.subheader("Judge runs")
        if os.path.exists(JUDGE_CSV):
            dfJ = pd.read_csv(JUDGE_CSV)
            st.dataframe(dfJ.tail(20), use_container_width=True)
            st.download_button("Download judge_runs.csv", dfJ.to_csv(index=False), "judge_runs.csv", "text/csv")
        else:
            st.write("No judge runs yet.")

    st.divider()
    st.subheader("Config snapshot")
    cfg = {
        "index_label": INDEX_LABEL,
        "openai_model_default": "gpt-4o-mini",
        "local_hf_default": "google/flan-t5-small",
        "prompt_sha": sha256_text(read_demo_prompt()),
        "data_dir": os.path.abspath(DATA_DIR),
        "logs_dir": os.path.abspath(LOG_DIR),
    }
    st.json(cfg)

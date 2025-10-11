# --- Shared imports ---
import os, time, textwrap, hashlib, json
from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Optional deps (handle gracefully)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# --- Paths & tiny setup ---
DATA_DIR = "data"
LOG_DIR = os.path.join(DATA_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
ANSWERS_CSV = os.path.join(LOG_DIR, "answers.csv")
JUDGE_CSV   = os.path.join(LOG_DIR, "judge_runs.csv")

def _ensure_csv(path: str, header: List[str]):
    if not os.path.exists(path):
        pd.DataFrame(columns=header).to_csv(path, index=False)

_ensure_csv(ANSWERS_CSV, ["timestamp","backend","model","top_k","temperature",
                          "max_tokens","persona","prompt_sha","latency_ms",
                          "question","answer"])
_ensure_csv(JUDGE_CSV,   ["timestamp","backend","model","top_k","temperature",
                          "max_tokens","persona","prompt_sha","accuracy","safety",
                          "empathy","clarity","robustness","rationale","question"])

# --- Data model ---
@dataclass
class Chunk:
    id: str
    text: str
    url: str
    title: str
    section: str

def _demo_chunks() -> List[Chunk]:
    return [
        Chunk("c1","Monitoring daily weight helps detect fluid buildup early in heart failure.",
              "https://example.com/hf-basics","HF Basics","Self-care"),
        Chunk("c2","Limiting sodium and tracking fluids can reduce swelling and breathlessness.",
              "https://example.com/hf-diet","Diet & Lifestyle","Diet"),
        Chunk("c3","Call 911 for chest pain, fainting, or severe shortness of breath.",
              "https://example.com/hf-safety","When to Seek Care","Safety"),
    ]

def load_chunks_parquet(path=os.path.join(DATA_DIR, "chunks.parquet")) -> List[Chunk]:
    if not os.path.exists(path):
        return _demo_chunks()
    df = pd.read_parquet(path)
    req = {"id","text","url","title","section"}
    if not req.issubset(df.columns):
        return _demo_chunks()
    return [Chunk(str(r.id), str(r.text), str(r.url), str(r.title), str(r.section))
            for _, r in df.iterrows()]

@st.cache_resource(show_spinner=False)
def get_minilm_encoder():
    if SentenceTransformer is None:
        return None  # UI will warn and suggest: pip install sentence-transformers
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_texts(texts: List[str], encoder) -> np.ndarray:
    if encoder is None:
        raise RuntimeError("Embedding model not available. Install sentence-transformers.")
    vecs = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)

def retrieve_topk(query: str, chunks: List[Chunk], encoder, k: int = 3) -> Tuple[pd.DataFrame, List[Chunk]]:
    if "chunk_embeddings" not in st.session_state:
        st.session_state["chunk_embeddings"] = embed_texts([c.text for c in chunks], encoder)
    qv = embed_texts([query], encoder)[0:1]
    sims = cosine_similarity(qv, st.session_state["chunk_embeddings"])[0]
    idxs = np.argsort(-sims)[:k]
    sel = [chunks[i] for i in idxs]
    df = pd.DataFrame([{
        "score": float(sims[i]),
        "url": chunks[i].url,
        "title": chunks[i].title,
        "section": chunks[i].section,
        "preview": textwrap.shorten(chunks[i].text, 160)
    } for i in idxs])
    return df, sel

def build_flan_prompt(question: str, selected: List[Chunk], system_text: str, persona_note: str) -> str:
    sources = []
    for i, c in enumerate(selected, 1):
        sources.append(f"[{i}] {c.text}\n(Title: {c.title}, URL: {c.url})")
    context = "\n\n".join(sources)
    return (
        f"{system_text}\n"
        f"Persona hint: {persona_note}\n\n"
        "Instruction: Using ONLY the provided context, write a clear, empathetic answer. "
        "Cite sources inline as [1], [2], ... and list them at the end with titles and URLs. "
        "If the answer is not in the context, say you don't know and suggest safe next steps.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n"
    )

def chunks_to_citations(selected: List[Chunk]) -> List[Dict]:
    return [{"url": c.url, "title": c.title, "section": c.section} for c in selected]

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

# --- Judge-Lite (deterministic) ---
def judge_lite(question: str, answer: str, citations: List[Dict]) -> Dict:
    a = (answer or "").lower()
    accuracy  = 2 + (1 if citations else 0) + (1 if any(k in a for k in ["weight","sodium","fluid","medication","symptom"]) else 0)
    safety    = 4 - (1 if any(k in a for k in ["dosage","mg","take x pills"]) else 0)
    empathy   = 3 + (1 if any(k in a for k in ["i understand","i’m sorry","it’s common","you’re not alone"]) else 0)
    clarity   = 3 + (1 if any(ch in answer for ch in ["\n- ", "\n1.", "\n2.", "\n\n"]) else 0)
    robustness= 3 + (1 if "i don't know" in a or "not in the provided context" in a else 0)
    clip = lambda x: max(1, min(5, x))
    scores = dict(
        accuracy=clip(accuracy), safety=clip(safety), empathy=clip(empathy),
        clarity=clip(clarity), robustness=clip(robustness),
        rationale="Heuristic rubric: accuracy favors grounded HF terms & citations; safety penalizes dosage; empathy/clarity reward tone & structure."
    )
    return scores

# --- Logging ---
def log_answer_row(backend, model, k, temp, max_tokens, persona, prompt_sha, latency_ms, question, answer):
    row = dict(
        timestamp=datetime.utcnow().isoformat(timespec="seconds"),
        backend=backend, model=model, top_k=k, temperature=temp,
        max_tokens=max_tokens, persona=persona, prompt_sha=prompt_sha,
        latency_ms=latency_ms, question=question, answer=answer
    )
    pd.DataFrame([row]).to_csv(ANSWERS_CSV, mode="a", header=False, index=False)

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

# -------- OpenAI adapter (cloud generator) --------
class OpenAIAdapter:
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY","").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1")

    def generate(self, question: str, selected_chunks: List[Chunk],
                 system_text: str, persona_note: str,
                 temperature: float, max_tokens: int):
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY.")

        prompt = build_flan_prompt(question, selected_chunks, system_text, persona_note)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": [
                {"role":"system","content": system_text},
                {"role":"user","content": prompt}
            ]
        }
        t0 = time.time()
        resp = requests.post(f"{self.base_url}/chat/completions",
                             headers=headers, data=json.dumps(payload), timeout=60)
        latency = int((time.time()-t0)*1000)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI API error: {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        citations = chunks_to_citations(selected_chunks)
        return answer, citations, latency, prompt

# -------- Streamlit UI (OpenAI only) --------
st.set_page_config(page_title="HeartWise — Simple RAG (OpenAI)", layout="wide")
st.sidebar.title("Simple RAG (OpenAI)")

st.header("Simple RAG (OpenAI)")
api_key = st.text_input("OpenAI API Key", type="password")
model   = st.selectbox("Model", ["gpt-4o-mini","gpt-4o"], index=0)
top_k   = st.slider("Top-k", 1, 10, 3)
temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
max_tokens  = st.slider("Max tokens", 64, 1024, 384, 32)
persona = st.radio("Persona", ["Patient-friendly","Clinician-brief","Caregiver-supportive"], index=0)

question = st.text_area("Question", value="What daily habits help reduce heart failure hospitalizations?")
colA, colB = st.columns(2)
retrieve_btn = colA.button("Retrieve Evidence")
generate_btn = colB.button("Generate Answer (OpenAI)")

chunks = load_chunks_parquet()
encoder = get_minilm_encoder()
st.caption(f"Index size: ~{len(chunks)} chunks")

if retrieve_btn:
    if not encoder:
        st.error("Install sentence-transformers to enable retrieval.")
    else:
        df_topk, selected = retrieve_topk(question, chunks, encoder, k=top_k)
        st.session_state["selected_chunks_openai"] = selected
        st.subheader("Top-k Retrieved Evidence")
        st.dataframe(df_topk, use_container_width=True)

if generate_btn:
    if not api_key.strip():
        st.error("Missing OPENAI_API_KEY.")
    else:
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        selected = st.session_state.get("selected_chunks_openai")
        if not selected:
            st.warning("Retrieve evidence first.")
        else:
            system_text = (
                "You are HeartWise, a compassionate health educator focused on U.S. heart failure self-care.\n"
                "- Prioritize clarity, empathy, and safety. Avoid dosages and individualized medical advice.\n"
                "- Use ONLY the provided context. If info is missing, say you don't know and suggest safe next steps.\n"
                "- Cite sources as [1], [2], ... and list titles + URLs at the end under 'Sources'.\n"
            )
            adapter = OpenAIAdapter(model)
            try:
                answer, citations, latency, full_prompt = adapter.generate(
                    question, selected, system_text, persona, temperature, max_tokens
                )
                with st.expander("Prompt (debug)", expanded=False):
                    st.code(full_prompt[:2000])

                st.subheader("Answer (OpenAI)")
                st.write(answer)

                st.subheader("Citations")
                for i, c in enumerate(citations, 1):
                    st.markdown(f"**[{i}] {c['title']}**  \n{c['url']}")

                scores = judge_lite(question, answer, citations)
                st.subheader("Judge-Lite")
                st.write(scores)

                prompt_sha = sha256_text(full_prompt)
                log_answer_row("openai", model, top_k, temperature, max_tokens, persona, prompt_sha, latency, question, answer)
                log_judge_row("openai", model, top_k, temperature, max_tokens, persona, prompt_sha, question, scores)
                st.success(f"Run saved · {latency} ms")
            except Exception as e:
                st.error(str(e)) 

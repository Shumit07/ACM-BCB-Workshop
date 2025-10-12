# ================================
# HeartWise — Simple RAG (Local)
# ================================
# - CPU-friendly local FLAN-T5 generation
# - Optional GPU via Colab (FastAPI + ngrok) through RemoteHFAdapter
# - MiniLM retrieval + grounded prompt with citations
# - Judge-Lite + CSV logging for reproducibility
#
# Usage:
#   pip install -r requirements.txt
#   streamlit run app_local.py
# ================================

# --- Shared imports ---
import os, time, textwrap, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Optional deps (handle gracefully)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ========= Paths & setup =========
DATA_DIR  = "data"
LOG_DIR   = os.path.join(DATA_DIR, "logs")
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


# ========= Data model & retrieval =========
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


# ========= Judge-Lite & logging =========
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


# ========= Local HF adapter (robust) =========
class LocalHFAdapter:
    """
    Minimal, robust FLAN-T5 adapter for CPU.
    - Trims each chunk in the prompt (helps small models)
    - Retries once without [#] citations if first pass is too short
    - Cleans leading echoes like 'Answer:' / 'Persona:'
    """
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self._tok = None
        self._model = None
        self._ready = False

    def preload(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch, gc

        # free previous
        try:
            del self._model; del self._tok
        except Exception:
            pass
        gc.collect()

        try_names = [self.model_name]
        if self.model_name == "google/flan-t5-large":
            try_names.append("google/flan-t5-base")  # fallback

        last_err = None
        for name in try_names:
            try:
                self._tok = AutoTokenizer.from_pretrained(name, use_fast=True)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    name, low_cpu_mem_usage=True, use_safetensors=True, torch_dtype="auto"
                )
                self._model.eval()
                self.model_name = name
                self._ready = True
                return
            except Exception as e:
                last_err = e
                try:
                    del self._model; del self._tok
                except Exception:
                    pass
                gc.collect()
        raise RuntimeError(f"Failed to load {self.model_name}. Last error: {last_err}")

    # --- helper: robust chunk text ---
    def _chunk_text(self, c) -> str:
        for attr in ("text", "content", "body", "chunk"):
            if hasattr(c, attr):
                v = getattr(c, attr)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        if isinstance(c, dict):
            for k in ("text", "content", "body", "chunk"):
                v = c.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return str(c)

    def _build_prompt(self, question: str, selected, system_text: str, persona: str, require_citations=True):
        style_map = {
            "Patient-friendly": "friendly, plain-language, supportive",
            "Clinician-brief": "concise, clinical, bullet-style where helpful",
            "Caregiver-supportive": "empathetic, practical, plain-language",
        }
        style = style_map.get(persona, "friendly, plain-language")
        def _trim(txt: str, n=280): return (txt[:n] + "…") if len(txt) > n else txt

        evidence_lines = []
        for i, c in enumerate(selected):
            evidence_lines.append(f"[{i+1}] {_trim(self._chunk_text(c))}")
        evidence = "\n\n".join(evidence_lines)

        tail = (
            "Task: Answer the question for a layperson in a {style} tone. Use ONLY the Context; "
            "if info is missing, say you don't know. Include inline citations like [1], [2]."
            if require_citations else
            "Task: Answer the question for a layperson in a {style} tone. Use ONLY the Context; "
            "if info is missing, say you don't know. Do NOT include bracketed citation markers."
        ).format(style=style)

        return (
            f"{system_text}\n\n"
            f"Context:\n{evidence}\n\n"
            f"{tail}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

    def generate(self, question: str, selected, temperature: float = 0.3,
                 max_tokens: int = 384, persona: str = ""):
        if not self._ready:
            self.preload()

        def _run(prompt: str, temp: float, max_new: int):
            inputs = self._tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
            do_sample = float(temp) > 0.0
            t0 = time.time()
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=int(max_new),
                do_sample=do_sample,
                temperature=float(temp) if do_sample else None,
                num_beams=1,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.12,
            )
            latency_ms = int((time.time() - t0) * 1000)
            text = self._tok.decode(out_ids[0], skip_special_tokens=True).strip()
            return text, latency_ms

        # Pass 1 — require citations
        prompt = self._build_prompt(question, selected, system_text="", persona=persona, require_citations=True)
        text, latency_ms = _run(prompt, temperature, max_tokens)

        # Retry if too short / only "[1]"
        looks_like_marker = text.strip().startswith("[") and text.strip().endswith("]") and len(text.strip()) <= 5
        if looks_like_marker or len(text) < 50:
            prompt2 = self._build_prompt(question, selected, system_text="", persona=persona, require_citations=False)
            text2, latency2 = _run(prompt2, max(temperature, 0.25), max(max_tokens, 384))
            if len(text2) > len(text):
                text, prompt, latency_ms = text2, prompt2, latency2

        # Clean leading echoes
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines and lines[0].lower().startswith(("persona:", "instruction:", "answer:")):
            lines[0] = lines[0].split(":", 1)[-1].strip()
        text = "\n".join(lines).strip()

        citations = chunks_to_citations(selected)
        return text, citations, latency_ms, prompt


# ========= Remote Colab adapter (optional GPU) =========
# Expect a FastAPI server in Colab exposing POST /generate
# Body: {"prompt": str, "max_new_tokens": int, "temperature": float}
import requests as _rq

class RemoteHFAdapter:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        # reuse local prompt builder for consistency
        self._prompt_builder = LocalHFAdapter()._build_prompt

    def generate(self, question, selected, system_text="", persona="", temperature=0.4, max_tokens=512):
        prompt = self._prompt_builder(question, selected, system_text=system_text, persona=persona, require_citations=True)
        r = _rq.post(
            f"{self.base_url}/generate",
            json={"prompt": prompt, "max_new_tokens": int(max_tokens), "temperature": float(temperature)},
            timeout=120
        )
        r.raise_for_status()
        text = (r.json() or {}).get("text", "").strip()
        if "Answer:" in text:
            text = text.split("Answer:", 1)[-1].strip()
        return text, chunks_to_citations(selected), 0, prompt


# ========= Streamlit UI =========
def ui_local_track():
    st.header("Simple RAG (Local / CPU or Colab GPU)")

    # Controls
    hf_model = st.selectbox("Local model", ["google/flan-t5-small","google/flan-t5-base","google/flan-t5-large"], index=1)
    col_p = st.columns(2)
    preload = col_p[0].button("Preload Local Model")
    use_gpu = col_p[1].checkbox("Use Colab GPU endpoint")
    col_url = st.columns([2,1])
    colab_url = col_url[0].text_input("Colab URL (from ngrok)", placeholder="https://xxxx.ngrok-free.app", value="")
    top_k   = st.slider("Top-k", 1, 10, 3)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    max_tokens  = st.slider("Max tokens", 64, 1024, 384, 32)
    persona = st.radio("Persona", ["Patient-friendly","Clinician-brief","Caregiver-supportive"], index=0)
    question = st.text_area("Question (Local)", value="What can patients do to reduce heart failure hospitalizations?")

    colA, colB = st.columns(2)
    retrieve_btn = colA.button("Retrieve Evidence")
    generate_btn = colB.button("Generate Answer")

    # Data
    chunks = load_chunks_parquet()
    encoder = get_minilm_encoder()
    st.caption(f"Index size: ~{len(chunks)} chunks")

    # Keep adapters in session
    if preload:
        try:
            st.session_state["local_hf"] = LocalHFAdapter(hf_model)
            st.session_state["local_hf"].preload()
            st.success(f"Model ready: {st.session_state['local_hf'].model_name}")
        except Exception as e:
            st.error(str(e))
            st.info("If missing deps: pip install transformers torch accelerate safetensors --upgrade")

    if retrieve_btn:
        if not encoder:
            st.error("Install sentence-transformers to enable retrieval.")
        else:
            df_topk, selected = retrieve_topk(question, chunks, encoder, k=top_k)
            st.session_state["selected_chunks_local"] = selected
            st.subheader("Top-k Retrieved Evidence (Local)")
            st.dataframe(df_topk, use_container_width=True)

    if generate_btn:
        selected = st.session_state.get("selected_chunks_local")
        if not selected:
            st.warning("Retrieve evidence first.")
            return

        try:
            if use_gpu and colab_url.strip():
                adapter = RemoteHFAdapter(colab_url.strip())
                answer, citations, latency, _ = adapter.generate(
                    question, selected, system_text="", persona=persona, temperature=temperature, max_tokens=max_tokens
                )
                backend = "remote_gpu"
                model_name = "colab_endpoint"
            else:
                # local
                if "local_hf" not in st.session_state or st.session_state["local_hf"].model_name != hf_model:
                    st.session_state["local_hf"] = LocalHFAdapter(hf_model)
                adapter = st.session_state["local_hf"]
                adapter.preload()  # lazy safe
                answer, citations, latency, _ = adapter.generate(
                    question, selected, temperature=temperature, max_tokens=max_tokens, persona=persona
                )
                backend = "local_hf"
                model_name = adapter.model_name

            st.subheader("Answer")
            st.write(answer)
            if backend == "local_hf":
                st.caption(f"Local model: {model_name} · Latency: {latency} ms")
            else:
                st.caption(f"Remote GPU via Colab: {colab_url.strip()}")

            st.subheader("Citations")
            for i, c in enumerate(citations, 1):
                st.markdown(f"**[{i}] {c['title']}**  \n{c['url']}")

            scores = judge_lite(question, answer, citations)
            st.subheader("Judge-Lite")
            st.write(scores)

            prompt_sha = sha256_text(answer[:200])  # local prompt omitted
            log_answer_row(backend, model_name, top_k, temperature, max_tokens, persona, prompt_sha, latency, question, answer)
            log_judge_row(backend, model_name, top_k, temperature, max_tokens, persona, prompt_sha, question, scores)
            st.success("Run saved")
        except Exception as e:
            st.error(str(e))


# ========= App shell =========
st.set_page_config(page_title="HeartWise — Simple RAG (Local)", layout="wide")
st.sidebar.title("Simple RAG — Local / GPU")
ui_local_track()




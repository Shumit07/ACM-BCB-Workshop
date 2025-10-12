# --- HeartWise Simple RAG (Local CPU or Remote Colab) ---
# Single-file Streamlit app with:
#  - Shared RAG core (chunks -> MiniLM -> retrieve -> prompt)
#  - Local CPU adapter (FLAN T5 small/base/large) with improved prompting
#  - Remote Colab adapter (call a small instruct model hosted on Colab/GPU)
#  - Judge-Lite + CSV logging

import os, time, textwrap, hashlib
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

# ---------------------- Paths & tiny setup ----------------------
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

# ---------------------- Data model & loading --------------------
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
    try:
        df = pd.read_parquet(path)
    except Exception:
        return _demo_chunks()
    req = {"id","text","url","title","section"}
    if not req.issubset(df.columns):
        return _demo_chunks()
    return [Chunk(str(r.id), str(r.text), str(r.url), str(r.title), str(r.section))
            for _, r in df.iterrows()]

# ---------------------- Embeddings & retrieval ------------------
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

def retrieve_topk(query: str, chunks: List[Chunk], encoder, k: int = 4) -> Tuple[pd.DataFrame, List[Chunk]]:
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

# -------------------------- Judge-Lite --------------------------
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

# ------------------------ Local HF adapter ----------------------
class LocalHFAdapter:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self._tok = None
        self._model = None
        self._ready = False

    def preload(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch, gc
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
                    name, low_cpu_mem_usage=True, use_safetensors=True, torch_dtype=torch.float32
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

    # Improved prompt to encourage longer, practical answers
    def _build_prompt(self, question: str, selected, system_text: str, persona: str, require_citations=True):
        style_map = {
            "Patient-friendly": "friendly, plain-language, supportive",
            "Clinician-brief": "concise, clinical, bullet-style where helpful",
            "Caregiver-supportive": "empathetic, practical, plain-language",
        }
        style = style_map.get(persona, "friendly, plain-language")
        def _trim(txt: str, n=280): return (txt[:n] + "…") if len(txt) > n else txt
        evidence_lines = [f"[{i+1}] {_trim(c.text)}" for i, c in enumerate(selected)]
        evidence = "\n\n".join(evidence_lines)

        if require_citations:
            tail = (
                "Task: Write a clear, supportive answer for a layperson in a {style} tone.\n"
                "- Use ONLY the Context; if info is missing, say you don't know.\n"
                "- DO NOT advise calling emergency services unless the Context explicitly lists red-flag symptoms.\n"
                "- Aim for 6–10 sentences and add a short bullet list of 3–5 practical actions.\n"
                "- Cite evidence inline like [1], [2]."
            ).format(style=style)
        else:
            tail = (
                "Task: Write a clear, supportive answer for a layperson in a {style} tone.\n"
                "- Use ONLY the Context; if info is missing, say you don't know.\n"
                "- Aim for 6–10 sentences and add a short bullet list of 3–5 practical actions."
            ).format(style=style)

        return f"{system_text}\n\nContext:\n{evidence}\n\n{tail}\n\nQuestion: {question}\n\nAnswer:"

    def generate(self, question: str, selected, temperature: float = 0.3,
                 max_tokens: int = 512, persona: str = ""):
        if not self._ready:
            self.preload()

        prompt1 = self._build_prompt(question, selected, system_text="", persona=persona, require_citations=True)
        inputs = self._tok(prompt1, return_tensors="pt", truncation=True, max_length=1024)

        # Longer, clearer outputs: sampling when temp>0; beams when temp==0
        gen_kwargs = dict(
            max_new_tokens=int(max_tokens),
            no_repeat_ngram_size=3,
            repetition_penalty=1.12,
            early_stopping=True,
        )
        if temperature > 0:
            gen_kwargs.update(dict(do_sample=True, temperature=float(temperature), top_p=0.9, typical_p=0.95, num_beams=1))
        else:
            gen_kwargs.update(dict(do_sample=False, num_beams=4))

        t0 = time.time()
        gen_ids = self._model.generate(**inputs, **gen_kwargs)
        latency_ms = int((time.time() - t0) * 1000)
        text = self._tok.decode(gen_ids[0], skip_special_tokens=True).strip()

        # Retry without bracket citations if too short
        looks_like_marker = text.strip().startswith("[") and text.strip().endswith("]") and len(text.strip()) <= 5
        if looks_like_marker or len(text) < 80:
            prompt2 = self._build_prompt(question, selected, system_text="", persona=persona, require_citations=False)
            inputs2 = self._tok(prompt2, return_tensors="pt", truncation=True, max_length=1024)
            gen_ids2 = self._model.generate(**inputs2, **gen_kwargs)
            text2 = self._tok.decode(gen_ids2[0], skip_special_tokens=True).strip()
            if len(text2) > len(text):
                text = text2

        citations = chunks_to_citations(selected)
        return text, citations, latency_ms, "(local prompt elided)"

# ----------------------- Remote Colab adapter -------------------
class RemoteHFAdapter:
    """
    Expects a FastAPI server in Colab exposing POST /generate with fields:
    { "prompt": str, "max_new_tokens": int, "temperature": float }
    """
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._prompt_builder = LocalHFAdapter()._build_prompt  # reuse same prompt style

    def generate(self, question, selected, temperature=0.4, max_tokens=512, persona=""):
        prompt = self._prompt_builder(question, selected, system_text="", persona=persona, require_citations=True)
        r = requests.post(
            f"{self.base_url}/generate",
            json={"prompt": prompt, "max_new_tokens": int(max_tokens), "temperature": float(temperature)},
            timeout=120
        )
        r.raise_for_status()
        text = r.json().get("text", "").strip()
        if "Answer:" in text:
            text = text.split("Answer:", 1)[-1].strip()
        citations = chunks_to_citations(selected)
        return text, citations, 0, "(remote colab prompt)"

# ------------------------------ UI ------------------------------
def ui_local_track():
    st.header("Simple RAG (Local CPU or Remote Colab)")

    run_mode = st.radio("Backend", ["Local CPU", "Remote Colab"], index=0, horizontal=True)

    if run_mode == "Local CPU":
        hf_model = st.selectbox(
            "Local model",
            ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"],
            index=1
        )
        preload = st.button("Preload Models")
    else:
        colab_url = st.text_input("Colab URL (e.g., https://xxxx.trycloudflare.com)", value="")
        hf_model = None
        preload = False

    top_k      = st.slider("Top-k", 1, 10, 4)
    temperature= st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    max_tokens = st.slider("Max tokens", 128, 1024, 512, 32)
    persona    = st.radio("Persona", ["Patient-friendly","Clinician-brief","Caregiver-supportive"], index=0)
    question   = st.text_area("Question", value="What can patients do to reduce heart failure hospitalizations?")

    colA, colB = st.columns(2)
    retrieve_btn = colA.button("Retrieve Evidence")
    generate_btn = colB.button("Generate Answer")

    chunks = load_chunks_parquet()
    encoder = get_minilm_encoder()
    st.caption(f"Index size: ~{len(chunks)} chunks")

    # preload local if requested
    if run_mode == "Local CPU" and preload:
        try:
            st.session_state["local_hf"] = LocalHFAdapter(hf_model)
            st.session_state["local_hf"].preload()
            st.success(f"Model ready: {st.session_state['local_hf'].model_name}")
        except Exception as e:
            st.error(str(e))
            st.info("If missing deps: pip install transformers torch accelerate safetensors --upgrade")

    # retrieval
    if retrieve_btn:
        if not encoder:
            st.error("Install sentence-transformers to enable retrieval.")
        else:
            df_topk, selected = retrieve_topk(question, chunks, encoder, k=top_k)
            st.session_state["selected_chunks"] = selected
            st.subheader("Top-k Retrieved Evidence")
            st.dataframe(df_topk, use_container_width=True)

    # generation
    if generate_btn:
        selected = st.session_state.get("selected_chunks")
        if not selected:
            st.warning("Retrieve evidence first.")
            return
        try:
            if run_mode == "Local CPU":
                if "local_hf" not in st.session_state:
                    st.session_state["local_hf"] = LocalHFAdapter(hf_model)
                adapter = st.session_state["local_hf"]
                adapter.preload()
                backend_key, model_label = "local_hf", adapter.model_name
                answer, citations, latency, _ = adapter.generate(
                    question, selected, temperature=temperature, max_tokens=max_tokens, persona=persona
                )
            else:
                if not colab_url:
                    st.error("Enter your Colab public URL.")
                    return
                adapter = RemoteHFAdapter(colab_url)
                backend_key, model_label = "remote_colab", "remote"
                answer, citations, latency, _ = adapter.generate(
                    question, selected, temperature=temperature, max_tokens=max_tokens, persona=persona
                )

            st.subheader("Answer")
            st.write(answer)
            if latency:
                st.caption(f"Latency: {latency} ms")
            st.subheader("Citations")
            for i, c in enumerate(citations, 1):
                st.markdown(f"**[{i}] {c['title']}**  \n{c['url']}")

            scores = judge_lite(question, answer, citations)
            st.subheader("Judge-Lite")
            st.write(scores)

            prompt_sha = sha256_text(answer[:200])
            log_answer_row(backend_key, model_label, top_k, temperature, max_tokens, persona, prompt_sha, latency, question, answer)
            log_judge_row(backend_key, model_label, top_k, temperature, max_tokens, persona, prompt_sha, question, scores)
            st.success("Run saved")
        except Exception as e:
            st.error(str(e))

# --------------------------- App shell --------------------------
st.set_page_config(page_title="HeartWise — Simple RAG (Local/Colab)", layout="wide")
st.sidebar.title("Simple RAG (Local or Remote Colab)")
ui_local_track()



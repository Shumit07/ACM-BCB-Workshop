📌 Overview

This repository provides a teaching demo and workshop scaffold for exploring Retrieval-Augmented Generation (RAG), lightweight judge evaluation, and agentic extensions for patient-support systems.

The current codebase focuses on:

A Streamlit app with six interactive tabs.

Demonstrations of web scraping, retrieval, and local vs API-based generation.

Logging and evaluation components for reproducibility.

It is not a production system; rather, it is a hands-on learning environment for building toward more autonomous “agentic” healthcare AI.

🗂️ App Structure (6 Tabs)

💬 HeartWise (Demo)
Live demo of the HeartWise assistant via AnythingLLM API.

🕸️ Web Scrape (Example)
Simple, respectful single-page fetch to show how raw context can be ingested.

🧩 Simple RAG (OpenAI)
Reproducible RAG pipeline using OpenAI Chat Completions.

🧩 Simple RAG (Local / Hugging Face)
Offline RAG pipeline running on CPU (MiniLM retrieval + FLAN-T5 generation).

🔬 Explore (Lab)
Interactive playground for personas, retrieval depth, and prompt variations.

🗂️ Ops & Logs
Lightweight reproducibility: CSV logs for answers and judge scores.

🧭 HeartWise Workshop System Flow
[START: User launches WorkShop.py]

│
▼
[App Boot]
│
└─── 1️⃣ Initialize Environment
  ├── Loads .env for API keys 🔐
  ├── Creates folders: data/, data/logs/, data/scrapes/, prompts/ 📁
  ├── Writes default heartwise_demo_prompt.txt 🧾
  └── Creates CSV logs if missing (answers.csv, judge_runs.csv) 🪶

[STREAMLIT UI SETUP]

│
└─── 2️⃣ Render Sidebar
  ├── Shows “Crisis Banner” (Safety Disclaimer) ⚠️
  ├── Displays file paths (Data, Logs, Scrapes) 📂
  └── Mentions .env usage for keys 🗝️

[TAB STRUCTURE INITIALIZED]

│
└─── 3️⃣ App Tabs Created
  - Tab 1: HeartWise (Demo)
  - Tab 2: Web Scrape (Example)
  - Tab 3: Simple RAG (OpenAI)
  - Tab 4: Simple RAG (Local HF)
  - Tab 5: Explore (Teaching Lab)
  - Tab 6: Ops & Logs

🧩 TAB 1 — HeartWise (Live Demo)

│
└─── 4️⃣ When User Clicks “Answer + Judge”
  ├── Calls _call_anythingllm(question) 🌐
  │ ├── Reads API endpoint + token from .env
  │ ├── POSTs to AnythingLLM API
  │ └── Returns JSON + latency (or error)
  │
  ├── Parses answer + citations (handles many JSON shapes) 🔍
  ├── Renders results on screen
  ├── Runs Judge-Lite (5 scores: accuracy, safety, empathy, clarity, robustness) 🎯
  └── Logs run → answers.csv and judge_runs.csv 📊

✅ Purpose: Illustrate what a production system looks like, not what we’ll build today.

🌐 TAB 2 — Web Scrape (Example)

│
└─── 5️⃣ When User Clicks “Plan Scrape” or “Run Scrape”
  ├── Plan Scrape: HEAD request to check page status + size
  ├── Run Scrape:
  │ ├── GET HTML → BeautifulSoup cleans it
  │ ├── Removes <script> + <style>
  │ ├── Extracts visible text 🧾
  │ ├── Saves Markdown snapshot → data/scrapes/ 📑
  │ └── Builds scrape_report.csv for reference
  └── Shows results + download button

✅ Purpose: Demonstrate responsible, single-page data ingestion for later RAG use.

🧠 TAB 3 — Simple RAG (OpenAI)

│
└─── 6️⃣ Workflow: Cloud RAG Pipeline
  ├── User enters API key + question
  ├── load_chunks_parquet() loads local knowledge base 📚
  ├── get_minilm_encoder() loads semantic model 🧩
  ├── retrieve_topk(query, chunks, encoder, k) finds relevant docs 🔍
  │ └── Displays Top-k evidence table
  │
  ├── Generate Answer:
  │ ├── Builds grounded prompt (context + citations)
  │ ├── OpenAIAdapter.generate(...) calls OpenAI Chat API
  │ └── Returns (answer, citations, latency, prompt)
  │
  ├── Renders answer + citations
  ├── Runs Judge-Lite scoring 🧮
  └── Logs run → CSVs ✅

✅ Purpose: Teach the core RAG pipeline — retrieval → prompt → generation → evaluation → logging.

💻 TAB 4 — Simple RAG (Local / Hugging Face)

│
└─── 7️⃣ Workflow: Offline RAG Pipeline
  ├── Choose model (flan-t5-base) + preload ⚙️
  │ └── LocalHFAdapter.preload() loads model + tokenizer on CPU
  │
  ├── “Retrieve Evidence” → same retrieve_topk() flow
  ├── “Generate Answer (Local)” → LocalHFAdapter.generate(...)
  │ ├── Builds prompt with/without citations
  │ ├── Generates text locally (no API)
  │ └── Handles retry if output too short
  │
  ├── Displays answer + citations
  ├── Judge-Lite → evaluate response
  └── Logs run (backend = “local_hf”)

✅ Purpose: Show offline privacy-friendly version of same RAG loop.

🧪 TAB 5 — Explore (Lab)

│
└─── 8️⃣ Interactive Experimentation
  ├── Lets user switch backend (OpenAI / Local)
  ├── Edit system prompt live 🧠
  ├── Adjust top-k, temperature, max tokens, persona
  ├── Retrieve → Generate → Judge using chosen backend
  └── Logs every run

✅ Purpose: Hands-on space to tweak parameters and observe RAG behavior.

🗂️ TAB 6 — Ops & Logs

│
└─── 9️⃣ After Session
  ├── Reads last 20 runs from answers.csv, judge_runs.csv 📊
  ├── Displays + allows downloads
  └── Shows config snapshot (model, paths, defaults)

✅ Purpose: Reinforce reproducibility & transparency.

[END: User completes workshop]

  ▼
✅ Outcome Summary

Students understand each stage of the RAG loop:
Retrieve → Ground → Generate → Evaluate → Log

They can run both cloud and local versions.

They’ve seen a small, safe scraping example.

They know how the app self-initializes and manages state.

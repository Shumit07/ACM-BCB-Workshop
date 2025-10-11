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

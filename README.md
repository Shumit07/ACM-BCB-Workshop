💡 HeartWise Simple-RAG Workshop
Educational Demonstration of Retrieval-Augmented Generation (RAG)

The HeartWise Simple-RAG Workshop introduces attendees to the core principles of Retrieval-Augmented Generation — the architecture behind explainable, context-aware large language models.
Rather than focusing on agentic automation, this version isolates the foundation: how LLMs retrieve, ground, and generate safe, evidence-based answers.

The workshop demonstrates two independent but parallel systems:

Path A — Cloud RAG (OpenAI): Uses the OpenAI API for inference.

Path B — Local RAG (CPU / GPU): Runs Hugging Face models locally or through a Colab GPU endpoint for those without API access.

Together, these pipelines form the educational backbone of the full HeartWise Agentic system — showing how retrieval, grounding, and reasoning interact before multi-agent layers are added.

🧠 Conceptual Overview

The Simple-RAG framework replicates the cognitive workflow of a responsible AI assistant:

Context Retrieval:
The system identifies and extracts the most relevant text segments (“chunks”) from a small curated dataset on heart-failure self-care.

Grounded Prompt Construction:
Retrieved evidence is formatted into a concise context window, ensuring the model’s output remains fact-anchored.

Controlled Generation:
The LLM (either OpenAI GPT or FLAN-T5) answers strictly within the retrieved context.
It cites sources inline ([1], [2], …) and refuses to speculate.

Transparent Evaluation:
Responses can be optionally scored using a heuristic “Judge-Lite” rubric (accuracy, safety, empathy, clarity, robustness).

## ⚙️ System Architecture

The diagram below shows the dual-path RAG workflow used in the workshop:


<img width="606" height="190" alt="image" src="https://github.com/user-attachments/assets/3fe8848d-1682-463c-ac66-122b72d1edd8" />


🧩 Dual-Path Demonstration
☁️ Path A — OpenAI RAG

For attendees with an OpenAI API Key

Retrieval → Prompt → Generation loop using gpt-4o-mini.

Demonstrates high-quality cloud inference with reproducible logging.

Focus: API integration, latency awareness, citation control.

💻 Path B — Local RAG (CPU / GPU / Colab)

For attendees without an API key

Uses flan-t5-base (or flan-t5-large) via Hugging Face Transformers.

Can optionally connect to a Colab GPU endpoint (RemoteHFAdapter).

Focus: privacy, transparency, model introspection.

☁️ Running the GPU (Colab) Version

If your local computer doesn’t have enough CPU power to run larger models, you can offload generation to a Colab GPU using ngrok and FastAPI.
This allows you to compare CPU vs GPU performance live during the workshop.

🧩 1️⃣ Open a New Google Colab Notebook

Visit Google Colab
 → click New Notebook

Copy the Colab code block from this repository (e.g., colab_server.py)

Paste it into a new Colab cell

⚙️ 2️⃣ Install Dependencies

Run the following cell first:

!pip -q install fastapi uvicorn transformers accelerate safetensors pyngrok --upgrade

🔐 3️⃣ Create a Free ngrok Account

Go to https://dashboard.ngrok.com/signup

After verifying your email, click
Getting Started → Your Authtoken

Copy your personal token — it looks like this:

2oxvxxxxxxxxxxxxxxxxxxxxxx_QwPxxxxxxxxxxxxxxxxxxxxx

🪄 4️⃣ Add the ngrok Token in Colab

In a new cell before running the FastAPI server, paste:

from pyngrok import ngrok
ngrok.set_auth_token("YOUR_AUTHTOKEN_HERE")

🚀 5️⃣ Start the FastAPI Server

Run the main Colab block:

!python colab_server.py


You should see output similar to:

🌐 Public URL: https://abcd1234.ngrok-free.app


Copy this public URL — it’s your live GPU endpoint.

💻 6️⃣ Connect Streamlit → Colab GPU

In your local app_local.py (Tab 4 of the app):

Paste the copied ngrok URL into the Colab URL input box

Choose the GPU option in the interface

Run your query — it will now use the remote GPU model

🎓 Teaching Objectives

By the end of the workshop, participants will:

Understand how retrieval reduces hallucination risk.

Compare cloud vs local model behavior on identical prompts.

Inspect how context size, temperature, and top-k affect factual grounding.

Recognize how this modular RAG block feeds into more advanced agentic systems.

🧩 Example Prompts

“What daily habits help reduce heart-failure hospitalizations?”

“Which symptoms indicate fluid retention in heart-failure patients?”

“When should a person with heart failure seek emergency care?”

⚖️ Ethical Framing

HeartWise Simple-RAG is an educational prototype.
It does not provide medical advice, diagnosis, or treatment.
Its purpose is to teach responsible AI design, transparency, and data stewardship within health-education contexts.

| Component                | Purpose                                      |
| ------------------------ | -------------------------------------------- |
| **Retriever**            | Encodes and ranks context snippets (MiniLM)  |
| **Generator**            | Produces grounded answer (OpenAI / FLAN-T5)  |
| **Evaluator (Optional)** | Heuristically scores answer quality          |
| **Interface**            | Streamlit UI for reproducible teaching demos |


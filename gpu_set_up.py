# ===============================
# 🚀 Colab GPU FastAPI Server 
# ===============================

!pip -q install fastapi uvicorn transformers accelerate safetensors pyngrok --upgrade

# ---- SETUP NGROK AUTHTOKEN ----
from pyngrok import ngrok, conf

# 1️⃣ Paste your ngrok authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTHTOKEN = "30wKcMAKMUhINpARmqC1GGzHzUG_6WsybuHJHi3PFgn7zDm71"

conf.get_default().auth_token = NGROK_AUTHTOKEN

# ---- IMPORTS ----
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Pick a small, capable instruct model (causal LM)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # or "microsoft/Phi-3-mini-4k-instruct"

# ---- LOAD MODEL ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

# ---- GENERATION FUNCTION ----
def generate_text(prompt: str, max_new_tokens: int = 512, temperature: float = 0.4):
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful, careful health educator."},
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out_ids = model.generate(
            inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0,
            temperature=float(temperature) if temperature > 0 else None,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if "Answer:" in text:
        text = text.split("Answer:", 1)[-1].strip()
    return text.strip()

# ---- FASTAPI SERVER ----
app = FastAPI()

class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.4

@app.post("/generate")
def generate(req: GenRequest):
    text = generate_text(req.prompt, req.max_new_tokens, req.temperature)
    return {"text": text}

# ---- START NGROK TUNNEL ----
public_url = ngrok.connect(8000, "http")
print("🌐 Public URL:", public_url.public_url)

# ---- RUN SERVER ----
import nest_asyncio, uvicorn, threading, time
nest_asyncio.apply()

def _run():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

threading.Thread(target=_run, daemon=True).start()
time.sleep(2)
print("✅ Uvicorn running on 0.0.0.0:8000 (serving /generate)")

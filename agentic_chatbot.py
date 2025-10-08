# C:/Users/htripathi/PycharmProjects/ACM_Agentic/agentic_chatbot.py
import requests
import json
import os
# Import the high-risk action logic to be called from the chatbot
from High_Risk import High_Risk_Patient_Action


def get_chatbot_response(patient_id: str, summary: str, chat_history: list, user_prompt: str, api_key: str, risk_status: str, retrieved_chunks: list = None) -> str:
    """
    Generates a response from the OpenAI API based on patient context and chat history.

    Args:
        patient_id: The ID of the patient.
        summary: The clinical summary for the patient.
        chat_history: A list of previous chat messages (e.g., [{"role": "user", "content": "..."}, ...]).
        user_prompt: The new message from the user.
        api_key: The user's OpenAI API key.
        risk_status: The patient's risk stratification ('high', 'moderate', 'low').
        retrieved_chunks: A list of Chunk objects retrieved based on the user's prompt.

    Returns:
        The assistant's response as a string.
    """
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = "gpt-4o-mini"

    # --- Intent Detection for "Take Action" ---
    action_keywords = ["take action", "do something", "help me", "send a notification", "contact my doctor"]
    prompt_lower = user_prompt.lower()
    if any(keyword in prompt_lower for keyword in action_keywords):
        if risk_status == 'high':
            # For high-risk patients, execute the action and return the result.
            action_message, er_locations = High_Risk_Patient_Action(patient_id)
            response_data = {"message": f"I have understood your request and taken action on your behalf. Here is the outcome:\n\n> {action_message}", "locations": er_locations}
            return response_data
        elif risk_status in ['low', 'moderate']:
            # For low/moderate risk, provide a reassuring, canned response.
            return f"I understand your concern. However, your current risk level is **{risk_status}**. The automated action protocol is reserved for high-risk situations to ensure urgent matters are prioritized. I recommend you continue to follow your care plan and discuss any non-urgent concerns with your doctor during your next visit."
        else:
            return "I cannot take action because the patient's risk status is unknown. Please ensure the risk stratification is correctly recorded."

    # Format the retrieved chunks if they exist
    evidence_context = "No additional evidence was retrieved for this query."
    if retrieved_chunks:
        sources = []
        for i, c in enumerate(retrieved_chunks, 1):
            sources.append(f"[{i}] {c.text}")
        evidence_context = "\n\n".join(sources)

    # Construct the system prompt with the patient's context
    system_prompt = (
        "You are HeartWise, a compassionate AI assistant for heart health education. "
        "You are speaking to a patient. Use their clinical summary, the retrieved evidence, and the ongoing conversation for context. "
        "Provide supportive, educational, and safe information. Do NOT provide medical advice, diagnoses, or dosage changes. "
        "If you use information from the retrieved evidence, you MUST cite it using the format [1], [2], etc. "
        "Encourage them to speak with their doctor for any medical concerns.\n\n"
        "--- PATIENT CONTEXT ---\n"
        f"Patient ID: {patient_id}\n"
        f"Clinical Summary: {summary}\n"
        "--- END PATIENT CONTEXT ---\n\n"
        "--- RETRIEVED EVIDENCE ---\n"
        f"{evidence_context}\n"
        "--- END RETRIEVED EVIDENCE ---"
    )

    # Prepare messages for the API call
    api_messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_prompt}]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": api_messages,
        "temperature": 0.5,
        "max_tokens": 250,
    }

    try:
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        full_response = resp.json()["choices"][0]["message"]["content"]
        return full_response

    except requests.exceptions.RequestException as e:
        # Handle network errors, timeouts, etc.
        return f"I'm sorry, I encountered a network error: {e}"
    except Exception as e:
        # Handle other errors, like API key issues or malformed requests
        return f"I'm sorry, I encountered an unexpected error: {e}"
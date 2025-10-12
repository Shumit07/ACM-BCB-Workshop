import requests
import json
import os
# Import the high-risk action logic to be called from the chatbot
from High_Risk import High_Risk_Patient_Action


def get_chatbot_response(patient_id: str, summary: str, chat_history: list, user_prompt: str, api_key: str,
                         risk_status: str, retrieved_chunks: list = None) -> str:
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
        The assistant's response as a string or dict.
    """
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = "gpt-4o-mini"

    # --- AI-Powered Intent Detection for "Take Action" ---
    # First, ask the AI if this message requires immediate action
    intent_check_prompt = f"""Based on the patient's message below, determine if they are requesting urgent medical action (such as contacting their doctor, sending notifications, or getting emergency help).

Patient Risk Level: {risk_status}
Patient Message: "{user_prompt}"

Respond with ONLY 'YES' if the patient is explicitly requesting action or help that requires intervention (like contacting their doctor, getting emergency services, or sending notifications).
Respond with ONLY 'NO' if they are asking informational questions, seeking general advice, or having a casual conversation.

Answer (YES or NO):"""

    try:
        intent_headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        intent_payload = {
            "model": model,
            "messages": [{"role": "user", "content": intent_check_prompt}],
            "temperature": 0.0,
            "max_tokens": 10,
        }

        intent_resp = requests.post(f"{base_url}/chat/completions", headers=intent_headers,
                                    data=json.dumps(intent_payload), timeout=30)
        intent_resp.raise_for_status()
        intent_decision = intent_resp.json()["choices"][0]["message"]["content"].strip().upper()

        # Check if the AI determined action is needed
        if "YES" in intent_decision:
            if risk_status == 'high':
                # For high-risk patients, execute the action and return the result.
                action_message, er_locations = High_Risk_Patient_Action(patient_id, api_key)
                response_data = {
                    "message": f"I have understood your request and taken action on your behalf. Here is the outcome:\n\n> {action_message}",
                    "locations": er_locations}
                return response_data
            elif risk_status in ['low', 'moderate']:
                # For low/moderate risk, provide a reassuring, canned response.
                return f"I understand your concern. However, your current risk level is **{risk_status}**. The automated action protocol is reserved for high-risk situations to ensure urgent matters are prioritized. I recommend you continue to follow your care plan and discuss any non-urgent concerns with your doctor during your next visit."
            else:
                return "I cannot take action because the patient's risk status is unknown. Please ensure the risk stratification is correctly recorded."

    except Exception as e:
        # If intent detection fails, log it and continue with normal response
        print(f"Intent detection failed: {e}. Proceeding with normal conversation.")

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

    # Prepare messages for the API call - Filter out sources from chat history
    filtered_chat_history = []
    for msg in chat_history:
        filtered_msg = {"role": msg["role"], "content": msg["content"]}
        filtered_chat_history.append(filtered_msg)

    api_messages = [{"role": "system", "content": system_prompt}] + filtered_chat_history + [
        {"role": "user", "content": user_prompt}]

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

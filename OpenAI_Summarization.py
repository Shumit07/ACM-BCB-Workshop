from openai import OpenAI
import math

OPENAI_API_KEY = r""

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"OpenAI client could not be initialized. Is the API key valid? Error: {e}")
    client = None


def Summary_Email(patient_data: dict) -> tuple:

    if not isinstance(patient_data, dict):
        return ("Error: Input must be a patient data dictionary.", None, None)

    if not client:
        return ("OpenAI client is not initialized.", None, None)

    # --- Part 1: Generate Clinical Summary (Live OpenAI Call) ---
    chat_log = patient_data.get("Chat Log", "")
    existing_summary = patient_data.get("Summarization", "")
    combined_text = f"Previous Summary:\n{existing_summary}\n\nFull Chat Log:\n{chat_log}"

    prompt = f"""
    Compose a clinical summary of the following patient data, written as a single, narrative paragraph 
    suitable for a doctor's review. The summary must be concise and integrate all key information 
    such as symptoms, history, risks, and the action plan. 
    Do not use any markdown formatting (like bolding, asterisks, or bullet points).

    ---
    PATIENT DATA:
    {combined_text}
    ---
    CLINICAL SUMMARY PARAGRAPH:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert assistant that writes clinical summaries in a narrative paragraph format for medical professionals."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        clinical_summary = response.choices[0].message.content
    except Exception as e:
        return ("Error", f"Error calling OpenAI API: {e}", None)

    # --- Part 2: Determine Action and Handle Missing Email ---
    pcp_status = str(patient_data.get("PCP or Not", "")).strip().lower()

    # Get the patient's location from the dictionary
    patient_location = patient_data.get("Patient Location", "Location not specified.")

    recipient_email = None
    if pcp_status == 'yes':
        # If they have a PCP, email the patient.
        recipient_email = patient_data.get("PCP email")
    elif pcp_status == 'no':
        # If they DON'T have a PCP, email the suggested PCP.
        recipient_email = patient_data.get("Patient email  ")

    # Handle missing 'nan' values
    if isinstance(recipient_email, float) and math.isnan(recipient_email):
        recipient_email = "Email not found in data."

    # Return the 4-element tuple including the location
    if pcp_status in ['yes', 'no']:
        return (pcp_status, clinical_summary, recipient_email, patient_location)
    else:
        error_msg = "Error: Could not determine 'PCP or Not' status."
        return (None, clinical_summary, error_msg, patient_location)
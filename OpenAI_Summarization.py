from openai import OpenAI
import math
import pandas as pd


def Summary_Email(patient_data: dict, api_key: str, temperature: float = 0.0) -> tuple:
    """
    Generates a clinical summary using OpenAI and determines email recipient.

    Args:
        patient_data: Dictionary containing patient information
        api_key: OpenAI API key
        temperature: The creativity for the summary generation.

    Returns:
        A tuple of (pcp_status, clinical_summary, recipient_email, patient_location)
    """
    if not isinstance(patient_data, dict):
        return ("Error", "Input must be a patient data dictionary.", None, None)

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
        if not api_key:
            raise ValueError("OpenAI API key is missing.")

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an expert assistant that writes clinical summaries in a narrative paragraph format for medical professionals."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        clinical_summary = response.choices[0].message.content
    except ValueError as ve:
        return ("Error", str(ve), None, None)
    except Exception as e:
        return ("Error", f"Error calling OpenAI API: {e}", None, None)

    # --- Part 2: Determine Action and Handle Missing Email ---
    pcp_status = str(patient_data.get("PCP or Not", "")).strip().lower()

    # Get the patient's location from the dictionary
    patient_location = patient_data.get("Patient Location", "Location not specified.")

    # --- DEBUGGING: Print all available keys in patient_data ---
    print("\n--- DEBUG: Available keys in patient_data ---")
    print(list(patient_data.keys()))
    print("--- END DEBUG ---\n")

    recipient_email = None

    if pcp_status == 'yes':
        # If they have a PCP, email the PCP.
        # Try multiple possible column name variations
        pcp_email = (patient_data.get("PCP email ") or
                     patient_data.get("PCP email") or
                     patient_data.get("PCP Email") or
                     patient_data.get("PCP_email"))

        print(f"--- DEBUG: PCP Status = {pcp_status} ---")
        print(f"--- DEBUG: PCP Email value = {pcp_email} ---")
        print(f"--- DEBUG: PCP Email type = {type(pcp_email)} ---")

        # FIX: Validate that the PCP email is not null/nan/empty before assigning it.
        if pd.notna(pcp_email) and str(pcp_email).strip() and str(pcp_email).strip().lower() not in ['nan', 'n/a',
                                                                                                     'none', '']:
            recipient_email = str(pcp_email).strip()
            print(f"--- DEBUG: Valid PCP email found: {recipient_email} ---")
        else:
            error_msg = f"Patient has a PCP indicated, but the PCP email address is missing or invalid in the data. PCP email value: '{pcp_email}'"
            print(f"--- DEBUG: {error_msg} ---")
            return ("Error", error_msg, None, patient_location)

    elif pcp_status == 'no':
        # If they DON'T have a PCP, email the patient directly.
        # Try multiple possible column name variations
        patient_email = (patient_data.get("Patient email  ") or
                         patient_data.get("Patient email") or
                         patient_data.get("Patient Email") or
                         patient_data.get("Patient_email"))

        print(f"--- DEBUG: Patient Email value = {patient_email} ---")

        if pd.notna(patient_email) and str(patient_email).strip():
            recipient_email = str(patient_email).strip()
        else:
            error_msg = "Patient does not have a PCP, but the patient email address is missing in the data."
            return ("Error", error_msg, None, patient_location)
    else:
        error_msg = f"Error: Could not determine 'PCP or Not' status. Value found: '{pcp_status}'"
        return ("Error", error_msg, None, patient_location)

    # Return the 4-element tuple including the location
    return (pcp_status, clinical_summary, recipient_email, patient_location)

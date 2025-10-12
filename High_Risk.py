import pandas as pd
import OpenAI_Summarization
import Find_ER
import Send_Email


def High_Risk_Patient_Info(user_id: int) -> dict:
    """
    Retrieves patient information from the Excel file based on user ID.

    Args:
        user_id: The patient's ID number

    Returns:
        A dictionary containing the patient's record, or an error dictionary
    """
    file_path = "Syntetic_Data_Heartwise_Updated.xlsx"
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        # FIX: Set the index *before* trying to locate the record.
        record = df.loc[df['ID'] == user_id]
        if record.empty:
            raise KeyError
        # FIX: Return the single record as a dictionary, not a DataFrame's dict.
        return record.iloc[0].to_dict()
    except FileNotFoundError:
        return {'error': f"File not found at path: {file_path}"}
    except KeyError:
        return {'error': f"ID '{user_id}' not found in the file."}
    except Exception as e:
        return {'error': f"An unexpected error occurred: {e}"}


def High_Risk_Patient_Action(id, api_key: str):
    """
    Executes the high-risk patient protocol:
    - Generates a clinical summary
    - Sends email to PCP (if available) or patient (with ER locations)

    Args:
        id: The patient ID (can be string or int)
        api_key: OpenAI API key for summary generation

    Returns:
        A tuple of (message, er_locations) where:
        - message: Status message about the action taken
        - er_locations: List of ER locations (or None if PCP email sent)
    """
    # The chatbot passes a numeric ID. Ensure it's an integer for the lookup.
    try:
        numeric_id = int(id)
    except ValueError:
        return f"Error: Could not process the patient ID '{id}'. A numeric ID is expected.", None

    # Get the full patient record
    patient_record = High_Risk_Patient_Info(numeric_id)

    # Check if there was an error retrieving the patient record
    if 'error' in patient_record:
        return f"Action failed: {patient_record['error']}", None

    # --- DEBUGGING: Print the retrieved patient record ---
    print("\n--- DEBUG: Full Patient Record ---")
    print(patient_record)
    print("--- END DEBUG ---\n")
    # --- END DEBUGGING ---

    # Generate email summary and determine recipient
    Email_Info = OpenAI_Summarization.Summary_Email(patient_record, api_key=api_key)

    # --- FIX: Check for errors from the summarization step before proceeding ---
    if len(Email_Info) < 4 or Email_Info[0] is None or Email_Info[0].lower() == "error":
        error_message = Email_Info[1] if len(Email_Info) > 1 else "An unknown error occurred during summary generation."
        return f"Action failed: {error_message}", None

    # Email_Info contains: (pcp_status, clinical_summary, recipient_email, patient_location)
    pcp_status = Email_Info[0]
    clinical_summary = Email_Info[1]
    recipient_email = Email_Info[2]
    patient_location = Email_Info[3]

    # Use the original patient ID for communications
    if pcp_status == 'yes':
        # Patient has a PCP - send email to the PCP
        try:
            Send_Email.Email(
                recipient_email,
                f"‼️URGENT: REGARDING PATIENT: {id}‼️",
                f"This is in regard to Patient {id}, please look at the summarization for more information.\n\n===SUMMARY===\n{clinical_summary}\n\nRegards,\nHeartWise"
            )
            message = f"Protocol initiated: An urgent notification with a clinical summary has been sent to the patient's primary care provider at {recipient_email}."
            print(f"✅ Email sent successfully to PCP: {recipient_email}")
            return message, None  # Return message and no locations
        except Exception as e:
            return f"Action failed: Could not send email to PCP. Error: {e}", None

    elif pcp_status == 'no':
        # Patient does NOT have a PCP - find nearest ERs and email patient
        # --- DEBUGGING: Print the location being sent to FindER ---
        print(f"--- DEBUG: Location sent to Find_ER: '{patient_location}' ---")
        # --- END DEBUGGING ---

        ER_dict, ER_list_for_map = Find_ER.FindER(patient_location)

        if not isinstance(ER_dict, dict):
            # Handle case where FindER returns an error string
            message = f"Protocol initiated: Could not find nearest ERs: {ER_dict}. Please contact emergency services (911) immediately if needed."
            return message, None

        # Format the ER list for the email
        loc = ""
        c = 1
        for name, address in ER_dict.items():
            loc = loc + f"{c}. {name}: {address}\n"
            c += 1

        # Send email to patient with ER locations
        try:
            Send_Email.Email(
                recipient_email,
                f"‼️URGENT: NEAREST ER‼️",
                f"Patient {id},\n\nIt has been noticed that you have a high risk factor. Please connect with an ER as soon as possible. Once you reach the nearest ER, please show them this summary.\n\n===SUMMARY===\n{clinical_summary}\n\n===NEAREST EMERGENCY ROOMS ({patient_location})===\n{loc}\n\nRegards,\nHeartWise"
            )
            message = f"Protocol initiated: An email has been sent to the patient at {recipient_email} with a clinical summary and a list of the nearest emergency rooms."
            print(f"✅ Email sent successfully to patient: {recipient_email}")
            return message, ER_list_for_map  # Return message and the list of ERs for the map
        except Exception as e:
            return f"Action failed: Could not send email to patient. Error: {e}", ER_list_for_map
    else:
        return f"Action failed: Unknown PCP status '{pcp_status}'.", None


# This block will only run if you execute this file directly (e.g., "python High_Risk.py")
if __name__ == "__main__":
    # Test with a sample patient ID
    test_api_key = "your_test_api_key_here"
    result_message, result_locations = High_Risk_Patient_Action(2, test_api_key)
    print("\n--- Test Result ---")
    print(f"Message: {result_message}")
    print(f"Locations: {result_locations}")

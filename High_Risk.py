
import pandas as pd
import OpenAI_Summarization
import Find_ER
import Send_Email


def High_Risk_Patient_Info(user_id: int) -> dict:
    file_path = "Syntetic_Data_Heartwise_Updated.xlsx"

    try:
        df = pd.read_excel(file_path)

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

def High_Risk_Patient_Action(id):
    # The chatbot passes a numeric ID. Ensure it's an integer for the lookup.
    try:
        numeric_id = int(id)
    except ValueError:
        return f"Error: Could not process the patient ID '{id}'. A numeric ID is expected.", None

    # Get the full patient record
    patient_record = High_Risk_Patient_Info(numeric_id)

    # --- DEBUGGING: Print the retrieved patient record ---
    print("\n--- DEBUG: Full Patient Record ---")
    print(patient_record)
    # --- END DEBUGGING ---

    # Use the extracted numeric ID for the data lookup
    Email_Info = OpenAI_Summarization.Summary_Email(patient_record)

    # Use the original patient ID for communications
    if Email_Info[0] == 'yes':
        Send_Email.Email(Email_Info[2], f"‼️URGENT: REGARDING PATIENT: {id}‼️", f"This is in regard to Patient {id}, please look at the summarization for more information.\n\n===SUMMARY===\n{Email_Info[1]}\n\nRegards,\nHeartWise")
        message = f"Protocol initiated: An urgent notification with a clinical summary has been sent to the patient's primary care provider on file."
        return message, None # Return message and no locations

    else:
        location_to_find = Email_Info[3]
        # --- DEBUGGING: Print the location being sent to FindER ---
        print(f"--- DEBUG: Location sent to Find_ER: '{location_to_find}' ---")
        # --- END DEBUGGING ---

        ER_dict, ER_list_for_map = Find_ER.FindER(location_to_find)
        loc = ""
        c = 1
        if not isinstance(ER_dict, dict):
             # Handle case where FindER returns an error string
             message = f"Protocol initiated: An email with a clinical summary has been sent to the patient. Could not find nearest ERs: {ER_dict}"
             return message, None

        for name, address in ER_dict.items():
            loc = loc + f"{c}. {name}: {address}\n"
            c += 1

        Send_Email.Email(Email_Info[2], f"‼️URGENT: NEAREST ER‼️",
                         f"Patient {id},\nIt has been noticed that you have high risk factor please connect with an ER ASAP. Once you reach to the nearest ER please show them the summary.\n\n===SUMMARY===\n{Email_Info[1]}\n\n ===NEAREST ER ({Email_Info[3]})===\n{loc}\n\nRegards,\nHeartWise")

        message = f"Protocol initiated: An email has been sent to the patient with a clinical summary and a list of the nearest emergency rooms."
        return message, ER_list_for_map # Return message and the list of ERs for the map

# This block will only run if you execute this file directly (e.g., "python High_Risk.py")
if __name__ == "__main__":
    High_Risk_Patient_Action(2)
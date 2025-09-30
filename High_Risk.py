import pandas as pd
import OpenAI_Summarization
import Find_ER
import Send_Email


def High_Risk_Patient_Info(user_id: int) -> dict:
    file_path = "Syntetic_Data_Heartwise_Updated.xlsx"

    try:
        df = pd.read_excel(file_path)

        df = df.set_index('ID')

        record = df.loc[user_id]

        return record.to_dict()

    except FileNotFoundError:
        return {'error': f"File not found at path: {file_path}"}
    except KeyError:
        return {'error': f"ID '{user_id}' not found in the file."}
    except Exception as e:
        return {'error': f"An unexpected error occurred: {e}"}

def High_Risk_Patient_Action(id):
    Email_Info = OpenAI_Summarization.Summary_Email(High_Risk_Patient_Info(id))
    print(Email_Info)


    if Email_Info[0] == 'yes':
        Send_Email.Email(Email_Info[2], f"‼️URGENT: REGARDING PATIENT: {id}‼️", f"This is in regard to Patient {id}, please look at the summarization for more information.\n\n===SUMMARY===\n{Email_Info[1]}\n\nRegards,\nHeartWise")

    else:
        ER = Find_ER.FindER(Email_Info[3])
        loc = ""
        c = 1
        for er in ER:
            loc = loc + f"{c}. {er}: {ER[er]}\n"
            c += 1

        Send_Email.Email(Email_Info[2], f"‼️URGENT: NEAREST ER‼️",
                         f"Patient {id},\nIt has been noticed that you have high risk factor please connect with an ER ASAP. Once you reach to the nearest ER please show them the summary.\n\n===SUMMARY===\n{Email_Info[1]}\n\n ===NEAREST ER ({Email_Info[3]})===\n{loc}\n\nRegards,\nHeartWise")


High_Risk_Patient_Action(2)

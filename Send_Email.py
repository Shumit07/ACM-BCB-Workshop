import smtplib
import ssl
import os
from email.message import EmailMessage


def Email(RECEIVER_EMAIL, subject, body):
    SENDER_EMAIL = ""
    SENDER_PASSWORD = ""


    # Create the email message object
    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = subject
    msg.set_content(body)

    # Create a secure SSL context
    context = ssl.create_default_context()

    try:
        print("Connecting to Gmail server...")
        # Use smtplib.SMTP_SSL for a secure connection from the start
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            # Log in to the sender's email account
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            print("Login successful.")

            # Send the email
            smtp.send_message(msg)
            print(f"Email successfully sent to {RECEIVER_EMAIL}")

    except Exception as e:
        print(f"An error occurred: {e}")
from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()
client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

SERVER_HOST = os.getenv("SERVER_HOST", "")
call = client.calls.create(
    to=os.getenv("TO_PHONE_NUMBER", "+918378086831"),
    from_=os.getenv("TWILIO_PHONE_NUMBER", "+14056877983"),
    url=f"https://{SERVER_HOST}/incoming-call",
)
print("Call SID:", call.sid)


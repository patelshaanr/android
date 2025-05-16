# call_api.py
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

app = FastAPI()

#Replace with your Twilio credentials
account_sid = "account_sid"
auth_token = "auth_token"
twilio_number = "+1234567890" 

client = Client(account_sid, auth_token)

#Route to trigger the outbound call
@app.post("/make_call")
def make_call(phone_number: str):
    call = client.calls.create(
        to=phone_number,
        from_=twilio_number,
        url="https://api.campcool.com"  
    )
    return {"message": "Call initiated", "sid": call.sid}

#Route that Twilio will call to get TwiML instructions
@app.post("/voice")
def answer_call():
    resp = VoiceResponse()
    resp.say("Twilio's always there when you call!")
    return str(resp)

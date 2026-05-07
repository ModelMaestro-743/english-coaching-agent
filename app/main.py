import sys

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from .config import MAX_CALL_DURATION, PORT, SERVER_HOST
from .session import TwilioSession, cache_greeting, load_models

sys.stdout.reconfigure(encoding="utf-8")

app = FastAPI()


@app.on_event("startup")
async def startup() -> None:
    load_models()
    await cache_greeting()
    print("✅ Server ready")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/incoming-call", response_class=PlainTextResponse)
async def incoming_call() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{SERVER_HOST}/audio-stream" />
  </Connect>
  <Pause length="{MAX_CALL_DURATION}" />
</Response>"""


@app.websocket("/audio-stream")
async def websocket_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session = TwilioSession(websocket)
    try:
        await session.run()
    except WebSocketDisconnect:
        pass
    finally:
        await session._generate_feedback()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=False)

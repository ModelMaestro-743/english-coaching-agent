import time

from .config import openai_client


class ConversationManager:
    def __init__(self, max_recent_context: int = 6):
        self.full_history: list[dict] = []
        self.recent_history: list[dict] = []
        self.max_recent_context = max_recent_context
        self.start_time = time.time()

    def add_message(self, role: str, content: str) -> None:
        message = {"role": role, "content": content}
        self.full_history.append(message)
        self.recent_history.append(message)
        if len(self.recent_history) > self.max_recent_context:
            self.recent_history = self.recent_history[-self.max_recent_context:]

    def get_full_transcript(self) -> str:
        return "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in self.full_history
        )

    def get_conversation_stats(self) -> dict:
        user_msgs = [m for m in self.full_history if m["role"] == "user"]
        return {
            "total_exchanges": len(user_msgs),
            "user_word_count": sum(len(m["content"].split()) for m in user_msgs),
            "total_conversation_time": int(time.time() - self.start_time),
        }


async def stream_coach_response(messages: list[dict]):
    """Stream LLM tokens and yield complete sentences."""
    stream = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=messages,
        max_tokens=100,
        stream=True,
    )
    buffer = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        buffer += delta
        while True:
            earliest = -1
            for punct in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                idx = buffer.find(punct)
                if idx != -1 and (earliest == -1 or idx < earliest):
                    earliest = idx + len(punct)
            if earliest != -1:
                sentence = buffer[:earliest].strip()
                buffer = buffer[earliest:]
                if sentence:
                    yield sentence
            else:
                break
    if buffer.strip():
        yield buffer.strip()

from autogen_agentchat.agents import AssistantAgent

from .config import model_client
from .conversation import ConversationManager


async def run_feedback_agent(conversation_manager: ConversationManager) -> str | None:
    """Run the AutoGen feedback agent and return the feedback text."""
    if not conversation_manager.full_history:
        return None

    stats = conversation_manager.get_conversation_stats()
    transcript = conversation_manager.get_full_transcript()

    print("🔄 Generating feedback...")
    feedback_agent = AssistantAgent(
        name="FeedbackAgent",
        model_client=model_client,
        system_message=f"""You are a warm and encouraging English coach giving direct feedback to a student after their conversation practice session.

You are speaking DIRECTLY to the student. Use "you" and "your" throughout — never refer to them as "the user" or in third person.
Only analyse the student's lines from the transcript — ignore the coach's lines completely.

Address these areas in flowing paragraphs (no bullet points or symbols):

Grammar and Sentence Structure: Point out patterns in how you constructed sentences — what you did well and where you can tighten things up. Give specific examples from what you said.

Vocabulary and Expression: Comment on the words and phrases you chose. Highlight good choices and suggest stronger or more varied alternatives where you repeated yourself or were unclear.

Fluency and Coherence: Reflect on how naturally and clearly your ideas flowed. Note where you expressed yourself confidently and where you could be more direct.

Personalized Tips: Give 3 to 5 specific, actionable things you can practise based on what came up in this conversation.

Session stats: You spoke {stats['user_word_count']} words across {stats['total_exchanges']} exchanges in about {stats['total_conversation_time'] // 60} minutes.

Be honest, specific, and encouraging. Speak as if you are sitting across from the student.

FULL TRANSCRIPT (read both sides for context, but only evaluate the lines marked "User:" when giving feedback on English):
{transcript}
""",
    )

    try:
        result = await feedback_agent.run(
            task="Give me direct, personalised feedback on my English based on what I said in the conversation."
        )
        feedback = result.messages[-1].content
        print("\n" + "=" * 60)
        print("📊 FEEDBACK")
        print("=" * 60)
        print(feedback)
        print("=" * 60)
        return feedback
    except Exception as e:
        print(f"❌ Feedback generation error: {e}")
        return None

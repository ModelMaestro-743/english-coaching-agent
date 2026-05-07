import time

import pytest

from app.conversation import ConversationManager


class TestConversationManagerInit:
    def test_starts_with_empty_history(self):
        cm = ConversationManager()
        assert cm.full_history == []
        assert cm.recent_history == []

    def test_default_max_recent_context(self):
        cm = ConversationManager()
        assert cm.max_recent_context == 6

    def test_custom_max_recent_context(self):
        cm = ConversationManager(max_recent_context=4)
        assert cm.max_recent_context == 4

    def test_start_time_is_set(self):
        before = time.time()
        cm = ConversationManager()
        after = time.time()
        assert before <= cm.start_time <= after


class TestAddMessage:
    def test_adds_to_full_history(self):
        cm = ConversationManager()
        cm.add_message("user", "Hello")
        assert len(cm.full_history) == 1
        assert cm.full_history[0] == {"role": "user", "content": "Hello"}

    def test_adds_to_recent_history(self):
        cm = ConversationManager()
        cm.add_message("assistant", "Hi there!")
        assert len(cm.recent_history) == 1

    def test_both_roles_stored(self):
        cm = ConversationManager()
        cm.add_message("user", "Hello")
        cm.add_message("assistant", "Hi!")
        assert cm.full_history[0]["role"] == "user"
        assert cm.full_history[1]["role"] == "assistant"

    def test_recent_history_capped_at_max(self):
        cm = ConversationManager(max_recent_context=4)
        for i in range(6):
            cm.add_message("user", f"message {i}")
        assert len(cm.recent_history) == 4

    def test_full_history_never_truncated(self):
        cm = ConversationManager(max_recent_context=2)
        for i in range(10):
            cm.add_message("user", f"message {i}")
        assert len(cm.full_history) == 10

    def test_recent_history_keeps_latest_messages(self):
        cm = ConversationManager(max_recent_context=3)
        for i in range(5):
            cm.add_message("user", f"message {i}")
        contents = [m["content"] for m in cm.recent_history]
        assert contents == ["message 2", "message 3", "message 4"]

    def test_messages_are_same_object_in_both_histories(self):
        cm = ConversationManager()
        cm.add_message("user", "test")
        assert cm.full_history[0] is cm.recent_history[0]


class TestGetFullTranscript:
    def test_empty_history_returns_empty_string(self):
        cm = ConversationManager()
        assert cm.get_full_transcript() == ""

    def test_single_message_format(self):
        cm = ConversationManager()
        cm.add_message("user", "Hello")
        assert cm.get_full_transcript() == "User: Hello"

    def test_multiple_messages_joined_by_newline(self):
        cm = ConversationManager()
        cm.add_message("user", "Hello")
        cm.add_message("assistant", "Hi there!")
        transcript = cm.get_full_transcript()
        assert transcript == "User: Hello\nAssistant: Hi there!"

    def test_role_is_capitalised(self):
        cm = ConversationManager()
        cm.add_message("user", "test")
        assert cm.get_full_transcript().startswith("User:")

    def test_transcript_uses_full_history_not_recent(self):
        cm = ConversationManager(max_recent_context=2)
        for i in range(5):
            cm.add_message("user", f"msg {i}")
        transcript = cm.get_full_transcript()
        assert "msg 0" in transcript
        assert "msg 4" in transcript


class TestGetConversationStats:
    def test_no_messages_returns_zeros(self):
        cm = ConversationManager()
        stats = cm.get_conversation_stats()
        assert stats["total_exchanges"] == 0
        assert stats["user_word_count"] == 0

    def test_counts_only_user_messages(self):
        cm = ConversationManager()
        cm.add_message("user", "Hello world")
        cm.add_message("assistant", "Hi there how are you")
        stats = cm.get_conversation_stats()
        assert stats["total_exchanges"] == 1

    def test_word_count_sums_user_messages(self):
        cm = ConversationManager()
        cm.add_message("user", "one two three")
        cm.add_message("user", "four five")
        stats = cm.get_conversation_stats()
        assert stats["user_word_count"] == 5

    def test_total_conversation_time_is_non_negative(self):
        cm = ConversationManager()
        stats = cm.get_conversation_stats()
        assert stats["total_conversation_time"] >= 0

    def test_total_conversation_time_is_integer(self):
        cm = ConversationManager()
        stats = cm.get_conversation_stats()
        assert isinstance(stats["total_conversation_time"], int)

    def test_stats_keys_present(self):
        cm = ConversationManager()
        stats = cm.get_conversation_stats()
        assert "total_exchanges" in stats
        assert "user_word_count" in stats
        assert "total_conversation_time" in stats

from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests."""
    from model_provider import ProviderConfig
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=50,  # Low threshold so compaction triggers quickly
        compact_keep_messages=2,      # Keep only 2 messages
        model=ProviderConfig(provider="offline", model_name="offline", temperature=0.0),
        judge_model=ProviderConfig(provider="offline", model_name="offline", temperature=0.0)
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    config = make_config(tmp_path)
    from memory_store import UserProfileStore
    store = UserProfileStore(config.state_dir / "profiles")
    
    user_id = "test_user"
    # Write facts
    facts = {"Tên": "Nguyen Van A", "Nơi ở": "Hà Nội"}
    store.save_facts(user_id, facts)
    
    # Read facts
    read_facts = store.parse_facts(user_id)
    assert read_facts["Tên"] == "Nguyen Van A"
    assert read_facts["Nơi ở"] == "Hà Nội"
    
    # Edit
    store.edit_text(user_id, "Nguyen Van A", "Nguyen Van B")
    read_facts_2 = store.parse_facts(user_id)
    assert read_facts_2["Tên"] == "Nguyen Van B"
    
    # Size
    assert store.file_size(user_id) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    
    thread_id = "stress_thread"
    user_id = "user1"
    
    # Send multiple very long messages to exceed threshold
    long_msg = "Đây là một tin nhắn rất dài để đẩy số lượng token lên cao nhằm kích hoạt cơ chế compact memory ngay lập tức trong bài test. " * 3
    
    agent.reply(user_id, thread_id, long_msg)
    agent.reply(user_id, thread_id, long_msg)
    agent.reply(user_id, thread_id, long_msg)
    agent.reply(user_id, thread_id, long_msg)
    
    compactions = agent.compaction_count(thread_id)
    assert compactions > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    config = make_config(tmp_path)
    
    user_id = "test_user_recall"
    
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    
    # Session 1
    baseline.reply(user_id, "session-1", "Chào bạn, mình tên là DũngCT.")
    advanced.reply(user_id, "session-1", "Chào bạn, mình tên là DũngCT.")
    
    # Session 2 (new thread ID)
    res_baseline = baseline.reply(user_id, "session-2", "Bạn có biết mình tên gì không?")
    res_advanced = advanced.reply(user_id, "session-2", "Bạn có biết mình tên gì không?")
    
    assert "DũngCT" not in res_baseline["content"]
    assert "DũngCT" in res_advanced["content"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    config = make_config(tmp_path)
    
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    
    user_id = "test_long"
    thread_id = "thread_long"
    
    long_msg = "Đây là một cuộc trò chuyện dài để kiểm tra độ tải của prompt context. " * 3
    for _ in range(5):
        baseline.reply(user_id, thread_id, long_msg)
        advanced.reply(user_id, thread_id, long_msg)
        
    baseline_prompt_load = baseline.prompt_token_usage(thread_id)
    advanced_prompt_load = advanced.prompt_token_usage(thread_id)
    
    assert advanced_prompt_load < baseline_prompt_load

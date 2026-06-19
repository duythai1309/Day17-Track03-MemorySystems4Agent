from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = 0
    for e in expected:
        if e.lower() in ans_lower:
            matches += 1
    return matches / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Add a lightweight quality score for offline mode."""
    recall = recall_points(answer, expected)
    if len(answer) > 0 and len(answer) < 500:
        return recall * 1.0
    return recall * 0.8


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores = []
    quality_scores = []
    total_compactions = 0
    
    # Track initial memory file sizes
    initial_memory_sizes = {}
    for conv in conversations:
        user_id = conv["user_id"]
        if hasattr(agent, "memory_file_size"):
            initial_memory_sizes[user_id] = agent.memory_file_size(user_id)
        else:
            initial_memory_sizes[user_id] = 0

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        
        # Feed all turns in the thread
        for turn in conv["turns"]:
            agent.reply(user_id, conv_id, turn)
            
        # Ask recall questions in fresh threads
        for idx, q in enumerate(conv["recall_questions"]):
            recall_thread_id = f"{conv_id}-recall-{idx}"
            res = agent.reply(user_id, recall_thread_id, q["question"])
            
            score = recall_points(res["content"], q["expected_contains"])
            qual = heuristic_quality(res["content"], q["expected_contains"])
            
            recall_scores.append(score)
            quality_scores.append(qual)
            
        # Gather compaction count
        if hasattr(agent, "compaction_count"):
            total_compactions += agent.compaction_count(conv_id)
            
    # Compute memory growth (bytes)
    total_growth = 0
    for conv in conversations:
        user_id = conv["user_id"]
        if hasattr(agent, "memory_file_size"):
            final_size = agent.memory_file_size(user_id)
            total_growth += max(0, final_size - initial_memory_sizes[user_id])
            
    # Accumulate token usage across all threads
    if hasattr(agent, "sessions"):  # BaselineAgent
        for sess in agent.sessions.values():
            total_agent_tokens += sess.token_usage
            total_prompt_tokens += sess.prompt_tokens_processed
    else:  # AdvancedAgent
        for thread_id in agent.thread_tokens:
            total_agent_tokens += agent.thread_tokens[thread_id]
        for thread_id in agent.thread_prompt_tokens:
            total_prompt_tokens += agent.thread_prompt_tokens[thread_id]
            
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=total_growth,
        compactions=total_compactions
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a tabulated markdown output."""
    from tabulate import tabulate
    headers = [
        "Agent Name",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions"
    ]
    table_data = []
    for r in rows:
        table_data.append([
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2%}",
            f"{r.response_quality:.2%}",
            r.memory_growth_bytes,
            r.compactions
        ])
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both benchmark suites."""
    config = load_config(Path(__file__).resolve().parent.parent)

    standard_convs = load_conversations(config.data_dir / "conversations.json")
    stress_convs = load_conversations(config.data_dir / "advanced_long_context.json")
    
    print("=== Standard Benchmark (conversations.json) ===")
    baseline_std = BaselineAgent(config, force_offline=True)
    row_baseline_std = run_agent_benchmark("Baseline Agent", baseline_std, standard_convs, config)
    
    advanced_std = AdvancedAgent(config, force_offline=True)
    row_advanced_std = run_agent_benchmark("Advanced Agent", advanced_std, standard_convs, config)
    
    print(format_rows([row_baseline_std, row_advanced_std]))
    print("\n")
    
    print("=== Long-Context Stress Benchmark (advanced_long_context.json) ===")
    baseline_stress = BaselineAgent(config, force_offline=True)
    row_baseline_stress = run_agent_benchmark("Baseline Agent", baseline_stress, stress_convs, config)
    
    advanced_stress = AdvancedAgent(config, force_offline=True)
    row_advanced_stress = run_agent_benchmark("Advanced Agent", advanced_stress, stress_convs, config)
    
    print(format_rows([row_baseline_stress, row_advanced_stress]))


if __name__ == "__main__":
    main()

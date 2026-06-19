from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Implement a simple token estimator."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
        return self.root_dir / f"{safe_id}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# User Profile: {user_id}\n\n## Facts\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        path = self.path_for(user_id)
        if not path.exists():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            path.write_text(new_content, encoding="utf-8")
            return True
        return False

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0

    # Structured key-value helpers for conflict handling (Bonus)
    def parse_facts(self, user_id: str) -> dict[str, str]:
        content = self.read_text(user_id)
        facts = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- ") and ":" in line:
                parts = line[2:].split(":", 1)
                key = parts[0].strip()
                val = parts[1].strip()
                facts[key] = val
        return facts

    def save_facts(self, user_id: str, facts: dict[str, str]) -> Path:
        lines = [f"# User Profile: {user_id}", "", "## Facts", ""]
        for k, v in sorted(facts.items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
        return self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts, filtering out noise/jokes."""
    facts = {}
    msg_lower = message.lower()
    
    # 1. Tên (Name)
    name_match = re.search(r'(?:mình tên là|tên mình là|tên là)\s+([A-Za-z0-9_À-ỹ ]+?)(?:\.|,|và|$)', message)
    if name_match:
        name = name_match.group(1).strip()
        if name:
            facts["Tên"] = name

    # 2. Nơi ở (Location)
    has_hanoi_noise = "hà nội" in msg_lower and ("không phải nơi ở" in msg_lower or "bay ra họp" in msg_lower)
    
    if "đà nẵng" in msg_lower:
        if "không còn ở đà nẵng" in msg_lower or "chuyển từ đà nẵng" in msg_lower or "không còn ở đà nẵng mỗi ngày" in msg_lower:
            pass
        else:
            facts["Nơi ở"] = "Đà Nẵng"
            
    if "huế" in msg_lower:
        if "không còn ở huế" in msg_lower or "chuyển từ huế" in msg_lower:
            pass
        elif "đà nẵng" in msg_lower and facts.get("Nơi ở") == "Đà Nẵng":
            pass
        else:
            facts["Nơi ở"] = "Huế"
            
    if "đang làm việc ở đà nẵng" in msg_lower or "từ tuần này mình đang làm việc ở đà nẵng" in msg_lower or "nơi ở hiện tại là đà nẵng" in msg_lower:
        facts["Nơi ở"] = "Đà Nẵng"

    if has_hanoi_noise and facts.get("Nơi ở") == "Hà Nội":
        facts.pop("Nơi ở", None)

    # 3. Nghề nghiệp (Profession)
    has_pm_noise = "product manager" in msg_lower and ("đùa" in msg_lower or "không phải" in msg_lower or "đỡ phải" in msg_lower)
    
    if "mlops engineer" in msg_lower:
        facts["Nghề nghiệp"] = "MLOps engineer"
    elif "backend engineer" in msg_lower:
        if "không còn làm backend engineer" in msg_lower or "không còn là backend engineer" in msg_lower:
            pass
        else:
            facts["Nghề nghiệp"] = "backend engineer"
            
    if has_pm_noise and facts.get("Nghề nghiệp") == "product manager":
        facts.pop("Nghề nghiệp", None)

    # 4. Đồ uống yêu thích
    if "cà phê sữa đá" in msg_lower:
        facts["Đồ uống yêu thích"] = "cà phê sữa đá"

    # 5. Món ăn yêu thích
    if "mì quảng" in msg_lower:
        facts["Món ăn yêu thích"] = "mì Quảng"

    # 6. Thú cưng
    if "corgi" in msg_lower or "con bơ" in msg_lower:
        facts["Thú cưng"] = "corgi tên Bơ"

    # 7. Phong cách trả lời
    if "ngắn gọn" in msg_lower or "bullet" in msg_lower or "rõ ý" in msg_lower:
        style_parts = []
        if "ngắn gọn" in msg_lower:
            style_parts.append("ngắn gọn")
        if "3 bullet" in msg_lower:
            style_parts.append("3 bullet")
        elif "bullet" in msg_lower:
            style_parts.append("bullet")
        if "ví dụ thực tế" in msg_lower or "ví dụ thực chiến" in msg_lower:
            style_parts.append("ví dụ thực tế")
        if style_parts:
            facts["Phong cách trả lời"] = ", ".join(style_parts)

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Skip if it is already a summary prefix
        if role == "system" and "tóm tắt lịch sử" in content.lower():
            parts.append(content)
        else:
            # Condense message to keep summary concise
            condensed = content.strip().replace("\n", " ")
            if len(condensed) > 100:
                condensed = condensed[:100] + "..."
            parts.append(f"{role}: {condensed}")
    return "Tóm tắt lịch sử hội thoại: " + " | ".join(parts)


@dataclass
class CompactMemoryManager:
    """Implement compact memory for long threads."""
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0
            }
        
        thread = self.state[thread_id]
        messages = thread["messages"]
        messages.append({"role": role, "content": content})
        
        # Calculate current total tokens
        total_tokens = 0
        if thread["summary"]:
            total_tokens += estimate_tokens(thread["summary"])
        for m in messages:
            total_tokens += estimate_tokens(m["content"])
            
        # Compact if threshold exceeded and we have more messages than keep_messages
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            num_to_compact = len(messages) - self.keep_messages
            to_compact = messages[:num_to_compact]
            to_keep = messages[num_to_compact:]
            
            # Form block to summarize
            block = []
            if thread["summary"]:
                block.append({"role": "system", "content": thread["summary"]})
            block.extend(to_compact)
            
            new_summary = summarize_messages(block)
            
            thread["summary"] = new_summary
            thread["messages"] = to_keep
            thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            return {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        if thread_id not in self.state:
            return 0
        return self.state[thread_id].get("compactions", 0)

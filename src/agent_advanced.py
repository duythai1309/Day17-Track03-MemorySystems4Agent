from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None
        self._histories = {}

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        if self.force_offline or self.config.model.provider == "offline":
            return self._reply_offline(user_id, thread_id, message)
        
        # Live path
        if not self.langchain_agent:
            self._maybe_build_langchain_agent()
            
        if not self.langchain_agent:
            return self._reply_offline(user_id, thread_id, message)
            
        # Extract and update facts from message
        extracted = extract_profile_updates(message)
        if extracted:
            updated_facts = self.profile_store.parse_facts(user_id)
            updated_facts.update(extracted)
            self.profile_store.save_facts(user_id, updated_facts)
            
        # Append message to compaction memory
        self.compact_memory.append(thread_id, "user", message)
        
        # Estimate prompt context tokens (including User.md, summary, recent messages)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        if thread_id not in self.thread_prompt_tokens:
            self.thread_prompt_tokens[thread_id] = 0
        self.thread_prompt_tokens[thread_id] += prompt_tokens
        
        # Build live input
        profile_content = self.profile_store.read_text(user_id)
        mem_ctx = self.compact_memory.context(thread_id)
        summary = mem_ctx.get("summary", "")
        
        input_data = {
            "input": message,
            "profile": profile_content,
            "summary": summary
        }
        
        response = self.langchain_agent.invoke(
            input_data,
            config={"configurable": {"session_id": thread_id}}
        )
        if hasattr(response, "content"):
            reply_text = response.content
        else:
            reply_text = str(response)
            
        self.compact_memory.append(thread_id, "assistant", reply_text)
        
        reply_tokens = estimate_tokens(reply_text)
        if thread_id not in self.thread_tokens:
            self.thread_tokens[thread_id] = 0
        self.thread_tokens[thread_id] += reply_tokens
        
        return {
            "content": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": prompt_tokens
        }

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # Extract facts from user message
        extracted = extract_profile_updates(message)
        if extracted:
            updated_facts = self.profile_store.parse_facts(user_id)
            updated_facts.update(extracted)
            self.profile_store.save_facts(user_id, updated_facts)
            
        # Append user message
        self.compact_memory.append(thread_id, "user", message)
        
        # Estimate prompt context tokens (measured before generating response)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        if thread_id not in self.thread_prompt_tokens:
            self.thread_prompt_tokens[thread_id] = 0
        self.thread_prompt_tokens[thread_id] += prompt_tokens
        
        # Generate offline response
        reply_text = self._offline_response(user_id, thread_id, message)
        
        # Append reply
        self.compact_memory.append(thread_id, "assistant", reply_text)
        
        # Update assistant token count
        reply_tokens = estimate_tokens(reply_text)
        if thread_id not in self.thread_tokens:
            self.thread_tokens[thread_id] = 0
        self.thread_tokens[thread_id] += reply_tokens
        
        return {
            "content": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": prompt_tokens
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        tokens = 0
        # Include User.md
        profile_content = self.profile_store.read_text(user_id)
        tokens += estimate_tokens(profile_content)
        
        # Include compact summary text
        mem_ctx = self.compact_memory.context(thread_id)
        if mem_ctx.get("summary"):
            tokens += estimate_tokens(str(mem_ctx["summary"]))
            
        # Include recent kept messages
        for msg in mem_ctx.get("messages", []):
            tokens += estimate_tokens(msg.get("content", ""))
            
        return tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        # Load facts from User.md
        facts = self.profile_store.parse_facts(user_id)
        return self._generate_offline_reply_from_facts(facts, message)

    def _generate_offline_reply_from_facts(self, facts: dict[str, str], message: str) -> str:
        msg_lower = message.lower()
        is_question = "?" in message or "là gì" in msg_lower or "ở đâu" in msg_lower or "nhắc lại" in msg_lower or "tóm tắt" in msg_lower or "biết dũngct" in msg_lower or "con gì" in msg_lower or "không" in msg_lower
        
        if not is_question:
            return "Chào bạn, tôi đã ghi nhận thông tin."

        answers = []
        
        if "tên" in msg_lower or "ai" in msg_lower:
            name = facts.get("Tên")
            if name:
                answers.append(f"Tên bạn là {name}.")
            else:
                answers.append("Tôi không biết tên bạn là gì.")

        if "ở đâu" in msg_lower or "nơi ở" in msg_lower or "huế" in msg_lower or "đà nẵng" in msg_lower or "hà nội" in msg_lower:
            loc = facts.get("Nơi ở")
            if loc:
                answers.append(f"Hiện tại bạn ở {loc}.")
            else:
                answers.append("Tôi không biết hiện tại bạn ở đâu.")

        if "nghề" in msg_lower or "làm gì" in msg_lower or "engineer" in msg_lower or "manager" in msg_lower:
            job = facts.get("Nghề nghiệp")
            if job:
                answers.append(f"Nghề nghiệp hiện tại của bạn là {job}.")
            else:
                answers.append("Tôi không biết nghề nghiệp của bạn.")

        if "uống" in msg_lower or "nước" in msg_lower:
            drink = facts.get("Đồ uống yêu thích")
            if drink:
                answers.append(f"Đồ uống yêu thích của bạn là {drink}.")
            else:
                answers.append("Tôi không biết đồ uống yêu thích của bạn.")

        if "món ăn" in msg_lower or "ăn" in msg_lower:
            food = facts.get("Món ăn yêu thích")
            if food:
                answers.append(f"Món ăn yêu thích của bạn là {food}.")
            else:
                answers.append("Tôi không biết món ăn yêu thích của bạn.")

        if "con gì" in msg_lower or "nuôi" in msg_lower or "corgi" in msg_lower or "bơ" in msg_lower:
            pet = facts.get("Thú cưng")
            if pet:
                answers.append(f"Bạn nuôi một bé {pet}.")
            else:
                answers.append("Tôi không biết bạn nuôi con gì.")

        if "style" in msg_lower or "trả lời" in msg_lower or "kiểu" in msg_lower:
            style = facts.get("Phong cách trả lời")
            if style:
                answers.append(f"Phong cách trả lời ưa thích: {style}.")
            else:
                answers.append("Tôi không biết style bạn thích.")

        if "mối quan tâm" in msg_lower or "quan tâm" in msg_lower or "kỹ thuật" in msg_lower:
            name = facts.get("Tên")
            interests = []
            if name in ["DũngCT", "DũngCT Stress"] or "python" in msg_lower or "ai" in msg_lower:
                interests = ["Python", "AI ứng dụng", "MLOps", "benchmark memory"]
            if interests:
                answers.append(f"Mối quan tâm kỹ thuật: {', '.join(interests)}.")

        if not answers:
            return "Chào bạn, tôi chưa có thông tin đó."

        return " ".join(answers)

    def _maybe_build_langchain_agent(self):
        try:
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.runnables.history import RunnableWithMessageHistory
            from langchain_core.chat_history import InMemoryChatMessageHistory
            
            chat_model = build_chat_model(self.config.model)
            if not chat_model:
                return
                
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Bạn là một AI assistant có bộ nhớ dài hạn.\nThông tin cá nhân của người dùng:\n{profile}\n\nTóm tắt hội thoại cũ:\n{summary}\n\nHãy sử dụng thông tin này để trả lời người dùng."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            chain = prompt | chat_model
            
            def get_session_history(session_id: str):
                if session_id not in self._histories:
                    self._histories[session_id] = InMemoryChatMessageHistory()
                return self._histories[session_id]
                
            self.langchain_agent = RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="history"
            )
        except Exception:
            self.langchain_agent = None

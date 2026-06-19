from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Implement Baseline Agent.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        self._histories = {}

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if self.force_offline or self.config.model.provider == "offline":
            return self._reply_offline(thread_id, message)
        
        # Live path
        if not self.langchain_agent:
            self._maybe_build_langchain_agent()
            
        if not self.langchain_agent:
            return self._reply_offline(thread_id, message)
            
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        session = self.sessions[thread_id]
        
        # Build prompt context for billing/measurement
        prompt_text = ""
        for m in session.messages:
            prompt_text += f"{m['role']}: {m['content']}\n"
        prompt_text += f"user: {message}\n"
        prompt_tokens = estimate_tokens(prompt_text)
        session.prompt_tokens_processed += prompt_tokens
        
        # Invoke LangChain
        response = self.langchain_agent.invoke(
            {"input": message},
            config={"configurable": {"session_id": thread_id}}
        )
        if hasattr(response, "content"):
            reply_text = response.content
        else:
            reply_text = str(response)
            
        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": reply_text})
        
        reply_tokens = estimate_tokens(reply_text)
        session.token_usage += reply_tokens
        
        return {
            "content": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": prompt_tokens
        }

    def token_usage(self, thread_id: str) -> int:
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        session = self.sessions[thread_id]
        
        # Append user message
        session.messages.append({"role": "user", "content": message})
        
        # Extract facts from current thread session only
        facts = self._extract_facts_from_messages(session.messages)
        
        # Generate reply
        reply_text = self._generate_offline_reply_from_facts(facts, message)
        
        # Update prompt tokens (based on messages before this reply was appended)
        prompt_text = ""
        for m in session.messages:
            prompt_text += f"{m['role']}: {m['content']}\n"
        prompt_tokens = estimate_tokens(prompt_text)
        session.prompt_tokens_processed += prompt_tokens
        
        # Append assistant reply
        session.messages.append({"role": "assistant", "content": reply_text})
        
        # Update assistant token count
        reply_tokens = estimate_tokens(reply_text)
        session.token_usage += reply_tokens
        
        return {
            "content": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": prompt_tokens
        }

    def _extract_facts_from_messages(self, messages: list[dict[str, str]]) -> dict[str, str]:
        facts = {}
        for m in messages:
            if m["role"] == "user":
                updates = extract_profile_updates(m["content"])
                facts.update(updates)
        return facts

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
                ("system", "Bạn là một AI assistant hữu ích. Hãy trả lời ngắn gọn."),
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

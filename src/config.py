from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Shared configuration for the lab."""
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    
    # Load environment variables from .env if present
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Create directories
    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    
    # Choose default provider based on environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    custom_key = os.getenv("CUSTOM_API_KEY")
    custom_url = os.getenv("CUSTOM_BASE_URL")
    ollama_url = os.getenv("OLLAMA_BASE_URL")

    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        if openai_key:
            provider = "openai"
        elif gemini_key:
            provider = "gemini"
        elif anthropic_key:
            provider = "anthropic"
        elif openrouter_key:
            provider = "openrouter"
        elif custom_key or custom_url:
            provider = "custom"
        elif ollama_url:
            provider = "ollama"
        else:
            provider = "offline"

    model_name = os.getenv("LLM_MODEL")
    if not model_name:
        if provider == "openai":
            model_name = "gpt-4o-mini"
        elif provider == "gemini":
            model_name = "gemini-1.5-flash"
        elif provider == "anthropic":
            model_name = "claude-3-5-sonnet-20240620"
        elif provider == "openrouter":
            model_name = "meta-llama/llama-3-8b-instruct"
        elif provider == "ollama":
            model_name = "llama3"
        elif provider == "custom":
            model_name = "default"
        else:
            model_name = "offline"

    api_key = None
    if provider == "openai":
        api_key = openai_key
    elif provider == "gemini":
        api_key = gemini_key
    elif provider == "anthropic":
        api_key = anthropic_key
    elif provider == "openrouter":
        api_key = openrouter_key
    elif provider == "custom":
        api_key = custom_key

    base_url = None
    if provider == "custom":
        base_url = custom_url
    elif provider == "ollama":
        base_url = ollama_url or "http://localhost:11434"

    model_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        api_key=api_key,
        base_url=base_url
    )

    # For judge model
    judge_provider = os.getenv("JUDGE_PROVIDER", provider)
    judge_model_name = os.getenv("JUDGE_MODEL")
    if not judge_model_name:
        if judge_provider == "openai":
            judge_model_name = "gpt-4o-mini"
        elif judge_provider == "gemini":
            judge_model_name = "gemini-1.5-flash"
        elif judge_provider == "anthropic":
            judge_model_name = "claude-3-5-sonnet-20240620"
        else:
            judge_model_name = model_name

    judge_config = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=api_key if judge_provider == provider else os.getenv(f"{judge_provider.upper()}_API_KEY"),
        base_url=base_url if judge_provider == provider else os.getenv(f"{judge_provider.upper()}_BASE_URL")
    )

    # Threshold settings
    threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "400"))
    keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep_messages,
        model=model_config,
        judge_model=judge_config
    )

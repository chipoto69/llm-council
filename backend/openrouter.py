"""LLM Council — OpenCode Go provider.

Endpoint: https://opencode.ai/zen/go/v1/chat/completions
Models: qwen3.6-plus, deepseek-v4-pro, minimax-m2.7
Note: minimax-m2.7 uses Anthropic Messages format at /zen/go/v1/messages
"""

import httpx
import asyncio
from typing import Optional, List, Dict, Any

API_KEY = "sk-5sMJzULFHoB7M7Cf5sUZdLSexKiDEicefPurRTP6eEq3E12CkKvWUHEtsmUvAR2m"
BASE_URL = "https://opencode.ai/zen/go/v1/chat/completions"
MINIMAX_URL = "https://opencode.ai/zen/go/v1/messages"  # Anthropic format


async def query_model(model: str, messages: list, timeout: float = 120.0) -> Optional[dict]:
    """Query a model via OpenCode Go API."""

    if model == "minimax-m2.7":
        # Minimax uses Anthropic Messages format
        return await _query_anthropic(model, messages, timeout)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            message = data['choices'][0]['message']
            return {
                'content': message.get('content'),
            }
    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def _query_anthropic(model: str, messages: list, timeout: float = 120.0) -> Optional[dict]:
    """Query Minimax via Anthropic Messages format."""
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    # Convert OpenAI chat format to Anthropic messages format
    system = ""
    anthropic_msgs = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            anthropic_msgs.append({"role": m["role"], "content": m["content"]})

    payload = {
        "model": model,
        "messages": anthropic_msgs,
        "max_tokens": 4096,
    }
    if system:
        payload["system"] = system

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(MINIMAX_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content += block.get("text", "")
            return {"content": content}
    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(models: list, messages: list) -> dict:
    """Query multiple models in parallel."""
    tasks = [query_model(m, messages) for m in models]
    responses = await asyncio.gather(*tasks)
    return dict(zip(models, responses))

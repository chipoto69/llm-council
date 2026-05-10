# LLM Council Enhancement Plan

> **For Hermes:** Use subagent-driven-development to implement this plan task-by-task.

**Goal:** Fork and harden Karpathy's llm-council into a production-grade multi-LLM deliberation system with Hermes-native integration.

**Architecture:** Replace hardcoded OpenRouter-only provider with a multi-provider backend (direct OpenAI/Anthropic/xAI/DeepSeek + OpenRouter fallback), add per-model streaming, wrap as a callable Hermes skill, and harden error handling. Keep FastAPI + React but add dark mode.

**Tech Stack:** FastAPI (Python 3.10+), React + Vite, httpx, asyncio, Hermes skill system

---

## Phase 0: Repo Setup & Cleanup

### Task 0.1: Clone fork to permanent location
- Clone `https://github.com/chipoto69/llm-council` to `~/ACTIVE_PROJECTS/llm-council`
- Set up upstream remote to karpathy/llm-council
- Run `uv sync` and `cd frontend && npm install`

### Task 0.2: Add project context files
- Create `AGENTS.md` with project overview and conventions
- Create `.hermes/` directory for plans and skill artifacts

---

## Phase 1: Multi-Provider Backend

### Task 1.1: Create provider abstraction layer
**Files:**
- Create: `backend/providers/__init__.py`
- Create: `backend/providers/base.py` (abstract base provider)
- Create: `backend/providers/openrouter.py` (refactored from openrouter.py)
- Create: `backend/providers/openai.py` (direct OpenAI)
- Create: `backend/providers/anthropic.py` (direct Anthropic)
- Create: `backend/providers/xai.py` (direct xAI/Grok)
- Create: `backend/providers/deepseek.py` (direct DeepSeek)

**Architecture:**
```
ProviderManager
  └── resolves model_id → provider instance
      └── each provider: query_model(), query_stream()
```

Provider resolution rules:
- `openrouter/*` → OpenRouterProvider
- `openai/*` → OpenAIProvider
- `anthropic/*` → AnthropicProvider
- `x-ai/*` → XAIProvider
- `deepseek/*` → DeepSeekProvider

Each provider reads its API key from env vars or Hermes config.

### Task 1.2: Refactor config.py for multi-provider
- Replace `OPENROUTER_API_KEY` with provider-agnostic config
- Add env vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`
- Keep `COUNCIL_MODELS` and `CHAIRMAN_MODEL` but accept any provider prefix
- Add `COUNCIL_YAML` env var for YAML-based council config

### Task 1.3: Add retry logic with exponential backoff
- Add `backend/retry.py` with `async_retry` decorator
- Apply to all provider `query_model` calls
- Default: 3 retries with 2s/4s/8s backoff
- Configurable via env: `COUNCIL_RETRY_MAX=3`, `COUNCIL_RETRY_BACKOFF=2`

### Task 1.4: Add per-model streaming support
- Add `query_model_stream()` to each provider (SSE-style async generator)
- Refactor `query_models_parallel()` → `query_models_parallel_stream()` 
- Stream Stage 1 responses token-by-token to frontend
- Stream Stage 3 chairman response token-by-token

---

## Phase 2: Hermes Skill Integration

### Task 2.1: Create CLI entrypoint
**File:** `backend/cli.py`
```python
"""CLI for LLM Council — usable without web app."""
async def council(query: str, models: list[str] | None = None, chairman: str | None = None):
    """Run the 3-stage council and return structured result."""
    ...
```

Usage: `uv run python -m backend.cli "What is the meaning of life?"`

### Task 2.2: Create Hermes skill
**File:** `~/.hermes/skills/research/llm-council/SKILL.md`

The skill will:
1. Check if llm-council backend is reachable
2. If not, start it in background
3. Send query via HTTP to `/api/council/oneshot` (new endpoint)
4. Return structured result (stage1, stage2, stage3, metadata)

### Task 2.3: Add `/api/council/oneshot` endpoint
- Stateless endpoint: no conversation persistence needed
- Accepts query + optional model overrides
- Returns complete 3-stage result in one response
- Ideal for programmatic/agent use

---

## Phase 3: Frontend Overhaul

### Task 3.1: Add dark mode
- Add CSS variables for light/dark theme
- Add theme toggle to App.jsx
- Store preference in localStorage
- Dark palette: bg #111, surface #1a1a2e, accent #4a9eff, text #e0e0e0

### Task 3.2: Add model configuration UI
- Add settings panel to select council models
- Dropdown with model presets from available providers
- Persist to localStorage
- Override via query params: `?models=openai/gpt-5.1,anthropic/claude-sonnet-4.5`

### Task 3.3: Wire streaming to frontend
- Replace batch `POST /message` with streaming `POST /message/stream`
- Show per-model loading spinners in Stage 1
- Show token-by-token streaming in Stage 1 tabs
- Show token-by-token streaming for Chairman in Stage 3

---

## Phase 4: Persistence Upgrade

### Task 4.1: Add Postgres storage backend (optional)
- Create `backend/storage_postgres.py` with asyncpg
- Auto-create tables on startup
- Fall back to JSON storage if Postgres unavailable
- Config via `DATABASE_URL` env var

### Task 4.2: Add conversation export
- Export to markdown
- Export to JSON
- Button in UI, CLI command

---

## Phase 5: Polish & Ship

### Task 5.1: Error boundary UI
- Show partial results when some models fail
- Display error cards for failed models with retry button
- Never fail the entire request due to partial failures

### Task 5.2: README & docs update
- Document multi-provider setup
- Document Hermes skill usage
- Document CLI usage
- Add configuration reference

### Task 5.3: Git tag and push
- Tag v0.2.0
- Push to origin
- Update upstream fork

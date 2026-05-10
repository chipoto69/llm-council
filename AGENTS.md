# LLM Council + Autocouncil

## What this is

A multi-model deliberation system with two modes:

### 1. Interactive Council (original Karpathy)
Web app where you submit a query → 3+ LLMs respond → they peer-review each other → chairman synthesizes. React frontend + FastAPI backend. Uses OpenRouter.

### 2. Autocouncil (Hermes-native, new)
Autonomous deliberation loop. Like `autoresearch` but for evaluation. Runs via Hermes CLI, so any model/provider works — local EXO, free OpenRouter, premium APIs.

```
uv run python -m backend.autocouncil "What is the best approach to X?" --json --quiet
```

Architecture:
- **Stage 1**: All council models produce first opinions (parallel)
- **Stage 2**: Each model anonymously peer-reviews and ranks the others
- **Stage 3**: Chairman synthesizes final answer from full context
- **Loop**: Repeat with previous synthesis as context until convergence or max rounds

## Council members

Any model Hermes supports. Pass as `model:provider` pairs:

```bash
# Premium mix
--models "grok-4.3:xai,deepseek-v4-pro:deepseek,z-ai/glm-5.1:openrouter"

# Free tier
--models "google/gemini-2.5-flash-lite:openrouter,qwen/qwen3-32b:openrouter"

# Local EXO models (once configured)
--models "mlx-community/Qwen2.5-7B:local,llama-3.1-8b:local"
```

## For agents

Drop-in judgment oracle. Any agent that can spawn a subprocess gets a multi-model deliberation:

```python
import subprocess, json
result = subprocess.run([
    "uv", "run", "python", "-m", "backend.autocouncil",
    "--models", "deepseek-v4-pro:deepseek,grok-4.3:xai",
    "--chairman", "gpt-5.5:openai-codex",
    "--max-rounds", "3",
    "--json", "--quiet",
    "Should I use Postgres or SQLite for this project?"
], capture_output=True, text=True)
data = json.loads(result.stdout)
print(data["final_answer"])
```

Returns: `{rounds, converged, convergence_reason, final_answer, answer_history, members, chairman}`

## Setup

```bash
cd ~/Desktop/ORGANIZED/ACTIVE_PROJECTS/llm-council
uv sync
echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env  # only if using OpenRouter models
```

## Project structure

```
backend/
  council.py          # Original 3-stage logic (OpenRouter only)
  autocouncil.py      # Hermes-native autonomous deliberation loop
  openrouter.py       # OpenRouter API client
  config.py           # Model config
  main.py             # FastAPI server
  storage.py          # JSON conversation storage
frontend/             # React + Vite web app
.hermes/plans/        # Enhancement plans
```

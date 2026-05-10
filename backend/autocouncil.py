"""
autocouncil.py — Hermes-native autonomous multi-model deliberation loop.

Each council member is a different Hermes model/provider combo, invoked via
`hermes -z "query" -m MODEL --provider PROVIDER`. This means you can use any
model Hermes supports — local EXO, Hugging Face MLX, free OpenRouter models,
or premium APIs — just by passing the right -m/--provider flags.

Usage:
    uv run python backend/autocouncil.py "What is the best way to X?"
    uv run python backend/autocouncil.py --models grok-4.3,deepseek-v4-pro,kimi-k2.5 "query"
    uv run python backend/autocouncil.py --max-rounds 3 "query"

Model format: "model_name:provider" (e.g., "grok-4.3:xai", "gemini-2.5-flash-lite:openrouter")
If provider omitted, uses the current default Hermes provider.
"""
import asyncio
import argparse
import json
import re
import sys
from typing import Optional
from dataclasses import dataclass, field


# Default council members as model:provider pairs
# These use models the user already has configured in Hermes
COUNCIL_MEMBERS = [
    ("grok-4.3", "xai"),
    ("deepseek-v4-pro", "deepseek"),
    ("kimi-k2.5", "kimi"),
]

CHAIRMAN = ("gpt-5.5", "openai-codex")


def parse_model_spec(spec: str) -> tuple[str, Optional[str]]:
    """Parse 'model:provider' or 'model' into (model, provider|None)."""
    if ":" in spec:
        model, provider = spec.split(":", 1)
        return model.strip(), provider.strip() or None
    return spec.strip(), None


async def query_hermes(model: str, provider: Optional[str], prompt: str, timeout: int = 180) -> Optional[str]:
    """
    Query Hermes with a specific model/provider combo.
    Uses: hermes -z "prompt" -m MODEL --provider PROVIDER
    """
    cmd = ["hermes", "-z", prompt, "-m", model]
    if provider:
        cmd.extend(["--provider", provider])
    
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            # Don't spam — just note failure
            return None
        return output
    except asyncio.TimeoutError:
        proc.kill()
        return None


async def query_all_parallel(members: list[tuple[str, Optional[str]]], prompt: str) -> dict[str, Optional[str]]:
    """Query all council members in parallel."""
    tasks = [query_hermes(m, p, prompt) for m, p in members]
    responses = await asyncio.gather(*tasks)
    result = {}
    for (model, provider), response in zip(members, responses):
        label = f"{model}" if not provider else f"{model} ({provider})"
        result[label] = response
    return result


def parse_ranking_section(text: str) -> list[str]:
    """Extract ranked labels from a review response."""
    ranking_section = ""
    if "FINAL RANKING:" in text:
        ranking_section = text.split("FINAL RANKING:")[1]
    else:
        ranking_section = text
    
    lines = ranking_section.strip().split("\n")
    ranking = []
    for line in lines:
        line = line.strip()
        # Match "1. Label" or "1) Label" or "1 - Label"
        match = re.match(r'^\d+[\.\)]\s*(.+?)$', line)
        if match:
            label = match.group(1).strip()
            # Remove trailing punctuation
            label = re.sub(r'[\.,;:!]+$', '', label)
            ranking.append(label)
        if len(ranking) >= 10:
            break
    
    return ranking


@dataclass
class ConvergenceState:
    round: int = 0
    prev_answer: Optional[str] = None
    prev_top: Optional[str] = None
    answer_history: list = field(default_factory=list)
    converged: bool = False
    convergence_reason: str = ""


def check_convergence(state: ConvergenceState, new_answer: str, rankings: dict) -> ConvergenceState:
    state.round += 1
    state.answer_history.append(new_answer)
    
    if state.round == 1:
        state.prev_answer = new_answer
        state.prev_top = list(rankings.keys())[0] if rankings else None
        return state
    
    new_top = list(rankings.keys())[0] if rankings else None
    
    # Criterion 1: Same top model 2 rounds in a row
    if state.prev_top and new_top and state.prev_top == new_top:
        state.converged = True
        state.convergence_reason = f"Stable winner: {new_top} leads 2 consecutive rounds"
        return state
    
    # Criterion 2: Answer length within 15% of previous
    if state.prev_answer and new_answer:
        ratio = len(new_answer) / max(len(state.prev_answer), 1)
        if 0.85 <= ratio <= 1.15 and state.round >= 2:
            state.converged = True
            state.convergence_reason = f"Answer stabilized (length ratio: {ratio:.2f})"
            return state
    
    state.prev_answer = new_answer
    state.prev_top = new_top
    return state


async def run_council_round(
    query: str,
    members: list[tuple[str, Optional[str]]],
    chairman: tuple[str, Optional[str]],
    round_num: int,
    previous_synthesis: Optional[str] = None,
    verbose: bool = True,
) -> tuple[dict, dict, str, dict]:
    """
    Execute one full council round (Stage 1 → 2 → 3).
    
    Returns: (stage1_responses, stage2_reviews, chairman_answer, aggregate_rankings)
    """
    
    # ── Stage 1: First opinions ──
    prompt = query
    if previous_synthesis and round_num > 1:
        prompt = (
            f"This is Round {round_num} of an iterative AI council deliberation.\n\n"
            f"Original question: {query}\n\n"
            f"Previous council synthesis (Round {round_num - 1}):\n{previous_synthesis[:2000]}\n\n"
            f"Your job: IMPROVE on the previous synthesis. Identify gaps, errors, "
            f"or shallow analysis. Provide a better, more complete answer.\n\n"
            f"Question: {query}"
        )
    
    stage1 = await query_all_parallel(members, prompt)
    valid = {k: v for k, v in stage1.items() if v is not None}
    
    if len(valid) < 2:
        return valid, {}, "", {}
    
    # ── Stage 2: Anonymized peer review ──
    labels = [chr(65 + i) for i in range(len(members))]  # A, B, C...
    model_keys = [f"{m}" + (f" ({p})" if p else "") for m, p in members]
    label_map = {}
    reverse_map = {}
    
    for i, key in enumerate(model_keys):
        if key in valid:
            lbl = f"Response {labels[i]}"
            label_map[lbl] = key
            reverse_map[key] = lbl
    
    anonymous_text = "\n\n".join([
        f"{reverse_map[key]}:\n{valid[key]}"
        for key in valid if key in reverse_map
    ])
    
    review_prompt = (
        f"You are evaluating responses to this question:\n\n\"{query}\"\n\n"
        f"Anonymized responses:\n\n{anonymous_text}\n\n"
        f"Your task:\n"
        f"1. Evaluate each response — what's strong, what's weak\n"
        f"2. Rank them best to worst\n\n"
        f"MUST end with exactly:\n"
        f"FINAL RANKING:\n"
        f"1. Response X\n"
        f"2. Response Y\n"
        f"..."
    )
    
    stage2 = await query_all_parallel(members, review_prompt)
    
    # Aggregate rankings
    votes = {}
    for reviewer, review in stage2.items():
        if not review:
            continue
        ranking = parse_ranking_section(review)
        for i, name in enumerate(ranking):
            if name in label_map:
                real_name = label_map[name]
                votes[real_name] = votes.get(real_name, 0) + max(3 - i, 1)
    
    aggregate = dict(sorted(votes.items(), key=lambda x: -x[1]))
    
    # ── Stage 3: Chairman synthesis ──
    cm, cp = chairman
    chair_label = f"{cm}" + (f" ({cp})" if cp else "")
    
    reviews_summary = "\n".join([
        f"**{r}:** {v[:200]}..."
        for r, v in stage2.items() if v
    ])
    
    chairman_prompt = (
        f"You are the Chairman of an AI Council. Synthesize the final answer.\n\n"
        f"Question: {query}\n\n"
        f"Individual responses:\n"
        + "\n".join(f"- {k}: {v[:300]}..." for k, v in valid.items())
        + f"\n\nPeer rankings (aggregate): {json.dumps(aggregate)}\n\n"
        f"Review highlights:\n{reviews_summary}\n\n"
        f"Produce a single comprehensive answer. Address the strongest points from "
        f"each contributor, note any disagreements, and give a clear final verdict."
    )
    
    chairman_resp = await query_hermes(cm, cp, chairman_prompt)
    chair_answer = chairman_resp if chairman_resp else (
        f"[Chairman unavailable — using top-ranked response]\n\n{list(valid.values())[0]}"
    )
    
    return valid, stage2, chair_answer, aggregate


async def run_autocouncil(
    query: str,
    models: Optional[list[str]] = None,
    chairman_model: Optional[str] = None,
    max_rounds: int = 5,
    verbose: bool = True,
):
    """Run the autonomous council deliberation loop."""
    
    # Parse council members
    if models:
        members = [parse_model_spec(m) for m in models]
    else:
        members = COUNCIL_MEMBERS
    
    # Parse chairman
    if chairman_model:
        chairman = parse_model_spec(chairman_model)
    else:
        chairman = CHAIRMAN
    
    state = ConvergenceState()
    
    member_labels = [f"{m}" + (f" ({p})" if p else "") for m, p in members]
    chair_label = f"{chairman[0]}" + (f" ({chairman[1]})" if chairman[1] else "")
    
    if verbose:
        print(f"\n{'='*65}")
        print(f"  AUTOCOUNCIL — Autonomous Multi-Model Deliberation")
        print(f"{'='*65}")
        print(f"  Query:    {query[:80]}{'...' if len(query) > 80 else ''}")
        print(f"  Council:  {', '.join(member_labels)}")
        print(f"  Chairman: {chair_label}")
        print(f"  Max:      {max_rounds} rounds")
        print(f"{'='*65}\n")
    
    for round_num in range(1, max_rounds + 1):
        if verbose:
            print(f"▸ Round {round_num}/{max_rounds}", end="", flush=True)
        
        try:
            prev = state.answer_history[-1] if state.answer_history else None
            s1, s2, answer, rankings = await run_council_round(
                query, members, chairman, round_num, prev, verbose
            )
            
            n_ok = len(s1)
            if verbose:
                print(f" → {n_ok} responded", end="")
                if rankings:
                    top3 = list(rankings.items())[:3]
                    print(f", top: {' > '.join(k.split()[0] for k,_ in top3)}", end="")
                print(f", chairman: {len(answer)} chars")
            
            state = check_convergence(state, answer, rankings)
            
            if state.converged:
                if verbose:
                    print(f"\n  ✓ CONVERGED: {state.convergence_reason}")
                break
                
        except Exception as e:
            if verbose:
                print(f"\n  ✗ ERROR: {e}")
            break
    
    if verbose:
        print(f"\n{'='*65}")
        print(f"  FINAL SYNTHESIS — {state.round} round(s)")
        if state.converged:
            print(f"  {state.convergence_reason}")
        else:
            print(f"  Max rounds reached, no convergence")
        print(f"{'='*65}\n")
        print(state.answer_history[-1] if state.answer_history else "(no answer)")
        print(f"\n{'='*65}\n")
    
    return {
        "query": query,
        "members": member_labels,
        "chairman": chair_label,
        "rounds": state.round,
        "converged": state.converged,
        "convergence_reason": state.convergence_reason,
        "final_answer": state.answer_history[-1] if state.answer_history else None,
        "answer_history": state.answer_history,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous multi-model deliberation via Hermes"
    )
    parser.add_argument("query", nargs="?", help="Question to deliberate on")
    parser.add_argument("--models", "-m", help="Comma-separated model:provider specs")
    parser.add_argument("--chairman", "-c", help="Chairman as model:provider")
    parser.add_argument("--max-rounds", "-n", type=int, default=5)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--json", "-j", action="store_true")
    
    args = parser.parse_args()
    
    if not args.query:
        args.query = input("Query: ").strip()
        if not args.query:
            print("No query provided.", file=sys.stderr)
            sys.exit(1)
    
    models = args.models.split(",") if args.models else None
    
    result = asyncio.run(run_autocouncil(
        query=args.query,
        models=models,
        chairman_model=args.chairman,
        max_rounds=args.max_rounds,
        verbose=not args.quiet,
    ))
    
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    
    return 0 if result["converged"] else 1


if __name__ == "__main__":
    sys.exit(main())

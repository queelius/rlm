"""CPU-testable terminal-RLOO math, not a collector, trainer or provenance auditor.

Inputs must first pass the existing native episode/request audit. This module checks
batch completeness and a common declared behavior-policy identity; it cannot prove
which weights were loaded. A caller must keep those weights fixed for collection
and pre-update replay. Backpropagate each credited call separately, accumulating
gradients, then at most one update for the batch. No optimizer is implemented here.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

TEMPERATURE = 0.5
EPISODES = 32


@dataclass(frozen=True)
class ActionCredit:
    task_id: str
    episode_id: str
    call_id: str
    advantage: float


def causal_inputs(record: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Exactly the saved prefix plus emitted[:-1], predicting every emitted token.

    This is the alignment used by rl_planner.root_logps, without that helper's
    temperature0.8 or64-episode denominator. No retokenization or EOS insertion.
    """
    prefix, emitted = record["input_token_ids"], record["output_token_ids"]
    if not prefix or not emitted or any(type(t) is not int or t < 0 for t in prefix + emitted):
        raise ValueError("nonempty exact native prefix and emitted token IDs required")
    if "request" in record and record["request"]["input_token_ids"] != prefix:
        raise ValueError("saved native prefix differs from request")
    return list(prefix) + list(emitted[:-1]), list(emitted)


def batch_credits(
    episodes: list[dict[str, Any]], calls: dict[str, dict[str, Any]]
) -> list[ActionCredit]:
    """Validate complete TRAIN8x4 and credit every emitted call, including errors.

    Flat groups contribute zero, but remain in the fixed32 denominator. No valid-
    action filter, reward normalization, per-turn mean or successful-path selection.
    """
    if len(episodes) != EPISODES or len({e["episode_id"] for e in episodes}) != EPISODES:
        raise ValueError("complete32 unique observed episodes required")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        if episode.get("observed") is not True or episode.get("native_score") not in (0, 1):
            raise ValueError("unavailable or ungraded episode cannot enter terminal RLOO")
        groups[episode["task_id"]].append(episode)
    if len(groups) != 8 or any(
        len(group) != 4 or {e["repeat"] for e in group} != {0, 1, 2, 3} for group in groups.values()
    ):
        raise ValueError("eight complete four-sample task groups required")
    result, seen, policies = [], set(), set()
    for task_id, group in groups.items():
        reward_sum = sum(e["native_score"] for e in group)
        for episode in group:
            advantage = float(episode["native_score"] - (reward_sum - episode["native_score"]) / 3)
            if not episode["call_ids"]:
                raise ValueError("observed trajectory has no emitted actions")
            for cid in episode["call_ids"]:
                if cid in seen or cid not in calls:
                    raise ValueError("reused or missing native action call")
                seen.add(cid)
                record = calls[cid]
                request = record["request"]
                if (
                    record.get("available") is not True
                    or record["call_id"] != cid
                    or record["episode_id"] != episode["episode_id"]
                    or request["task_id"] != task_id
                ):
                    raise ValueError("unavailable or mismatched episode action")
                causal_inputs(record)
                sampling = request["sampling"]
                if any(
                    sampling.get(k) != v
                    for k, v in {"temperature": 0.5, "top_p": 1.0, "top_k": 0}.items()
                ):
                    raise ValueError("behavior sampling must match untruncated temperature0.5")
                policy = tuple(
                    request.get(k)
                    for k in ("model_manifest_sha256", "adapter_sha256", "adapter_commit_sha256")
                )
                if not all(policy):
                    raise ValueError("declared frozen behavior-policy identity missing")
                policies.add(policy)
                result.append(ActionCredit(task_id, episode["episode_id"], cid, advantage))
    if seen != set(calls):
        raise ValueError("orphan calls imply incomplete or unplanned trajectories")
    if len(policies) != 1:
        raise ValueError("mixed behavior-policy identities within batch")
    return result


def action_loss(logits: Any, targets: list[int], advantage: float) -> Any:
    """Suffix-only emitted-token SUM atT0.5 divided by32, with autograd intact."""
    import torch

    if logits.ndim != 3 or tuple(logits.shape[:2]) != (1, len(targets)) or not targets:
        raise ValueError("logits shape must be exactly[1, emitted_count, vocabulary]")
    if not math.isfinite(advantage):
        raise ValueError("finite RLOO advantage required")
    target = torch.tensor([targets], dtype=torch.long, device=logits.device)
    logps = torch.log_softmax(logits.float() / TEMPERATURE, dim=-1)
    selected = logps.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return -float(advantage) * selected.sum() / EPISODES


def replay_action_loss(model: Any, record: dict[str, Any], advantage: float) -> Any:
    """One-call differentiable replay; earlier prompt positions are never targets."""
    import torch

    ids, targets = causal_inputs(record)
    tensor = torch.tensor([ids], dtype=torch.long, device=model.device)
    logits = model(input_ids=tensor, use_cache=False, logits_to_keep=len(targets)).logits
    return action_loss(logits, targets, advantage)

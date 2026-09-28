"""Read-only native reward/credit and sampled-action analysis; never training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rl_resume_20260928"))

import probe_common as p  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "rl_signal_existing_trainer", HERE.parent / "rl_resume_20260928/train.py"
)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def action_key(action: dict, projected: bool = False) -> str:
    """Native argument identity modulo notes/finish wording, not future-state identity."""
    value = {k: v for k, v in action.items() if k != "note"}
    if value["action"] == "finish":
        value.pop("message", None)
    if projected and value["action"] == "craft":
        value.pop("ingredients")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def parsed(text: str) -> dict | None:
    try:
        return p.c.bridge.parse_action(text)
    except (ValueError, TypeError):
        return None


def entropy(counts) -> float:
    total = sum(counts)
    return -sum((n / total) * math.log(n / total) for n in counts if n) if total else 0.0


def prefix_hash(call: dict) -> str:
    return hashlib.sha256(json.dumps(call["input_token_ids"]).encode()).hexdigest()


def choice_diversity(calls: list[dict], require_identical: bool = False) -> dict:
    prefixes = {prefix_hash(call) for call in calls}
    if require_identical and len(prefixes) != 1:
        raise ValueError("initial prefixes differ; not an identical-policy-state comparison")
    semantic, projected = Counter(), Counter()
    invalid = 0
    for call in calls:
        action = parsed(call["text"])
        if action is None:
            invalid += 1
            key = json.dumps({"invalid_schema_raw_text": call["text"]}, sort_keys=True)
            semantic[key] += 1
            projected[key] += 1
        else:
            semantic[action_key(action)] += 1
            projected[action_key(action, projected=True)] += 1
    return dict(
        samples=len(calls),
        exact_prefixes=len(prefixes),
        unique_raw_texts=len({call["text"] for call in calls}),
        unique_semantic_actions=len(semantic),
        unique_projected_choices=len(projected),
        invalid_schema=invalid,
        empirical_semantic_entropy_nats=entropy(semantic.values()),
        empirical_projected_entropy_nats=entropy(projected.values()),
        semantic_action_counts=dict(semantic),
        projected_choice_counts=dict(projected),
    )


def member_spans(text: str) -> list[tuple[int, int, str]]:
    """Exact top-level member character ranges, including key/colon/value, no comma."""
    decoder = json.JSONDecoder()
    spans = []
    pos = len(text) - len(text.lstrip())
    if text[pos] != "{":
        raise ValueError("JSON object required")
    pos += 1
    while True:
        while text[pos].isspace():
            pos += 1
        if text[pos] == "}":
            break
        start = pos
        key, pos = decoder.raw_decode(text, pos)
        while text[pos].isspace():
            pos += 1
        if text[pos] != ":":
            raise ValueError("invalid JSON member")
        pos += 1
        while text[pos].isspace():
            pos += 1
        _, pos = decoder.raw_decode(text, pos)
        label = (
            "ingredients"
            if key == "ingredients"
            else "decisions"
            if key in {"action", "target_item", "output_count"}
            else "other_fields"
        )
        spans.append((start, pos, label))
        while text[pos].isspace():
            pos += 1
        if text[pos] == "}":
            break
        if text[pos] != ",":
            raise ValueError("invalid JSON separator")
        pos += 1
    return spans


def span_counts(text: str, offsets: list[tuple[int, int]]) -> dict[str, int]:
    """Do not force cross-member punctuation tokens into an arbitrary field."""
    spans, counts = member_spans(text), Counter()
    for start, end in offsets:
        overlap = [(a, b, label) for a, b, label in spans if a < end and start < b]
        contained = [label for a, b, label in overlap if a <= start and end <= b]
        if len(contained) == 1:
            label = contained[0]
        elif any(label == "ingredients" for _, _, label in overlap):
            label = "ingredient_boundary"
        elif overlap:
            label = "other_boundary"
        else:
            label = "structural"
        counts[label] += 1
    return dict(counts)


def craft_tokens(call: dict, tokenizer) -> dict:
    """Offsets count only when re-encoding exactly reproduces the sampled text IDs."""
    encoded = tokenizer(call["text"], add_special_tokens=False, return_offsets_mapping=True)
    emitted = call["output_token_ids"]
    nonspecial = [t for t in emitted if t not in tokenizer.all_special_ids]
    if encoded["input_ids"] != nonspecial:
        return dict(exact=False, reason="retokenization_does_not_equal_emitted_ids")
    counts = span_counts(call["text"], encoded["offset_mapping"])
    counts["special_tokens"] = len(emitted) - len(nonspecial)
    if sum(counts.values()) != len(emitted):
        raise ValueError("token attribution failed conservation")
    return dict(exact=True, emitted_tokens=len(emitted), counts=counts)


def credit_totals(rows: list[dict]) -> dict:
    return dict(
        calls=len(rows),
        episodes=len({r["episode_id"] for r in rows}),
        tasks=len({r["task_id"] for r in rows}),
        tokens=sum(r["tokens"] for r in rows),
        advantage_weighted_tokens=sum(r["advantage"] * r["tokens"] for r in rows),
        absolute_advantage_weighted_tokens=sum(abs(r["advantage"]) * r["tokens"] for r in rows),
    )


def error_type(history: dict) -> str:
    feedback = history["feedback"]
    if "response" in history:
        return "invalid_schema"
    if isinstance(feedback, str) and feedback.startswith("Rejected action:"):
        return "rejected_action"
    if isinstance(feedback, str) and feedback.startswith("Error:"):
        return "native_action_error"
    if isinstance(feedback, list) and any(isinstance(x, dict) and "error" in x for x in feedback):
        return "query_item_error"
    return "no_explicit_error"


def entropy_stats(ids: list[str], metrics: dict) -> dict:
    hs = [h for cid in ids for h in metrics[cid]["token_entropies"]]
    logps = [v for cid in ids for v in metrics[cid]["logps"]]
    return dict(
        calls=len(ids),
        tokens=len(hs),
        mean_sampled_prefix_vocabulary_entropy_nats=mean(hs) if hs else None,
        mean_sampled_token_surprisal_nats=-mean(logps) if logps else None,
        mean_first_token_entropy_nats=(
            mean(metrics[cid]["token_entropies"][0] for cid in ids) if ids else None
        ),
    )


def attribution_summary(rows: list[dict]) -> dict:
    exact = [r for r in rows if r["token_attribution"]["exact"]]
    counts = sum((Counter(r["token_attribution"]["counts"]) for r in exact), Counter())
    total = sum(counts.values())
    lower = counts["ingredients"]
    upper = lower + counts["ingredient_boundary"]
    return dict(
        calls=len(rows),
        exact_calls=len(exact),
        excluded_calls=len(rows) - len(exact),
        emitted_tokens=total,
        counts=dict(counts),
        ingredient_fraction_lower=lower / total if total else None,
        ingredient_fraction_upper=upper / total if total else None,
        decision_member_fraction=counts["decisions"] / total if total else None,
    )


def analyze(directory: Path) -> dict:
    from transformers import AutoTokenizer

    plan_path = directory / "PLAN.json"
    plan = p.read(plan_path)
    terminals = list(directory.glob("TERMINAL-*.json"))
    if len(terminals) != 1 or not p.read(terminals[0])["complete"]:
        raise ValueError("one completed owner terminal required; no partial-reward substitution")
    calls, credits, _, admission = trainer.load_batch(
        directory, {"collection_plan_sha256": p.sha(plan_path)}
    )
    native = p.read(directory / "NATIVE-AUDIT.json")
    episodes = [p.read(directory / "episodes" / (j["episode_id"] + ".json")) for j in plan["jobs"]]
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"], local_files_only=True, trust_remote_code=False
    )
    metrics, metrics_sha = {}, {}
    for cid, call in calls.items():
        path = directory / "generation-logps" / (cid + ".json")
        value = p.read(path)
        n = len(call["output_token_ids"])
        if len(value["logps"]) != n or len(value["token_entropies"]) != n:
            raise ValueError("captured probability/entropy lengths differ from emitted tokens")
        if not all(math.isfinite(x) for x in value["logps"] + value["token_entropies"]):
            raise ValueError("nonfinite captured probability/entropy")
        metrics[cid], metrics_sha[str(path)] = value, p.sha(path)
    credit = {v.call_id: v.advantage for v in credits}
    call_rows, episode_rows, first_ids = [], [], []
    for episode in episodes:
        if episode["node_ids"] != ["n0"]:
            raise ValueError("this census is scoped to flat root episodes")
        checked = native["audits"][episode["episode_id"]]
        if not checked["replayed"] or checked["native_score"] != episode["native_score"]:
            raise ValueError("episode reward differs from native replay")
        node = p.read(directory / "nodes" / (episode["episode_id"] + "-n0.json"))
        if node["call_ids"] != episode["call_ids"]:
            raise ValueError("node/episode history ordering differs")
        assists = {row["call_id"]: row for row in node.get("execution_assists", [])}
        first_ids.append(episode["call_ids"][0])
        local = []
        for step, (cid, history) in enumerate(
            zip(node["call_ids"], node["public_history"], strict=True)
        ):
            call, action = calls[cid], parsed(calls[cid]["text"])
            executed = history.get("action")
            if action is not None:
                if cid in assists:
                    if assists[cid]["requested_action"] != action:
                        raise ValueError("binder requested action differs from emitted JSON")
                    if assists[cid]["executed_action"] != executed:
                        raise ValueError("binder executed action differs from native history")
                elif action != executed:
                    raise ValueError("raw action differs from native history")
            elif call["text"] != history["response"]:
                raise ValueError("schema failure text differs from history")
            row = dict(
                call_id=cid,
                episode_id=episode["episode_id"],
                task_id=episode["task_id"],
                repeat=episode["repeat"],
                step=step,
                reward=episode["native_score"],
                advantage=credit[cid],
                tokens=len(call["output_token_ids"]),
                action=action["action"] if action else "invalid_schema",
                error=error_type(history),
                exact_prefix_sha256=prefix_hash(call),
                requested_action=action,
                executed_action=executed,
                binder_changed_arguments=bool(action and action != executed),
                binding_reason=assists.get(cid, {}).get("binding_reason"),
            )
            if row["action"] == "craft":
                row["token_attribution"] = craft_tokens(call, tokenizer)
            call_rows.append(row)
            local.append(row)
        episode_rows.append(
            dict(
                episode_id=episode["episode_id"],
                task_id=episode["task_id"],
                repeat=episode["repeat"],
                seed=episode["seed"],
                reward=episode["native_score"],
                advantage=credit[episode["call_ids"][0]],
                status=episode["status"],
                **credit_totals(local),
            )
        )
    first_set = set(first_ids)
    initial = {}
    for task in native["groups"]:
        ids = [cid for cid in first_ids if calls[cid]["request"]["task_id"] == task]
        first_h = [metrics[cid]["token_entropies"][0] for cid in ids]
        initial[task] = dict(
            **choice_diversity([calls[cid] for cid in ids], require_identical=True),
            call_ids=ids,
            exact_prefix_sha256=prefix_hash(calls[ids[0]]),
            first_token_entropy_range_nats=[min(first_h), max(first_h)],
        )
    later_prefixes = defaultdict(list)
    for row in call_rows:
        if row["call_id"] not in first_set:
            later_prefixes[row["exact_prefix_sha256"]].append(row["call_id"])
    repeated = [
        dict(
            prefix_sha256=key,
            call_ids=ids,
            **choice_diversity([calls[cid] for cid in ids], require_identical=True),
        )
        for key, ids in later_prefixes.items()
        if len(ids) > 1
    ]
    crafts = [row for row in call_rows if row["action"] == "craft"]
    credited = [row for row in call_rows if row["advantage"]]
    categories = sorted({(r["action"], r["error"]) for r in call_rows})
    result = dict(
        schema="textcraft-rl-signal-20260928-v1",
        directory=str(directory),
        execution_mode=plan["execution_mode"],
        update=plan["update"],
        adapter=plan["adapter"],
        input_sha256={
            str(path): p.sha(path)
            for path in [
                plan_path,
                directory / "NATIVE-AUDIT.json",
                directory / "SUMMARY.json",
                terminals[0],
            ]
        },
        source_sha256={
            str(path): p.sha(path)
            for path in [
                Path(__file__),
                Path(trainer.__file__),
                Path(p.__file__),
                Path(p.rl.loss_math.__file__),
            ]
        },
        generation_sidecar_sha256=metrics_sha,
        verified_native_receipts=len(native["receipt_sha256"]),
        planned_episodes=32,
        native_successes=native["successes"],
        native_success_rate=native["successes"] / 32,
        mixed_groups=native["mixed_groups"],
        groups={
            task: dict(info, episodes=[r for r in episode_rows if r["task_id"] == task])
            for task, info in native["groups"].items()
        },
        physical_cost=admission["physical_cost"],
        native_errors=native["errors"],
        episode_status_counts=dict(Counter(e["status"] for e in episodes)),
        all_calls=credit_totals(call_rows),
        credited_calls=credit_totals(credited),
        by_credit_sign={
            label: credit_totals([r for r in call_rows if predicate(r["advantage"])])
            for label, predicate in (
                ("positive", lambda a: a > 0),
                ("negative", lambda a: a < 0),
                ("zero", lambda a: a == 0),
            )
        },
        by_action_error={
            f"{action}/{error}": {
                label: credit_totals(
                    [
                        r
                        for r in call_rows
                        if r["action"] == action
                        and r["error"] == error
                        and predicate(r["advantage"])
                    ]
                )
                for label, predicate in (
                    ("positive", lambda a: a > 0),
                    ("negative", lambda a: a < 0),
                    ("zero", lambda a: a == 0),
                )
            }
            for action, error in categories
        },
        entropy={
            "all": entropy_stats(list(calls), metrics),
            "initial": entropy_stats(first_ids, metrics),
            "later": entropy_stats([cid for cid in calls if cid not in first_set], metrics),
            "credited": entropy_stats([r["call_id"] for r in credited], metrics),
            "interpretation": "Full-vocabulary conditional entropy at sampled token prefixes "
            "after temperature0.5; not sequence/task entropy or a learning change.",
        },
        initial_identical_prefix_diversity=initial,
        later_diversity=dict(
            calls=len(calls) - len(first_ids),
            exact_prefixes=len(later_prefixes),
            repeated_prefix_groups=len(repeated),
            repeated_prefix_groups_with_semantic_variation=sum(
                r["unique_semantic_actions"] > 1 for r in repeated
            ),
            repeated_prefix_groups_with_projected_variation=sum(
                r["unique_projected_choices"] > 1 for r in repeated
            ),
            repeated_prefixes=repeated,
            caveat="Different histories/inventories/budgets are different policy states; "
            "only exact saved input-token-prefix matches are grouped.",
        ),
        craft_token_attribution={
            "schema_valid_all": attribution_summary(crafts),
            "schema_valid_nonzero_credit": attribution_summary(
                [r for r in crafts if r["advantage"]]
            ),
            "native_successful_crafts": attribution_summary(
                [r for r in crafts if r["error"] == "no_explicit_error"]
            ),
            "binder_changed_arguments_calls": sum(r["binder_changed_arguments"] for r in crafts),
            "method": "Exact emitted IDs after removing actual special tokens must equal "
            "native tokenizer re-encoding. Whole tokens wholly inside key:JSON-value member "
            "spans are counted; tokens crossing ingredient boundaries form an explicit upper "
            "bound. JSON syntax outside members and EOS remain separate. Decision members are "
            "action/target_item/output_count. These are costs, not justified gradient masks.",
        },
        call_rows=call_rows,
        episode_rows=episode_rows,
        limitation="Collection diagnostics only: no optimizer result, learning improvement, "
        "held-out generalization or recursion benefit. Semantic keys ignore formatting/note "
        "and finish wording; projected craft keys also drop ingredients but are NOT equivalent "
        "actions under the raw environment. Four samples give sparse empirical diversity.",
    )
    if result["all_calls"]["tokens"] != native["physical_cost"]["completion_tokens"]:
        raise ValueError("physical emitted-token count differs from credit census")
    return result


def compare(raw: dict, binder: dict) -> dict:
    left = {(e["task_id"], e["repeat"]): e for e in raw["episode_rows"]}
    right = {(e["task_id"], e["repeat"]): e for e in binder["episode_rows"]}
    if set(left) != set(right) or raw["adapter"] != binder["adapter"]:
        raise ValueError("paired task/repeat/checkpoint identity differs")
    pairs = []
    for key, a in left.items():
        b = right[key]
        if a["seed"] != b["seed"]:
            raise ValueError("paired execution seeds differ")
        pairs.append(
            dict(
                task_id=key[0],
                repeat=key[1],
                reward_delta=b["reward"] - a["reward"],
                calls_delta=b["calls"] - a["calls"],
                tokens_delta=b["tokens"] - a["tokens"],
            )
        )
    return dict(
        schema="textcraft-rl-collection-interface-comparison-20260928-v1",
        direction="binder minus raw; same warm model, no training effect",
        pairs=len(pairs),
        wins=sum(r["reward_delta"] > 0 for r in pairs),
        losses=sum(r["reward_delta"] < 0 for r in pairs),
        ties=sum(r["reward_delta"] == 0 for r in pairs),
        success_rate_difference=mean(r["reward_delta"] for r in pairs),
        mixed_groups={"raw": raw["mixed_groups"], "binder": binder["mixed_groups"]},
        credited_calls={"raw": raw["credited_calls"], "binder": binder["credited_calls"]},
        physical_cost={"raw": raw["physical_cost"], "binder": binder["physical_cost"]},
        rows=pairs,
        limitation="Eight exposed TRAIN clusters in one world; four execution seeds are not "
        "32 independent tasks. Assistance changes trajectories and gradient token dose.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--raw-report", type=Path)
    parser.add_argument("--binder-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable report exists: " + str(args.output))
    if args.collection:
        report = analyze(args.collection.resolve())
        summary = {
            k: report[k]
            for k in ("native_successes", "mixed_groups", "physical_cost", "credited_calls")
        }
    elif args.raw_report and args.binder_report:
        report = compare(p.read(args.raw_report), p.read(args.binder_report))
        report["input_sha256"] = {
            str(path.resolve()): p.sha(path) for path in (args.raw_report, args.binder_report)
        }
        summary = {k: report[k] for k in ("wins", "losses", "ties", "success_rate_difference")}
    else:
        parser.error("supply --collection or both report paths")
    p.c.save(args.output, report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

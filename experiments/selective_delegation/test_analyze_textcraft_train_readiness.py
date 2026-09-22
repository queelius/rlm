import importlib.util
import json


def test_readiness_keeps_incomplete_unknown_and_counts_mixed_despite_action_errors():
    assert importlib.util.find_spec("analyze_textcraft_train_readiness") is not None
    import analyze_textcraft_train_readiness as a

    jobs = [dict(episode_id=f"{p}-{r}", task_id=p, repeat=r) for p in ("a", "b") for r in range(4)]
    rows = {
        "a-0": dict(observed=True, native_score=0, errors={}),
        "a-1": dict(observed=False, native_score=None, errors={}),
        **{
            f"b-{r}": dict(
                observed=True, native_score=int(r == 0), errors={"native_action_error": 1}
            )
            for r in range(4)
        },
    }
    audits = {
        eid: dict(replayed=True, native_score=row["native_score"], errors=row["errors"])
        for eid, row in rows.items()
        if row["observed"]
    }
    calls = [
        dict(
            episode_id="a-2",
            available=True,
            usage={"prompt_tokens": 7, "completion_tokens": 3},
            started=1,
            ended=2,
        )
    ]
    result = a.group_readiness(jobs, rows, calls, audits)
    assert result["a"]["classification"] == "incomplete"
    assert (result["a"]["observed"], result["a"]["missing"], result["a"]["unavailable"]) == (
        1,
        2,
        1,
    )
    assert result["a"]["success_rate_bounds"] == [0, 0.75]
    assert result["a"]["cost"]["calls"] == 1  # Orphan call still costs resources.
    assert result["b"]["classification"] == "mixed"
    assert result["b"]["successes"] == 1 and result["b"]["observed_failures"] == 3
    assert result["b"]["error_episode_counts"]["native_action_error"] == 4


def test_saved_native_episode_regrades_before_group_count():
    import analyze_textcraft_train_readiness as a
    from transformers import AutoTokenizer

    root = a.reader.c.ROOT
    output = root / "textcraft-trained-readout-001"
    plan = a.audit.read(output / "PLAN.json")
    job = next(j for j in plan["jobs"] if j["episode_id"] == "t00-r0-flat-original")
    task = json.loads((root / "textcraft-inputs-001/tasks.jsonl").read_text().splitlines()[0])
    row = a.audit.read(output / "episodes" / (job["episode_id"] + ".json"))
    calls = {cid: a.audit.read(output / "calls" / (cid + ".json")) for cid in row["call_ids"]}
    nodes = {
        nid: a.audit.read(output / "nodes" / (job["episode_id"] + "-" + nid + ".json"))
        for nid in row["node_ids"]
    }
    tokenizer = AutoTokenizer.from_pretrained(a.reader.c.BASE, local_files_only=True)
    native = a.audit.audit_episode(
        task,
        job,
        row,
        calls,
        nodes,
        plan,
        a.reader.c.inputs.sha(output / "PLAN.json"),
        tokenizer,
        a.reader.c.bridge.load_world(),
    )
    assert native["replayed"] and native["native_score"] == 0
    group = a.group_readiness(
        [job], {job["episode_id"]: row}, list(calls.values()), {job["episode_id"]: native}
    )[task["id"]]
    assert group["classification"] == "all_zero" and group["successes"] == 0
    assert group["cost"]["calls"] == 73

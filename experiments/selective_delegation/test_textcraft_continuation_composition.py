def test_actual_saved_train_plan_runtime_tasks_and_native_batch():
    import rl_textcraft_terminal as rl
    from transformers import AutoTokenizer

    source = rl.c.ROOT / "textcraft-train-readiness-001"
    plan = rl.audit.read(source / "PLAN.json")
    tasks = {t["id"]: t for t in rl.readiness.runtime_tasks(plan)}
    assert len(tasks) == 8
    tokenizer = AutoTokenizer.from_pretrained(
        rl.c.BASE, local_files_only=True, trust_remote_code=False
    )
    episodes, calls, credits, audits = rl.native_batch(
        source, plan, tokenizer, tasks, rl.c.bridge.load_world()
    )
    assert len(episodes) == len(audits) == 32
    assert len(calls) == len(credits) == 770
    assert sum(x.advantage != 0 for x in credits) == 388

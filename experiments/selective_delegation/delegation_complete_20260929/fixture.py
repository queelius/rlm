"""Real saved screen prefixes and two complete native trees through actual CPU request capture."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit  # noqa: E402
import complete as c  # noqa: E402


class SavedScreenModel:
    device = "cpu"

    def __init__(self, tokenizer, mode):
        self.tokenizer, self.spec, self.matched = tokenizer, None, 0
        self.source = c.ROOT / f"textcraft-decomp-flexible-{mode}-20260928-001"
        self.saved = {
            i: c.read(self.source / "calls" / f"decomp-val494-{mode}-r0-c{i:03d}.json")
            for i in range(32)
        }

    def parameters(self):
        return iter(())

    def generate(self, input_ids, attention_mask, generation_config):
        import torch

        index = self.spec["global_call_index"]
        if index < 32:
            saved = self.saved[index]
            if self.spec["prompt"] != saved["request"]["prompt"] or (
                input_ids[0].tolist() != saved["input_token_ids"]
            ):
                raise ValueError("actual saved-screen prompt/token IDs differ")
            tokens = saved["output_token_ids"]
            self.matched += 1
        else:
            if index != 32 or self.spec["depth"] != 0:
                raise ValueError("fixture expects one root finish after32 recorded responses")
            tokens = self.tokenizer.encode(
                '{"action":"finish","message":"scripted fixture cutoff"}',
                add_special_tokens=False,
            ) + [self.tokenizer.eos_token_id]
        if len(tokens) > generation_config.max_new_tokens:
            raise ValueError("fixture exceeded real response limit")
        return torch.tensor([input_ids[0].tolist() + tokens], dtype=torch.long)


def success_model(tokenizer):
    previous = sys.modules.get("run")
    sys.modules["run"] = c.flex
    try:
        old = c.data.load(c.flex.OLD / "fixture.py", "complete_success_fixture")
    finally:
        if previous is None:
            sys.modules.pop("run", None)
        else:
            sys.modules["run"] = previous
    old.routing = c.flex.routing
    return old.ScriptedModel(tokenizer, inject_errors=True)


def one_case(output, mode, *, saved):
    plan, tasks, world, tokenizer, collector, _ = c.build(mode, output, fixture=True)
    plan["fixture_kind"] = (
        "actual32response_prefix_then_scripted_finish"
        if saved
        else ("public_only_scripted_success_with_root_child_errors")
    )
    c.persist(output / "PLAN.json", plan)

    class Client(collector.NativeClient):
        def call(self, spec):
            self.model.spec = spec
            return super().call(spec)

    def model():
        return SavedScreenModel(tokenizer, mode) if saved else success_model(tokenizer)

    client = Client(
        model(),
        tokenizer,
        output,
        time.time() + 120,
        plan["model_manifest_sha256"],
        c.sha(output / "PLAN.json"),
    )
    matched, references = 0, {}
    for job in plan["jobs"]:
        client.model = model()
        row = collector.episode(tasks[0], job, client, world, output, time.time() + 120)
        if not row["observed"]:
            raise ValueError("scripted native fixture must complete: " + str(row["failure"]))
        if saved:
            matched += client.model.matched
            source = client.model.source
            for path in [source / "PLAN.json", *(source / "calls").glob("*.json")]:
                references[str(path)] = c.sha(path)
        elif row["native_score"] != 1:
            raise ValueError("public-only scripted construction failed")
    report = audit.analyze(output, terminal=False)
    if report["observed"] != 2 or report["full_native_replays"] != 2:
        raise ValueError("both complete roots must independently replay")
    if report["physical_cost"]["calls"] != client.returned or client.failed:
        raise ValueError("fixture global returned-call accounting differs")
    report["matched_saved_calls"] = matched
    report["saved_screen_sha256"] = references
    c.persist(output / "NATIVE-AUDIT.json", report)
    return {
        k: report[k]
        for k in (
            "output",
            "routing",
            "observed",
            "successes",
            "full_native_replays",
            "matched_saved_calls",
            "physical_cost",
            "role_cost",
            "errors",
        )
    }


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError("fresh immutable fixture directory required")
    cases = [one_case(output / mode, mode, saved=True) for mode in c.MODES]
    success = one_case(output / "scripted-success", "fixed", saved=False)
    result = dict(
        schema="complete-delegation-actual-request-fixture-20260929-v1",
        passed=True,
        scientific_model_calls=0,
        GPU_loaded=False,
        saved_screen_prefixes=cases,
        scripted_success=success,
        source_sha256={str(path): c.sha(path) for path in c.HERE.glob("*.py")},
        interpretation="Each new-seed fixture reuses all32 actual saved screen prompts, exact "
        "input/output token IDs, then one explicitly scripted finish (not model efficacy). "
        "Seeds/request identities change as declared; prompts/token IDs do not. Both roots "
        "replay full native trees. Separate two-job public-only scripted construction tests "
        "successful root/child work and charged schema/native errors. No scientific generation.",
    )
    c.persist(output / "FIXTURE.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report = run(parser.parse_args().output)
    print(json.dumps({k: report[k] for k in ("passed", "scientific_model_calls", "GPU_loaded")}))

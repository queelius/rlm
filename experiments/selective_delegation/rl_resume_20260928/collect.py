"""Fresh FP16 native TRAIN rollouts; raw/binder share model, seeds and budgets."""

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path

import probe_common as p


class MetricCapture(p.rl.GenerationCapture):
    def generate(self, **kwargs):
        import torch

        result = self.model.generate(**kwargs, return_dict_in_generate=True, output_scores=True)
        prefix = kwargs["input_ids"].shape[1]
        self.tokens = result.sequences[0, prefix:].detach().cpu().tolist()
        self.logps, self.entropies = [], []
        for score, token in zip(result.scores, self.tokens, strict=True):
            logp = torch.log_softmax(score.float(), -1)
            self.logps.append(float(logp[0, token]))
            terms = torch.where(torch.isfinite(logp), logp.exp() * logp, 0)
            self.entropies.append(float(-terms.sum()))
        return result.sequences


class MetricsClient(p.c.NativeClient):
    def __init__(self, model, *args, **kwargs):
        self.capture = MetricCapture(model)
        super().__init__(self.capture, *args, **kwargs)

    def call(self, spec):
        record = super().call(spec)
        if record["available"]:
            if self.capture.tokens != record["output_token_ids"]:
                raise ValueError("generation score/token mismatch")
            p.c.save(
                self.output / "generation-logps" / (spec["call_id"] + ".json"),
                dict(
                    call_sha256=p.sha(self.output / "calls" / (spec["call_id"] + ".json")),
                    logps=self.capture.logps,
                    token_entropies=self.capture.entropies,
                    entropy_policy="Full vocabulary after temperature0.5; top_p1, top_k0",
                ),
            )
        return record


def run(args):
    plan, rows = p.prepare_collection(args, Path(__file__))
    if args.prepare_only:
        print(
            json.dumps(
                dict(
                    prepared=True,
                    GPU_loaded=False,
                    output=str(args.output),
                    episodes=plan["planned_episodes"],
                    plan_sha256=p.sha(args.output / "PLAN.json"),
                )
            )
        )
        return
    if list(args.output.glob("OWNER-*.json")) or (args.output / "calls").exists():
        raise ValueError("existing attempt; use a separately named explicit retry")
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    end = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + plan["budget_seconds"], end - 600)
    if deadline <= time.time() + 180:
        raise ValueError("insufficient allocation margin")
    collector, _ = p.implementation(args.mode)
    collector.STOP = False
    lock = (p.c.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner, started = uuid.uuid4().hex[:12], time.time()
    p.c.save(
        args.output / f"OWNER-{owner}.json",
        dict(
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            started=started,
            deadline=deadline,
            allocation_end=end,
            source=str(Path(__file__).resolve()),
            source_sha256=plan["source_sha256"],
        ),
    )
    model = base = client = None
    complete, failure, native = False, None, None

    def stop(*_):
        collector.STOP = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("parent must assign exactly one GPU")
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(
            p.c.BASE, local_files_only=True, trust_remote_code=False
        )
        base = AutoModelForCausalLM.from_pretrained(
            p.c.BASE,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            base,
            plan["adapter"]["path"],
            adapter_name="textcraft_action",
            is_trainable=False,
            autocast_adapter_dtype=True,
        )
        p.rl.set_mode(model, training=False)
        p.c.save(
            args.output / "LOAD.json",
            dict(
                base_dtype=str(model.get_base_model().dtype),
                lora_dtypes=sorted(
                    {str(v.dtype) for n, v in model.named_parameters() if "lora_" in n}
                ),
                adapter=plan["adapter"],
                gpu=torch.cuda.get_device_name(),
                cuda=torch.version.cuda,
                python=sys.version,
                executable=sys.executable,
                versions={
                    name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "peft", "safetensors")
                },
            ),
        )
        client = MetricsClient(
            model,
            tokenizer,
            args.output,
            deadline,
            plan["model_manifest_sha256"],
            p.sha(args.output / "PLAN.json"),
            adapter=plan["adapter"],
        )
        world, lookup = p.c.bridge.load_world(), {t["id"]: t for t in rows}
        for job in plan["jobs"]:
            if (args.output / "STOP").exists():
                raise TimeoutError("authenticated owner STOP file")
            row = collector.episode(
                lookup[job["task_id"]], job, client, world, args.output, deadline
            )
            if not row["observed"]:
                raise RuntimeError(row["failure"])
        # Release model memory before independent CPU replay; no model outcome is a substitute.
        client = model = base = None
        gc.collect()
        torch.cuda.empty_cache()
        native = p.audit_collection(args.output, plan, rows, tokenizer, world, args.mode)
        p.c.save(args.output / "NATIVE-AUDIT.json", native)
        complete = True
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        client = model = base = None
        gc.collect()
        torch.cuda.empty_cache()
        p.c.save(
            args.output / "SUMMARY.json",
            dict(
                complete=complete,
                failure=failure,
                native_audit=str(args.output / "NATIVE-AUDIT.json") if native else None,
                collection=p.c.summarize(args.output, plan),
            ),
        )
        p.c.save(
            args.output / f"TERMINAL-{owner}.json",
            dict(
                complete=complete,
                failure=failure,
                stopped=collector.STOP,
                ended=time.time(),
                elapsed_seconds=time.time() - started,
                deadline=deadline,
            ),
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("raw", "binder"), required=True)
    parser.add_argument("--phase", choices=("collect", "readout"), default="collect")
    parser.add_argument("--update", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=p.WARM)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1.25)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())

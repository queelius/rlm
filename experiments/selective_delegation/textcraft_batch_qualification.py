"""Prospective 15-minute native serving qualification; never an automatic GPU launch."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

import eval_textcraft_public as public

c = public.collector
SEED = 2026092241
ALIAS = "textcraft-public056-checkpoint23"
SIDE = c.ROOT.parent
OLD = SIDE / "strict-rlm-temperature-adherence-v1"
HELPER = OLD / "scripts/launch.py"
PRIME = Path("/project/alex_phd/envs/prime-rl-5990b1b")
DRIVER = Path("/export/software/system/nvidia/580.126.20")
save, sha = c.save, c.inputs.sha


def select(rows):
    ordered = sorted(rows, key=lambda r: (len(r["request"]["input_token_ids"]), r["call_id"]))
    if len(ordered) < 32:
        raise ValueError("at least32 fixed original-prompt requests required")

    def key(row):
        return hashlib.sha256(f"{SEED}:{row['call_id']}".encode()).hexdigest()

    middle = len(ordered) // 2
    return sorted(ordered[:middle], key=key)[:16] + sorted(ordered[middle:], key=key)[:16]


def request_body(row):
    req = row["request"]
    return dict(
        model=ALIAS,
        prompt=req["input_token_ids"],
        max_tokens=req["cap"],
        temperature=0.5,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        seed=req["seed"],
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=1,
        stream=False,
        echo=False,
        return_token_ids=True,
        skip_special_tokens=True,
        truncate_prompt_tokens=None,
        ignore_eos=False,
        stop_token_ids=[151645],
        min_tokens=0,
    )


def normalize(row, response, tokenizer):
    if response.get("model") != ALIAS or len(response.get("choices", [])) != 1:
        raise ValueError("native model/single-choice identity differs")
    choice, usage = response["choices"][0], response["usage"]
    inputs, outputs = choice.get("prompt_token_ids"), choice.get("token_ids")
    if inputs != row["request"]["input_token_ids"] or usage["prompt_tokens"] != len(inputs):
        raise ValueError("native input IDs/usage mismatch or absent echo")
    if (
        not isinstance(outputs, list)
        or not 1 <= len(outputs) <= row["request"]["cap"]
        or any(type(t) is not int for t in outputs)
        or usage["completion_tokens"] != len(outputs)
    ):
        raise ValueError("native output IDs/cap/usage mismatch")
    text = tokenizer.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    action, error = None, None
    try:
        action = c.bridge.parse_action(text)
    except (ValueError, TypeError) as exc:
        error = str(exc)
    return dict(
        input_token_ids=inputs,
        output_token_ids=outputs,
        text=text,
        provider_request_id=response["id"],
        finish_reason=choice["finish_reason"],
        usage=usage,
        valid_action=action is not None,
        action=action,
        parse_error=error,
        hf_output_ids_equal=outputs == row.get("output_token_ids"),
    )


def freeze(output):
    from transformers import AutoTokenizer

    if output.exists():
        raise FileExistsError("immutable qualification inputs exist")
    reference = c.ROOT / "textcraft-public-discovery-readout-001"
    plan = json.loads((reference / "PLAN.json").read_text())
    if plan["training_plan_sha256"] != public.PLAN_SHA:
        raise ValueError("actual057 fixedpublic056 reference required")
    binding = public.shared.endpoint(
        Path(plan["adapter"]["path"]),
        training_plan_sha256=public.PLAN_SHA,
        rows_sha256=public.ROWS_SHA,
    )
    paths = sorted((reference / "calls").glob("*.json"))
    inventory = [
        dict(json.loads(p.read_text()), source_path=str(p), source_sha256=sha(p)) for p in paths
    ]
    candidates = [r for r in inventory if r["request"].get("prompt_profile") == "original"]
    chosen = select(candidates)  # Selection reads input lengths/IDs only, never responses/scores.
    selected_ids = {row["call_id"] for row in chosen}
    warmup = sorted(
        [row for row in candidates if row["call_id"] not in selected_ids],
        key=lambda row: hashlib.sha256(f"{SEED}:warm:{row['call_id']}".encode()).hexdigest(),
    )[:5]
    if len(warmup) != 5:
        raise ValueError("five disjoint outcome-blind warmup requests required")
    output.mkdir(parents=True)
    save(
        output / "SELECTION.json",
        dict(
            seed=SEED,
            selected_call_ids=[r["call_id"] for r in chosen],
            rule="Split inputlength-ranked original057 requests at medianrank;16 SHA(seed:id) "
            "fromeach half; no correctness/response filtering",
            candidates=len(candidates),
            inventory_sha256={r["source_path"]: r["source_sha256"] for r in candidates},
            selected_before_response_audit=True,
            reference_plan_sha256=sha(reference / "PLAN.json"),
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(c.BASE, local_files_only=True)
    for row in chosen + warmup:
        req = row["request"]
        actual = tokenizer.apply_chat_template(
            [dict(role="user", content=req["prompt"])],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        if actual != req["input_token_ids"] or len(actual) + req["cap"] > 8192:
            raise ValueError("saved057 input/template/context mismatch; no replacement")
        if req["adapter_sha256"] != binding["sha256"] or not row["available"]:
            raise ValueError("selected historical native reference unavailable or wrong adapter")
    save(output / "REQUESTS.json", chosen)
    save(output / "WARMUP.json", warmup)
    save(
        output / "PLAN.json",
        dict(
            schema="textcraft-batch-qualification-v1",
            status="PROPOSED_NOT_GPU_ACCEPTED",
            requests_sha256=sha(output / "REQUESTS.json"),
            warmup_sha256=sha(output / "WARMUP.json"),
            selection_sha256=sha(output / "SELECTION.json"),
            seed=SEED,
            adapter=binding,
            model=str(c.BASE),
            model_manifest_sha256=plan["model_manifest_sha256"],
            tokenizer_sha256={
                str(p): sha(p) for p in sorted(c.BASE.glob("*token*")) if p.is_file()
            },
            template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            source_sha256={
                str(p.resolve()): sha(p)
                for p in (
                    Path(__file__),
                    Path(c.__file__),
                    Path(c.bridge.__file__),
                    Path(public.__file__),
                    Path(public.shared.__file__),
                    HELPER,
                )
            },
            measured_requests=64,
            warmup_requests=5,
            max_model_calls=69,
            cap_seconds=900,
            schedule="One shared warmup:1then4 disjoint requests excludedfrom32; "
            "then32concurrency1,32concurrency4; prefixcacheoff throughout",
            prefix_caching=False,
            model_dtype="bfloat16",
            serving_lora_dtype="bfloat16",
            stored_lora_dtype="float32; casting is a declared difference from HF",
            engine_seed=0,
            hf_eos_token_id=tokenizer.eos_token_id,
            installed_serving_source_sha256={
                str(p): sha(p)
                for p in (
                    PRIME
                    / "lib/python3.12/site-packages/vllm/entrypoints/openai/completion/protocol.py",
                    PRIME
                    / "lib/python3.12/site-packages/vllm/entrypoints/openai/completion/serving.py",
                    Path(
                        "/project/alex_phd/research-cache/repos/prime-rl/packages/prime-rl-configs/src/"
                        "prime_rl/configs/inference.py"
                    ),
                )
            },
            batch_invariant=True,
            sampling=request_body(chosen[0]) | {"prompt": None},
            admission="All64 measured native contracts valid; request throughput gain>=1.5; "
            "concurrency4 valid-action count>=concurrency1 and nonzero. "
            "Only admits a later freshly matched backend pilot, not scientific equivalence.",
            reference="Saved057 HF responses/costs; not recollected "
            "and not a same-backend baseline",
        ),
    )


def load_helper():
    spec = importlib.util.spec_from_file_location("batch_qualified_launch_helpers", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def server_config(output):
    value = json.loads((OLD / "configs/inference-replica0.json").read_text())
    value["vllm"].update(
        model=str(c.BASE),
        dtype="bfloat16",
        lora_dtype="bfloat16",
        max_loras=1,
        max_cpu_loras=1,
        max_lora_rank=8,
        max_num_seqs=4,
        max_model_len=8192,
        enable_prefix_caching=False,
        generation_config="vllm",
        api_key=["batch-local-only"],
        tool_call_parser=None,
        reasoning_parser=None,
    )
    value["output_dir"] = str(output / "launcher")
    return value


def run(prepared, output):
    import psutil
    from transformers import AutoTokenizer

    plan = json.loads((prepared / "PLAN.json").read_text())
    if sha(prepared / "REQUESTS.json") != plan["requests_sha256"]:
        raise ValueError("immutable request inventory changed")
    for path, expected in plan["source_sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError("source changed: " + path)
    for path, expected in plan["installed_serving_source_sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError("installed serving source changed: " + path)
    for path, expected in plan["tokenizer_sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError("tokenizer changed")
    if sha(c.BASE / "local-research-manifest.json") != plan["model_manifest_sha256"]:
        raise ValueError("model manifest changed")
    adapter = plan["adapter"]
    if sha(Path(adapter["path"]) / "adapter_model.safetensors") != adapter["sha256"]:
        raise ValueError("adapter changed")
    rows = json.loads((prepared / "REQUESTS.json").read_text())
    if sha(prepared / "WARMUP.json") != plan["warmup_sha256"]:
        raise ValueError("shared warmup inventory changed")
    warmup = json.loads((prepared / "WARMUP.json").read_text())
    if output.exists() or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("new output and main-assigned exclusive GPU required")
    deadline = min(time.time() + 900, int(os.environ["SLURM_JOB_END_TIME"]) - 600)
    if deadline < time.time() + 300:
        raise ValueError("insufficient allocation margin")
    helper = load_helper()
    if not DRIVER.is_dir() or any(not helper._port_free(p) for p in (18601, 18611, 18621)):
        raise ValueError("qualified driver absent or existing service ports occupied")
    tokenizer = AutoTokenizer.from_pretrained(c.BASE, local_files_only=True)
    lock = (c.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True)
    process, failure, measured = None, None, {}
    started = time.time()
    save(
        output / "OWNER.json",
        dict(
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            started=started,
            deadline=deadline,
            source=str(Path(__file__).resolve()),
            source_sha256=sha(Path(__file__)),
            input_plan_sha256=sha(prepared / "PLAN.json"),
        ),
    )

    def stop(*_):
        raise TimeoutError("qualification owner cap/signal")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, max(1, deadline - time.time() - 60))
    try:
        environment = helper._server_environment(helper._environment(), 0)
        environment["LD_LIBRARY_PATH"] = str(DRIVER / "lib")
        environment["PATH"] = str(DRIVER / "bin") + ":" + environment["PATH"]
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_BATCH_INVARIANT="1")
        for name in (
            "VLLM_CACHE_ROOT",
            "VLLM_CONFIG_ROOT",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "HUMMING_CACHE_DIR",
            "HUMMING_TMP_DIR",
        ):
            environment[name] = str(output / "cache" / name.lower())
        config_path = output / "inference.json"
        save(config_path, server_config(output))
        command = [str(PRIME / "bin/inference"), "@", str(config_path)]
        with (output / "service.log").open("x") as log:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        save(
            output / "SERVER.json",
            dict(
                pid=process.pid,
                create_time=psutil.Process(process.pid).create_time(),
                command=command,
            ),
        )
        os.environ["TEXTCRAFT_BATCH_LOCAL_KEY"] = "batch-local-only"
        descriptor = dict(
            port=18601,
            replica=0,
            api_key_env="TEXTCRAFT_BATCH_LOCAL_KEY",
            model_alias=ALIAS,
            adapter={"path": adapter["path"]},
        )
        helper._wait_endpoint_model(descriptor, process, str(c.BASE), timeout=240)
        helper._load_adapter(descriptor, timeout=60)
        helper._wait_endpoint_model(descriptor, process, ALIAS, timeout=30)
        with urlopen(
            Request(
                "http://127.0.0.1:18601/v1/models",
                headers={"Authorization": "Bearer batch-local-only"},
            ),
            timeout=10,
        ) as response:
            cards = json.load(response)
        save(output / "MODELS.json", cards)
        card = next(row for row in cards["data"] if row["id"] == ALIAS)
        if card.get("root") != adapter["path"] or card.get("parent") != str(c.BASE):
            raise ValueError("native alias does not bind exact public056 adapter/base")
        engine_lines = [
            line
            for line in (output / "service.log").read_text().splitlines()
            if "Initializing a V1 LLM engine" in line
        ]
        if not engine_lines or any(
            "enable_prefix_caching=False" not in line for line in engine_lines
        ):
            raise ValueError("effective engine prefix-cache-off evidence absent")
        save(output / "ENGINE-CONFIG.json", dict(lines=engine_lines, prefix_caching=False))
        aborted = threading.Event()

        def send(pair):
            if aborted.is_set():
                raise RuntimeError("earlier native failure; remaining calls cancelled")
            label, row = pair
            request = request_body(row)
            record = dict(
                label=label,
                hf_call_id=row["call_id"],
                request=request,
                started=time.time(),
                available=False,
            )
            save(output / "starts" / (label + ".json"), record)
            try:
                remaining = deadline - time.time() - 60
                if remaining <= 0:
                    raise TimeoutError("no response/cleanup headroom")
                wire = Request(
                    "http://127.0.0.1:18601/v1/completions",
                    data=json.dumps(request).encode(),
                    method="POST",
                    headers={
                        "Authorization": "Bearer batch-local-only",
                        "Content-Type": "application/json",
                    },
                )
                with urlopen(wire, timeout=min(90, remaining)) as response:
                    raw = json.load(response)
                record["raw_response"] = raw
                record.update(normalize(row, raw, tokenizer), available=True)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                aborted.set()
            record["ended"] = time.time()
            save(output / "calls" / (label + ".json"), record)
            if not record["available"]:
                raise RuntimeError(record["error"])
            return record

        def phase(name, chosen, concurrency):
            tick = time.time()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                result = list(
                    pool.map(send, [(f"{name}-{i:02d}", r) for i, r in enumerate(chosen)])
                )
            receipt = dict(wall_seconds=time.time() - tick, records=result, concurrency=concurrency)
            save(output / (name + ".json"), receipt)
            if not all(r["available"] for r in result):
                raise RuntimeError("native transport/identity failure; no retry or promotion")
            return receipt

        phase("warm1", warmup[:1], 1)
        phase("warm4", warmup[1:], 4)
        measured["c1"] = phase("c1", rows, 1)
        measured["c4"] = phase("c4", rows, 4)
        gain = measured["c1"]["wall_seconds"] / measured["c4"]["wall_seconds"]
        valid = {k: sum(r["valid_action"] for r in v["records"]) for k, v in measured.items()}
        save(
            output / "RESULT.json",
            dict(
                gain=gain,
                valid_actions=valid,
                admitted_to_future_pilot=gain >= 1.5 and valid["c4"] >= valid["c1"] > 0,
                measured_calls=64,
                max_total_calls=69,
                reference="saved057HF only",
                phases={
                    name: dict(
                        wall_seconds=phase["wall_seconds"],
                        requests_per_second=32 / phase["wall_seconds"],
                        prompt_tokens=sum(r["usage"]["prompt_tokens"] for r in phase["records"]),
                        output_tokens=sum(
                            r["usage"]["completion_tokens"] for r in phase["records"]
                        ),
                        output_tokens_per_second=sum(
                            r["usage"]["completion_tokens"] for r in phase["records"]
                        )
                        / phase["wall_seconds"],
                        native_contract_errors=sum(not r["available"] for r in phase["records"]),
                        invalid_actions=sum(not r["valid_action"] for r in phase["records"]),
                        hf_output_ids_equal=sum(r["hf_output_ids_equal"] for r in phase["records"]),
                    )
                    for name, phase in measured.items()
                },
                historical_acquisition=dict(
                    calls=32,
                    service_seconds=sum(r["ended"] - r["started"] for r in rows),
                    prompt_tokens=sum(r["usage"]["prompt_tokens"] for r in rows),
                    output_tokens=sum(r["usage"]["completion_tokens"] for r in rows),
                ),
                claim="Serving-throughput qualification; "
                "no success/generalization/equivalence claim",
            ),
        )
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if process is not None:
            helper._stop(process)
        released = all(helper._port_free(p) for p in (18601, 18611, 18621))
        save(
            output / "TERMINAL.json",
            dict(
                failure=failure,
                released=released,
                elapsed_seconds=time.time() - started,
                scientific_model_calls=len(list((output / "calls").glob("*.json"))),
                complete=failure is None and released,
            ),
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.freeze_only:
        freeze(args.output.resolve())
    elif args.prepared:
        run(args.prepared.resolve(), args.output.resolve())
    else:
        parser.error("--prepared or --freeze-only required; no implicit launch")

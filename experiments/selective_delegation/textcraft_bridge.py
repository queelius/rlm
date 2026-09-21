"""Strict JSON bridge to pinned trusted TextCraft methods, never model-code execution."""

import ast
import asyncio
import contextvars
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MethodType, SimpleNamespace

CACHE = Path(
    "/project/alex_phd/research-cache/repos/platoon-rao-d9c5857d3a0a056ebc9b047241a2a0c9515aafbe"
)
PACKAGE = CACHE / "plugins/textcraft/platoon/textcraft"
ENV = PACKAGE / "env.py"
GENERATOR = PACKAGE / "synth_recipe_generator.py"
EXPECTED = {
    ENV: "c58ffad58e535d91def7291db148fd8f6433a8b3a915642f634a5b75d6d587d0",
    GENERATOR: "1c33e0a6f61759eb3a8eb33b35f0b88361155525f5f575885b53acbd011efb0e",
    CACHE / "LICENSE": "e7107f1e1b3a0c65cbbae6c496d9e71b570025bdddebb3d402a6080dd916bcba",
}
FINISH_MESSAGE = contextvars.ContextVar("textcraft_native_finish_message", default=None)


def selected_methods():
    tree = ast.parse(ENV.read_text())
    selected = []
    allowed = {
        "TextCraftCodeExecutor": {"craft", "get_info", "view_inventory"},
        "TextCraftEnv": {"evaluate"},
    }
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in allowed:
            selected.extend(
                n
                for n in cls.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name in allowed[cls.name]
            )
    if {n.name for n in selected} != {"craft", "get_info", "view_inventory", "evaluate"}:
        raise ValueError("trusted method inventory differs")
    return selected


def trusted_provenance():
    actual = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in EXPECTED}
    if any(actual[str(path)] != expected for path, expected in EXPECTED.items()):
        raise ValueError("trusted upstream source changed; review before execution")
    return {
        "commit": "d9c5857d3a0a056ebc9b047241a2a0c9515aafbe",
        "source_sha256": actual,
        "license": "MIT",
        "method_ast_sha256": {
            n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
            for n in selected_methods()
        },
        "model_code_execution": False,
        "lifecycle": "exact selected official method bodies, not full Platoon/IPython lifecycle",
    }


@lru_cache(maxsize=1)
def native_functions():
    trusted_provenance()
    # Only statically allowlisted, reviewed source ASTs enter compile. No model strings enter here.
    nodes = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *selected_methods(),
    ]
    namespace = {"finish_message": FINISH_MESSAGE}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(ENV), "exec"
        ),
        namespace,
    )
    return {name: namespace[name] for name in ("craft", "get_info", "view_inventory", "evaluate")}


def load_world():
    trusted_provenance()
    name = "pinned_textcraft_synth_generator_d9c5857d"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, GENERATOR)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    module.set_naming_mode(semantic=False)
    world = module.SynthRecipeDatabase()
    world.generate_all_recipes(seed=42, items_per_domain_tier=25)
    return world


class BudgetExceeded(RuntimeError):
    """Observed policy budget exhaustion, distinct from external/native errors."""


@dataclass
class Budget:
    max_calls: int = 96
    max_output_tokens: int = 8192
    calls: int = 0
    output_tokens: int = 0

    def reserve(self, per_call=256):
        if self.calls >= self.max_calls or self.output_tokens >= self.max_output_tokens:
            raise BudgetExceeded("global tree call/token budget exhausted")
        return min(per_call, self.max_output_tokens - self.output_tokens)

    def charge(self, emitted_tokens):
        allowed = self.reserve()
        if type(emitted_tokens) is not int or not 0 <= emitted_tokens <= allowed:
            raise ValueError("native token count outside requested cap; external error")
        self.calls += 1
        self.output_tokens += emitted_tokens


def quantities(value):
    return (
        isinstance(value, dict)
        and 1 <= len(value) <= 32
        and all(
            isinstance(k, str) and 0 < len(k) <= 100 and type(v) is int and 0 < v <= 1000000
            for k, v in value.items()
        )
    )


def parse_action(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate field")
            result[key] = value
        return result

    action = json.loads(text, object_pairs_hook=unique)
    if not isinstance(action, dict):
        raise ValueError("expected exact action JSON object")
    schemas = {
        "get_info": {"items"},
        "view_inventory": set(),
        "craft": {"ingredients", "target_item", "output_count"},
        "finish": {"message"},
        "delegate": {"targets", "context"},
    }
    name = action.get("action")
    if (
        not isinstance(name, str)
        or name not in schemas
        or set(action) - {"note"} != schemas[name] | {"action"}
    ):
        raise ValueError("unknown action or exact-field schema violation")
    if "note" in action and (not isinstance(action["note"], str) or len(action["note"]) > 1000):
        raise ValueError("note must be a bounded string")
    if name == "get_info" and (
        not isinstance(action["items"], list)
        or not 1 <= len(action["items"]) <= 32
        or any(not isinstance(i, str) or not 0 < len(i) <= 100 for i in action["items"])
    ):
        raise ValueError("items must be a bounded nonempty string list")
    if name == "craft" and (
        not quantities(action["ingredients"])
        or not isinstance(action["target_item"], str)
        or not quantities({action["target_item"]: action["output_count"]})
    ):
        raise ValueError("craft requires positive integer quantities, not booleans")
    if name == "delegate" and (
        not quantities(action["targets"])
        or not isinstance(action["context"], str)
        or len(action["context"]) > 2000
    ):
        raise ValueError("delegate requires targets and bounded context")
    if name == "finish" and (
        not isinstance(action["message"], str) or len(action["message"]) > 1000
    ):
        raise ValueError("finish message must be a bounded string")
    return action


INSTRUCTION = (
    "You control a crafting inventory. Produce the requested targets IN ADDITION to their "
    "quantities at the start of this task. Recipes are discovered with get_info; initial "
    "root materials suffice but intermediate items may need crafting. Recipe outputs have "
    "fixed batch sizes: output_count must be a multiple of result_count, and ingredients "
    "must exactly equal the per-batch ingredients times that number of batches. Shared "
    "inventory changes immediately; verify quantities. Finish explicitly with a nonempty "
    "message only when done. Return ONLY one exact JSON object using one schema below. "
    "An optional short string note is allowed on EVERY action. No Python or extra fields.\n"
    '{"action":"get_info","items":["item"]}\n'
    '{"action":"view_inventory"}\n'
    '{"action":"craft","ingredients":{"item":2},"target_item":"product","output_count":2}\n'
    '{"action":"finish","message":"done"}\n'
    '{"action":"delegate","targets":{"item":2},"context":"brief useful public context"}\n'
    "Delegation is allowed only below max_agent_depth. Root depth is0; a child is1 and "
    "grandchild2. Children use the SAME inventory and GLOBAL call/token budget, receive "
    "a fresh context, and return after finishing. They must produce their requested targets "
    "in addition to the inventory at their own start. No free actions, retries or repairs. "
    "All model responses, including errors/delegation/finish/returns, consume the global budget. "
    "Rejected actions do not change inventory; use the feedback to choose your next action.\n"
)


def public_prompt(frame, history, context="", goal=None):
    return INSTRUCTION + json.dumps(
        {
            "goal": goal,
            "target_items": frame.targets,
            "inventory_at_task_start": frame.initial_inventory,
            "current_inventory": frame.inventory,
            "agent_depth": frame.depth,
            "max_agent_depth": frame.max_depth,
            "global_calls_remaining": frame.budget.max_calls - frame.budget.calls,
            "global_output_tokens_remaining": frame.budget.max_output_tokens
            - frame.budget.output_tokens,
            "delegated_context": context,
            "history": history,
        },
        ensure_ascii=False,
    )


def initial_prompt(task, policy):
    if policy not in ("flat", "recursive"):
        raise ValueError("unknown fixed policy")
    frame = SimpleNamespace(
        targets=dict(task["misc"]["target_items"]),
        initial_inventory=dict(task["misc"]["initial_inventory"]),
        inventory=dict(task["misc"]["initial_inventory"]),
        depth=0,
        max_depth=0 if policy == "flat" else 2,
        budget=Budget(),
    )
    return public_prompt(frame, [], goal=task["goal"])


class Frame:
    """One context; all children share inventory and the same charged model budget."""

    def __init__(self, world, inventory, targets, budget, max_depth, depth=0):
        self.world, self.inventory, self.targets = world, inventory, dict(targets)
        self.initial_inventory = dict(inventory)
        self.budget, self.max_depth, self.depth = budget, max_depth, depth
        self.finished, self.message = False, None
        self.executor = SimpleNamespace(recipe_db=world, inventory=inventory)
        self.native = native_functions()

    def apply(self, action):
        # Revalidate even direct callers; the collector charges the response BEFORE parsing/apply.
        action = parse_action(json.dumps(action))
        if self.finished:
            raise ValueError("frame already finished")
        name = action["action"]
        if name == "delegate":
            return self.delegate(action["targets"])
        if name == "finish":
            self.finished, self.message = True, action["message"]
            return "Finished."
        method = MethodType(self.native[name], self.executor)
        if name == "craft":
            return method(action["ingredients"], (action["target_item"], action["output_count"]))
        if name == "get_info":
            return method(action["items"])
        return method()

    def delegate(self, targets):
        if self.depth >= self.max_depth:
            raise ValueError("maximum delegation depth exceeded")
        if not quantities(targets):
            raise ValueError("invalid delegated target quantities")
        return Frame(
            self.world, self.inventory, targets, self.budget, self.max_depth, depth=self.depth + 1
        )

    def score(self):
        state = SimpleNamespace(
            _skip_subagent_reward_computation=False,
            _task=SimpleNamespace(
                id="textcraft_root" if self.depth == 0 else "child",
                misc={"target_items": self.targets},
            ),
            _state=SimpleNamespace(finished=self.finished),
            _initial_inventory=self.initial_inventory,
            _code_executor=self.executor,
        )
        token = FINISH_MESSAGE.set(self.message)
        try:
            return asyncio.run(self.native["evaluate"](state))
        finally:
            FINISH_MESSAGE.reset(token)


def replay_gold(task, world):
    misc = task["misc"]
    frame = Frame(
        world, dict(misc["initial_inventory"]), misc["target_items"], Budget(), max_depth=2
    )
    records = []
    for step in misc["gold_trajectory"]:
        action = {
            "action": "craft",
            "ingredients": step["ingredients"],
            "target_item": step["target"][0],
            "output_count": step["result_count"],
        }
        response = frame.apply(action)
        records.append({"action": action, "feedback": response})
        if response.startswith("Error:"):
            break
    frame.apply({"action": "finish", "message": "Gold replay complete"})
    score, details = frame.score()
    return {
        "task_id": task["id"],
        "depth": misc["max_depth"],
        "native_score": score,
        "native_details": details,
        "steps": records,
        "gold_craft_steps": len(misc["gold_trajectory"]),
        "gold_craft_plus_finish_calls": len(misc["gold_trajectory"]) + 1,
        "one_info_per_craft_upper_bound_calls": 2 * len(misc["gold_trajectory"]) + 1,
        "interpretation": "Constructive gold trace, not a proof of globally minimal action count; "
        "host oracle never enters model prompt.",
    }

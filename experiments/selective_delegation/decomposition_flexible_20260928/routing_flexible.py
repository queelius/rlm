"""Optional discovery order; preserve native one-boundary delegation semantics."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

OLD = Path(__file__).resolve().parent.parent / "decomposition_20260928"
sys.path.insert(0, str(OLD))
import routing as original  # noqa: E402

native = original.native
MARKER = original.MARKER


def make_bridge(mode):
    prior = original.make_bridge(mode)

    class Frame(prior.Frame):
        def routing(self):
            control = super().routing()
            required = control["required_next_action"]
            if required is not None and required["action"] == "get_info":
                control["required_next_action"] = None
            control["discovery_order"] = "model_chosen; suggested queries are optional"
            control["abort_requested"] = self.refusals["delegate"] >= 2
            return control

        def delegate(self, targets):
            if self.depth >= self.max_depth or self.delegated:
                raise ValueError("one public helper boundary only")
            self.delegated = True
            return Frame(
                self.world,
                self.inventory,
                targets,
                self.budget,
                self.max_depth,
                depth=self.depth + 1,
            )

    def public_prompt(frame, history, context="", goal=None):
        control = frame.routing()
        frame.announced = control
        guidance = (
            "\nDiscover recipes and craft in your chosen order. The query under decision "
            "is only a suggestion, not a mandatory action. When required_next_action is "
            "a delegate, emit that delegate with the exact targets/counts; context may "
            "be paraphrased. Otherwise solve directly. Candidate facts use only returned "
            "recipes and stock. Observed depth is a lower bound, not hidden world depth. "
            "Every response and helper consumes the same global budget."
        )
        return (
            native.public_prompt(frame, history, context=context, goal=goal)
            + guidance
            + MARKER
            + json.dumps(control)
        )

    namespace = dict(vars(prior))
    namespace.update(Frame=Frame, public_prompt=public_prompt, __file__=__file__)
    return SimpleNamespace(**namespace)

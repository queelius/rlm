# Plain-language explanations and likely questions

## Slide1: What are we trying to teach?

We want a model to find missing information, choose useful steps and finish a
task. A crafting game makes success easy to check: the requested items either
exist in the inventory or they do not. The pickaxe example is simplified and
illustrative; it is not a literal saved training case.

**Why does this belong in RLM research?** A recursive language model can ask
helpers to solve smaller tasks. Before adding helpers here, we are examining
whether one model can use the environment reliably. These tests are flat:
they do not establish a benefit from recursion or learned decomposition.

## Slide2: What did training look like?

We used Qwen3-4B-Instruct-2507, a four-billion-parameter language model. We trained
small added sets of weights called LoRA adapters. Supervised fine-tuning means
showing the model a situation and teaching it to produce the demonstrated next
action. We do not just add examples to the prompt at test time.

The training data were scripted TextCraft demonstrations, not human-written
answers or demonstrations collected from another large language model. Both
packages use32 training tasks and366 training steps, with23 parameter updates,
one pass through the data and learning rate0.0001. We repeated training using
two different random seeds. See the pinned training PLAN files for full details.

The original teacher can use the underlying recipes when constructing its
demonstrations, and it still shows public recipe lookups before crafting.
The alternative discovers prerequisite recipes through the available interface
to determine the route itself. This is not lookup versus no lookup.
The student must learn from the observations
in those examples. This does **not** mean the original student was given an
all-knowing tool at evaluation time.

**Is this a clean experiment on looking things up alone?** No. The packages also
differ in action order, observations and token counts. Equal step counts do not
mean equal amounts of text or compute. Our current claim is about the teaching
packages, not proof of a single mechanism.

## Slide3: How strong is the main result?

Each training repeat was tested on16 evaluation tasks with two attempts per task.
The first comparison is5 versus15 successes; the second is3 versus10, all out of32.
Every outcome is known. The advantage is31.25 and21.875 percentage points.

Intervals obtained by resampling whole tasks are15.625–50 and9.375–37.5 percentage
points. Attempts on one task are related, so we do not treat every attempt as an
independent task. These are exploratory intervals, not a preregistered discovery
or proof after correcting for every experiment in the larger campaign.

We found and corrected a quantity error in the original demonstrations, then
retrained. The advantage remained. The panel has been examined during our research;
it is not a final untouched test set. Also, the two training repeats share tasks
and base weights. They are not independent datasets.

## Slide4: What does changing the world mean?

We keep the goal identities but change recipe assignments and starting supplies.
There are eight goals with two attempts each in each of three worlds. Both
training seeds favor discovery demonstrations in all three worlds.

The limitation is important: these comparisons used the uncorrected original
teacher. Slide3 controls the quantity mistake; slide4 changes the world. Until
the new control finishes, we have not shown both protections in one experiment.
All worlds also share one synthetic generator. Success on a new dataset or
different kind of environment remains untested by these comparisons.

## Slide5: What change to the RLM are we proposing?

Today the model chooses a crafting action and writes its detailed tool arguments.
The proposed interface separates those responsibilities. The model chooses the
item and amount; code uses a recipe the model already looked up to fill in the
ingredients correctly. It does not secretly look up recipes or solve the plan.

This might remove copying mistakes, but it cannot fix a bad choice of item or
missing supplies. Earlier one-step audits found both types of problem. Repairing
one action is not proof that an entire task would succeed. The next experiment
should compare end-to-end success using fixed tasks, seeds and model weights.

**Why not just do more reinforcement learning?** We are still interested in it.
But a larger update did not improve this task, and positive-only versus signed
credit tied at15/32, with three wins and three losses on paired attempts. The
full-panel difference interval is−12.5 to+12.5 points. That does not prove the
methods are equivalent; it shows no improvement in this small comparison.
Positive-only evaluation required extra collection time to finish four missing
cases. All extra calls, including an interrupted attempt, remain charged.

**Did the notebook help?** Not reliably. With full history, adding a recipe
notebook changed successes15→14 for one model,15→6 for another and14→14 for the
RL-trained model. With only four recent interactions, changes were10→14,
10→8 and13→13. This motivates learning how to use a tool, rather than assuming
that adding information automatically improves performance.

**What could become publishable?** A controlled finding about which demonstrations
teach usable information-gathering behavior, or about when separating planning
from exact tool execution helps. Both need broader tests and a careful prior-art
comparison. Learnable demonstrations and planning/execution separation already
have prior work; neither general idea should be claimed as new.

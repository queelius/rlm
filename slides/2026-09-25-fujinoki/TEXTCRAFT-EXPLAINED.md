# Understanding our TextCraft experiments

## The thirty-second explanation

“We give a small language model a goal in a crafting game. It has to look up
recipes, make the needed parts, and finish the item. The game checks whether it
really succeeded. We tested two things: how to teach the model to find the
steps, and whether code should handle the exact ingredient lists. Both changes
helped in our small tests. Now we are checking whether those benefits hold on
additional goals and whether they work together.”

## What does the model actually do?

It receives a goal, its current inventory, and a record of previous actions and
the game's replies. It chooses one action at a time: request recipe information,
craft something, or finish. It sends structured commands rather than physically
moving around a graphical game. The environment returns what happened and
updates the inventory. The model then chooses its next action.

Here is an invented, simplified example—not a literal saved task or recipe:

1. The goal is to make a pickaxe, and the inventory contains some wood.
2. The model asks for the pickaxe recipe and learns it needs planks and a stick.
3. It asks how to make those parts, then makes them.
4. It assembles the pickaxe and finishes.
5. The game checks whether the requested item and quantity were actually made.

Saying “I made it” does not count as success. Looking up the right recipe does
not count either. The task must actually be completed under the game's rules.

## Why use a crafting game?

We are not primarily interested in crafting. We want a manageable experiment
where actions depend on earlier actions, information must be requested, mistakes
have observable consequences, and success can be checked automatically.

That lets us distinguish a convincing explanation from a working sequence of
actions. It is a useful laboratory, not evidence that a method already works on
all real-world planning problems.

## Experiment one: change the worked examples

We trained copies of the same starting language model to imitate example actions.
This is supervised fine-tuning, not reinforcement learning.

Both sets of examples came from programs. Both showed recipe lookups. The
difference was how the program found the sequence of steps:

- **Teacher knows the recipes:** it can use recipe knowledge to plan the route
  before showing the learner the lookups and crafting actions.
- **Teacher discovers the recipes:** it uses the available lookup interface to
  discover the parts and their recipes, building the route from those observations.

The second approach shows a way to find a solution, not just steps selected by
a teacher with advance knowledge. That is our motivation, not yet a proven
explanation of the results: the examples also differ in action order and text.

A simplified training row might say:

> Situation: Make a pickaxe. You do not know its recipe yet.
>
> Action to imitate: Ask for the pickaxe recipe.

Actual rows also contain inventory and history; the output is a structured tool
command. Training changes model weights. At test time the model must choose its
own actions on the evaluation tasks; we do not supply the worked solution.

**Finding:** discovery examples led to more completed tasks in both training
repeats. The first teacher's known quantity error was corrected before the
comparison. This is evidence about two teaching approaches, not proof that one
isolated feature caused the difference.

## Experiment two: change the tool, not the weights

Sometimes the model read a recipe correctly but supplied the wrong ingredients
when trying to craft the item. We let ordinary code fill those details from a
recipe the model had already looked up.

Using the invented example: the model chooses “make one pickaxe.” Code supplies
the ingredient list from the previously observed recipe. The model still chooses
the item and amount. It still needs to request information and make missing parts.

The intervention does not invent a plan, reveal hidden recipes, create free
materials, or turn a wrong goal into the right one. The game still checks whether
the action is valid. An early finish or a bad choice of next item can still fail.

**Finding:** more tasks succeeded without any additional model training. A first
completed comparison on additional goals also favored the assisted tool. This
suggests execution errors matter for overall success; it does not prove the
model became a better planner.

## Where does reinforcement learning fit?

In our separate crafting RL experiments, the model tried tasks and received
feedback based on whether the whole attempt succeeded. We then updated its
weights from its own attempts, rather than giving it a scripted next action.

Those small RL runs did not establish a useful improvement. A successful attempt
could contain many rejected actions before recovery, so final success alone
does not identify which actions were helpful. This is one plausible difficulty,
not a proven explanation or evidence that RL cannot work.

The next proposed RL question is: if code handles recipe details, can reward-based
training focus more effectively on useful choices? We have not tested that yet.

## What is running now?

We are comparing both teaching methods, with and without ingredient assistance,
on additional goals and recipe settings. These are evaluations of already-trained
models, not new RL runs.

This asks two things: does assistance help beyond the first goals, and does the
discovery-training advantage remain when execution details are handled by code?
If the gap narrows, execution reliability may explain part of it. If it remains,
choosing steps and gathering information remain possible explanations.

## How does this connect to RLMs?

RLMs can use code and delegate smaller problems to helper models. These recent
crafting comparisons use a single model with tools—no recursive helper calls.
They study basic action and information-gathering skills that could support
later RLM work. They do not establish learned recursion or better delegation.

The larger research question is which decisions need a learned model, which
details ordinary code should handle, and when further decomposition is useful.

## What can I safely say in the meeting?

Say: “How we teach the model and how we design its tools both affected whether
it finished the task. The early results are promising, but still limited to one
game. We are testing broader goals before claiming a general method.”

Avoid: “We solved planning,” “RL learned to recurse,” “code assistance improved
the model's weights,” or “we doubled performance in general.” None follows from
these experiments.

If asked about novelty: “Using code for exact details is not new by itself. We
are investigating when this division of work helps and how it interacts with
learning. Establishing that would require broader tests and comparison with
related research.”

For exact counts and evidence, see the [speaker guide](speaker-guide.md) and
[new-goal result](new-goal-result.md).

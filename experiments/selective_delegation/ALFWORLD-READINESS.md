# ALFWorld text-only CPU readiness

2026-09-21. **CPU ready; no GPU screen accepted or launched.** This does not replace
the accepted MuSiQue RL continuation/readout or the canonical Hotpot panel.
Setup and checks took approximately ten minutes within the 20-minute bound.

## What actually ran

The official pinned ALFWorld text engine reset and executed public commands in
three manually operated seen-development games. These are **manual traces, not
model-policy performance**. Only `AlfredDemangler(shuffle=False)` wrapped the
engine; no `AlfredExpert`, expert policy, PDDL facts, or annotation plans were
requested or projected. Native `won`, reward, and termination were recorded
separately from the public feedback/action-list projection.

| Manual game | Newly observed evidence and useful next action | Actions | Native won |
|---|---|---:|---|
| Book → sidetable, scene329 | Opening drawer1 revealed a cellphone, not a book. Visiting bed1 revealed books1/2; pickup then became useful. | 6 | true |
| Heat apple → diningtable, scene26 | Countertops1/2 and opened fridge contained no apple. Diningtable2 revealed apple1; pickup replaced further search, followed by heating and placement. | 11 | true |
| Clean butterknife → countertop, scene8 | Countertop1 lacked the target; countertop2 revealed butterknife1. Pickup, sink cleaning, and placement completed the goal. | 7 | true |

All raw actions and public observations remain in `readiness/manual-book-001.json`,
`manual-heat-001.json`, and `manual-clean-001.json` under the external dataset path
below. Game paths and SHA-256 identities are in each trace and the manifest.
The manual choices used the returned public interface, not annotated solutions.
These examples demonstrate observation-conditioned search; they do **not** show
that a learned hierarchy is needed. A fixed find→transform→place recipe with a
reactive search routine could suffice. Adaptive planning and ALFWorld agents are
established prior work, not a novelty claim.

### Admissible commands are affordance assistance, not just syntax help

At the sinkbasin in the cleaning game, feedback said it contained nothing, but
the native list offered `slice potato 1 with butterknife 1`. A separate public
replay actually issued that action: feedback confirmed that potato1 was sliced.
Thus this is executable information beyond visible object feedback, not merely
an unexecutable template. That additional trace is preserved as
`manual-affordance-check-001.json`; it was deliberately stopped after five actions,
with native `won=false` because slicing was not the cleaning goal.

For the proposed screen, expose **identical native admissible-command lists to
both policies** and label the interface *affordance-assisted*. Do not claim that
the agent inferred all object affordances or that the observation alone is a
complete state description. No oracle plan, hidden facts, game path, task folder,
trajectory annotation, intermediate expert reward, or full state belongs in a
model prompt. Keep the ordinary initial room/goal observation unchanged.

## Eight-game candidate inventory—not accepted GPU work

`readiness/MANIFEST.json` freezes a candidate panel from official `valid_seen`
(seen-development, **not an unseen test**). Sort official eligible paths, shuffle
with Python seed **2026092177**, then take four simple placement, two heating,
and two cleaning tasks by their native task-type prefix. No outcome filtering.
All eight initial resets succeeded and returned native `won=false`.

| Task | Scene | Initial actions listed | Feedback characters |
|---|---:|---:|---:|
| Egg → microwave | 4 | 25 | 475 |
| Tissuebox → toilet | 426 | 20 | 408 |
| Knife → sidetable | 3 | 27 | 490 |
| Mug → sidetable | 329 | 12 | 271 |
| Heat tomato → fridge | 24 | 44 | 739 |
| Heat egg → fridge | 14 | 23 | 446 |
| Clean butterknife → countertop | 8 | 42 | 715 |
| Clean cloth → drawer | 423 | 24 | 494 |

Measured maxima **at these eight resets**: 44 commands, 859 characters for the
JSON action list, 739 feedback characters, and 1,643 characters for the combined
public JSON projection. These are not maxima over unexecuted trajectories, token
counts, or full policy prompts. Reset time ranged approximately 0.52–2.76 seconds.
Future collection must record actual tokenizer lengths and enforce the same
context policy in both arms; repeated historical admissible lists are unnecessary.

There are eight distinct scene IDs within this panel, but `valid_seen` scenes
overlap training by design. One selected game is the manually inspected cleaning
game; the manifest explicitly preserves that overlap. It is **not replaced**
after observing success. The panel is exposed exploratory development, not clean
confirmation. An initial attempt to impose eight unique scenes on an unseen-split
quota failed before any candidate reset; no result was selected from it. The
current seen-development panel follows the later requested seed/split choice.

A possible later screen remains 8 games × 2 sampling seeds × 2 policies = 32 episodes:
equally informed flat reactive policy versus adaptive goal manager/worker, the
same action vocabulary, 50 environment actions and 2,048 generated tokens per
episode, counting manager calls. Native `won` is the success metric; termination
alone is insufficient. Report parse/grounding failures and cost separately. This
document supplies environment readiness, not an implemented model collector or
evidence that either policy is better. Main decides whether to accept that screen.

## Reproducible assets and environment

- Official repository: <https://github.com/alfworld/alfworld.git>, exact commit
  `aaba6870f86c5be6a08a491f32a50b906227bc3e`, cloned and inspected before execution;
  `git status --porcelain` clean after setup.
- Clone: `/project/alex_phd/research-cache/repos/alfworld-aaba6870f86c5be6a08a491f32a50b906227bc3e`.
- External environment project:
  `/project/alex_phd/envs/alfworld-aaba6870f86c5be6a08a491f32a50b906227bc3e`;
  interpreter `.venv/bin/python`, Python 3.12.12, uv 0.12.9, ALFWorld 0.5.0,
  TextWorld 1.7.0, fast-downward-textworld 20.6.4. No Torch/model/GPU installation
  or visual extras were needed. Environment size approximately 319 MB.
- `pyproject.toml` and `uv.lock` are outside the clone; reproduce with
  `/project/alex_phd/tools/uv-current/uv sync --frozen` from that environment
  project, using the retained pinned local clone. Lock SHA-256:
  `5001ba013310e3b50312e177610905b24b1d87d43020603904260b9e1c650a2e`.
- Dataset/cache root:
  `/project/alex_phd/research-cache/datasets/alfworld-text-0.4.2-20260921`.
  Only two official release archives were acquired, approximately 109 MB compressed
  and 1.8 GB total cache including extracted metadata. No images, THOR environment,
  detector weights, pretrained agents, or seq2seq data were downloaded.

| Official release asset | Bytes | SHA-256 |
|---|---:|---|
| [Text games,0.4.2](https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip) | 36,507,267 | `5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf` |
| [Trajectory metadata,0.2.2](https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip) | 72,018,818 | `25171f16e20ad7b048c47275c45b0babf3aa1cbab29cec97387922350a9844bc` |

The full metadata archive contains expert annotations; acquisition is **not**
permission to expose them. It is needed by the official game inventory loader;
the runtime text-game files already embed their domain/problem/grammar, so no
separate PDDL archive or default all-assets downloader was required.

The official `AlfredTWEnv.collect_game_files` filter verified 3,553 train,
140 `eval_in_distribution`/`valid_seen`, 134 `eval_out_of_distribution`/`valid_unseen`.
The text archive also contains 200 `valid_train` games; the proposed screen does
not use them. The official loader checks supported task types and the saved
`solvable` flag; those host-only checks are not an expert solution in the prompt.
The eight reset checks establish engine readiness, not that every task is solvable
under a 50-action model budget.

Repository license is MIT; retained LICENSE SHA-256
`0bdf8c0558499c192b6ab55818e99e91ef0eb02c6e2d19d907b3f2e14df40590`.
TextWorld and other dependencies retain their own notices; Fast Downward is
GPLv3. No independent data-license notice was identified in these release
archives, so do not infer that every dependency/asset is MIT. All installed package
versions and available license metadata are recorded in the external manifest.

Authoritative external readiness manifest:
`readiness/MANIFEST.json`, SHA-256
`7c0a86472377cd73af66585ec04dff4010c75de3a2346e9f2988a86840eb76c5`.
It binds inspected source files, environment files, archives, traces, candidate
games, exact split inventories, public reset observations, and reset timings.
`readiness/build_manifest.py` and `manual_text_trace.py` are retained CPU scripts;
the manifest refuses overwrite. Set `ALFWORLD_DATA` to the dataset/cache root.

Minimal engine seam (already exercised):

```python
env = textworld.start(game_path,
    textworld.EnvInfos(won=True, admissible_commands=True),
    wrappers=[AlfredDemangler(shuffle=False)])
state = env.reset()
state, reward, done = env.step(public_command)
# Prompt projection: feedback + admissible_commands + public history only.
# Host metric: bool(state.won). Always close the environment.
```

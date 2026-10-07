# Bounded official GRPO comparison, recorded before launch

Decision time: 2026-10-07 12:52 UTC. Exploratory, not confirmatory.

Question: With the same small training budget and question-only input, does
the authors' standard GRPO option behave differently from Dr. GRPO?

Start from the original Qwen2.5-Math-1.5B base. Use the same 512 canonical
training questions, learner seed 43, eight answers per question, 32 updates,
4,096 sampled answers, learning rate 1e-6, and 3,000-token answer cap.
Change only the official critic_type option from drgrpo to grpo. Keep the
authors' source at dfca49dd460ee7cc8e4a5a162c876a7fd6993b87 unchanged.
Actor sampling is time-seeded, so randomness is not fully paired.

Primary endpoint: full official MATH-500 question-only evaluation of the
prescribed final step_00033 checkpoint after exactly 32 completed updates.
Compare with base 305/500 and Dr. GRPO runs 314/500 and 306/500, retaining all.
Chat-style full500 is secondary and runs only if at least five minutes remain.
No favorable checkpoint selection. The 64 monitoring questions are part of
these 500; all 500 have already been inspected, so this is exploratory.

Training cap: 55 minutes plus 60 seconds termination grace. Whole sequence
ends by 14:00 UTC plus at most 60 seconds termination grace. A partial/failed
run is diagnostic evidence, not a fixed-endpoint score. Save native model,
optimizer, monitoring answers and replay buffers; preserve failed attempts.

The objective switch changes both response-length averaging and reward
standard-deviation normalization. Even with the same learning rate, update
scale changes. A difference cannot identify either mechanism separately.
One seed cannot establish superiority. A negligible effect deprioritizes this
small-budget comparison; a large effect motivates fresh matched training
replicates and a separate gradient-scale comparison after the meeting.

Launch script SHA256:
d61cd7080902ef33b90af1ca0922b5bcf28dec4c94563d90beb5f407df185a26

Inputs: data-oat/train512-raw-grpo43 has exactly four byte-identical canonical
dataset files and 512 verified rows, with no copied map caches at preparation.
Heavy output: /home/atowell/research-runs/r1-zero-replication-20261007/subset512-raw-grpo43
Training log: subset512-raw-grpo43.log
Final evaluations: eval500-rawgrpo43-final-no-template and eval500-rawgrpo43-final-chat.
The five-slide deck is already published and will not be delayed for this run.

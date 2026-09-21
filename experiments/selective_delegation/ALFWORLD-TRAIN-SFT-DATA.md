# Fixed ALFWorld TRAIN action data

`alfworld-train-expert-raw-001` fixes 36 official TRAIN TextWorld games: six
per supported hand-coded family, selected by SHA256(`2026092199:path`) after
excluding the six earlier feasibility games. All selected games are retained:
30 won within the 50-executed-action cap and six reached that cap. Future SFT
uses the 30 successful games only, so it is explicitly a success-filtered TRAIN
demonstration dataset, not a representative action distribution.

`alfworld-train-sft-inputs-001` contains 524 public-observation action targets.
Each input is the exact indexed flat prompt used by source026, with initial and
current public feedback, admissible list, and prior executed action/feedback.
Neutral tokenizer accounting retained all histories (maximum 3,208 prompt tokens)
under the 8,192 context with 512 reserve and 128 output budget. Targets are
strict `{"action_index":int}` JSON plus EOS; prompt/target/loss-mask token
lengths are recorded. No expert plan, PDDL/task metadata, hidden facts, or won
state appears in a prompt.

This only prepares a possible fixed, one-epoch LoRA SFT prior. It is not an
accepted trainer or GPU reservation. Any later evaluation must use the same
public bridge, preserve action/token caps, and separate TRAIN from DEV.

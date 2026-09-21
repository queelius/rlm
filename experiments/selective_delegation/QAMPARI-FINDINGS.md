# QAMPARI fixed-pool screen: no established fan-out gain

Completed 2026-09-21 17:28:49 UTC. Sixteen hash-selected packaged development
questions, two repeats each; all five naturally selected question types retained.
Both arms receive the same first 200 BM25 passages. This tests fixed-input
fan-out, not global retrieval, adaptive search, or a novel PIG architecture.

## Complete native readout

| Arm | Valid attempts | Precision | Recall | F1 | Calls | Total tokens | Native seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct, 200 passages | 22/32 | .2882 | .2115 | .1863 | 32 | 1,035,930 | 623.98 |
| Four 50-passage maps, exact union | 24/32 | .1650 | .2440 | .1661 | 128 | 1,041,608 | 450.35 |

Map minus direct F1 is **−.0201**, parent-bootstrap 95% interval
**[−.0755, +.0274]**; precision −.1232 [−.2390, −.0250], recall +.0325
[−.0438, +.1178]. Bootstrap uses 20,000 draws, seed 2026092190, resampling
16 parents with their two repeats together. It does not establish independence
of shared entities/documents. Five tiny type strata are descriptive, not evidence
that composition specifically benefits.

There are 11 F1 wins and 10 losses for map: eight wins/nine losses with both
responses valid; three wins/one loss involving protocol failure. Over the full
32-attempt denominator, both-valid changes contribute −.0315 F1 and protocol
changes +.0113. The small validity advantage does not conceal a positive
content result. Map takes approximately 28% less measured native inference time
here, despite four times as many sequential calls and slightly more total tokens;
this is measured execution cost, not algorithmic sample efficiency.

All 160 planned calls and 64 episodes are present; no inference errors, missing
outcomes, unresolved starts, or unlinked calls. Owner elapsed time was 1,110.06s.
The independent audit rebuilt prompts/input IDs, seeds, sampling settings,
digests, native decoding and usage, and regraded with the official entity-set
metric. No answer repair or permissive parsing was used.

## Cap hits mostly expose degeneration, not just insufficient space

Every invalid response ended at its output cap: ten direct calls at 1,024 tokens,
twelve map calls at 256 tokens affecting eight map attempts. No time-cap stops.
All 22 raw heads/tails were inspected. All ten direct calls repeat completed
strings; eight of twelve map calls do too. These are text diagnostics, not
recovered answer lists or changed grades. Counting complete quoted string spans
after the `answers` key gives, for example:

- Colin Welland directors: `Colin Tilley` repeats 163 and 164 times in direct
  repeats 0/1. Both remain invalid at the cap.
- Pharaohs: direct repeats loop `Pharaoh (module)` 139 times and
  `Pharaoh (Prus novel)` 97 times; a map block repeats `Pharaoh Kaiba` seven times.
- Island countries: direct repeat 1 repeats `USA` 254 times.
- Geothermal stations: direct repeat 0 has 171 complete strings but only 34
  distinct strings, cycling through a station list.

Some map tails are non-repeating but drift into generic dynasty labels, unrelated
places, or `Reef of ...` templates. Longer caps could close some responses, but
the evidence does not justify treating all cap hits as useful unfinished lists.

## Grounded help/harm, including both repeats

**Harm: distractors survive union.** For Scott Z. Burns screenplays
(`00f495d67c6fbcf65a80641c`), direct returns the same six credited films in both
repeats, F1 .800/.800. Map retains all six but adds an unproduced adaptation,
`The Library`, and title variants: F1 .667/.632. Passage d114 explicitly calls
The Library a **play**; d66 describes plans for a future adaptation, not writing
the 1954 film. Repeat 1 also retains a trailing-space duplicate of The Informant!
under the frozen exact-string union. This is observed type/relation filtering
and representation error, not a demonstrated need for cross-block planning.

**Help, but low precision: more explicit contest names.** For the Baltimore
Orioles intersection question (`1c931b8ac91c44d01927dcbc`), direct mainly lists
years/Temple Cup and scores zero both times. Map's last block supplies three
credited ALCS names in both repeats, raising recall to .5 and F1 to .150/.136.
It also invents a sequence of ALCS years, so precision is only .088/.079.
This single intersection example cannot establish an intersection benefit.

**A concrete relation failure.** For Rabanus Maurus's students
(`27b23a5937225538cd3ffdcc`), map's last block answers Marie Curie in both repeats;
d155 discusses a famous student of the Flying University, not Rabanus. Map also
loses Lupus Servatus: F1 .333/.333 versus direct .462/.462. Splitting evidence
has not solved relevance checking.

The New London composition case improves F1 slightly (.286/.247 → .318/.286)
while losing recall (.818 → .636 both times). We did **not** establish a necessary
cross-block join whose absence explains a measured loss; do not label these
examples proof that recursive composition would rescue the panel.

Host-only literal aliases appear for 100/168 canonical entities, parent-macro
64.04%. This is neither supporting-evidence verification nor an achievable
ceiling. Alias incompleteness matters: the Pakistan armament question's reference
requires names with parenthesized abbreviations, so both arms receive zero for
ordinary forms such as Heavy Industries Taxila and Pakistan Ordnance Factories.
Do not equate every official mismatch or missing literal alias with reader failure.

## Ranked next decisions

1. **Retire cap doubling after the complete trace inspection.** Longer versions
   of the observed loops are unlikely to answer the scientific question. The
   22-cap-plus-ten-control proposal is not prioritized or implemented.
2. Qualify the direct baseline once using the model's documented sampling
   defaults: temperature .7, top-p .8, top-k 20. Reissue all 32 original direct
   slots with the same prompts, seeds, 1,024-token cap, model, strict parser and
   official metric. This is a packaged sampling comparison, not an isolated
   temperature effect or a new independent panel. A repetition penalty of 1.1
   was considered but is not the proposed control: HF also penalizes prompt
   tokens, potentially hurting copying of evidence names. Cap 30 minutes,
   expected 5–10 minutes but conservatively allow the original direct cost
   (approximately 10.4 native minutes). No implementation or GPU acceptance yet.
   If direct substantially improves validity/content, give map the same sampling
   package; otherwise end this fan-out branch rather than sweep chunk sizes.
3. Do not train a router or recursive controller from this screen. Establish a
   repeatable benefit on new questions before allocating that optimization budget.

## Evidence and reproducibility

Research store `R` is
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Native run: `R/qampari-reading-001`; frozen input: `R/qampari-inputs-001`.
Independent report: `R/analysis-qampari-001.json/.md` (JSON SHA-256
`2e0896c933e8c41817ae6325c0e4a354e3f7bd1b130a87a324e3c95c8b9d15d1`).
PLAN SHA-256: `47631abaa069691032980716e959d006f71688ac669643b19d399db174601456`.
Analyzer: `R/analysis-source-qampari-001/analyze_qampari.py`; five focused tests
passed before outcomes. The report binds every consumed receipt and small source
dependency. Reproduce to a **new** report destination with `--output`, `--cases`,
and `--report`; no GPU/model-weight loading is needed.

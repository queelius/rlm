# Eight-case supplied-evidence audit

This LLM-assisted manual diagnostic found seven apparently insufficient negatives and one negative with apparent alternative explicit support. It does **not** estimate label-noise prevalence, create new gold answers, or alter any score, TRAIN reward, or held-out panel.

Before reading documents or model outcomes for this audit, eight of the original exposed panel's 32 parents were frozen by ascending SHA256(`2026092204:` + parent ID). All 20 public documents in each positive and negative variant were read, displaying identical shared text once with both variant-specific document IDs. Official decomposition/support annotations were inspected afterward as host-only metadata. No external knowledge was used to supply missing facts; no model-output receipts were consumed.

Immutable selection and detailed reasons/hashes are in `R/sufficiency-semantic-audit-001/SELECTION.json` and `AUDIT.json`. Original cases SHA256: `ed6f40e7fe23403cbf8d8d708d9375a27acf79c9c6436050a3f7334edb2e50a0`. Selection SHA256: `eb432e3ff60a1532bd9907dfebb733a5a5cc62395d8765e1fb369789ff1e48e5`. Here `N` and `P` denote negative and positive public-document IDs, not original supporting indices.

| Selected parent | Negative assessment | Supplied-text reason |
|---|---|---|
| `2hop__156700_63853` | Apparently insufficient | Kraai→Orange River is present in N:d13, but Orange's source is absent. P:d14 supplies Thaba Putsoa. |
| `2hop__130984_55721` | Apparently insufficient | The negative gives other high schools, not Sandy High School's location or Oregon rainfall. P:d3 and P:d5 supply the chain. |
| `2hop__156702_51769` | Apparently insufficient | N:d13 establishes Patuxent→Chesapeake Bay, but no retriever lifespan appears. P:d6 supplies that fact. |
| `2hop__91667_67223` | Apparent alternative explicit support | N:d12 itself identifies Ted telling the story in2030 and his married life with Tracy Mosby. The designated narrator paragraph is absent, but this retained paragraph reconnects both facts. |
| `2hop__56270_68396` | Apparently insufficient | Neither951→California nor the state's national-population percentage is supplied. Negative951 mentions concern population/year/baseball, not the area-code relation. P:d11 and P:d18 supply the chain. |
| `2hop__161507_77849` | Apparently insufficient | N:d9 contains the gold Chief Justice name, but Masherbrum→Pakistan is absent. P:d12 supplies that missing bridge. A correct name mention is not full-question support. |
| `2hop__619265_72380` | Apparently insufficient | N:d17 contains the gold date under Ray Donovan episodes, but The Bag or the Bat→Ray Donovan is absent. P:d18 supplies that link. |
| `2hop__153532_72380` | Apparently insufficient | N:d18 identifies the episode's series and2013 pilot date, but no season-four date is supplied. P:d9 adds the date excerpt. |

The Ted/Tracy example was already known; it happened to rank fourth under the frozen selection. It is a reconfirmation, not another independent discovery. Its retained paragraph describes an alternate ending, so the conclusion is apparent support from the supplied passage—not an adjudication of outside television canon.

Two additional positive-context qualifications surfaced. The retriever paragraph lists multiple survey statistics (UK median10.75, UK mean9.85, US mean9.4 years), while the generic question's gold selects just9.4. Both Ray Donovan positives contain a flattened episode37 row giving online June20 and Showtime June26,2016 dates, without explicitly identifying season four or its premiere. That missing table context must not be filled from memory. The two Ray Donovan questions also share atomic component72380 and near-identical meaning, so eight sampled parent IDs are not eight independent semantic situations.

Decision relevance: designated-support deletion does not guarantee every negative lacks an alternative chain, but literal answer occurrence is also an unsafe noise detector. This sample qualifies the interpretation of official paired sufficiency reward; it does not establish that label noise explains any observed SFT/RL gain or conservatism. Keep current training and evaluation contracts unchanged. Any broader label-quality study would require a separately designed annotation protocol, not opportunistic correction of these eight cases.

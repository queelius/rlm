# E31 pre-registration (written 2026-10-22 04:50 UTC, before any row of this run exists)

Question: does choosing the MuSiQue routine by haystack length, as a statistics table would, beat the single
default routine on questions not used to choose it?

Rule, read off E22d (fixed reader, task seed 0): 8k -> filter+BM25 ("both"), 32k -> BM25 top 60, 128k -> both.
Test set: task seed 1 (90 dev questions, 3 shared with seed 0), same reader (1,500 tokens + fallback), same
server for all arms. Primary metric: mean answer F1 of the routed policy minus "both" everywhere, paired over
the 90 questions (it differs only at 32k). Decision: routed - both > 0 with paired z > 2 promotes length
routing for this family; |z| < 2 means the seed-0 gap at 32k (+0.063) was selection noise and the
single routine stays; also report tokens per question (BM25-60 costs about a quarter of "both" at 32k).

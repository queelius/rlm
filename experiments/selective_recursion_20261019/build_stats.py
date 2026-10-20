"""Build the routine-outcome statistics table from logged runs (teaching/train splits only).

Usage: build_stats.py --out stats.json RUN[:routine] ...   e.g. teach-lib-009 sweep-001-t0
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skills import stats  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("runs", nargs="+")
a = p.parse_args()
records = []
for spec in a.runs:
    run, _, routine = spec.partition(":")
    records += stats.collect(Path(run), routine=routine or None)
table = stats.summarize(records)
Path(a.out).write_text(json.dumps(dict(sources=a.runs, records=len(records), table=table), indent=1))
print(f"{len(records)} records, {len(table)} signatures -> {a.out}")

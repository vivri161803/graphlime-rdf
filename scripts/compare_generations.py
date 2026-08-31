"""Compare two generations of results/*.jsonl field-by-field (M8 determinism
evidence): every numeric field must match exactly; git_commit may differ
(it records the HEAD at generation time, not the math).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
families = sorted(p.name for p in a_dir.glob("*.jsonl"))
assert families, f"no jsonl in {a_dir}"
failures = 0
for name in families:
    a_lines = (a_dir / name).read_text().splitlines()
    b_path = b_dir / name
    if not b_path.exists():
        print(f"{name}: MISSING in {b_dir}")
        failures += 1
        continue
    b_lines = b_path.read_text().splitlines()
    if len(a_lines) != len(b_lines):
        print(f"{name}: row count {len(a_lines)} vs {len(b_lines)}")
        failures += 1
        continue
    diffs = 0
    for la, lb in zip(a_lines, b_lines, strict=True):
        ra, rb = json.loads(la), json.loads(lb)
        ra.pop("git_commit"), rb.pop("git_commit")
        if ra != rb:
            diffs += 1
    status = "OK" if diffs == 0 else f"{diffs} differing rows"
    if diffs:
        failures += 1
    print(f"{name}: {len(a_lines)} rows, {status}")
print("DETERMINISM " + ("VERIFIED" if failures == 0 else f"FAILED ({failures} families)"))
sys.exit(1 if failures else 0)

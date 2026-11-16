"""Python 3.11 rejects unhashable dataclass defaults (`x: FooConfig = FooConfig()`); habitat-lab 0.3.x was written for 3.9.
Rewrite them to `field(default_factory=FooConfig)` in the given trees. Idempotent.  python patches/patch_py311_dataclass.py DIR..."""
import re, sys
from pathlib import Path
PAT = re.compile(r"^(\s+\w+\s*:\s*[\w\.\[\], ]+?\s*=\s*)([A-Z]\w*Config)\(\)\s*$", re.M)
for root in sys.argv[1:]:
    for p in Path(root).rglob("*.py"):
        s = p.read_text()
        if not PAT.search(s):
            continue
        s2 = PAT.sub(lambda m: f"{m.group(1)}field(default_factory={m.group(2)})", s)
        if not re.search(r"^from dataclasses import .*\bfield\b", s2, re.M) and "dataclasses.field" not in s2:
            s2 = "from dataclasses import field\n" + s2 if not s2.startswith("#!") else s2
        p.write_text(s2); print("patched", p, len(PAT.findall(s)))

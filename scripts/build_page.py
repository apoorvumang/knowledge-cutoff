#!/usr/bin/env python3
"""Inline report_data.json into report_template.html -> report_explorer.html.

The data goes inside a <script type="application/json"> block; we escape '<'
as \\u003c (valid inside JSON strings) so a response containing '</script>'
can't terminate the block early.
"""
import base64
import glob
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tpl = open(os.path.join(HERE, "report_template.html"), encoding="utf-8").read()
data = open(os.path.join(HERE, "report_data.json"), encoding="utf-8").read()
data = data.replace("<", "\\u003c")

# Real provider logos (assets/logos/<provider>.svg) -> base64 data: URIs, so each
# vendor SVG is fully isolated (no cross-logo CSS class / gradient-id collisions).
MIME = {".svg": "image/svg+xml", ".png": "image/png"}
logos = {}
for path in sorted(glob.glob(os.path.join(HERE, "assets", "logos", "*.svg")) +
                   glob.glob(os.path.join(HERE, "assets", "logos", "*.png"))):
    key, ext = os.path.splitext(os.path.basename(path))
    b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
    logos[key] = f"data:{MIME[ext]};base64," + b64

answers = open(os.path.join(HERE, "report_answers.json"), encoding="utf-8").read()
answers_esc = answers.replace("<", "\\u003c")

# Model count is derived, not hardcoded, so prose can't drift as models are added.
nmodels = len(json.loads(open(os.path.join(HERE, "report_data.json"), encoding="utf-8").read())["models"])

def render(answers_literal: str, answers_url: str) -> str:
    o = tpl.replace("__DATA__", data)
    o = o.replace("__LOGOS__", json.dumps(logos))
    o = o.replace("__ANSWERS__", answers_literal)
    o = o.replace("__ANSWERS_URL__", answers_url)
    o = o.replace("__NMODELS__", str(nmodels))
    assert "__" not in o.replace("__", "", 0) or True
    return o

out = render(answers_literal="null", answers_url="report_answers.json")
# Local build artifact for the claude.ai Artifact: must be a single self-contained
# file, because the Artifact CSP blocks fetches -- so the answers stay inlined there.
dest = os.path.join(HERE, "report_explorer.html")
standalone = render(answers_literal=answers_esc, answers_url="")
open(dest, "w", encoding="utf-8").write(standalone)

# The GitHub Pages copy (committed, served at /docs) instead ships the answers as a
# sibling file the page fetches after first paint, which is ~89% of the bytes.
docs = os.path.join(HERE, "docs")
os.makedirs(docs, exist_ok=True)
open(os.path.join(docs, "index.html"), "w", encoding="utf-8").write(out)
open(os.path.join(docs, "report_answers.json"), "w", encoding="utf-8").write(answers)
open(os.path.join(docs, ".nojekyll"), "w").write("")  # serve files as-is
print(f"wrote {os.path.relpath(dest, HERE)} ({len(standalone)/1024:.0f} KB, self-contained)")
print(f"wrote docs/index.html ({len(out)/1024:.0f} KB initial)"
      f" + docs/report_answers.json ({len(answers)/1024:.0f} KB deferred)")

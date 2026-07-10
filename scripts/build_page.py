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
logos = {}
for path in sorted(glob.glob(os.path.join(HERE, "assets", "logos", "*.svg"))):
    key = os.path.splitext(os.path.basename(path))[0]
    b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
    logos[key] = "data:image/svg+xml;base64," + b64

out = tpl.replace("__DATA__", data)
out = out.replace("__LOGOS__", json.dumps(logos))
# Local build artifact (used for the claude.ai Artifact) ...
dest = os.path.join(HERE, "report_explorer.html")
open(dest, "w", encoding="utf-8").write(out)
# ... and the GitHub Pages copy (committed, served at /docs).
docs = os.path.join(HERE, "docs")
os.makedirs(docs, exist_ok=True)
open(os.path.join(docs, "index.html"), "w", encoding="utf-8").write(out)
open(os.path.join(docs, ".nojekyll"), "w").write("")  # serve files as-is
print(f"wrote {os.path.relpath(dest, HERE)} and docs/index.html ({len(out)/1024:.0f} KB)")

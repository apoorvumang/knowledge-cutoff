#!/usr/bin/env python3
"""Inline report_data.json into report_template.html -> report_explorer.html.

The data goes inside a <script type="application/json"> block; we escape '<'
as \\u003c (valid inside JSON strings) so a response containing '</script>'
can't terminate the block early.
"""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tpl = open(os.path.join(HERE, "report_template.html"), encoding="utf-8").read()
data = open(os.path.join(HERE, "report_data.json"), encoding="utf-8").read()
data = data.replace("<", "\\u003c")
out = tpl.replace("__DATA__", data)
dest = os.path.join(HERE, "report_explorer.html")
open(dest, "w", encoding="utf-8").write(out)
print(f"wrote {os.path.relpath(dest, HERE)} ({len(out)/1024:.0f} KB)")

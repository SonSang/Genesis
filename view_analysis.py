"""Render a numbered-hierarchy analysis file in multiple formats.

Input format (rigid_solver_analysis.txt):
    1. Forward
    2. Backward
    2-1. kernel_prepare_backward_substep
    2-1-1. func_load_adjoint_cache

    free-form description lines (any non-numbered line) attaches to the
    most recent node.

Usage:
    python view_analysis.py [path]                 # ANSI-colored tree
    python view_analysis.py --plain                # no color
    python view_analysis.py --markdown > out.md    # markdown
    python view_analysis.py --html > out.html      # standalone HTML
    python view_analysis.py --html --open          # write + open in browser
"""
import re
import sys
from pathlib import Path


NUMBER_RE = re.compile(r"^(\d+(?:-\d+)*)\.\s*(.*)$")

# 256-color palette by depth. Each row is one level deeper.
# Picked for contrast on dark and light terminals.
DEPTH_COLORS = [
    39,    # depth 1 — bright cyan
    141,   # depth 2 — purple
    220,   # depth 3 — gold
    78,    # depth 4 — green
    203,   # depth 5 — coral red
    81,    # depth 6 — light teal
    213,   # depth 7 — pink
    229,   # depth 8 — pale yellow
]
DESC_COLOR = 244  # gray for free-form description lines
TREE_COLOR = 240  # darker gray for tree branch characters


def ansi(code: int, s: str) -> str:
    return f"\x1b[38;5;{code}m{s}\x1b[0m"


def bold(s: str) -> str:
    return f"\x1b[1m{s}\x1b[0m"


def parse(path: Path):
    """Parse lines into (depth, number, title, description_lines)."""
    nodes = []  # list of dicts: {depth, number, title, desc}
    pending_desc = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip():
            if pending_desc and pending_desc[-1] != "":
                pending_desc.append("")
            continue
        m = NUMBER_RE.match(line.strip())
        if m:
            # Attach any pending description to the previous node
            if nodes and pending_desc:
                # strip trailing blanks
                while pending_desc and pending_desc[-1] == "":
                    pending_desc.pop()
                nodes[-1]["desc"] = pending_desc
            pending_desc = []
            number = m.group(1)
            title = m.group(2)
            depth = number.count("-") + 1
            nodes.append({"depth": depth, "number": number, "title": title, "desc": []})
        else:
            pending_desc.append(line)

    # Trailing description
    if nodes and pending_desc:
        while pending_desc and pending_desc[-1] == "":
            pending_desc.pop()
        nodes[-1]["desc"] = pending_desc

    return nodes


def render(nodes, use_color: bool) -> str:
    """Render the tree.

    For each node we draw the indent prefix based on which ancestor depths
    still have more siblings to come.
    """
    # Precompute: for each node, is it the last sibling at its depth?
    # Determined by looking ahead for the next node at the same depth with
    # the same parent prefix.
    is_last = [False] * len(nodes)
    for i, node in enumerate(nodes):
        depth = node["depth"]
        parent_prefix = node["number"].rsplit("-", 1)[0] if depth > 1 else ""
        # Look ahead: any subsequent node with same depth AND same parent prefix?
        last = True
        for j in range(i + 1, len(nodes)):
            nd = nodes[j]
            if nd["depth"] < depth:
                break  # left this subtree
            if nd["depth"] == depth:
                nd_parent = nd["number"].rsplit("-", 1)[0] if nd["depth"] > 1 else ""
                if nd_parent == parent_prefix:
                    last = False
                    break
        is_last[i] = last

    # For each depth, track whether an ancestor at that depth still has
    # more siblings to come (→ draw "│", else " ").
    out_lines = []

    def color(code, s):
        return ansi(code, s) if use_color else s

    def tree_chr(s):
        return color(TREE_COLOR, s) if use_color else s

    # ancestor_more[d] = True means "at depth d there will be a later sibling"
    ancestor_more = {}

    for i, node in enumerate(nodes):
        d = node["depth"]
        # Build the prefix from depth 1 .. d-1, then the connector at depth d
        prefix = ""
        for anc_d in range(1, d):
            prefix += tree_chr("│   ") if ancestor_more.get(anc_d, False) else "    "
        connector = tree_chr("└── ") if is_last[i] else tree_chr("├── ")
        prefix += connector

        depth_color = DEPTH_COLORS[min(d - 1, len(DEPTH_COLORS) - 1)]
        label = f"{node['number']}. {node['title']}"
        styled_label = color(depth_color, bold(label) if use_color else label)
        out_lines.append(prefix + styled_label)

        # Update ancestor tracker for this depth (children inherit)
        ancestor_more[d] = not is_last[i]
        # Reset deeper levels (they'll be re-set when traversed)
        for deeper in list(ancestor_more.keys()):
            if deeper > d:
                del ancestor_more[deeper]

        # Description lines: indent under this node
        if node["desc"]:
            desc_indent = ""
            for anc_d in range(1, d):
                desc_indent += tree_chr("│   ") if ancestor_more.get(anc_d, False) else "    "
            desc_indent += tree_chr("│   ") if not is_last[i] else "    "
            for desc_line in node["desc"]:
                if desc_line == "":
                    out_lines.append(desc_indent)
                else:
                    out_lines.append(desc_indent + color(DESC_COLOR, desc_line))

    return "\n".join(out_lines)


def render_markdown(nodes) -> str:
    """Render as markdown with heading hierarchy (# .. ###### for depths 1-6).

    Deeper levels use bold list items so structure is preserved.
    Descriptions render as regular paragraphs under each heading.
    """
    lines = []
    for node in nodes:
        d = node["depth"]
        label = f"{node['number']}. {node['title']}"
        if d <= 6:
            lines.append(f"{'#' * d} {label}")
        else:
            # Beyond h6, use bold list items with indentation
            indent = "  " * (d - 7)
            lines.append(f"{indent}- **{label}**")
        if node["desc"]:
            lines.append("")
            for desc_line in node["desc"]:
                lines.append(desc_line)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg: #fafafa;
  --fg: #1f2328;
  --muted: #6b7280;
  --border: #e5e7eb;
  --code-bg: #f3f4f6;
  --c1: #0891b2;  /* cyan */
  --c2: #7c3aed;  /* purple */
  --c3: #b45309;  /* gold/amber */
  --c4: #059669;  /* green */
  --c5: #dc2626;  /* coral red */
  --c6: #0284c7;  /* sky blue */
  --c7: #db2777;  /* pink */
  --c8: #ca8a04;  /* dark yellow */
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #0d1117;
    --fg: #e6edf3;
    --muted: #9ca3af;
    --border: #30363d;
    --code-bg: #161b22;
    --c1: #22d3ee;
    --c2: #c4b5fd;
    --c3: #fbbf24;
    --c4: #4ade80;
    --c5: #f87171;
    --c6: #60a5fa;
    --c7: #f472b6;
    --c8: #facc15;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: -apple-system, "Segoe UI", "Noto Sans KR", sans-serif;
  background: var(--bg);
  color: var(--fg);
  max-width: 1100px;
  margin: 0 auto;
  padding: 32px 24px 80px 24px;
  line-height: 1.55;
  font-size: 15px;
}}
h1.doc-title {{
  font-size: 1.4em;
  border-bottom: 1px solid var(--border);
  padding-bottom: 8px;
  margin-bottom: 24px;
  color: var(--fg);
}}
.node {{
  margin: 6px 0;
  padding-left: 14px;
  border-left: 3px solid transparent;
}}
.node-title {{
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 5px;
  display: inline-block;
  font-family: "SF Mono", "Menlo", "Consolas", monospace;
  font-size: 0.95em;
}}
.node-num {{ opacity: 0.55; margin-right: 0.4em; font-weight: 500; }}
.node-desc {{
  color: var(--muted);
  margin: 6px 0 8px 12px;
  white-space: pre-wrap;
  font-family: inherit;
}}
{depth_css}
.controls {{
  position: sticky; top: 0; background: var(--bg);
  padding: 8px 0 12px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
  font-size: 13px;
  color: var(--muted);
}}
.controls button {{
  font-size: 12px; padding: 4px 10px; margin-right: 6px;
  border: 1px solid var(--border); border-radius: 4px;
  background: var(--code-bg); color: var(--fg); cursor: pointer;
}}
.controls button:hover {{ background: var(--border); }}
.legend {{ float: right; }}
.legend .chip {{
  display: inline-block; padding: 1px 8px; margin-left: 4px;
  border-radius: 10px; font-size: 11px; font-weight: 600;
}}
@media (max-width: 700px) {{
  body {{ padding: 16px 12px; }}
  .legend {{ display: none; }}
}}
</style>
</head>
<body>

<h1 class="doc-title">{title}</h1>

<div class="controls">
  <button onclick="document.querySelectorAll('.node-desc').forEach(e => e.style.display='none')">Hide descriptions</button>
  <button onclick="document.querySelectorAll('.node-desc').forEach(e => e.style.display='block')">Show descriptions</button>
  <span class="legend">
    <span class="chip" style="background:var(--c1);color:white">1</span>
    <span class="chip" style="background:var(--c2);color:white">2</span>
    <span class="chip" style="background:var(--c3);color:white">3</span>
    <span class="chip" style="background:var(--c4);color:white">4</span>
    <span class="chip" style="background:var(--c5);color:white">5</span>
    <span class="chip" style="background:var(--c6);color:white">6</span>
  </span>
</div>

{body}

</body>
</html>
"""


def render_html(nodes, title: str) -> str:
    """Render as standalone HTML with per-depth color coding + indentation."""
    # Build per-depth CSS
    depth_css_parts = []
    colors = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    for d in range(1, 13):  # support up to depth 12
        c = colors[min(d - 1, len(colors) - 1)]
        indent_px = (d - 1) * 28
        depth_css_parts.append(
            f".node.depth-{d} {{ margin-left: {indent_px}px; border-left-color: var(--{c}); }}\n"
            f".node-title.depth-{d} {{ color: var(--{c}); background: color-mix(in srgb, var(--{c}) 12%, transparent); }}"
        )
    depth_css = "\n".join(depth_css_parts)

    body_parts = []
    for node in nodes:
        d = node["depth"]
        num = html_escape(node["number"])
        title_text = html_escape(node["title"])
        body_parts.append(f'<div class="node depth-{d}">')
        body_parts.append(
            f'  <span class="node-title depth-{d}">'
            f'<span class="node-num">{num}.</span>{title_text}'
            f'</span>'
        )
        if node["desc"]:
            desc = "\n".join(node["desc"])
            body_parts.append(f'  <div class="node-desc">{html_escape(desc)}</div>')
        body_parts.append("</div>")
    body = "\n".join(body_parts)

    return HTML_TEMPLATE.format(title=html_escape(title), depth_css=depth_css, body=body)


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


def main():
    args = sys.argv[1:]
    use_color = True
    mode = "tree"  # tree | markdown | html
    open_browser = False
    path = Path("rigid_solver_analysis.txt")

    for a in args:
        if a == "--plain":
            use_color = False
        elif a == "--markdown" or a == "--md":
            mode = "markdown"
        elif a == "--html":
            mode = "html"
        elif a == "--open":
            open_browser = True
        elif a.startswith("--"):
            print(f"unknown flag: {a}", file=sys.stderr)
            sys.exit(2)
        else:
            path = Path(a)

    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        sys.exit(1)

    nodes = parse(path)
    if not nodes:
        print("(no nodes parsed)", file=sys.stderr)
        sys.exit(0)

    if mode == "markdown":
        print(render_markdown(nodes), end="")
    elif mode == "html":
        title = path.stem.replace("_", " ")
        html = render_html(nodes, title)
        if open_browser:
            out_path = path.with_suffix(".html")
            out_path.write_text(html, encoding="utf-8")
            print(f"wrote {out_path}", file=sys.stderr)
            import webbrowser
            webbrowser.open(out_path.absolute().as_uri())
        else:
            print(html)
    else:
        print(render(nodes, use_color))


if __name__ == "__main__":
    main()

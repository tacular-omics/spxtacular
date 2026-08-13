#!/usr/bin/env python3
"""Emit paper.md and paper.bib: the files the Journal of Open Source Software takes.

WHY THIS EXISTS. JOSS ingests a Markdown file with a YAML front matter block and
a BibTeX file beside it, and runs them through pandoc itself. This scaffold
authors in Typst. Hand-maintaining both would put the same paper in two files,
which is the one failure mode the rest of this directory is built to prevent: the
copy is what goes stale, and the copy that goes stale is always the one nobody
rebuilt before submitting.

So paper.md is GENERATED, from exactly the sources everything else here reads:

    config.typ     title, authors, ORCIDs, affiliations, keywords, date
    paper.typ      the prose between the BODY START / BODY END markers
    stats.json     every #s("id") resolved to its rendered value
    assets.json    every fig("id") resolved to its path
    references.bib the bibliography, copied to joss/paper.bib

WHY A SUBDIRECTORY. Two reasons, and both are load-bearing. JOSS resolves a
figure path relative to the paper file, so everything the submission needs sits
beside paper.md and a reviewer can read the bundle without reasoning about the
rest of the repository. And a second `.bib` at the manuscript root would collide
with the one already there: `just prose-check` scans every root `*.bib` as one
bibliography and would report all thirteen entries as duplicated. Suppressing
that would have switched off a real check to make room for a copy.

Everything under joss/ is GENERATED and TRACKED, like figures/ and si/, because
JOSS needs it present in the repository and its reviewers must be able to read
it without a Typst install. `--check` re-derives all three files and fails if
what is committed differs, which is what makes "regenerate before submitting" a
gate rather than a ritual. The copied figure is a copy that cannot go stale
quietly: `just verify` compares its bytes on every run.

WHAT IS DELIBERATELY DROPPED. The abstract (JOSS has none), the Supporting
Information (JOSS has none either), and the back matter that the JOSS template
supplies for itself. The SI is still in the PDF, and the Reproducibility section
in it names the commands, so nothing that only lives there is load-bearing for
the submission.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "config.typ"
PAPER = ROOT / "paper.typ"
BIB_IN = ROOT / "references.bib"

# The submission bundle. Self-contained: paper.md, the bibliography it names,
# and every figure it references, all beside each other.
OUT_DIR = ROOT / "joss"
MD_OUT = OUT_DIR / "paper.md"
BIB_OUT = OUT_DIR / "paper.bib"

BODY_START = re.compile(r"(?m)^// >>> BODY START.*$")
BODY_END = re.compile(r"(?m)^// <<< BODY END.*$")


# ---------------------------------------------------------------------------
# config.typ
# ---------------------------------------------------------------------------

def _typst_string(name: str, src: str) -> str:
    """Read `#let <name> = "..."` out of config.typ."""
    m = re.search(rf'#let\s+{re.escape(name)}\s*=\s*"((?:[^"\\]|\\.)*)"', src)
    if not m:
        raise SystemExit(f"config.typ has no string binding for {name}")
    return m.group(1).replace('\\"', '"')


def _typst_list(name: str, src: str) -> list[str]:
    """Read `#let <name> = ("a", "b")` out of config.typ."""
    m = re.search(rf"#let\s+{re.escape(name)}\s*=\s*\((.*?)\)\s*(?:\n\n|\n#|\Z)",
                  src, re.S)
    if not m:
        raise SystemExit(f"config.typ has no list binding for {name}")
    return re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))


def _authors(src: str) -> list[dict[str, str | bool]]:
    """Read the paper-authors array, one dict per author.

    Parsed by splitting on the record braces rather than with one regex over the
    whole block: an author who happens to lack an email or an ORCID must come out
    as an author with a missing field, not shift every later field by one.
    """
    m = re.search(r"#let\s+paper-authors\s*=\s*\((.*?)\n\)\s*\n", src, re.S)
    if not m:
        raise SystemExit("config.typ has no paper-authors block")
    out: list[dict[str, str | bool]] = []
    for rec in re.findall(r"\(\s*\n(.*?)\n\s*\)", m.group(1), re.S):
        a: dict[str, str | bool] = {}
        for key, val in re.findall(r'(\w+)\s*:\s*"((?:[^"\\]|\\.)*)"', rec):
            a[key] = val
        if re.search(r"corresponding\s*:\s*true", rec):
            a["corresponding"] = True
        if "name" in a:
            out.append(a)
    if not out:
        raise SystemExit("paper-authors parsed as empty")
    return out


def front_matter() -> str:
    src = CONFIG.read_text()
    title = _typst_string("paper-title", src)
    keywords = _typst_list("paper-keywords", src)
    date = _typst_string("paper-date", src)
    authors = _authors(src)

    # Affiliations numbered in first-appearance order, the same rule config.typ
    # applies for the PDF, so the two front matters cannot disagree.
    affils: list[str] = []
    for a in authors:
        if a["affiliation"] not in affils:
            affils.append(str(a["affiliation"]))

    lines = ["---", f"title: {yaml_str(title)}", "tags:"]
    lines += [f"  - {yaml_str(k)}" for k in keywords]
    lines.append("authors:")
    for a in authors:
        lines.append(f"  - name: {yaml_str(str(a['name']))}")
        if a.get("orcid"):
            lines.append(f"    orcid: {a['orcid']}")
        if a.get("corresponding"):
            lines.append("    corresponding: true")
        lines.append(f"    affiliation: {affils.index(str(a['affiliation'])) + 1}")
    lines.append("affiliations:")
    for i, affil in enumerate(affils, start=1):
        lines.append(f"  - name: {yaml_str(affil)}")
        lines.append(f"    index: {i}")
    lines.append(f"date: {yaml_str(date)}")
    lines.append("bibliography: paper.bib")
    lines.append("---")
    return "\n".join(lines)


def yaml_str(s: str) -> str:
    """Quote for YAML. Single quotes, doubled inside, which JOSS's own examples use."""
    return "'" + s.replace("'", "''") + "'"


# ---------------------------------------------------------------------------
# paper.typ -> Markdown
# ---------------------------------------------------------------------------

def _values() -> dict[str, str]:
    """Rendered display strings for every #s("id"), from stats-rendered.json.

    Reads the RENDERED file rather than formatting the values here, so a number
    in paper.md is the same string the PDF prints. Two formatters would be two
    answers to "what does this number look like".
    """
    p = ROOT / "stats-rendered.json"
    if not p.is_file():
        raise SystemExit(
            "stats-rendered.json is missing. It is generated: run `just joss`, "
            "which renders it first, rather than this script directly.")
    return {k: v["display"] for k, v in json.load(p.open())["values"].items()}


def _assets() -> dict[str, str]:
    p = ROOT / "assets.json"
    if not p.is_file():
        return {}
    return {k: v["path"] for k, v in json.load(p.open())["values"].items()}


def _balanced(text: str, start: int) -> tuple[str, int]:
    """Return the contents of the (...) or [...] beginning at `start`, and the index after it."""
    open_ch = text[start]
    close_ch = {"(": ")", "[": "]"}[open_ch]
    depth, i = 0, start
    while i < len(text):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    raise SystemExit(f"unbalanced {open_ch} in paper.typ at offset {start}")


def _figures(body: str, assets: dict[str, str]) -> str:
    """Rewrite `#figure(fig("id", ...), caption: [...]) <fig:x>` as a JOSS image.

    JOSS renders `![caption](path)` and cross-references it as \\autoref{fig:x},
    which is what the label on the Typst figure becomes.
    """
    out, i = [], 0
    while True:
        j = body.find("#figure(", i)
        if j < 0:
            out.append(body[i:])
            return "".join(out)
        out.append(body[i:j])
        inner, after = _balanced(body, j + len("#figure"))

        m = re.search(r'fig\(\s*"([^"]+)"', inner)
        cap = re.search(r"caption:\s*\[", inner)
        label = re.match(r"\s*<([^>]+)>", body[after:])
        if not m or not cap or not label:
            # A table figure, or one this rewriter does not understand. Dropping
            # it silently would lose content, so say so and keep going.
            print(f"note: skipped a #figure() with no fig() id or no label "
                  f"at offset {j}", file=sys.stderr)
            out.append(body[j:after])
            i = after
            continue

        caption, _ = _balanced(inner, cap.end() - 1)
        path = assets.get(m.group(1))
        if path is None:
            raise SystemExit(f"figure id {m.group(1)} is not declared in assets.json")
        FIGURES.add(path)
        # Rewritten to sit beside paper.md, because JOSS resolves an image path
        # relative to the paper file.
        out.append(f'![{inline(caption).strip()}]({path}){{ #{label.group(1)} }}')
        i = after + label.end()


def inline(text: str) -> str:
    """Typst inline markup -> Markdown, for a fragment with no block structure.

    Every call pattern is written to tolerate the line breaks typstyle inserts:
    `just fmt` reflows to 80 columns and will split `#s("id")` across three
    lines, which a regex written against the one-line form leaves in the output
    verbatim. That is the same class of bug tests/fixture.typ exists to catch for
    the word counter and the narrator.
    """
    text = re.sub(r'#s\(\s*"([^"]+)"\s*,?\s*\)', lambda m: VALUES[m.group(1)], text)
    text = re.sub(r'#lit\(\s*"([^"]*)"\s*,?\s*\)', r"\1", text)
    text = re.sub(r'#link\(\s*"([^"]+)"\s*,?\s*\)', r"<\1>", text)
    text = re.sub(r"@(fig|tbl|sec):([A-Za-z0-9_-]+)", r"\\autoref{\1:\2}", text)
    # Citations. The key charset deliberately excludes `.`, so a citation ending
    # a sentence does not swallow the full stop into the key.
    text = re.sub(r"(?<![\w`])@([A-Za-z][A-Za-z0-9_:-]*)", r"[@\1]", text)
    text = re.sub(r"\]\s*\[@", "; @", text)
    text = re.sub(r"\*([^*\n]+)\*", r"**\1**", text)      # Typst strong
    text = re.sub(r"(?<![\w\\])_([^_\n]+)_(?![\w])", r"*\1*", text)  # Typst emphasis
    return text


def body_markdown() -> str:
    src = PAPER.read_text()
    a, b = BODY_START.search(src), BODY_END.search(src)
    if not a or not b:
        raise SystemExit(
            "paper.typ is missing the BODY START / BODY END markers; there is no "
            "way to tell the prose from the front and back matter.")
    body = src[a.end():b.start()]

    body = _figures(body, _assets())

    out: list[str] = []
    # Fenced code blocks pass through untouched, INCLUDING their line breaks:
    # they are the one place Typst markup characters are literal text, and the
    # one place the paragraph reflow below would destroy meaning rather than
    # whitespace.
    for i, chunk in enumerate(re.split(r"(```.*?```)", body, flags=re.S)):
        if i % 2:
            out.append(chunk.strip())
            continue
        chunk = re.sub(r"(?m)^(=+) ", lambda m: "#" * len(m.group(1)) + " ", chunk)
        chunk = re.sub(r"(?m)\s*<(?:sec|fig|tbl):[^>]+>\s*$", "", chunk)
        # Typst wraps prose at 80 columns and treats a single newline as a
        # space. Markdown mostly does too, but a wrapped line that happens to
        # begin with a numeral or a bracket reads as a list to some parsers, so
        # each paragraph is reflowed onto one line.
        paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", inline(chunk))]
        out.append("\n\n".join(p for p in paras if p))
    return "\n\n".join(c for c in out if c.strip())


def render() -> tuple[str, str]:
    md = front_matter() + "\n\n" + body_markdown() + "\n"
    return md, BIB_IN.read_text()


def figure_pairs() -> list[tuple[Path, Path]]:
    """(source, destination) for every figure paper.md references.

    Populated by body_markdown() as it rewrites each #figure(), so a figure that
    stops being referenced stops being copied instead of lingering in the bundle.
    """
    return [(ROOT / rel, OUT_DIR / rel) for rel in sorted(FIGURES)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed paper.md/paper.bib differ from "
                         "what the sources produce now")
    args = ap.parse_args()

    md, bib = render()
    figures = figure_pairs()

    if args.check:
        stale = [name for name, path, want in
                 (("joss/paper.md", MD_OUT, md), ("joss/paper.bib", BIB_OUT, bib))
                 if not path.is_file() or path.read_text() != want]
        stale += [str(dst.relative_to(ROOT)) for src, dst in figures
                  if not dst.is_file() or dst.read_bytes() != src.read_bytes()]
        # A figure left in the bundle that paper.md no longer references would
        # ship a file nothing explains, so an extra is as stale as a missing one.
        wanted = {dst for _, dst in figures}
        stale += [str(p.relative_to(ROOT))
                  for p in sorted((OUT_DIR / "figures").glob("*"))
                  if p.is_file() and p not in wanted]
        if stale:
            for name in stale:
                print(f"STALE:    {name} does not match the manuscript -- "
                      f"regenerate: just joss")
            return 1
        print("the JOSS bundle in joss/ matches the manuscript")
        return 0

    OUT_DIR.mkdir(exist_ok=True)
    MD_OUT.write_text(md)
    BIB_OUT.write_text(bib)
    for src, dst in figures:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    words = len(body_markdown().split())
    print(f"wrote joss/paper.md ({words} words), joss/paper.bib and "
          f"{len(figures)} figure(s) for JOSS submission")
    return 0


VALUES = _values() if (ROOT / "stats-rendered.json").is_file() else {}

# Filled in by _figures() during body_markdown(); read by figure_pairs().
FIGURES: set[str] = set()

if __name__ == "__main__":
    raise SystemExit(main())

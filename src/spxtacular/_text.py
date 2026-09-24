"""
Backend-neutral rich text for figure labels.

A :class:`RichText` is a tuple of runs, each run a piece of text plus a
typographic role: normal, subscript, superscript, italic or bold. Both plotting
backends render the same runs -- plotly as ``<sub>``/``<sup>`` HTML, matplotlib
as mathtext -- so an ion label reads ``y₇²⁺`` in either one, and the label
placer can measure it without a renderer.

Measuring text without a renderer
---------------------------------
Collision avoidance needs every label's size *before* anything is drawn, and it
has to give the same answer for both backends. :func:`text_width` uses the
Helvetica advance widths (the metrics Arial and Liberation Sans share), with
sub- and superscripts at 70 % size. It over-estimates slightly for narrower
faces, which errs on the side of a little extra space rather than an overlap.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Literal

RunRole = Literal["n", "sub", "sup", "it", "bf"]

#: Scale of a sub- or superscript relative to the surrounding text.
SCRIPT_SCALE: float = 0.7

# Helvetica advance widths in 1/1000 em, ASCII 32..126. Arial and Liberation
# Sans are metric-compatible with it, which is why it is a safe default.
_HELVETICA: tuple[int, ...] = (
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,  # sp ! " # $ % & ' ( ) * + , - . /
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556,  # 0-9
    278, 278, 584, 584, 584, 556, 1015,  # : ; < = > ? @
    667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833,  # A-M
    722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611,  # N-Z
    278, 278, 278, 469, 556, 333,  # [ \ ] ^ _ `
    556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833,  # a-m
    556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500,  # n-z
    334, 260, 334, 584,  # { | } ~
)  # fmt: skip

_EXTRA_WIDTHS: dict[str, int] = {
    "\u2212": 584,  # minus sign
    "±": 584,  # plus-minus
    "\u00d7": 584,  # multiplication sign
    "\u2013": 556,  # en dash
    "—": 1000,  # em dash
    "·": 278,  # middle dot
    "√": 549,  # square root
    "\u2009": 167,  # thin space
}

#: Bold faces run a little wider than regular.
_BOLD_FACTOR: float = 1.06


def char_width(ch: str) -> float:
    """Advance width of one character in em units."""
    code = ord(ch)
    if 32 <= code <= 126:
        return _HELVETICA[code - 32] / 1000.0
    return _EXTRA_WIDTHS.get(ch, 556) / 1000.0


@dataclass(frozen=True)
class RichText:
    """Text made of typographic runs, rendered identically by every backend."""

    runs: tuple[tuple[str, RunRole], ...]

    # -- construction -----------------------------------------------------

    @classmethod
    def plain(cls, text: str) -> RichText:
        return cls(((text, "n"),)) if text else cls(())

    @classmethod
    def coerce(cls, value: RichText | str | None) -> RichText:
        if value is None:
            return cls(())
        if isinstance(value, RichText):
            return value
        return cls.plain(str(value))

    def __add__(self, other: RichText | str) -> RichText:
        other_rt = RichText.coerce(other)
        return RichText(self.runs + other_rt.runs)

    def __bool__(self) -> bool:
        return any(text for text, _ in self.runs)

    # -- output -----------------------------------------------------------

    @property
    def text(self) -> str:
        """The characters only, with no markup."""
        return "".join(text for text, _ in self.runs)

    def html(self) -> str:
        """Plotly markup: ``<sub>``, ``<sup>``, ``<i>``, ``<b>``."""
        tags = {"sub": "sub", "sup": "sup", "it": "i", "bf": "b"}
        out: list[str] = []
        for text, role in self.runs:
            esc = html.escape(text, quote=False).replace("\n", "<br>")
            tag = tags.get(role)
            out.append(f"<{tag}>{esc}</{tag}>" if tag else esc)
        return "".join(out)

    def mathtext(self) -> str:
        """Matplotlib markup. Scripts become mathtext, everything else stays plain.

        Plain runs are escaped so a literal ``$`` in user text cannot switch
        mathtext on by accident.
        """
        out: list[str] = []
        for text, role in self.runs:
            if role == "n":
                out.append(text.replace("\\", "\\\\").replace("$", r"\$"))
                continue
            if role == "sub":
                out.append(f"$_{{{_math_escape(text)}}}$")
            elif role == "sup":
                out.append(f"$^{{{_math_escape(text)}}}$")
            elif role == "it":
                out.append(f"${_math_escape(text, 'mathit')}$")
            else:
                out.append(f"${_math_escape(text, 'mathbf')}$")
        return "".join(out)

    # -- metrics ----------------------------------------------------------

    def width(self, size_pt: float) -> float:
        """Estimated rendered width in points at font size ``size_pt``."""
        return text_width(self, size_pt)

    def height(self, size_pt: float) -> float:
        """Estimated line box height in points, including script overhang."""
        lines = max(1, self.text.count("\n") + 1)
        has_script = any(role in ("sub", "sup") for _, role in self.runs)
        return size_pt * (1.18 * lines + (0.22 if has_script else 0.0))


def _math_escape(text: str, font: str = "mathrm") -> str:
    """Escape text for a mathtext group so it renders in ``font`` (upright by default).

    ``+`` and ``-`` would otherwise get binary-operator spacing ("2 +"); braces
    make them ordinary symbols. Letters go through ``\\mathrm`` so they are not
    set in math italic.
    """
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append("\\" + font + "{" + "".join(buf) + "}")
            buf.clear()

    for ch in text:
        if ch == "+":
            flush()
            out.append("{+}")
        elif ch in ("-", "\u2212"):
            flush()
            out.append("{-}")
        elif ch == " ":
            buf.append("\\ ")
        elif ch in "{}$\\_^%#&":
            buf.append("\\" + ch if ch not in "\\^" else "")
        else:
            buf.append(ch)
    flush()
    return "".join(out)


def text_width(text: RichText | str, size_pt: float) -> float:
    """Estimated width in points; the widest line for multi-line text."""
    rich = RichText.coerce(text)
    widths = [0.0]
    for chunk, role in rich.runs:
        scale = SCRIPT_SCALE if role in ("sub", "sup") else 1.0
        factor = _BOLD_FACTOR if role == "bf" else 1.0
        for i, line in enumerate(chunk.split("\n")):
            if i:
                widths.append(0.0)
            widths[-1] += sum(char_width(c) for c in line) * size_pt * scale * factor
    return max(widths)


# ---------------------------------------------------------------------------
# mzPAF -> RichText
# ---------------------------------------------------------------------------


def _formula(formula: str) -> RichText:
    """``H2O`` -> H + sub(2) + O. Digits following a letter are subscripts."""
    runs: list[tuple[str, RunRole]] = []
    i = 0
    while i < len(formula):
        ch = formula[i]
        if ch.isdigit() and i > 0 and (formula[i - 1].isalpha() or formula[i - 1] == ")"):
            j = i
            while j < len(formula) and formula[j].isdigit():
                j += 1
            runs.append((formula[i:j], "sub"))
            i = j
            continue
        if runs and runs[-1][1] == "n":
            runs[-1] = (runs[-1][0] + ch, "n")
        else:
            runs.append((ch, "n"))
        i += 1
    return RichText(tuple(runs))


def _charge_sup(charge: int) -> RichText:
    if charge in (0, 1):
        return RichText(())
    if charge == -1:
        return RichText((("\u2212", "sup"),))
    sign = "+" if charge > 0 else "\u2212"
    return RichText(((f"{abs(charge)}{sign}", "sup"),))


def _loss_text(loss: object) -> RichText:
    """One neutral loss or gain: ``-H2O`` -> −H₂O, ``+NH3`` -> +NH₃, masses as numbers."""
    count = int(getattr(loss, "count", -1) or -1)
    sign = "\u2212" if count < 0 else "+"
    n = abs(count)
    prefix = f"{sign}{n if n > 1 else ''}"
    formula = getattr(loss, "base_formula", None)
    if formula:
        return RichText.plain(prefix) + _formula(str(formula))
    reference = getattr(loss, "base_reference", None)
    if reference:
        return RichText.plain(f"{prefix}[{reference}]")
    mass = getattr(loss, "base_mass", None)
    if mass is not None:
        return RichText.plain(f"{prefix}{float(mass):.2f}")
    return RichText.plain(prefix + "?")


#: Display weight of an annotation relative to a plain singly-charged backbone ion.
#: Used to rank labels when there is not room for all of them.
_PRIORITY_CHARGED = 0.7
_PRIORITY_LOSS = 0.45
_PRIORITY_ISOTOPE = 0.35
_PRIORITY_OTHER = 0.4


@dataclass(frozen=True)
class IonLabel:
    """A parsed ion label ready to draw."""

    rich: RichText
    series: str  #: ``"b"``, ``"y"``, ``"p"``, ``"i"``, ``"m"``, ``"?"`` ...
    priority: float  #: 1.0 for a plain 1+ backbone ion, less for derived ions
    raw: str  #: the source string (mzPAF)


def mzpaf_to_label(text: str) -> IonLabel:
    """Turn one mzPAF annotation into display text and a priority.

    ``b3`` -> b₃, ``y7^2`` -> y₇²⁺, ``y5-H2O`` -> y₅−H₂O, ``p^3`` -> [M+3H]³⁺,
    ``IC`` -> Im(C). Anything the parser rejects is shown as written.
    """
    raw = str(text).strip()
    try:
        import paftacular as pft

        ann = pft.parse(raw)
    except Exception:
        return IonLabel(RichText.plain(raw), "?", _PRIORITY_OTHER, raw)

    ion = ann.ion_type
    kind = type(ion).__name__
    charge = int(getattr(ann, "charge", 1) or 1)
    losses = list(getattr(ann, "neutral_losses", None) or ())
    isotopes = list(getattr(ann, "isotopes", None) or ())
    priority = 1.0

    if kind == "PeptideIon":
        series = str(getattr(ion, "series", "?"))
        position = getattr(ion, "position", None)
        rich = RichText.plain(series)
        if position is not None:
            rich = rich + RichText(((str(position), "sub"),))
        for loss in losses:
            rich = rich + _loss_text(loss)
        rich = rich + _charge_sup(charge)
    elif kind == "PrecursorIon":
        series = "p"
        sign = "+" if charge > 0 else "\u2212"
        z = abs(charge)
        inner = RichText.plain(f"[M{sign}{z if z > 1 else ''}H")
        for loss in losses:
            inner = inner + _loss_text(loss)
        rich = inner + "]" + RichText((((f"{z}" if z > 1 else "") + sign, "sup"),))
        priority = 0.5
    elif kind == "ImmoniumIon":
        series = "i"
        aa = str(getattr(ion, "amino_acid", "?"))
        mod = getattr(ion, "modification", None)
        rich = RichText.plain(f"Im({aa}{'*' if mod else ''})")
        for loss in losses:
            rich = rich + _loss_text(loss)
        priority = 0.8
    elif kind == "InternalFragment":
        series = "m"
        seq = getattr(ion, "sequence", None)
        start = getattr(ion, "start_position", None)
        end = getattr(ion, "end_position", None)
        rich = RichText.plain(f"int {seq}") if seq else RichText.plain(f"m{start}:{end}")
        for loss in losses:
            rich = rich + _loss_text(loss)
        rich = rich + _charge_sup(charge)
        priority = _PRIORITY_OTHER
    else:
        return IonLabel(RichText.plain(raw), "?", _PRIORITY_OTHER, raw)

    if charge > 1 and kind != "PrecursorIon":
        priority *= _PRIORITY_CHARGED
    if losses:
        priority *= _PRIORITY_LOSS
    if isotopes:
        total = sum(int(getattr(iso, "count", 1) or 1) for iso in isotopes)
        rich = rich + RichText.plain(f"+{total if abs(total) > 1 else ''}i")
        priority *= _PRIORITY_ISOTOPE
    return IonLabel(rich, series, priority, raw)


def best_label(text: str) -> IonLabel:
    """Choose the annotation to draw for a peak that matched several ions.

    A peak can carry ``"y7^2<br>b3-H2O"``. Drawing both stacks two labels on one
    stick; the one with the highest display priority is drawn and the rest stay
    in the hover and the plot table.
    """
    parts = [p for p in str(text).replace("<br>", ",").split(",") if p.strip()]
    if not parts:
        return IonLabel(RichText(()), "?", 0.0, "")
    labels = [mzpaf_to_label(p) for p in parts]
    return max(labels, key=lambda lab: lab.priority)

"""Graphical abstract for spxtacular, drawn from stages.json (real pipeline output)."""
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).parent
SANS, MONO = "Inter", "JetBrains Mono"

# The approved figure is set in Inter and JetBrains Mono. matplotlib falls back to
# DejaVu Sans without a word when a family is missing, which would silently produce a
# figure that is not the approved one, so a missing face is an error here. `just
# paper-fonts` fetches both from their pinned releases into fonts/.
FONT_SOURCES = (
    (SANS, HERE / "fonts/i/extras/ttf", "Inter-*.ttf"),
    (MONO, HERE / "fonts/j/fonts/ttf", "JetBrainsMono-*.ttf"),
)
for family, directory, pattern in FONT_SOURCES:
    faces = [f for f in directory.glob(pattern) if "Italic" not in f.name and "NL" not in f.name]
    if not faces:
        raise SystemExit(f"no {family} faces in {directory}: run `just paper-fonts` first")
    for f in faces:
        fm.fontManager.addfont(str(f))
for family, _, _ in FONT_SOURCES:
    try:
        fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
    except ValueError as exc:
        raise SystemExit(f"matplotlib cannot resolve the font family {family!r}: {exc}") from exc
plt.rcParams.update({"font.family": SANS, "svg.fonttype": "none", "pdf.fonttype": 42})

D = json.load(open(HERE / "data/stages.json"))
INK, SUB, FAINT, HAIR = "#111827", "#6B7280", "#C7CCD4", "#E5E7EB"
BLUE, RED = "#0061FF", "#DC1D1D"
# Charge is ordinal, so spxtacular's own palette encodes it as one hue's lightness rather
# than as separate colours (theme.py, charge_color). Taking it from the library keeps the
# abstract from inventing a scheme the package does not use.
from spxtacular.theme import charge_color  # noqa: E402
Z1, Z2 = charge_color(1), charge_color(2)
PROTON = 1.007276

# Band A is drawn on a square-root intensity scale, the usual one for spectral display and
# the transform cosine() itself applies. The peaks denoise() removes are the small ones,
# 2.2e-3 to 1.2e-2 of the base peak against 1.2e-2 to 1 for the peaks it keeps, so on a
# linear axis the removal is invisible and panel 2 looks like panel 1. A log scale makes
# them visible but lifts every small peak to a similar height, which stops the panels
# looking like spectra at all.

def logh(rel):
    """Stick height for a relative intensity, on a square-root scale."""
    return np.sqrt(np.clip(np.asarray(rel, dtype=np.float64), 0, 1))

W, H = 12.0, 7.9
fig = plt.figure(figsize=(W, H), dpi=300)
fig.patch.set_facecolor("white")
R = fig.canvas.get_renderer()

def T(x, y, s, size=10, color=INK, weight="regular", family=SANS, ha="left", va="baseline", **k):
    return fig.text(x / W, y / H, s, fontsize=size, color=color, fontweight=weight, fontfamily=family, ha=ha, va=va, **k)

def width_in(t):
    return t.get_window_extent(R).width / fig.dpi

def AX(x, y, w, h):
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK); ax.spines["bottom"].set_linewidth(0.8)
    ax.set_yticks([]); ax.patch.set_alpha(0)
    ax.tick_params(axis="x", colors=SUB, labelsize=9, length=2.5, width=0.7, pad=2)
    for l in ax.get_xticklabels(): l.set_fontfamily(SANS)
    return ax

def sticks(ax, x, y, color, lw=0.8, z=2):
    ax.vlines(x, 0, y, colors=color, linewidths=lw, zorder=z)

def line(x0, y0, x1, y1, color=HAIR, lw=0.8):
    fig.add_artist(Line2D([x0 / W, x1 / W], [y0 / H, y1 / H], color=color, lw=lw))

def chip(x, y, n):
    r = 0.105
    fig.add_artist(Circle((x / W, y / H), r / W, transform=fig.transFigure, fc=INK, ec="none"))
    # circle in figure coords is an ellipse unless corrected: draw with a text bbox instead
def header(x, y, n, call, caption, maxw=None):
    num = T(x, y, str(n), size=8.5, color="white", weight="bold", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.32", fc=INK, ec="none"))
    num.set_position(((x + 0.11) / W, (y + 0.045) / H))
    call_t = T(x + 0.32, y, call, size=12.6, color=BLUE, weight="bold", family=MONO)
    if maxw is not None and width_in(call_t) > maxw - 0.32:
        raise SystemExit(f"call {call!r} is {width_in(call_t):.2f} in wide, panel is {maxw - 0.32:.2f} in")
    cap = T(x, y - 0.235, caption, size=10.0, color=SUB)
    # A caption wider than its panel runs into the next panel's caption, which reads as one
    # sentence rather than as a drawing bug. Fail here instead.
    if maxw is not None and width_in(cap) > maxw:
        raise SystemExit(f"caption {caption!r} is {width_in(cap):.2f} in wide, panel is {maxw:.2f} in")

def arrow(x0, x1, y):
    fig.add_artist(FancyArrowPatch((x0 / W, y / H), (x1 / W, y / H), transform=fig.transFigure,
                                   arrowstyle="-|>,head_length=3.2,head_width=1.8", lw=0.9, color="#9CA3AF", mutation_scale=1.6))

# ------------------------------------------------------------------ header
M = 0.45
t1 = T(M, 7.16, "spx", size=31, color=BLUE, weight="extra bold")
T(M + width_in(t1), 7.16, "tacular", size=31, color=RED, weight="extra bold")
T(M, 6.82, "One chainable Spectrum object, from the peaks an instrument records to a scored peptide-spectrum match.",
  size=12, color="#374151")

# ------------------------------------------------------------------ band A: four stages
raw, den, dec, neu = D["raw"], D["denoised"], D["deconvoluted"], D["neutral"]
rmz, rin, kept = np.array(raw["mz"]), np.array(raw["intensity"]), np.array(raw["kept"])
dmz, din = np.array(den["mz"]), np.array(den["intensity"])
cmz, cin, cz = np.array(dec["mz"]), np.array(dec["intensity"]), np.array(dec["charge"])
n_charged = int((cz > 0).sum())

GAP = 0.36
wS = w3 = (W - 2 * M - 3 * GAP) / 4
x1 = M; x2 = x1 + wS + GAP; x3 = x2 + wS + GAP; x4 = x3 + w3 + GAP
YH, Y0, HP = 6.30, 4.66, 1.22
XL = (90, 1460)

header(x1, YH, 1, "fetch_usi(usi)", f"{len(rmz)} peaks, as acquired", wS)
ax = AX(x1, Y0, wS, HP); sticks(ax, rmz, logh(rin / rin.max()), INK, 0.55)
ax.set_ylabel("\u221a intensity", fontsize=9.0, color=SUB, labelpad=2)
ax.set_xlim(*XL); ax.set_ylim(0, 1.05); ax.set_xticks([400, 800, 1200])

header(x2, YH, 2, ".denoise()", f"{len(dmz)} peaks above the noise floor", wS)
ax = AX(x2, Y0, wS, HP)
sticks(ax, dmz, logh(din / rin.max()), INK, 0.65, 2)
sticks(ax, rmz[~kept], logh(rin[~kept] / rin.max()), "#EF6B6B", 0.8, 3)
ax.set_xlim(*XL); ax.set_ylim(0, 1.05); ax.set_xticks([400, 800, 1200])

header(x3, YH, 3, ".deconvolute()", f"{n_charged} clusters given a charge state", w3)
ax = AX(x3, Y0, w3, HP)
u = cz < 0
sticks(ax, cmz[u], logh(cin[u] / cin.max()), FAINT, 0.55, 1)
sticks(ax, cmz[cz == 1], logh(cin[cz == 1] / cin.max()), Z1, 1.0, 2)
sticks(ax, cmz[cz == 2], logh(cin[cz == 2] / cin.max()), Z2, 1.3, 3)
# The charge key. Deconvolution's whole output is the charge state, so the panel has to say
# which colour is which: the ordinal ramp puts 1+ and 2+ one lightness step apart on purpose,
# which is legible beside a label and not on its own.
for row, (zc, col) in enumerate(((1, Z1), (2, Z2))):
    ax.text(0.99, 0.97 - row * 0.17, f"{int((cz == zc).sum())} at {zc}+", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.6, color=col, fontweight="semibold")
ax.text(0.99, 0.97 - 2 * 0.17, f"{int((cz < 0).sum())} unassigned", transform=ax.transAxes,
        ha="right", va="top", fontsize=8.6, color=SUB)
ax.set_xlim(*XL); ax.set_ylim(0, 1.05); ax.set_xticks([400, 800, 1200])

header(x4, YH, 4, ".decharge().merge()", f"{len(neu['mass'])} neutral masses", wS)
ax = AX(x4, Y0, wS, HP)
# Drawn from the pipeline's own neutral array, not recomputed from the charged peaks, so
# the panel shows the merge: an ion seen at 1+ and at 2+ gives one neutral mass, not two.
nmass, nint = np.array(neu["mass"]), np.array(neu["intensity"])
sticks(ax, nmass, logh(nint / nint.max()), INK, 1.0)
ax.set_xlim(90, 1460); ax.set_ylim(0, 1.05); ax.set_xticks([400, 800, 1200])

ya = Y0 + HP * 0.45
arrow(x1 + wS + 0.07, x2 - 0.07, ya); arrow(x2 + wS + 0.07, x3 - 0.07, ya); arrow(x3 + w3 + 0.07, x4 - 0.07, ya)

# ------------------------------------------------------------------ band B: hero
line(M, 4.18, W - M, 4.18)
ms = D["matches"]; sc = D["scores"]; pep = D["peptide"]; n = len(pep)
wHero = 7.05
header(M, 3.78, 5, ".score(fragments)", f"{len(ms)} of {D['n_fragments']} theoretical b and y fragment ions found, within {D['tolerance_ppm']:.0f} ppm")
nm, ni = np.array(neu["mass"]), np.array(neu["intensity"])
XH = (200, 1380)
axh = AX(M, 1.34, wHero, 1.95)
matched_mass = {round(m["mass"], 4) for m in ms}
um = np.array([round(v, 4) not in matched_mass for v in nm])
sticks(axh, nm[um], ni[um] / ni.max(), FAINT, 0.9, 1)
SUBS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
for m in ms:
    col = BLUE if m["ion"] == "b" else RED
    hgt = m["intensity"] / ni.max()
    axh.vlines(m["mass"], 0, hgt, colors=col, linewidths=1.9, zorder=3)
    axh.text(m["mass"], hgt + 0.035, f'{m["ion"]}{str(m["pos"]).translate(SUBS)}', ha="center", va="bottom", fontsize=9.2, color=col, fontweight="semibold")
axh.set_xlim(*XH); axh.set_ylim(0, 1.22); axh.set_xticks([]); axh.spines["bottom"].set_color(FAINT)
# mass error strip
axe = AX(M, 0.86, wHero, 0.40)
axe.axhline(0, color=FAINT, lw=0.7, zorder=0)
for m in ms:
    axe.plot(m["mass"], m["ppm"], "o", ms=3.4, color=BLUE if m["ion"] == "b" else RED, mec="none")
axe.set_xlim(*XH); axe.set_ylim(-8, 8); axe.set_xticks([400, 600, 800, 1000, 1200])
axe.set_yticks([-5, 0, 5]); axe.tick_params(axis="y", colors=SUB, labelsize=8, length=2, width=0.6, pad=1.5)
axe.spines["left"].set_visible(True); axe.spines["left"].set_color(FAINT); axe.spines["left"].set_linewidth(0.7)
axe.set_xlabel("neutral mass (Da)", fontsize=9.5, color=SUB, labelpad=1)
axe.set_ylabel("Δ mass (ppm)", fontsize=9.0, color=SUB, labelpad=2)

# sequence ladder, centred over the hero plot
bset = {m["pos"] for m in ms if m["ion"] == "b"}; yset = {m["pos"] for m in ms if m["ion"] == "y"}
step = 0.33; xs0 = M + wHero - 0.20 - step * (n - 1); yL = 3.06
for i, aa in enumerate(pep):
    T(xs0 + i * step, yL, aa, size=14.5, color=INK, weight="semibold", family=MONO, ha="center", va="center")
for i in range(1, n):  # cleavage after residue i
    xc = xs0 + (i - 0.5) * step
    if i in bset:
        line(xc, yL + 0.02, xc, yL + 0.19, BLUE, 1.3); line(xc, yL + 0.19, xc - 0.11, yL + 0.19, BLUE, 1.3)
    if (n - i) in yset:
        line(xc, yL - 0.02, xc, yL - 0.19, RED, 1.3); line(xc, yL - 0.19, xc + 0.11, yL - 0.19, RED, 1.3)
T(xs0 - 0.42, yL + 0.14, "b", size=9.5, color=BLUE, weight="bold", ha="right", va="center")
T(xs0 - 0.42, yL - 0.14, "y", size=9.5, color=RED, weight="bold", ha="right", va="center")

# ------------------------------------------------------------------ band B right: numbers and code
xr = M + wHero + 0.55; wr = W - M - xr
stats = [(f"{len(ms)}/{D['n_fragments']}", "fragment ions"), (f"{sc['hyperscore']:.1f}", "hyperscore"), (f"{sc['mean_ppm_error']:.1f} ppm", "mean mass error")]
for k, (big, small) in enumerate(stats):
    xk = xr + k * wr / 3
    T(xk, 3.50, big, size=19, color=INK, weight="bold")
    T(xk, 3.27, small, size=8.4, color=SUB)

code = [
    [("import", SUB), (" spxtacular ", INK), ("as", SUB), (" spx, peptacular ", INK), ("as", SUB), (" pt", INK)],
    [],
    [("fragments = pt.", INK), ("fragment", BLUE), ('("VLHPLEGAVVIIFK", charges=0)', INK)],
    [('ppm = dict(tolerance=20, tolerance_unit="ppm")', INK)],
    [],
    [("scores = (", INK)],
    [("    spx.", INK), ("fetch_usi", BLUE), ("(usi)", INK)],
    [("    .", INK), ("denoise", BLUE), ('(method="iterative_median")', INK)],
    [("    .", INK), ("deconvolute", BLUE), ("(**ppm)", INK)],
    [("    .", INK), ("decharge", BLUE), ("()", INK)],
    [("    .", INK), ("merge", BLUE), ('(mz_tolerance=20, mz_tolerance_unit="ppm")', INK)],
    [("    .", INK), ("score", BLUE), ("(fragments, **ppm)", INK)],
    [(")", INK)],
]
cs = 7.7; lh = 0.172; ctop = 2.78; pad = 0.16
probe = T(0, 0, "M" * 10, size=cs, family=MONO); cw = width_in(probe) / 10; probe.remove()
fig.add_artist(FancyBboxPatch(((xr - pad) / W, (ctop - lh * (len(code) - 1) - 0.14) / H), (W - M - xr + pad) / W, (lh * (len(code) - 1) + 0.40) / H,
                              boxstyle="round,pad=0,rounding_size=0.006", transform=fig.transFigure, fc="#F5F6F8", ec="none"))
for r, row in enumerate(code):
    col = 0
    for txt, c in row:
        T(xr + col * cw, ctop - r * lh, txt, size=cs, color=c, family=MONO, weight="medium" if c == BLUE else "regular")
        col += len(txt)

# ---------------------------------------------------- footer: where the data came from
line(M, 0.42, W - M, 0.42)
prov = [("REAL DATA, PUBLIC SPECTRUM", 8.0, "bold", SANS, INK),
        (D["usi"], 8.6, "regular", MONO, SUB),
        (f'{D["peptide"]}, precursor charge 2', 8.6, "regular", SANS, SUB)]
SEP, PAD = "·", 0.10
widths = []
for s, size, weight, family, _ in prov:
    probe = T(0, -1, s, size=size, weight=weight, family=family)
    widths.append(width_in(probe)); probe.remove()
probe = T(0, -1, SEP, size=7.8); w_sep = width_in(probe); probe.remove()
x = (W - (sum(widths) + 2 * (w_sep + 2 * PAD))) / 2
for i, (s, size, weight, family, color) in enumerate(prov):
    T(x, 0.20, s, size=size, color=color, weight=weight, family=family)
    x += widths[i]
    if i < len(prov) - 1:
        T(x + PAD, 0.20, SEP, size=7.8, color=SUB)
        x += w_sep + 2 * PAD

out = sys.argv[1]
fig.savefig(out + ".png", dpi=300, facecolor="white")
fig.savefig(out + ".pdf", facecolor="white")
print("saved", out)

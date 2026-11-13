"""Run the real spxtacular pipeline on a public spectrum and cache every stage for the figure."""
import json, pathlib, sys
import numpy as np
import spxtacular as spx
import peptacular as pt

HERE = pathlib.Path(__file__).parent
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
USI = "mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555"
PEP = "VLHPLEGAVVIIFK"
TOL = 20.0

raw = spx.fetch_usi(USI)
den = raw.denoise(method="iterative_median")
dec = den.deconvolute(charge_range=(1, 3), tolerance=TOL, tolerance_unit="ppm")
neu = dec.decharge().merge(mz_tolerance=TOL, mz_tolerance_unit="ppm")
frags = pt.fragment(PEP, ion_types=("b", "y"), charges=0)
matches = spx.match_fragments(neu, frags, tolerance=TOL, tolerance_unit="ppm")
has_method = hasattr(neu, "score")
scores = neu.score(frags, tolerance=TOL, tolerance_unit="ppm") if has_method else spx.score(neu, frags, tolerance=TOL, tolerance_unit="ppm")
print("Spectrum.score method:", has_method, "| match method:", hasattr(neu, "match_fragments"), hasattr(neu, "match"))

def arr(a): return [float(x) for x in np.asarray(a)]
kept = np.isin(np.round(raw.mz, 6), np.round(den.mz, 6))
ch = np.asarray(dec.charge)

# isotope envelopes in the denoised spectrum behind deconvoluted peaks, for the zoom lens
lens = []
for z in (2, 1):
    idx = np.where(ch == z)[0]
    best = None
    for i in idx[np.argsort(-np.asarray(dec.intensity)[idx])]:
        m0 = float(dec.mz[i])
        peaks = []
        for k in range(6):
            t = m0 + k * 1.003355 / z
            j = int(np.argmin(np.abs(den.mz - t)))
            if abs(den.mz[j] - t) / t * 1e6 <= TOL: peaks.append((float(den.mz[j]), float(den.intensity[j])))
            else: break
        if len(peaks) >= (3 if z == 1 else 2) and (best is None or len(peaks) > len(best[1])): best = (m0, peaks)
        if best and len(best[1]) >= 4: break
    if best: lens.append({"z": z, "mono": best[0], "peaks": best[1]})
    print("lens z", z, best)

ms = []
for m in matches:
    f = m.fragment
    ms.append({"ion": str(getattr(f.ion_type, "value", f.ion_type)), "pos": int(f.position), "mass": float(m.peak_mz),
               "intensity": float(m.peak_intensity), "ppm": float(m.ppm_error)})

# reporter ions from a second public spectrum (CPTAC iTRAQ 4-plex reference)
rep_usi = "mzspec:PXD000966:CPTAC_CompRef_00_iTRAQ_05_2Feb12_Cougar_11-10-09.mzML:scan:12298"
rs = spx.fetch_usi(rep_usi)
rep = []
for name, mz in (("114", 114.1107), ("115", 115.1077), ("116", 116.1111), ("117", 117.1144)):
    j = int(np.argmin(np.abs(rs.mz - mz))); rep.append({"channel": name, "mz": float(rs.mz[j]), "intensity": float(rs.intensity[j])})
print("reporters", rep)

d = {"usi": USI, "peptide": PEP, "precursor_charge": 2, "tolerance_ppm": TOL,
     "raw": {"mz": arr(raw.mz), "intensity": arr(raw.intensity), "kept": [bool(x) for x in kept]},
     "denoised": {"mz": arr(den.mz), "intensity": arr(den.intensity)},
     "deconvoluted": {"mz": arr(dec.mz), "intensity": arr(dec.intensity), "charge": [int(x) for x in ch]},
     "neutral": {"mass": arr(neu.mz), "intensity": arr(neu.intensity)},
     "n_fragments": len(frags), "matches": ms, "scores": {k: float(v) for k, v in scores.items()},
     "lens": lens, "reporter": {"usi": rep_usi, "channels": rep}}
json.dump(d, open(OUT / "stages.json", "w"))
print("raw", len(raw.mz), "kept", int(kept.sum()), "den", len(den.mz), "dec", len(dec.mz), {int(z): int((ch == z).sum()) for z in np.unique(ch)},
      "neutral", len(neu.mz), "matched", len(ms), "of", len(frags))
print({k: round(float(v), 3) for k, v in scores.items()})
print([(m["ion"], m["pos"], round(m["ppm"], 2)) for m in ms])
# does neutral keep origin charge?
print("neutral charge attr:", None if getattr(neu, "charge", None) is None else np.unique(np.asarray(neu.charge)))

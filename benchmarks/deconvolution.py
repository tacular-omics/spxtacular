"""Report accuracy, runtime, and Python-tracked peak memory on analytic fixtures.

Run from the checkout with uv run --locked python benchmarks/deconvolution.py.
Timing is reported without a machine-dependent CI threshold.
"""

import json
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np

import spxtacular as spx


def main() -> None:
    cases = json.loads((Path(__file__).resolve().parents[1] / "tests/reference/carbon_envelopes.json").read_text())
    results = []
    for case in cases:
        spectrum = spx.Spectrum(
            mz=np.asarray(case["mz"]),
            intensity=np.asarray(case["intensity"]),
            im=np.asarray(case["im"]) if "im" in case else None,
        )
        model = spx.IsotopeModel(atoms_per_da={}, fixed_composition={"C": case["carbon_count"]})

        def run(spectrum=spectrum, model=model, case=case):
            return spectrum.deconvolute(
                isotope_model=model,
                ionization_model=case["ionization_model"],
                charge_range=(1, 6),
                tolerance=5.0,
                min_intensity=case["min_intensity"],
                min_score=0.4,
            )

        start = time.perf_counter()
        output = run()
        first_call = time.perf_counter() - start
        timings = []
        for _ in range(10):
            start = time.perf_counter()
            run()
            timings.append(time.perf_counter() - start)
        tracemalloc.start()
        run()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert output.charge is not None
        assigned = output.charge > 0
        np.testing.assert_array_equal(output.charge[assigned], case["expected_charges"])
        masses = output.decharge().mz
        np.testing.assert_allclose(masses, case["expected_masses"], atol=1e-4, rtol=0)
        assert np.count_nonzero(output.charge == -1) == case["expected_singletons"]
        np.testing.assert_allclose(output.intensity.sum(), spectrum.intensity.sum(), rtol=1e-12)
        results.append(
            {
                "name": case["name"],
                "first_call_seconds": first_call,
                "median_warm_seconds": statistics.median(timings),
                "python_tracked_peak_bytes": peak,
                "assigned_clusters": int(assigned.sum()),
                "max_mass_error_da": float(np.max(np.abs(masses - case["expected_masses"]))),
                "singleton_count": int(np.count_nonzero(output.charge == -1)),
            }
        )
    print(json.dumps({"python": platform.python_version(), "spxtacular": spx.__version__, "cases": results}, indent=2))


if __name__ == "__main__":
    main()

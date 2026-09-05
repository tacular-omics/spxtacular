# Dependency review for spxtacular 0.7.0

Checked September 4, 2026 against the official PyPI JSON metadata for every direct and
transitive dependency. Prereleases, development releases, and yanked releases were excluded.
The release lock contains 95 dependency packages. All 21 declared runtime, optional, and
development dependencies resolve to their latest stable releases. Of the transitive packages,
NetworkX and SciPy remain at their newest compatible versions because of upstream constraints.
The refresh upgrades 37 packages and adds cloudpickle as a transitive dependency.
Rechecked newly published peptacular 3.3.0 and tdfpy 4.0.1 during release preparation.

## Direct dependencies

These versions were installed for validation. The build backend is resolved separately from
the project lock. Its latest available version is listed as well.

| Package | Previous lock | Release lock or build release | PyPI |
| --- | --- | --- | --- |
| fisher-py | 2.0.2 | 2.0.2 | [Release](https://pypi.org/project/fisher-py/2.0.2/) |
| hatchling | Separate build environment | 1.32.0 | [Release](https://pypi.org/project/hatchling/1.32.0/) |
| jsonschema | 4.26.0 | 4.26.0 | [Release](https://pypi.org/project/jsonschema/4.26.0/) |
| matchms | 0.33.1 | 0.33.1 | [Release](https://pypi.org/project/matchms/0.33.1/) |
| mkdocs | 1.6.1 | 1.6.1 | [Release](https://pypi.org/project/mkdocs/1.6.1/) |
| mkdocs-material | 9.7.6 | 9.7.7 | [Release](https://pypi.org/project/mkdocs-material/9.7.7/) |
| mzmlpy | 0.7.0 | 0.9.0 | [Release](https://pypi.org/project/mzmlpy/0.9.0/) |
| numba | 0.65.0 | 0.67.0 | [Release](https://pypi.org/project/numba/0.67.0/) |
| numpy | 2.4.4 | 2.5.2 | [Release](https://pypi.org/project/numpy/2.5.2/) |
| paftacular | 1.1.0 | 1.2.0 | [Release](https://pypi.org/project/paftacular/1.2.0/) |
| pandas | 3.0.2 | 3.0.5 | [Release](https://pypi.org/project/pandas/3.0.5/) |
| peptacular | 3.2.0 | 3.3.0 | [Release](https://pypi.org/project/peptacular/3.3.0/) |
| plotly | 6.6.0 | 7.0.0 | [Release](https://pypi.org/project/plotly/7.0.0/) |
| pytest | 9.0.3 | 9.1.1 | [Release](https://pypi.org/project/pytest/9.1.1/) |
| pytest-cov | 7.1.0 | 7.1.0 | [Release](https://pypi.org/project/pytest-cov/7.1.0/) |
| pytest-timeout | 2.4.0 | 2.4.0 | [Release](https://pypi.org/project/pytest-timeout/2.4.0/) |
| pytest-xdist | 3.8.0 | 3.8.0 | [Release](https://pypi.org/project/pytest-xdist/3.8.0/) |
| ruff | 0.15.9 | 0.16.6 | [Release](https://pypi.org/project/ruff/0.16.6/) |
| spectrl | 1.0.0 | 1.1.0 | [Release](https://pypi.org/project/spectrl/1.1.0/) |
| spectrum-utils | 0.5.0 | 0.5.0 | [Release](https://pypi.org/project/spectrum-utils/0.5.0/) |
| tdfpy | 3.0.0 | 4.0.1 | [Release](https://pypi.org/project/tdfpy/4.0.1/) |
| ty | 0.0.29 | 0.0.78 | [Release](https://pypi.org/project/ty/0.0.78/) |

The peptacular minimum is now 3.3.0, tdfpy is 4.0.1, paftacular is 1.2.0,
mzMLPy is 0.9.0, and spectrl is 1.1.0.
Other existing supported minimums remain unchanged. The lock records the newer versions
used for development and release checks.

## Constraints that prevent the absolute latest versions

| Package | Locked | Latest stable | Reason |
| --- | --- | --- | --- |
| NetworkX | 3.4.2 | 3.6.1 | matchms 0.33.1 requires networkx >=3.4.2,<3.5 |
| SciPy | 1.16.3 | 1.18.1 | matchms 0.33.1 and sparsestack 0.7.1 require scipy <1.17 |

These constraints come from the published [matchms metadata](https://pypi.org/pypi/matchms/0.33.1/json)
and [sparsestack metadata](https://pypi.org/pypi/sparsestack/0.7.1/json). Both packages are current.
Forcing newer NetworkX or SciPy versions would violate their declared compatibility.

## Compatibility validation

- Full suite on Linux with Python 3.12.3, 3.13.9, and 3.14.0: 1,141 passed on each, zero skips.
- All optional extras installed, including real Bruker and Thermo reader integrations. Thermo used .NET 8.0.30.
- Installed dependency metadata checks passed in all three environments.
- Current Ruff lint, formatting, and ty checks passed. Type checking also passed on Python 3.12 and 3.14.
- Strict documentation and distribution builds passed. GitHub workflow validation passed.
- Isolated wheel tests outside the checkout passed on all three Python versions: 459 passed, eight expected optional-backend skips per run.
- A separate Python 3.12 wheel check with the lowest installable direct dependencies also passed 459 tests with the same eight expected skips.

mzMLPy 0.9 preserves the native numeric dtype of decoded arrays. Four new integration cases
verify that float32, float64, int32, and int64 inputs become independent float64 arrays at
the spxtacular reader boundary. Three additional cases verify tdfpy fractional-scan mobility
transport and peptacular position-free precursor and neutral fragment scoring. Bruker precursor
mobility can change because tdfpy no longer truncates the recorded scan coordinate.

Strict resource-warning checks also exposed an HTTP-error response body that was not closed
by the USI fetcher. The error path now closes it, a regression test covers cleanup, and the
broad resource-warning suppression was removed. The current type checker also required explicit narrowing
of optional reader modules and chromatogram sequences, plus a more precise test helper.

The CI full-suite matrix now covers Python 3.12 through 3.14. Windows and macOS wheel jobs
are configured but were not executed locally. This review establishes the behavior covered
by the tests and package metadata, not every possible scientific input or optional platform.

Reproduce the locked environment with `uv sync --locked --all-extras`. Run the checks in
`CONTRIBUTING.md`, and use the dedicated Thermo job to require its runtime instead of skipping.

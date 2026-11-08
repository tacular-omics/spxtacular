# Contributing to spxtacular

Bug reports, feature proposals, documentation fixes, and code contributions are welcome. Use the
[GitHub issue tracker](https://github.com/tacular-omics/spxtacular/issues) for bugs, questions, and
support. Please include a minimal example, the spxtacular and Python versions, and the complete
error message when reporting a defect.

## Development setup

`pip install spxtacular` is the install for using the library. Developing it requires Python 3.12
or newer and uses [uv](https://docs.astral.sh/uv/) for the environment and
[just](https://github.com/casey/just) as the task runner, so that every command runs against the
locked development environment:

```bash
git clone https://github.com/tacular-omics/spxtacular.git
cd spxtacular
just install          # uv sync --locked
```

`just install` is `uv sync --locked`, which installs the development group. That group happens to
carry most of the optional libraries already, so almost the whole suite runs from it, but not all
of them, and a test whose library is absent skips instead of running.

The extras are `bruker` (tdfpy), `mzml` (mzmlpy), `thermo` (fisher-py), `spectrl`, `interop`
(matchms, spectrum-utils), `numba`, `matplotlib`, `plotly-export` (kaleido and plotly), and `all`.
Install them with `uv sync --locked --all-extras`, and do that if your change touches figures,
static export, or an optional reader.

Read every outcome pytest prints, not the collected total alone, because the two kinds of skip
look different. A test gated inside a function or a fixture is collected and then skipped, so the
collected total does not move. A module whose `pytest.importorskip` runs at import, at the top
level of the file, is never imported at all, so its tests leave the collected total altogether and
the whole file reports as one skip. spxtacular has both shapes, so neither number answers on its
own. What separates them is a subtraction: add the passed, failed, skipped, xfailed, xpassed and
error counts, leave out any subtest count, and subtract the collected total. The difference is the
number of files skipped at collection, and it is zero when every optional library is installed.
`uv pip list` says what is actually installed, which is a measurement of the environment rather
than of the declarations, and the `SKIPPED` lines pytest prints under `-ra` name every test that
did not run and why.

Thermo is the one no sync can fix: `fisher-py` installs from the development group, but reading a
`.raw` file needs a .NET 8 runtime on the machine, so those tests skip without it. CI has a job
that installs .NET 8 and sets `SPXTACULAR_REQUIRE_THERMO=1`, where they cannot skip.

`just --list` prints every recipe, and the
[development guide](https://tacular-omics.github.io/spxtacular/development/) explains them one by
one. Without `just`, run the command inside each recipe directly (`just --show test` prints it).

Install the commit hooks once per clone with `uvx pre-commit install`. On every
commit they run `ruff check`, `ruff format --check` and a few file checks
(`uvx pre-commit run --all-files` runs them on everything). Type checks and
tests are not in the hooks; run `just check` for those.

`just test` is the fast default run: it skips tests marked `@pytest.mark.slow` (subprocess
import checks, memory checks, and the matchms/spectrum_utils interop tests, whose imports
JIT-compile numba code) and runs 30 hypothesis examples per property test. `just test-all`
runs everything the way CI does (`RUN_SLOW=1 HYPOTHESIS_PROFILE=thorough`; `--run-slow`
works too). Mark a new test `slow` only if it still takes more than about 2 s after shrinking
its input.

Before opening a pull request, run the same checks used by continuous integration:

```bash
just lint
just fmt-check
just check
just test-cov
just docs-build
```

CI also builds and tests the installed wheel outside the checkout on Linux, Windows, and macOS
with Python 3.12 through 3.14. A separate Python 3.12 job resolves the lowest direct dependencies
available as binary wheels. To run the wheel check locally:

```bash
uv build --out-dir wheel-dist
uv run --no-project python .github/scripts/check_wheel.py wheel-dist
uv run --no-project python .github/scripts/check_wheel.py wheel-dist --lowest
```

The dedicated Thermo job installs .NET 8 and sets `SPXTACULAR_REQUIRE_THERMO=1` so backend
failures cannot become successful skipped tests. Other jobs may skip optional reader tests.
Run `just benchmark` for timing, mass accuracy, assignment checks, and Python-tracked memory
on the analytical fixtures. Their provenance and limitations are in `tests/reference/README.md`.

Changes should include tests for new behavior and user-facing documentation when the public API
changes. Add a concise entry to `CHANGELOG.md` for changes that users need to know about. Keep pull
requests focused, and explain the scientific or practical motivation as well as the implementation.

Small test fixtures that can be redistributed may be committed to `tests/data`. Do not contribute
patient data, confidential data, or instrument files whose license does not permit redistribution.
For large public datasets, open an issue first so that an external archival location can be chosen.

## Project decisions and conduct

The maintainer reviews contributions and makes final decisions about scope, API compatibility, and
releases. Design discussions should take place in public issues or pull requests whenever possible.
Participants are expected to be respectful, constructive, and welcoming. Harassment, personal
attacks, and discriminatory conduct are not acceptable.

For a suspected security vulnerability, use GitHub's private security-advisory reporting for this
repository instead of opening a public issue.

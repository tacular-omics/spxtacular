# Development

`pip install spxtacular` is the install for *using* the library, and it is all a user needs.
Working *on* spxtacular uses a clone, [uv](https://docs.astral.sh/uv/) for the environment, and
[just](https://github.com/casey/just) as the task runner: every recipe wraps `uv run --locked`, so
each command runs against the locked development environment rather than whatever happens to be on
`PATH`.

## Set up a clone

```bash
git clone https://github.com/tacular-omics/spxtacular.git
cd spxtacular
just install              # uv sync --locked: the package plus the dev dependency group
uvx pre-commit install    # once per clone
```

Python 3.12 or newer is required; uv downloads a matching interpreter if the machine has none.
`just install` installs the test, lint, type-check and docs tools along with every optional backend
the tests exercise (tdfpy, mzmlpy, fisher-py, spectrl, matchms, spectrum_utils, numba), so a fresh
clone can run the whole suite. `uv sync --locked --all-extras` adds the two figure extras,
matplotlib and kaleido, which the figure tests and the gallery need.

The commit hooks run `ruff check`, `ruff format --check` and a few file checks on every commit;
`uvx pre-commit run --all-files` runs them over the whole tree. Type checks and tests are
deliberately not hooks, because they are slower than a commit should be.

## The recipes

`just --list` prints all of them. These are the ones a contributor runs:

| Command | What it does |
|---|---|
| `just install` | `uv sync --locked`: create or update `.venv` from the lock file |
| `just test` | the fast test run: slow tests skipped, 30 hypothesis examples per property |
| `just test-all` | every test the way CI runs it (`RUN_SLOW=1 HYPOTHESIS_PROFILE=thorough`) |
| `just test-cov` | `test-all` plus `coverage.xml` and `junit.xml` |
| `just lint` | `ruff check src tests .github/scripts benchmarks` |
| `just fmt-check` | `ruff format --check`: what CI enforces |
| `just format` | sort imports and format; **rewrites files** |
| `just check` | `ty check src tests`: the type check only, not a full check |
| `just docs` | build and serve the documentation on `localhost:8003` |
| `just docs-build` | `mkdocs build --strict`: fails on a broken link or anchor |
| `just benchmark` | deconvolution accuracy, timing and memory on the analytical fixtures |

Running `just` with no recipe does `lint`, `format`, `check` and `test` in order. It includes
`format`, so it rewrites files; `just fmt-check` is the read-only counterpart.

A single test file runs directly: `uv run --locked pytest tests/test_core.py -k deconvolute`.
Tests marked `@pytest.mark.slow` are skipped by default and run with `--run-slow` or `RUN_SLOW=1`.

## Before opening a pull request

```bash
just lint
just fmt-check
just check
just test-cov
just docs-build
```

That is the same set continuous integration runs. `just docs-build` matters for a documentation
change: `--strict` turns a dead cross-page link or a missing heading anchor into a failure.

Every Python example in the guides and in `README.md` is executed by `pytest tests/test_docs.py`,
page by page, in one namespace per page. A block that cannot run says so in its fence title
(a signature listing, or one needing your own data file, network access or the Thermo .NET
runtime), and those titles come from a closed set, so "not executed" cannot become a place to park
a broken example.

## What CI adds

CI builds and tests the installed wheel outside the checkout on Linux, Windows and macOS with
Python 3.12 through 3.14, and resolves the lowest supported direct dependencies in a separate
Python 3.12 job. A dedicated job installs .NET 8 and sets `SPXTACULAR_REQUIRE_THERMO=1` so a
Thermo backend failure cannot pass as a skipped test; other jobs skip the optional reader tests.
To run the wheel check locally:

```bash
uv build --out-dir wheel-dist
uv run --no-project python .github/scripts/check_wheel.py wheel-dist
uv run --no-project python .github/scripts/check_wheel.py wheel-dist --lowest
```

## Contribution guidelines

[`CONTRIBUTING.md`](https://github.com/tacular-omics/spxtacular/blob/main/CONTRIBUTING.md) covers
the rest: what belongs in a pull request, the rules for test fixtures and data, how releases and
API-compatibility decisions are made, conduct, and how to report a suspected security
vulnerability privately.

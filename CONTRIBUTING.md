# Contributing to spxtacular

Bug reports, feature proposals, documentation fixes, and code contributions are welcome. Use the
[GitHub issue tracker](https://github.com/tacular-omics/spxtacular/issues) for bugs, questions, and
support. Please include a minimal example, the spxtacular and Python versions, and the complete
error message when reporting a defect.

## Development setup

spxtacular requires Python 3.12 or newer and uses [uv](https://docs.astral.sh/uv/) to manage its
development environment:

```bash
git clone https://github.com/tacular-omics/spxtacular.git
cd spxtacular
uv sync --locked
```

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
changes. Add a concise entry to `HISTORY.md` for changes that users need to know about. Keep pull
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

# Copilot instructions

- Use an existing `just` recipe for project commands; inspect `just --list` first.
- When no recipe exists, use `uv` for dependency management and `uv run` for Python tools.
- Do not invoke `python`, `pip`, `pytest`, npm, yarn, gem, or another package manager directly.
- Run `just test`, `just lint`, `just format`, and `just check` as appropriate.
- Type production code comprehensively with Python 3.12 syntax and concrete generic types.
- Prefer frozen slotted dataclasses and functional transformations when practical.
- Prefer clear names and small helpers over implicit behavior or clever one-liners.
- Keep tests focused; exhaustive typing is not required in tests.
- Keep docstrings concise, explain non-obvious reasons, and document raised exceptions.
- Do not repeat information already expressed by type hints.
- Do not leave placeholder TODO comments; raise `NotImplementedError` with a reason when necessary.
- Keep responses concise and assume the reader is proficient with Python.

Repository architecture and invariants are documented in `CLAUDE.md`.

# Repository Guidelines

## Project Structure & Module Organization
Core simulation code lives in `src/mujoco_mcp`, covering servers, scene loaders, RL adapters, and visualization. Reusable automation lives in `scripts/`, interactive demonstrations in `examples/`, and regression suites in `tests/` plus top-level `test_*.py` integrations. Performance suites sit under `benchmarks/`, while build artifacts land in `dist/`.

## Build, Test, and Development Commands
- `pip install -e .[dev]` — install the package plus lint/type/test tooling.
- `scripts/setup-dev.sh` — bootstrap MuJoCo paths and editor-friendly environment variables.
- `python mujoco_viewer_server.py` — launch the default viewer server for manual verification.
- `python scripts/quick_internal_test.py` — fast smoke test covering MCP endpoints.
- `pytest` or `pytest tests/test_v0_8_basic.py` — run targeted suites defined in `pytest.ini`.
- `./run_all_tests.sh` — full battery including compliance, RL, and performance checks.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation, type hints on public APIs, and docstrings for new modules. `black` and `isort` enforce a 100-character line ceiling and import order; run `scripts/format.sh` before committing. `ruff` handles lint rules (`scripts/quality-check.sh` runs ruff, mypy, and pytest together). Use `snake_case` for modules/functions, `CamelCase` for classes, and prefix optional CLI scripts with verbs (e.g., `setup-dev`).

## Testing Guidelines
Add unit tests under `tests/` mirroring the package path, and place whole-system scenarios with the other root-level `test_*.py` files. Name tests descriptively (`test_<feature>_<behavior>`) so they surface cleanly in CI. Run `python scripts/quick_internal_test.py` before every PR and expand to `pytest -k <area>` for focused changes. Features touching realtime control or RL loops should include a deterministic seed and, when practical, assertions on sensor payloads or reward trends.

## Commit & Pull Request Guidelines
Commit messages follow `type: imperative summary` (e.g., `fix: resolve built-in open misuse`). Squash small fixes locally to keep history readable. Pull requests need a concise description, linked issues, a summary of tests executed (`pytest`, smoke scripts, or manual viewer checks), and screenshots or logs for UI/viewer adjustments. Highlight configuration changes (env vars like `MUJOCO_MENAGERIE_PATH`) and note resource-heavy tests when reviewers should run them manually.

## Security & Configuration Tips
Never commit MuJoCo license keys or proprietary Menagerie assets; reference paths instead. Use `.env` files ignored by git for credentials and viewer ports. When adding new agents or servers, document required environment variables in `ARCHITECTURE.md` and add sanity checks so viewers refuse to start without critical paths.

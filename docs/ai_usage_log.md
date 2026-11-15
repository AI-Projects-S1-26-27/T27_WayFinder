# AI-usage log

This log matches Section 7 of the report (`docs/report/REPORT.md`).

## Tool

Claude Code (Anthropic), running the Claude Opus 5.5 model, used during the project's implementation and write-up
phase.

## What it was used for

| Area | Use |
|---|---|
| Implementation | Implementing and refactoring code: the pipeline, search algorithms, training and evaluation code, the FastAPI service, the web interface and the user-study tooling |
| Software engineering | Packaging the project: requirements, `pyproject.toml`, Makefile, README, `.gitignore` |
| Data | Advice on choosing datasets for the study area (OpenStreetMap, ESA WorldCover, Copernicus GLO-30) |
| Tests | Writing the test suite and `scripts/verify_routes.py` |
| Evaluation | Running the pipeline and the benchmark evaluation |
| Report | Drafting and editing `docs/report/REPORT.md` |

## What stayed with the team

The project idea, the proposal (problem, objectives, algorithms, evaluation plan and the Week-4 gate), the Week-1
data acquisition and coverage audit, and the decision of what to report.

## How the output was verified

- Test suite (`python -m pytest`, 87 tests): every search compared against exact optima (networkx reference and
  backward Dijkstra), on a fixture graph and on the real graph.
- Focal Search bound checked with an adversarial, wildly inadmissible guide; the bound held in all 7,000 bounded
  benchmark runs.
- `h_adm` admissibility and consistency checked on every edge, and `h_adm ≤ h*` on all 8,500,000 labels.
- Features, turn penalties and costs cross-checked against an independent re-implementation from the raw GraphML;
  numpy inference checked against PyTorch.
- `scripts/verify_routes.py`: the app's default route equals the exact optimum on 147/147 random trips.
- Full pipeline re-run from a fresh clone: models rebuilt byte-identical, all results identical except timings.

## Errors found in AI-written code

Found by the checks above and fixed:

- The web app defaulted to Focal Search at W = 1.5, so it did not always show the best route. It now defaults to
  exact A\*.
- The route cache rounded slider weights, so two nearby settings could share a route. It now uses exact weights.
- The focal-floor count included goal states through floating-point ties. Goal states are now excluded.

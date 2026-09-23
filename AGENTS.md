# AGENTS.md

This is a PyTorch time-series forecasting portfolio project.

## Before Editing

- Run `git status --short --branch`.
- Do not overwrite user changes.
- Keep changes focused, reproducible, and scoped to the requested task.

## Verification

- After Python edits, run:

  ```bash
  python3 -m py_compile src/*.py src/Informer_Models/*.py src/Informer_Utils/*.py
  ```

- If dependencies and data are available, run a small smoke test before summarizing work.

## Do Not Commit

- Datasets
- Checkpoints
- `.pt`, `.pth`, or `.pckl` model files
- Environment folders
- Generated caches
- Paper PDFs or the `literature/` folder

## Project Goal

Make this a clean, reproducible ML portfolio project suitable for grad-school applications. Prefer clear baselines, honest limitations, and reproducible commands over broad but unfinished claims.

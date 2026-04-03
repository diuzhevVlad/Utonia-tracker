# AGENTS.md
## Scope
- This file applies to the whole repository at `/home/vladislav/Documents/Utonia-tracker`.
- The repo is a Python 3.10 codebase for Utonia point-cloud inference, visualization demos, and tracking experiments.
- Treat this repo as script-driven rather than CI-driven: validation is mostly smoke testing through real entry points.
## Instruction Files
- Root `AGENTS.md` exists and should be kept current when repo conventions change.
- No Cursor rules were found in `.cursor/rules/`.
- No `.cursorrules` file was found.
- No Copilot instructions were found in `.github/copilot-instructions.md`.
## Project Layout
- `utonia/`: main package code.
- `utonia/tracker.py`: best reference for newer style in this repo.
- `utonia/model.py`: large core model implementation with older conventions.
- `utonia/transform.py`: large transform registry and preprocessing pipeline.
- `scripts/`: practical smoke/integration entry points.
- `demo/`: interactive demos from the README.
- `data/`: local sample assets; `.gitignore` ignores `data/`, so do not assume datasets are committed.
- `environment.yml`: recommended conda environment.
- `setup.py`: package install entry point.
## Environment And Build
- Recommended standalone setup:
```bash
conda env create -f environment.yml --verbose
conda activate utonia
```
- `environment.yml` pins `python=3.10`, `pytorch=2.5.0`, `pytorch-cuda=12.4`, and `numpy<=1.26.4`.
- README notes that FlashAttention should normally be installed; code falls back to `enable_flash=False` in some paths.
- Package-mode install from source:
```bash
python setup.py install
```
- Demo and script usage often expects the repo root on `PYTHONPATH`:
```bash
export PYTHONPATH=./
```
- There is no `Makefile`, `tox`, `nox`, or PEP 517 build config in this repo.
- Quick import smoke check after install:
```bash
python -c "import utonia; print(utonia.__all__)"
```
- Do not invent `poetry`, `hatch`, or `make` workflows here.
## Lint And Static Validation
- No committed formatter or linter config was found.
- No `pyproject.toml`, `ruff`, `black`, `isort`, `flake8`, `mypy`, or `pyright` config files are present.
- There is no official lint command.
- Safe syntax validation for the whole codebase:
```bash
python -m compileall utonia scripts demo
```
- Narrow validation is preferred over broad validation when the change is localized.
## Test Commands
- No `pytest` or `unittest` suite is checked in.
- In practice, tests are smoke tests through `scripts/` and selected `demo/` entry points.
- Fast import smoke test:
```bash
python -c "import utonia, utonia.model, utonia.transform, utonia.tracker"
```
- Whole-repo syntax smoke test:
```bash
python -m compileall utonia scripts demo
```
- PCA smoke test against the sample asset already in the repo:
```bash
python scripts/pca_test.py data/000300.bin
```
- PCA smoke test against one frame in a sequence:
```bash
python scripts/pca_test.py /path/to/sequence_or_velodyne_dir --frame 300
```
- Similarity smoke test:
```bash
python scripts/similarity_test.py /path/to/sequence_or_velodyne_dir
```
- KITTI tracker smoke test:
```bash
python scripts/kitti_tracker_test.py /path/to/trackkitti/training/velodyne/0000
```
- Standard README demo smoke commands are `python demo/0_pca_indoor.py` through `python demo/7_pca_outdoor.py` with `PYTHONPATH=./`.
## Single Test Guidance
- There is no single-test selector like `pytest path::test_name` in this repo.
- Here, “run a single test” means “run one script with one concrete dataset input”.
- Use `python scripts/pca_test.py data/000300.bin` for the fastest real feature-extraction smoke test.
- Use `python scripts/pca_test.py /path/to/sequence --frame 300` when you need one exact frame.
- Use `python scripts/similarity_test.py ...` for similarity or feature-normalization changes.
- Use `python scripts/kitti_tracker_test.py ...` for `utonia/tracker.py` changes.
- If the change only affects imports, packaging, or syntax, use `python -m compileall utonia scripts demo`.
## Runtime Notes
- Many code paths assume CUDA is available, though several scripts fall back to CPU.
- Hugging Face downloads are used for pretrained weights.
- KITTI and SemanticKITTI-style scripts require local datasets.
- `scripts/*.py` import `rerun`, but `rerun` is not declared in `environment.yml`; if a script fails on import, that is the first dependency to check.
- Demo visualization depends on packages such as `open3d`, `opencv-python`, `camtools`, and `trimesh`.
- `demo/8_pca_video.py` has extra VGGT-related setup documented in `README.md`.
- GUI-heavy demos are not suitable for headless validation.
## Code Style Overview
- Follow the surrounding file before applying any global preference.
- Prefer the cleaner style in `utonia/tracker.py` when adding new modules or small helpers.
- Keep changes minimal, local, and task-focused.
- Avoid opportunistic refactors in `model.py` or `transform.py` unless required for correctness.
- Preserve existing copyright/license headers in files that already have them.
## Imports
- Group imports as standard library, third-party, then local package imports.
- Separate groups with one blank line.
- Inside the package, prefer explicit relative imports such as `from .model import load`.
- Avoid unused imports in new code, even if some older files are loose about this.
- Common aliases in this repo are `import numpy as np`, `import torch`, and `import torch.nn.functional as F`.
## Formatting
- Use 4-space indentation.
- Keep formatting Black-compatible even though Black is not configured here.
- Wrapped calls usually use trailing commas and one argument per line.
- Do not reflow unrelated legacy code just to enforce a strict line width.
- Keep comments short and only where they materially help with dense logic.
## Types
- Python 3.10 features are available.
- Prefer built-in generics like `list[...]`, `dict[...]`, and `tuple[...]`.
- Prefer `X | None` over `Optional[X]` in new code.
- Add type hints for new public functions and non-trivial internal helpers.
- Use `np.ndarray` and `torch.Tensor` annotations where they clarify tensor/array expectations.
- `@dataclass` is already used for lightweight state in `utonia/tracker.py` and is acceptable for similar state holders.
## Naming
- Use `snake_case` for functions, methods, variables, and helpers.
- Use `PascalCase` for classes.
- Use `UPPER_CASE` for module-level constants such as `MODELS` or `DEVICE`.
- Preserve established point-cloud keys like `coord`, `color`, `normal`, `segment`, `batch`, `offset`, `grid_coord`, `feat`, `inverse`, `pooling_parent`, and `pooling_inverse`.
- Prefer descriptive names unless a short domain term is already conventional in the file, such as `feat`, `coord`, `idx`, or `pcd`.
## Data And Tensor Handling
- Avoid mutating caller-owned numpy arrays unless the function contract clearly allows it.
- Existing code commonly uses `coord.copy()` before transforms; follow that pattern for raw input arrays.
- When color or normal is unavailable, current scripts fill them with `np.zeros_like(coord)`.
- Use `torch.inference_mode()` or `@torch.no_grad()` for inference-only paths.
- Move tensors to device explicitly.
- Existing CUDA paths often use `.cuda(non_blocking=True)` or `.to(self.device)`.
- Normalize embeddings with `F.normalize(...)` when cosine similarity or prototype matching depends on unit-length features.
## Error Handling
- Validate external paths and inputs early.
- Raise standard exceptions with short, actionable messages.
- Existing code commonly uses `FileNotFoundError`, `TypeError`, `KeyError`, `ValueError`, and `RuntimeError`.
- Use assertions for internal invariants when failure indicates programmer error rather than bad user input.
- Include the failing path, key, or value in the message when practical.
## Comments And Docstrings
- Public classes and non-obvious public methods should have concise docstrings.
- Do not add comments that merely restate the code.
- Short comments are useful before dense tensor manipulation or coordinate-frame logic.
- Keep docstrings factual; avoid tutorial-style prose inside library code.
## File-Specific Guidance
- `utonia/`: keep code import-safe and reusable.
- `utonia/tracker.py`: preferred style reference for new tracking work.
- `utonia/model.py` and `utonia/transform.py`: large legacy-heavy files; match local conventions when editing them.
- `scripts/`: favor clear CLI arguments, explicit file validation, and narrow smoke-test behavior.
- `demo/`: example-oriented code; preserve the existing flow unless the task is demo-specific.
## Change Strategy For Agents
- Start with the smallest correct change.
- Read the touched file carefully before editing; this repo mixes newer typed code with older library code.
- Prefer fixing the targeted path over introducing new abstractions.
- Verify with the narrowest command that exercises the modified behavior.
- For tracking changes, prefer `scripts/kitti_tracker_test.py` over unrelated demos.
- For feature-extraction changes, prefer `scripts/pca_test.py` or `scripts/similarity_test.py`.
- Do not add new tooling/config files unless the user explicitly asks for them.
- If validation depends on unavailable datasets, GPU, or GUI support, say so explicitly in your final note.

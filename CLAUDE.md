# CLAUDE.md - Guidelines for AI Assistants

## Development Commands
- Deploy reward model: `sky launch serve_rm/serve_rm.yaml`
- Jupyter notebook: `jupyter notebook best_hn/generate_titles/prepare_data.ipynb`
- Run Python module: `python -m best_hn.generate_titles.<module_name>`
- Run a single test: `pytest -xvs test_file.py::test_function_name`

## Code Style
- Imports: stdlib first, third-party packages second, local modules last
- Naming: snake_case for variables/functions, PascalCase for classes
- Type hints: Use explicit type annotations for function parameters and returns
- Error handling: Use try/except with specific exceptions, avoid bare except
- Models: Use Pydantic for request/response models
- Docstrings: Google style docstrings for classes and functions
- Line length: 88 characters (Black default)

## Project Structure
- `best_hn/generate_titles`: Main package for HN title generation
- `serve_rm`: FastAPI server for reward model inference
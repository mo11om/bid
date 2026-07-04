# bid — environment

## Conda environment

- Env name: `bridge-eval` (python 3.10), created by `setup_env.sh` (conda if
  available, else falls back to a local `.venv`).
- Interpreter path (this box): `/home/mo1om/code/miniconda/envs/bridge-eval/bin/python`
  — use this directly for one-off scripts/commands instead of `conda activate`
  when running non-interactively.
- Activate interactively: `conda activate bridge-eval`
  (conda hook lives at `~/code/miniconda/bin/conda` / `~/code/miniconda3/bin/activate`).
- Setup / re-provision: `bash setup_env.sh` — verifies core imports
  (`redeal`, `endplay`, `pydantic`, `openai`, `matplotlib`, `pandas`) after install.

## Dependencies (`requirements.txt` / `environment.yml`)

- `redeal` — deal generation. **Not on PyPI** — installed from
  `git+https://github.com/anntzer/redeal`, so `git` must be on `PATH`.
- `endplay>=0.5.0` — double-dummy solver (DDS quality scoring, self-play rollout).
- `pydantic>=2.0` — structured LLM output / dataset schema.
- `openai>=1.0.0` — OpenAI-compatible client (vLLM backend, and the legacy
  Ollama `/v1` path — prefer native `/api/chat` for Ollama, see below).
- `requests>=2.28.0` — Ollama native `/api/chat` path.
- `matplotlib>=3.7.0` — accuracy-vs-HCP reporting.
- `pandas>=2.0.0` — tabular aggregation.
- `pytest>=7.3.0` — test suite (130 tests, no network/solver required —
  `endplay`/`redeal` integration paths are covered via fake-module injection).

## Model backends (not conda, but environment-adjacent)

- **Ollama** (default, `--backend ollama`): native endpoint
  `http://localhost:11434/api/chat`. Model tag is case-sensitive
  (`Gemma4:26b`, not `gemma4:26b`) — `ollama list` to confirm exact tags.
  Untagged names do NOT alias to an installed tag; a mismatch silently
  404s every request and falls back to all-`Pass`.
- **vLLM** (`--backend vllm`): OpenAI-compatible endpoint
  `http://localhost:8000/v1/chat/completions`.
- Override host/port with `--base-url`.
- No API keys needed for local backends; `Config.api_key` is a placeholder
  string only (the openai client requires it non-empty).

## Running things

```bash
bash setup_env.sh
conda activate bridge-eval

python run_eval.py --mode eval_local --model Gemma4:26b
pytest
```

See `README.md` for full CLI flags and architecture.

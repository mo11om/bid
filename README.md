# Bridge AI Evaluation System

Evaluate how well a **local** LLM (served via vLLM or Ollama) bids at contract
bridge. Each position gives the model one seat's hand plus the auction so far and
asks for a call. Calls are judged two ways:

1. **Exact accuracy** — does the call match the reference ("expert") call?
2. **Quality (1-IMP rule)** — when it differs, both auctions are rolled out by
   LLM self-play, the settled contracts are double-dummy scored on the full deal,
   and the call is *acceptable* if the contracts are within the configured
   threshold (default: 1 IMP via the WBF table).

## Setup

```bash
bash setup_env.sh          # conda env 'bridge-eval' (py3.10) + deps
conda activate bridge-eval
```

Dependencies: `redeal` (deal generation), `endplay` (double-dummy solver),
`pydantic`, `openai`, `requests`, `matplotlib`, `pandas`, `pytest`.

> **Note:** `redeal` is not published to PyPI — `requirements.txt` installs it
> from GitHub (`git+https://github.com/anntzer/redeal`), so `git` must be on
> your PATH. Everything else comes from PyPI.

## Usage

```bash
# Phase 1 — generate a synthetic dataset (4 records per deal)
python run_eval.py --mode mock_data --count 25

# Phase 2/3 — evaluate a local model and write an accuracy-vs-HCP report
python run_eval.py --mode eval_local --model gemma4:26b            # Ollama (default)
python run_eval.py --mode eval_local --backend vllm --model <name> # vLLM

# Phase 4 — constrained scenarios (force N/S HCP bands)
python run_eval.py --mode scenario_test --count 10 \
    --north-hcp 12 16 --south-hcp 6 9
```

Useful flags: `--threshold-mode {imp,score}`, `--threshold-n N`, `--no-dds`
(exact-match only, skips the solver), `--results-json out.json`,
`--think {auto,on,off}` (see below), `--backend {ollama,vllm}` (see below),
`--prompt-style {base,knowledge,examples}` (see below),
`--no-retry-illegal` (disable the one corrective re-ask after an FSM-rejected
call; illegal calls then fall straight back to Pass).

### Prompt style (`--prompt-style`)

Three prompt variants, A/B-tested live on the Ben-SAYC benchmark set
(Gemma4:26b, `--think off`, temp 0, retry-on-illegal enabled):

| Style | Contents | bench25 | bench150 |
|---|---|---|---|
| `base` | hand features + auction roles | 60% | — |
| `knowledge` | + SAYC reference guide | 60% | — |
| `examples` (default) | + targeted rules + generic few-shot examples | **72%** | **62%** |

(bridge-llm-bench's Gemini Flash Lite scores 68.7% on the same 150 positions
with its simplified P22 prompt — whose examples include test-set deals.)

With full-deal DDS quality scoring (see below), bench150 rises to **69.3%
within 1 IMP** and **82.0% under the asymmetric rule** — 19 of the 57
oracle-exact "misses" are lines that *beat* the oracle by more than 1 IMP.

This mirrors bridge-llm-bench's ablation finding: removing all examples cost
that benchmark −29pts, removing all rules cost −0.7pts. Only the ablation-kept
rule blocks (penalty doubles, 5-level, suit choice) are ported; the rules that
measurably hurt (competitive-bidding, takeout-X-response, "when not to
compete") are deliberately absent. All few-shot hands are freshly composed —
`tests/test_prompt_examples.py` enforces that none appears in the benchmark
data (the benchmark's own example block leaked test-set deals, inflating its
headline number).

### Ben-SAYC benchmark datasets

`scripts_convert_bench.py` converts the sibling `bridge-llm-bench` data into
this harness's JSONL schema, using the Ben SAYC engine as oracle:

```bash
python scripts_convert_bench.py --set 25    # data/bench25_bensayc.jsonl
python scripts_convert_bench.py --set 150   # data/bench150_bensayc.jsonl
```

The CSVs list positions deal by deal, rotating through the four seats, so each
deal's first four rows contain all four hands — the converter reassembles them
into `deal_pbn`/`all_hands` (validated to 52 unique cards), which enables the
full rollout + double-dummy quality scoring on these sets (dealer assumed N,
vulnerability None; the truncated last deal of the 25-set stays exact-match
only). `scripts_vote_eval.py` adds k-sample majority voting on top of the same
harness (`--k 9 --temp 0.5 --prompt-style examples`).

### DDS acceptability rule (`--dds-rule`)

`asymmetric` (default): a differing call is acceptable when its rolled-out
contract is within `threshold_n` IMPs of the expert's *or better for the
model's side* — beating the oracle is not a failure. `symmetric` is the legacy
`|delta| <= threshold` rule, kept for comparison. Internally the NS-perspective
IMP delta is re-oriented by the acting seat (`model_gain_imp` in `--detail`
output): without that, "better" would be inverted for E/W seats.

### Backend (`--backend`)

Two backends are supported, selected via `--backend`:

| Flag | Endpoint | Protocol |
|---|---|---|
| `ollama` (default) | `http://localhost:11434/api/chat` | Ollama native |
| `vllm` | `http://localhost:8000/v1/chat/completions` | OpenAI-compatible |

**Use `--backend ollama` for Ollama models.** The native `/api/chat` endpoint
correctly honours `think:false` to suppress reasoning. The OpenAI-compatible
`/v1` adapter on Ollama ignores `think:false` for thinking models (verified
with `qwen3.6:27b`) — reasoning runs on every call regardless, burning latency
without any benefit.

Override the URL with `--base-url` when running on a non-default host or port.

### Thinking mode (`--think`)

Some Ollama models (qwen3, deepseek-r1, gpt-oss, ...) support a reasoning/
"thinking" toggle. `--think` is tri-state:

- `auto` (default) — the `think` field is omitted entirely; the server or
  model picks its own default.
- `off` — force thinking off. Recommended for bidding: it's faster and
  cheaper, and `BridgeBid.thinking` already captures the model's one-line
  rationale in the structured output. Only works correctly with
  `--backend ollama`.
- `on` — force thinking on.

The on-disk response cache keys on `(model, temperature, think, prompt)`, so
flipping this flag never serves a response generated under the other setting.
As a safety net, `LocalLLMClient._parse_bid` strips any inline
`<think>...</think>` block before extracting JSON, in case a backend inlines
reasoning into `content` instead of a separate field.

### If every bid comes back `Pass`

`LocalLLMClient.get_bid` never raises — any failure (dead endpoint, wrong
model name, unparsable output, illegal call) falls back to `Pass` so one bad
position can't crash an entire run. That robustness has a downside: a fully
broken setup (e.g. `--model llama3` when Ollama only has `llama3:8b`
installed — Ollama does **not** alias untagged names to a tag) makes *every*
call 404 and silently produces an all-`Pass` report that looks like real
results.

Two safeguards:

- **`LocalLLMClient.verify_connection()`** runs automatically before
  `eval_local` / `scenario_test` start. It lists models from the endpoint and
  checks `config.model` is actually one of them, failing fast with an
  actionable message (and the list of what *is* installed) instead of
  grinding through the whole dataset. Run `ollama list` to see exact tags.
- **`fallback_reason`** is attached to every result (`"none"`,
  `"transport_error"`, `"parse_error"`, or `"illegal_call"`), and the summary
  reports aggregate `fallback_counts` / `fallback_pass_rate`. A non-zero rate
  printed after a run means the model isn't actually bidding for some
  positions — rerun with `--detail` to see each fallback's exact cause in
  `llm_thinking` (e.g. `"parse error: unrecognized call: '2'"`).

## Architecture

```
src/
  config.py                 # endpoint, model, threshold knobs
  bridge.py                 # pure hand/bid primitives (HCP, shape, LTC, ...)
  schema/dataset.py         # Pydantic: MockDealRecord, BridgeBid
  data/mock_generator.py    # redeal -> per-seat records -> JSONL
  harness/
    fsm_guardrail.py        # BiddingFSM legality check
    prompt_builder.py       # masked prompt (active seat only)
    llm_client.py           # LocalLLMClient + self-play rollout
    llm_cache.py            # on-disk response cache
  evaluation/
    metrics.py              # accuracy, settle, DDS scoring, 1-IMP rule
    reporter.py             # matplotlib accuracy-vs-HCP chart
  simulation/scenario_runner.py
run_eval.py                 # CLI orchestrator
```

### Key design points

- **Masking is structural.** `MockDealRecord` carries the *full* deal (needed by
  the solver and the rollout), but the prompt is built only from
  `masked_view()` — the active seat's hand, HCP, and auction. The other hands
  never reach the model. (`tests/test_prompt_masking.py`)
- **The prompt injects derived features.** To spare the model error-prone
  card-counting, `ContextBuilder` states each suit's explicit length, the
  distribution (`compute_shape`), hand type (`classify_shape`: Balanced /
  Two-suited / Unbalanced), and LTC (`compute_ltc`) — all derived only from the
  active seat's own hand. The auction is annotated with table roles
  (`annotate_auction`: RHO/Partner/LHO/**You** — the cycle has period 4 since
  the model's own earlier calls are part of the history; a period-3 cycle
  mislabeled who opened in every auction of 4+ calls), derived purely from
  auction order so masking still holds.
- **A bid becomes a contract via self-play rollout.** `LocalLLMClient` bids the
  remaining seats until the auction closes, then `endplay` double-dummy scores
  the settled contract. Rollout calls are cached to keep runs cheap.
- **Threshold is configurable.** Default is a true IMP delta (`imp`); `score`
  compares raw duplicate points.
- **Placeholder expert.** The mock reference is a trivial heuristic (12+ HCP in
  first seat → 1NT, else Pass), to be replaced by real expert/BBO data later.
  Because it is shape-blind, **exact-accuracy against it is now misleading**: a
  model that correctly opens 1H on a 5-5 hand (rather than 1NT) is scored as
  "wrong". Prefer the DDS-acceptable rate, and see `docs/SESSION_LOG.md` §11 for
  the full error analysis.

## Tests

```bash
pytest          # 101 tests
```

Most tests run with no network and no solver. The `endplay` and `redeal`
integration paths are covered by injecting fake modules
(`tests/test_dds_path.py`, `tests/test_redeal_path.py`), so the wiring is
verified even before the heavy C extensions are installed.

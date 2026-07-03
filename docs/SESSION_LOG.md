# Bridge AI Evaluation System — Session Log

This document consolidates everything discussed and decided while building this
project: the original request, the design decisions made through brainstorming,
what was actually built, and a fix applied afterward. Read this if you want the
full "why" behind the code without re-reading the whole chat.

---

## 1. Original Request (verbatim spec)

The user supplied a 4-phase "Master Actionable Architecture" spec for a Bridge
AI Evaluation System, asking for:

- **Env**: conda env `bridge-eval` (py3.10), packages `redeal endplay pydantic
  openai matplotlib pandas`.
- **Directory tree**: `bridge-eval/{data,src/{schema,data,harness,evaluation,
  simulation},run_eval.py}`.
- **Phase 1 — Data pipeline**: Pydantic `MockDealRecord` / `BridgeBid` models;
  `generate_mock_dataset()` using `redeal` to deal random boards, compute HCP /
  shape / LTC, mock an `expert_bid` heuristic, write JSONL.
- **Phase 2 — LLM harness**: `BiddingFSM.is_valid_bid()` legality guardrail;
  `ContextBuilder.build_prompt()` masked prompt (active seat only); `LocalLLMClient`
  hitting an OpenAI-compatible local endpoint (vLLM `:8000` / Ollama `:11434`),
  returning structured `BridgeBid`, falling back to `Pass` on FSM rejection.
- **Phase 3 — Evaluation**: `calculate_accuracy()`; `evaluate_with_dds()` — the
  "1-IMP rule": when the LLM's bid differs from the expert's, double-dummy score
  both contracts via `endplay.dds` and accept if the score delta is small;
  `generate_report()` — matplotlib accuracy-vs-HCP-range chart.
- **Phase 4 — Simulation**: `ScenarioRunner` for forced-HCP scenarios via
  `redeal`; `run_eval.py` CLI with `mock_data` / `eval_local` / `scenario_test`
  modes.
- **Definition of Done**: conda env creates cleanly; directory structure
  followed exactly; Phase 1 HCP/Shape/LTC correct; Phase 2 client returns valid
  `BridgeBid`; Phase 3 imports `endplay` and computes DD scores on mismatch;
  `run_eval.py` orchestrates the whole pipeline.

---

## 2. Brainstorming — Context Discovered

Before designing, the working directory was explored:

- `bid/` (this project) was **empty**.
- A **mature sibling project**, `../bridge-llm-bench`, already implements most
  of this: DDS/IMP scoring (`metrics/dd_scoring.py`), HCP parsing
  (`parsers/hand_parser.py`), auction validation (`validation/game_validator.py`),
  11 LLM provider clients, and a reported **80% bidding accuracy** result.
- Neither `redeal` nor `endplay` was installed in the ambient Python 3.13
  interpreter — expected, since the spec calls for a dedicated py3.10 conda env.

This raised an explicit scoping question, answered by the user:

> **Decision: build fresh and standalone.** Do not import from
> `bridge-llm-bench`. The new `bid/` project must stand on its own, even though
> it duplicates some proven logic (the WBF IMP table was used as a reference,
> not a dependency).

## 3. Brainstorming — Resolved Ambiguities

The spec's Phase 3 (`evaluate_with_dds(deal_pbn, llm_bid, expert_bid)`) was
underspecified: a single bid like `"1NT"` isn't a scorable contract by itself —
double-dummy scoring needs a full settled contract (level, strain, declarer).
Two decisions resolved this:

1. **How does a bid become a scorable contract?**
   → **Roll out the full auction via LLM self-play.** `LocalLLMClient` bids all
   remaining seats (using their real hands) until the auction closes (3 passes),
   for both the LLM's line and the expert's line. This is the most realistic of
   three options considered (the others: treat the bid as the final contract
   outright, or score against the DD-best contract in that strain).

2. **What does "1-IMP rule" actually mean?**
   → **Configurable, default = true IMP delta ≤ 1**, converting both raw
   duplicate scores to IMPs via the WBF table. A literal "score points ≤ 1"
   mode is also available behind the same flag (`--threshold-mode score`),
   since the spec's literal wording and its own "1-IMP" name don't match —
   raw score points jump in increments of 10/20+, so a literal points-based
   threshold would almost never accept anything.

Two implementation-process decisions were also made:

- **Don't install the conda env during the build.** Generate
  `requirements.txt` / `environment.yml` / `setup_env.sh`; the user runs setup
  themselves. (Avoids long installs / sandbox network limits during
  implementation.)
- **Extend `MockDealRecord` beyond the spec.** Add `deal_pbn`, `dealer`,
  `vulnerability`, `all_hands` — required so the self-play rollout and DDS
  solver have full deal information, while the prompt layer still shows the
  model only its own hand, HCP, and the auction so far (masking is enforced in
  code via `masked_view()`, not by what the record omits).

## 4. What Was Built

```
bid/
├── data/                          # datasets, llm cache, report PNGs (gitignored)
├── docs/
│   └── SESSION_LOG.md             # this file
├── src/
│   ├── config.py                  # endpoint, model, IMP threshold knobs
│   ├── bridge.py                  # pure hand/bid primitives (HCP, shape, LTC, FSM helpers)
│   ├── schema/dataset.py          # Pydantic: MockDealRecord, BridgeBid
│   ├── data/mock_generator.py     # redeal-driven synthetic data generation
│   ├── harness/
│   │   ├── fsm_guardrail.py       # BiddingFSM.is_valid_bid
│   │   ├── prompt_builder.py      # ContextBuilder.build_prompt (masked)
│   │   ├── llm_client.py          # LocalLLMClient.get_bid + rollout_auction (self-play)
│   │   └── llm_cache.py           # on-disk JSON response cache
│   ├── evaluation/
│   │   ├── metrics.py             # accuracy, settle_contract, DDS scoring, 1-IMP rule
│   │   └── reporter.py            # matplotlib accuracy-vs-HCP bar chart
│   └── simulation/scenario_runner.py  # forced-HCP scenario generation + eval
├── tests/                         # 38 pytest tests (offline-runnable, see below)
├── run_eval.py                    # CLI: --mode {mock_data, eval_local, scenario_test}
├── requirements.txt
├── environment.yml
├── setup_env.sh
└── README.md
```

Key points worth remembering:

- **Masking is structural, not just a convention.** `MockDealRecord` carries the
  full deal (for the solver/rollout), but `ContextBuilder.build_prompt()` reads
  *only* `record.masked_view()` — the active seat's hand, HCP, and auction.
  Covered by `tests/test_prompt_masking.py`, which asserts other seats'
  distinctive holdings never appear in the rendered prompt.
- **Self-play rollout drives DDS.** `LocalLLMClient.rollout_auction()` extends
  `current_bidding + [first_bid]` by asking the model to bid every remaining
  seat from its real hand until 3 passes close the auction (capped by
  `config.max_rollout_calls` as a safety valve). Run twice per mismatch (once
  for the LLM's bid, once for the expert's), then both contracts are settled
  and double-dummy scored on the same `deal_pbn`.
- **LLM calls are cached on disk** (`src/harness/llm_cache.py`, keyed by a hash
  of model+temperature+prompt) because rollout multiplies calls per mismatched
  position — without caching, re-running an eval would re-bid every auction
  from scratch.
- **The WBF IMP table and duplicate-score formula were ported** (not imported)
  from `bridge-llm-bench/bridge_llm_bench/metrics/dd_scoring.py` into
  `src/evaluation/metrics.py`, per the "fresh and standalone" decision.
- **`expert_bid` is a placeholder heuristic** (12+ HCP and no contract on the
  table yet → open `1NT`, else `Pass`) — explicitly a stand-in until real
  expert/BBO reference data is wired in. This is the most likely next piece of
  work and was flagged as out of scope for this build.

## 5. Verification Performed

The ambient interpreter here is Python 3.13 without `redeal`/`endplay`
installed (by design — those belong in the py3.10 conda env). Verification was
layered accordingly:

- **32 pure-logic tests** run directly: HCP/shape/LTC math, bid normalization
  and ranking, the `BiddingFSM` legality guardrail, prompt masking, contract
  settling, the WBF IMP table, and the full evaluation loop — all with no
  external dependencies.
- **Reporter** verified against a real `matplotlib` install (added temporarily
  to confirm the chart actually renders).
- **`endplay` DDS composition** (settle → trick lookup → score → IMP threshold)
  verified by injecting a **fake `endplay` module** with the same surface
  (`Deal`, `Denom`, `Player`, `calc_dd_table`) — `tests/test_dds_path.py`.
- **`redeal` interop + JSONL round-trip** verified the same way, with a fake
  `redeal.Deal.prepare()` — `tests/test_redeal_path.py`.
- **CLI** smoke-tested: `--help`, invalid `--mode` rejection, and a graceful
  (non-crashing) exit when the dataset file is missing.

All 38 tests pass; `python -m compileall` is clean across `src/`, `run_eval.py`,
and `tests/`.

## 6. Post-Build Fix: `redeal` Is Not on PyPI

After the build, running `setup_env.sh` failed with:

```
ERROR: Could not find a version that satisfies the requirement redeal>=0.4.0 (from versions: none)
ERROR: No matching distribution found for redeal>=0.4.0
```

**Root cause:** `redeal` (by anntzer) has never been published to PyPI — the
PyPI JSON endpoint for it 404s. It's only distributed from its GitHub repo,
[anntzer/redeal](https://github.com/anntzer/redeal). (`endplay` *is* genuinely
on PyPI — version 0.5.12, wheels for Python 3.9–3.13 — so that pin was left
unchanged.)

**Fix applied:**

- `requirements.txt`: `redeal>=0.4.0` → `redeal @ git+https://github.com/anntzer/redeal`
  (PEP 508 direct-reference syntax, works with `pip install -r`).
- `environment.yml`: same swap in the `pip:` section.
- `setup_env.sh`: added a `git` on-PATH prerequisite check (needed for the git
  install), and changed the per-package failure hint to point at
  `pip install -r requirements.txt` instead of a bare `pip install redeal`
  (which would just re-hit the nonexistent PyPI package).
- `README.md`: added a note that `redeal` comes from GitHub and `git` must be
  available.

**Also fixed in the same pass — per-package import diagnostics.** The original
`setup_env.sh` verified dependencies with a single combined import statement
(`import redeal, endplay, pydantic, ...`), which fails opaquely on the *first*
missing package and hides the rest. It now loops over each package
individually, printing `OK <pkg> <version>` or `FAIL <pkg>` per line, collects
*all* failures, and reports a complete missing-package list at the end instead
of stopping at the first one.

**Still true / worth knowing when running setup for real:**
- `setup_env.sh` needs outbound network access to `github.com` to clone
  `redeal` — a restricted/offline environment will fail this step even with
  the fix applied.
- `redeal` bundles Bo Haglund's DDS solver as a git submodule, but this project
  only uses `redeal` for *dealing and hand evaluation* — actual double-dummy
  scoring goes through `endplay`, so a finicky DDS submodule build inside
  `redeal` should not block the eval pipeline.

---

## 7. Follow-up: Configurable Ollama "Thinking" Mode

Some Ollama-served models (qwen3, deepseek-r1, gpt-oss, etc.) support a
reasoning/"thinking" toggle on the request. This was previously not exposed at
all — `LocalLLMClient` always called the endpoint the same way regardless of
whether the served model supported or defaulted to thinking mode, which (a)
couldn't be disabled for speed/cost, and (b) risked broken JSON parsing if a
backend ever inlined reasoning into the response content.

**Change:** added `Config.think: Optional[bool] = None`, tri-state:

- `None` ("auto") — the `think` field is **omitted from the request entirely**,
  so servers that don't recognize it (vLLM) never see it.
- `True` / `False` — forwarded to the OpenAI-compatible endpoint via
  `extra_body={"think": ...}`, which Ollama reads as a top-level extra
  parameter.

Wired through:
- `src/config.py` — the new field.
- `src/harness/llm_client.py._raw_call()` — only adds `extra_body` when
  `think is not None`; `_parse_bid()` now strips any inline
  `<think>...</think>` block before extracting JSON, as a defensive measure
  in case a backend inlines reasoning into `content` regardless of the
  request setting.
- `src/harness/llm_cache.py` — cache key extended to
  `(model, temperature, think, prompt)`, so toggling the flag can never
  silently serve a response generated under the other setting.
- `run_eval.py` — new `--think {auto,on,off}` CLI flag, default `auto`.
- `README.md` — documented under "Thinking mode (`--think`)".

Recommended default for this project: `off`. Bidding doesn't benefit from
extended reasoning latency, and `BridgeBid.thinking` already captures a
one-line rationale in the structured output itself.

Added 8 new tests (`tests/test_llm_cache.py`, `tests/test_llm_client.py`) —
all offline, using a fake OpenAI client/cache to verify the `extra_body`
wiring and the `<think>` tag stripping without touching a network. Full suite
is now 46 tests, all passing.

---

## 8. Follow-up: Game-Sequence Detail Capture (`--detail`)

The DDS rollout (§7's sibling concept, Phase 3) computes each line's complete
auction, settled contract, and double-dummy score internally — but
`evaluate_with_dds` only ever returned the accept/reject boolean, discarding
that detail. There was no way to see "what actually happened" in a rolled-out
game.

**Change:** split the rollout logic into `dds_details()` (returns the full
comparison: `llm_auction`, `expert_auction`, `llm_contract`/`expert_contract`
via new `format_contract()`, both NS scores, `imp_delta`, `acceptable`) with
`evaluate_with_dds()` now a thin wrapper that just reads `["acceptable"]`.
`evaluate_dataset()` gained a `detail: bool` parameter: when true, every result
also carries `dealer`/`vulnerability`/`current_bidding`, and any rolled-out
position carries a `dds` block with the full sequences above. `--detail`
exposed on both `eval_local` and `scenario_test`; best paired with
`--results-json`. Added 4 tests (`tests/test_detail.py`); suite reached 50.

---

## 9. Follow-up: Diagnosing and Fixing "Always Pass"

Live-tested against the user's actual running Ollama (`localhost:11434`,
models `llama3:8b` and `qwen3.6:27b` installed). This surfaced a real bug:

**Root cause found:** `Config.model` defaulted to `"llama3"`, but Ollama had
only `"llama3:8b"` installed — Ollama does **not** alias an untagged name to
an installed tag, so every request 404'd. `LocalLLMClient.get_bid` never
raises (by design, so one bad position doesn't crash a whole run) — it caught
the 404 and fell back to `Pass` for every single position, producing a report
that *looked* like real results but was 100% silent fallback. Confirmed by
deliberately reproducing it: `model="llama3"` → every call
`NotFoundError: model 'llama3' not found` → `Pass`. With the correct tag
`"llama3:8b"`, the same prompts returned real structured bids (`1NT`, `2C`,
etc.) — the pipeline itself was sound; the config default was wrong for this
machine.

A second, separate risk was also identified: the single generic fallback
message (`"parse/transport error"`) made transport failures, parse failures,
and FSM rejections indistinguishable, and `evaluate_dataset` discarded
`BridgeBid.thinking` entirely — so even with the model fixed, a *future*
all-Pass run would have been just as opaque to diagnose.

**Fixes applied:**

- `src/config.py` — default `model` changed `"llama3"` → `"llama3:8b"`
  (matches what's actually installed); docstring now warns that Ollama tags
  are mandatory and not aliased.
- `src/harness/llm_client.py`:
  - `get_bid` now catches the transport call and the parse call separately,
    tagging `thinking` with a distinct prefix per failure mode:
    `"transport error: ..."`, `"parse error: ..."`, or
    `"illegal call '<bid>' -> Pass"`. Still never raises.
  - New `verify_connection()` — lists models from the endpoint via
    `client.models.list()` and checks `config.model` is present, raising a
    `RuntimeError` with the full available-models list and an Ollama-tagging
    hint if not. Intended as a one-time preflight before evaluating a dataset.
- `src/evaluation/metrics.py`:
  - New `classify_fallback(thinking) -> str` mapping the prefixes above to
    `"transport_error"` / `"parse_error"` / `"illegal_call"` / `"none"`.
  - `evaluate_dataset` now captures the full `BridgeBid` (not just `.bid`),
    attaches `fallback_reason` to **every** result (not gated by `detail`),
    and (when `detail=True`) also attaches `llm_thinking` — the literal error
    message. The summary gained `fallback_counts` (dict of the four
    categories) and `fallback_pass_rate`.
- `run_eval.py`:
  - `cmd_eval_local` and `cmd_scenario_test` both call
    `client.verify_connection()` immediately after constructing the client,
    before the per-record loop; a `RuntimeError` is printed to stderr and the
    command exits `1` — confirmed live to fail in ~0.6s instead of grinding
    through the dataset.
  - `print_summary` now prints `fallback_counts`/`fallback_pass_rate` and a
    one-line hint to rerun with `--detail` when the rate is non-zero.

**Verified live** (not just unit tests): `verify_connection()` correctly
passed against the real endpoint with the corrected default, correctly raised
with the old `"llama3"` name (listing the two real installed models), and the
CLI fail-fast path was timed at ~0.6s. A full `eval_local --detail` run against
`llama3:8b` on a 4-position toy dataset caught a **genuine live parse failure**
mid-run — the model returned a bid of `"2"` for one seat, which surfaced as
`fallback_reason: "parse_error"`, `llm_thinking: "parse error: unrecognized
call: '2'"` — exactly the diagnosis this fix exists to provide, on real model
output, not a contrived test case.

Added 8 new tests (`tests/test_llm_client.py`: fallback-reason
differentiation + `verify_connection` success/missing-model/dead-endpoint;
`tests/test_eval_loop.py`: `classify_fallback` + aggregation). Suite now 58
tests, all passing.

---

## 10. Follow-up: Dual-backend Support (Ollama native + vLLM)

### Discovery

Live-tested with `qwen3.6:27b` against both Ollama endpoints simultaneously:

- **Native `/api/chat`** with `think:false` → `message` has no `thinking` field
  (reasoning suppressed correctly).
- **OpenAI-compat `/v1/chat/completions`** with `think:false` → `message.reasoning`
  still present and non-empty (reasoning runs regardless of the flag).

Root cause: Ollama's `/v1` adapter does not translate the `think` field into
its internal inference control for thinking models. The **`extra_body={"think":
false}`** the code was already sending had no effect — qwen3.6:27b (and other
thinking models) reasoned on every call, paying full latency without the
reasoning being used. The `message.reasoning` field the server returned was
also silently dropped (code only read `choices[0].message.content`).

vLLM speaks only `/v1` and has no native Ollama API, so a single code path
cannot handle both correctly.

### Changes

**`src/config.py`**
- New constant: `OLLAMA_NATIVE_BASE_URL = "http://localhost:11434"`
- New `Config` field: `backend: str = "ollama"  # "ollama" | "vllm"`
- Default `base_url` changed from `OLLAMA_BASE_URL` (`/v1`) to
  `OLLAMA_NATIVE_BASE_URL` (no `/v1` suffix)
- `__post_init__` validates `backend` is `"ollama"` or `"vllm"`, mirroring the
  existing `threshold_mode` check

**`src/harness/llm_client.py`**
- Added `import requests as _requests`
- `_raw_call` is now a dispatcher:
  ```python
  if backend == "vllm"   → _raw_call_vllm   (OpenAI SDK, unchanged behaviour)
  if backend == "ollama" → _raw_call_ollama  (requests.post /api/chat)
  else                   → ValueError
  ```
- `_raw_call_ollama`: POSTs to `{base_url}/api/chat` with `format:"json"` and
  `think` field (omitted when `None`); reads `message.content` for the answer
  and merges `message.thinking` into `BridgeBid.thinking` when the model's own
  JSON didn't fill it
- `verify_connection` also branches: vLLM uses OpenAI SDK `models.list()`;
  Ollama uses `GET {base_url}/api/tags` and checks `models[].name`

**`run_eval.py`**
- New `--backend {ollama,vllm}` CLI flag (default `"ollama"`)
- `build_config` auto-selects `OLLAMA_NATIVE_BASE_URL` or `VLLM_BASE_URL` when
  `--base-url` is not explicitly provided; user override still works

**`requirements.txt`**
- Added `requests>=2.28.0` (Ollama native path; was present in the conda env
  as a transitive dep but not pinned)

### Tests

`tests/test_llm_client.py` fully rewritten (7 old tests → 18 tests):

- 3 vLLM tests confirm `extra_body` wiring (renamed with `test_vllm_` prefix,
  now explicitly set `backend="vllm"`)
- 3 Ollama tests confirm `think` field presence/absence in the native payload
  (mock `requests.post` — no network)
- 1 Ollama test confirms server-side reasoning is surfaced into `BridgeBid.thinking`
- Fallback-reason tests (`transport`, `parse`, `illegal_call`) switched to
  Ollama path via `mock.patch` on `requests.post`
- `verify_connection` tests split: 3 for vLLM (OpenAI SDK mock), 3 for Ollama
  (`requests.get` mock)

Full suite: **65 tests, all passing** (up from 58).

### Verification performed

```bash
# Confirmed think=False suppresses reasoning via native path (no 'thinking' key)
# Confirmed get_bid returns correct bid with no reasoning overhead
# Confirmed bad --backend rejected at startup (ValueError + argparse)
```

---

## 11. Follow-up: Prompt-Quality Remediation + Error Analysis

### Context

A 10-deal live eval of **`gemma4:26b`** (`--think off`; default model changed
from `llama3:8b` → `gemma4:26b` this session) surfaced a 15% fallback-to-Pass
rate with three distinct failure modes:

- **3 illegal calls** — the model tried to *open* (1NT/1S) when partner had
  already opened; it did not realize it was *responding*. `BiddingFSM` caught
  these and turned them into `Pass`.
- **3 parse errors** — the model emitted a valid JSON object followed by
  trailing prose or a second object; `json.loads` failed with "Extra data".
- **3 light openings** — judgment (opened on 10–11 HCP).

A pasted multi-phase remediation plan was pressure-tested against this data.
Its CoT-math track was rejected (observed failures were *not* arithmetic, and
we run think=off), and its infra-vendoring and BBO-oracle tracks were
de-scoped. Only the **Prompt Quality** track was pursued.

### Changes

**`src/bridge.py`** — new `classify_shape(shape) -> "Balanced" | "Two-suited"
| "Unbalanced"`, pure string logic over the existing `compute_shape` output.

**`src/harness/prompt_builder.py`** — all in `_render`, so both `build_prompt`
and the rollout's `build_prompt_parts` benefit:
- **Feature injection:** each suit rendered with explicit length + spaced cards
  (`♠ 5 cards (A K 7 5 3)`), plus a `Distribution: 5-3-3-2 · Balanced · LTC 8`
  line, all derived from the active seat's own hand (masking intact).
- **Auction roles:** new `annotate_auction(history)` labels prior calls by
  table role (RHO/Partner/LHO, back-counted from the active seat — no dealer
  needed) and emits a one-line summary ("Partner opened 1NT. You are
  RESPONDING, not opening."). Directly targets the illegal-call failures.
- **`SYSTEM_INSTRUCTION`** now says "output exactly one JSON object, no markdown,
  nothing before or after."

**`src/harness/llm_client.py`** — `_parse_bid` replaced the greedy
`_JSON_RE = \{.*\}` match with `json.JSONDecoder().raw_decode()` from the first
`{`, which parses the first complete object and ignores trailing content —
fixing the "Extra data" parse errors. `_JSON_RE` removed (now unused).

### Regression result (clean run, warm cache)

| Metric | Before | After |
|---|---|---|
| Fallback-to-Pass | 15.0% | **0.0%** |
| — parse_error | 3 | **0** |
| — illegal_call | 3 | **0** |
| DDS-acceptable | 80.0% | 80.0% |
| Exact accuracy | 75.0% | 65.0% |

Both targeted failure modes were fully eliminated. Exact accuracy *dropped* 10
pts — but this is **not a regression**; it exposed the toy oracle (see below).
Quality (DDS-acceptable) held flat.

> Note: interim numbers seen via the background Monitor (77.5%, parse=1,
> illegal=1) were unreliable — several stale `run_eval.py` processes were racing
> on a half-populated cache. The clean single-process run above is the trusted
> result.

### Error analysis (14 non-exact positions, post-fix)

With fallbacks at zero, every miss is now a real bid the model chose. They sort
into three buckets:

- **Bucket A — model correct, oracle wrong (5/14):** b8-N (16 HCP 5-4 → 1H),
  b9-N (13 HCP 5-5 → 1H), b6-S (15 HCP 6-4 → 1D), b5-N (14 HCP 6♣ → 1C), b1-E
  (10 HCP 6-4, Rule-of-20=20 → 1S). 1NT requires a balanced hand; the model
  correctly refuses it on these shapes. Three of them (b1-E +13, b9-N +5, b8-N
  +2 IMP) actually **beat** the heuristic on DDS yet count as failures.
- **Bucket B — genuine model errors, light openings (5/14):** b1-S (7 HCP → 1H,
  −830 on DDS), b10-E (10 HCP, factually wrong reasoning about 1NT range), b2-N
  (8 HCP → 1C), b10-W (8 HCP → 1C), b2-W (11 HCP → 1D). The model opens
  sub-Rule-of-20 hands.
- **Bucket C — borderline / DDS-neutral (4/14):** b1-W (19 → 2NT), b2-E, b2-S,
  b6-E — style differences within ~1 IMP.

### Root causes (ranked)

1. **The toy `expert_bid` heuristic is now the dominant error source.** It is
   shape-blind, so it penalizes the model for good bridge. Exact-accuracy
   against it is actively misleading → strongest case yet for a real/BBO oracle.
2. **Light-opening bias** is the model's one real weakness (~5 overbids on
   7–11 HCP hands). Prompt-addressable: inject the **Rule of 20** or a stated
   minimum-to-open, the same way shape features fixed the 1NT-on-unbalanced
   problem.
3. **DDS metric is symmetric when it shouldn't be.** `evaluate_with_dds`
   accepts iff `|IMPΔ| ≤ threshold`, so a *superior* contract (b1-E +13, b9-N
   +5) scores identically to a blunder. An asymmetric rule ("accept if within
   threshold *or better*") would surface Bucket A correctly.

### Tests

Added 11 tests (65 → **76**, all passing): `classify_shape` cases in
`test_bridge.py`; feature-injection + `annotate_auction` role cases (opening /
responding-to-partner / opponent-opened) in `test_prompt_masking.py`; and
`_parse_bid` trailing-data / leading-prose / no-object cases in
`test_llm_client.py`.

---

## 12. Follow-up: Closing the Accuracy Gap vs bridge-llm-bench

### Context

On 25 Ben-SAYC benchmark positions (converted from the sibling
`bridge-llm-bench` project, which shares the oracle), gemma4:26b sat at 52%
baseline / 56% with a SAYC-knowledge block, while the benchmark's Gemini Flash
Lite journey reached 70.7% (N=150) and a headline 80% (N=50 + voting). The
benchmark's ablation data dictated the porting strategy: **removing all
examples cost it −29.3pts; removing all rules cost −0.7pts.** Three rule
blocks (competitive bidding, takeout-X response, "when not to compete")
measurably *hurt*.

Two integrity findings about the benchmark itself:

- **Its example block leaks test-set deals.** P22's examples include
  `S:Q52 H:6543 D:K732 C:AJ | P P 1S 1NT → X` — literally the bench-0/bench-4
  deal. Its own log flags P21 as "INVALID — hardcoded test hands", but P22
  kept several. Part of its headline number is leakage.
- Decision here: **generic examples only**, with
  `tests/test_prompt_examples.py::test_no_example_hand_appears_in_benchmark_data`
  as a permanent fence (it normalizes hand formats and checks every example
  hand against both benchmark CSV/JSONL sets).

### The bug that mattered: auction roles had period 3, not 4

`annotate_auction` back-counted roles as `RHO, Partner, LHO` repeating — but a
bridge auction rotates **four** seats; the model's own earlier calls are in
the history. Every call 4+ positions back was mislabeled, including **who
opened** — the one fact the summary line asserts. On bench25, 12 of 25
positions have 4+ call auctions, and the mislabeling precisely explains the
observed misses: bench-8 (partner opened 1S → labeled "LHO opened" → model
passed instead of raising to 3S), bench-9 (partner's 1NT overcall labeled
LHO → passed instead of 3NT), bench-18 (the model's OWN 1H opening labeled
"RHO opened 1H"), bench-19 (RHO's opening labeled "partner opened").

Fix: `_ROLE_CYCLE = ("RHO", "Partner", "LHO", "You")`, modulo 4, plus a
dedicated summary branch ("YOU opened 1H earlier — you are the OPENER choosing
a rebid"). This alone moved bench25 from 52% → 60% with the lean base prompt.

### Prompt styles (Config.prompt_style / --prompt-style)

The env-flag knowledge toggle became a first-class tri-state knob:

- `base` — hand features + auction roles (the previous default).
- `knowledge` — + SAYC reference guide (ported verbatim from the benchmark).
- `examples` (new default) — + the two ablation-kept rule blocks (penalty
  doubles, 5-level, opening suit choice) + 17 generic few-shot examples
  covering the observed error buckets: light-opening restraint (Rule of 20),
  weak-2 suit quality, competitive doubles, transfers over partner's 1NT
  overcall, competitive jump raises, 3NT-with-stopper over overcalls, jumps
  after partner's takeout X, penalty-pass discipline. Each example hand is
  validated by tests for 13 cards and correct HCP annotation.

### Results (Gemma4:26b, --think off, temp 0, exact match, no DDS)

| Config | bench25 | bench150 |
|---|---|---|
| old baseline (period-3 roles) | 52% | — |
| old + SAYC knowledge (env flag) | 56% | — |
| old + vote k=9 t=0.5 | 48% | — |
| `base` (role fix only) | 60% | — |
| `knowledge` | 60% | — |
| `examples`, no retry | 76%¹ | 63.3%¹ |
| **`examples` + retry-on-illegal (default)** | **72%** | **62.0%** |
| `examples` + retry + vote k=9 t=0.5 | 72% | — |
| bridge-llm-bench Flash Lite reference | — | P22 68.7% / P20 70.7% |

Majority voting adds exactly nothing on top of the examples prompt (72% →
72%) — replicating the benchmark's own finding that voting only rescued its
*weak* prompt (P18+vote 80% vs P20+vote +0). With a strong prompt the model's
errors are systematic, not noisy, so sampling cannot vote them away.

¹ Inflated by *accidentally correct* Passes: the FSM converts an illegal
(insufficient) bid straight to Pass, and on several positions Pass happened to
be the oracle call even though the model had chosen to overbid (3 such
positions on bench25, 3 net on bench150). The retry numbers are the honest
ones — every scored call is one the model actually chose and was legal.

### Retry-on-illegal (`Config.retry_illegal`, `--no-retry-illegal`)

10.7% of bench150 positions had the model choosing an *insufficient* bid
(e.g. 2D when the auction stood at 2S) that the FSM silently turned into
Pass. Rather than stating legality rules up-front (see failed refinement
below), `get_bid` now re-asks **once** with the rejection appended ("your
previous answer '2D' is an ILLEGAL call here — the auction stands at 2S ...").
A second failure falls back to Pass exactly as before. Because the corrective
turn only runs on positions that were already lost, it cannot perturb
legal-call behavior. Result: illegal-call fallbacks 14 → 0 on bench150,
4 → 0 on bench25.

### The refinement that failed (kept for the record)

A "Legality: the auction stands at 2S — bid higher or Pass" line was added to
fix 4 illegal-call fallbacks (insufficient bids silently FSM'd to Pass). It
**regressed bench25 76% → 60%**: the model over-anchored, re-opening light
hands ("any bid from 1C up is legal" read as an invitation) and passing
competitive positions. Reverted; a warning comment sits at the site in
`prompt_builder.py`. Same lesson as the benchmark's P21: with a small model,
each added instruction is a behavioral lever, not documentation — never add
prompt text without re-measuring.

### Remaining bench25 misses (retry-era, 7/25)

- bench-5 (2H vs 2D): bids the 5-card suit naturally instead of transferring
  over partner's 1NT overcall.
- bench-6 (3C vs Pass): real overbid, previously hidden behind an FSM Pass.
- bench-8/9 (Pass vs 3S/3NT): reasoning identifies the layout correctly but
  stays conservative below game.
- bench-18/19 (2NT vs 3C, 2D vs 3D): right read of the auction, wrong rebid
  choice / no jump.
- bench-23 (Pass vs 3D): declines to compete at the 3-level.

The failure profile is now *judgment* (conservatism, transfer discipline),
not mechanics (roles, parsing, legality) — the same profile the benchmark hit
after its example stage.

---

## 13. Follow-up: Deal Reconstruction, DDS Quality Metric, and Two Negative A/Bs

### Error analysis that drove this (bench150, examples+retry, 57 misses)

- Openings nearly solved: 84% exact (21/25); competitive seats 58% (72/125).
- 74% of competitive misses were **under-competing** (27 passes + 12 timid
  underbids of 53) — June's over-bidding got over-corrected into passivity.
- Convention clusters: splinters (5 misses), strong-2C continuations (4), one
  long competitive deal at 11/14.
- Exact-match noise: auctions are seeded from **WBridge5** bids (~44%
  SAYC-agreement per the benchmark's readme) and Ben answers them with
  sometimes-implausible calls (a 5-HCP hand scored wrong for not bidding 6H).
  Estimated 5–10 of 57 misses punished the sounder call.

### Deal reconstruction (scripts_convert_bench.py)

The bench CSVs list positions deal by deal, rotating through the seats — each
deal's first four rows are the four hands. The converter now groups rows into
deals (new deal = empty auction), rebuilds `all_hands`/`deal_pbn` via the
existing `to_pbn_hand`/`_deal_to_full_pbn` helpers, and validates 52 unique
cards per deal (all 14/14 pass; the 25-set's truncated last deal falls back to
exact-match only with a warning). This unlocks rollout + double-dummy scoring
on the benchmark sets for the first time. Dealer is assumed N, vulnerability
None — not recorded in the CSVs.

### Asymmetric DDS rule (`Config.dds_rule`, default `asymmetric`)

Closes SESSION_LOG open item #3. The trap: `imp_delta` is **NS-perspective**,
so "model better" must be re-oriented by the acting seat
(`seat_sign = +1 for N/S, −1 for E/W`; `model_gain = seat_sign * imp`) or the
rule inverts for E/W bidders. Acceptable ⇔ `model_gain >= -threshold_n`.
`--dds-rule symmetric` preserves the legacy `|delta| <= n` for comparison;
`model_gain_imp` is reported in `--detail` output. Covered by fake-endplay
tests including the seat-orientation case (`tests/test_dds_path.py`).

### Two negative A/Bs (prompt examples — both REJECTED)

Per the error analysis, four examples were added and gate-tested (keep only if
bench25 holds AND bench150 exact improves):

| Variant | bench25 | bench150 | Verdict |
|---|---|---|---|
| committed block (baseline) | 72% | 62.0% | — |
| + 2 anti-passivity + 2 convention | 56% | 59.3% | rejected |
| + 2 convention only (splinter, 2C) | 68% | not run | rejected |

The anti-passivity examples ("don't sell out with a fit") made the model
over-compete on unrelated positions — a 2C overcall on a flat 12, pulling a
penalty double. The convention examples regressed bench25 even though
splinter/2C sequences never occur there: **any added example shifts behavior
globally, not just on its target pattern.** Both reverted; a warning comment
sits at the site in `prompt_builder.py`. This is the third confirmation of
the P21/legality-line law, now measured per-group. The passivity bucket is
not prompt-addressable for this model.

### Results: exact vs quality on bench150 (Gemma4:26b, examples+retry)

| Metric | bench150 |
|---|---|
| Exact match | 62.0% (93/150) |
| DDS-acceptable, symmetric (within 1 IMP) | 69.3% (104/150) |
| **DDS-acceptable, asymmetric (within 1 IMP or better)** | **82.0% (123/150)** |

Split of the 57 exact-misses by rolled-out contract quality:

| Bucket | Count |
|---|---|
| model's line **better by >1 IMP** | **19** |
| within ±1 IMP (inconsequential) | 11 |
| worse by >1 IMP (real errors) | 27 |

**A third of the "misses" are the model outbidding the oracle.** Top cases:
bench-73 Pass → 4S-by-them beats oracle's failing 5H (+6 IMP); bench-9's
"conservative" Pass beats the oracle's 3NT (+5); bench-85/123's lower partial
beats the oracle's failing game (+5 each). The under-competing bucket from the
§13 error analysis was therefore substantially **oracle noise, not model
weakness** — which also explains why the anti-passivity examples failed their
A/B: they pushed the model toward a noisy target. Real errors (worse by
>1 IMP) are 27/150 = 18% of positions.

Exact-match against this WBridge5-seeded oracle understates the model by
~20 points. The honest headline for Gemma4:26b on Ben-SAYC-150 is
**82% quality-acceptable / 62% oracle-exact**, and metric work (this section)
mattered more than another round of prompt work would have.

---

## Open Items for Future Work

1. **Replace the placeholder `expert_bid` heuristic** (§4,
   `src/data/mock_generator.py::heuristic_call`) with real expert or BBO-derived
   reference bids — now the single biggest gap (see §11 root cause #1).
2. **Add Rule-of-20 guidance to the prompt** to curb the light-opening bias
   (§11 root cause #2).
3. **Make the DDS acceptability rule asymmetric** so a model that beats the
   expert is not scored as a failure (§11 root cause #3).

---
name: gomoku-ai-debug
description: Diagnose and fix Gomoku engine playing-strength regressions from game records and suspicious moves. Use when requests mention records, YiXin losses or analysis, self-play anomalies, missed wins or blocks, candidate generation, threat evaluation, PVS, VCF/VCT, ProofSearch, root review or safety, search budgets, or structural regression positions and tests. Respect diagnosis-only requests and do not edit code until authorized. Do not use for generic UI, packaging, documentation-only, launcher-only, or unrelated refactoring work.
---

# Gomoku AI Debugging

Use records as evidence to find the earliest mechanism-level failure. Improve the
general class of positions, never one filename, sequence, hash, or coordinate.

## Protect Project State

1. Inspect Git status, local instructions, relevant code and tests before acting.
2. Preserve all unrelated tracked and untracked user files.
3. Treat the current contents of `records/` as a strict retention boundary:
   - never restore, regenerate, copy back, or re-add deleted or moved records;
   - never rewrite a record during diagnosis;
   - never stage record or analysis output unless the user explicitly requests it.
4. Put any historical regression position in a minimal fixture under
   `tests/positions/`, or construct the board directly in a test.
5. Discover the current branch, version, runtime and configuration each time.
   Do not encode a standing branch, release number, commit, Python path or test
   count in the diagnosis.

## Respect the Requested Scope

- For analysis, diagnosis, review or route-planning requests, remain read-only.
- For change or fix requests, reproduce and diagnose before editing production
  logic.
- Do not run a real match, create analysis output, change settings, commit or push
  unless the request or established project workflow authorizes that action.
- State what remains uncertain instead of silently expanding the task.

## Establish Reliable Evidence

1. Select the latest relevant record using timestamps and record metadata, or use
   the exact record named by the user.
2. Pair an analysis file with its exact `source_path`; do not guess by filename
   similarity alone.
3. Verify rules, board size, engines, colors, depth, time limits and move-count
   convention.
4. Treat external-engine evaluation as bounded evidence, not unquestionable truth.
   Check completed depth, alignment, paired-child completion and evaluation
   volatility before calling a move decisive.
5. If the oracle changes mate/non-mate conclusions across adjacent positions,
   recommend longer repeated analysis before patching to that conclusion.

## Find the Earliest Substantive Mistake

1. Replay the game and reconstruct the position immediately before each plausible
   turning point, including side to move and ordered move history.
2. Separate:
   - a forced consequence of an earlier loss;
   - a candidate omission;
   - incorrect threat classification;
   - PVS horizon or score propagation failure;
   - VCF/VCT or ProofSearch incompleteness;
   - search-budget exhaustion;
   - root-review or fallback selection failure;
   - diagnostic/provenance inconsistency.
3. Prefer the earliest move whose correction could change the later trajectory.
4. Confirm the reported engine move under comparable settings when practical.
5. Compare plausible alternatives relationally; avoid assuming one oracle move is
   uniquely correct without proof.

## Trace the Decision Pipeline

For every serious candidate, answer these questions in order:

1. **Generated:** Was the move present in the legal and root-relevant pools?
2. **Classified:** What own/opponent threat profile and frontier kind did it have?
3. **Sourced:** Which candidate source admitted it: ordinary, mandatory defense,
   frontier, prevention, counterattack, expansion or bridge?
4. **Retained:** Did a cap, mode switch, whitelist, pruning rule or stale scan remove
   it?
5. **Searched:** What depth, window, extension, score and principal variation did it
   receive?
6. **Verified:** Did VCF/VCT, ProofSearch and root safety return proven, disproven or
   unknown? Preserve UNKNOWN semantics.
7. **Selected:** Did root review compare it, and did a fallback override stronger
   tactical evidence?
8. **Recorded:** Do selected move, candidate provenance, proof result and reason
   agree? Treat disagreement as a separate observability defect.

Inspect only the components relevant to the evidence. Common project locations are:

- `engine/search.py` for orchestration, PVS, budgets and final selection;
- `engine/root_candidates.py` for candidate modes, sources and caps;
- `engine/root_review.py` and `engine/root_safety.py` for finalist arbitration;
- `engine/proof_search.py` for proven/disproven/unknown behavior;
- `engine/threats.py` and `engine/evaluator.py` for tactical semantics;
- `engine/time_manager.py` for deadlines and reserved budgets;
- `engine/native_core.py` and `native/` for accelerated/reference parity;
- `arena.py`, `cvc_analysis.py` and `engine/records.py` for recorded evidence.

## State the Root Cause Before Editing

Write the diagnosis in this form:

> When **[structural condition]** occurs, **[component]** violates
> **[expected invariant]**, causing **[incorrect candidate/search behavior]** and
> making the root prefer **[class of inferior moves]**.

Do not modify production logic until this statement is supported by a reproducible
position, record fields, code paths or focused instrumentation.

## Implement a General Fix

1. Restore the violated invariant with the smallest reversible change.
2. Never special-case a coordinate, exact sequence, record filename, position hash
   or one-off version.
3. Do not broadly tune parameters until evidence shows the defect is parameter-level.
4. Preserve UNKNOWN as unknown; never reinterpret budget exhaustion as safe, false,
   proven or disproven.
5. Keep candidate membership structurally stable; use history, killers and TT hints
   for ordering unless correctness proves they must affect membership.
6. Keep diagnostic provenance synchronized when expansion or filtering changes the
   effective root set.
7. Remove disposable instrumentation before delivery unless it adds durable,
   bounded audit value.

## Add Structural Regression Coverage

1. Make the old mechanism fail and the corrected invariant pass.
2. Prefer assertions such as:
   - a critical class of defensive move remains in the root set;
   - a threat kind or source is preserved;
   - UNKNOWN does not become a definitive conclusion;
   - an expanded candidate retains provenance;
   - a serious challenger reaches root review;
   - a proven unsafe move is rejected.
3. Avoid exact scores unless score encoding itself is the invariant.
4. Use exact-move assertions only when the position proves that move or a defined
   equivalent set is required.
5. Add a negative or boundary assertion when it prevents an overbroad fix.

## Validate Proportionately

1. Locate a working Python interpreter; verify it instead of trusting a WindowsApps
   alias.
2. Run the focused regression first.
3. Run the relevant subsystem tests next.
4. Run the complete suite when appropriate:

   `python -B -m unittest discover -s tests -p "test_*.py" -v`

5. Report the exact runner count, failures, errors and skips from current output.
6. If candidate breadth, Proof/VCF frequency, pruning or NativeCore changes, run the
   relevant benchmark or parity test once and report exact results.
7. Do not claim playing-strength improvement from unit tests alone. Request a fresh
   self-play or YiXin game when real-game evidence is still needed.

## Maintain Performance Discipline

- Measure candidate-count, node-count and elapsed-time impact when widening search.
- Prefer source-aware bounded widening over globally increasing a cap.
- Diagnose low proof reuse before caching deadline-censored UNKNOWN results.
- Optimize native hotspots only after candidate and search semantics are correct.
- Do not repeatedly run long matches when a focused position answers the question.

## Report in Evidence Order

Report:

1. record and earliest substantive mistake;
2. reconstructed position and stronger behavior;
3. mechanism-level root cause and supporting pipeline evidence;
4. changed files and invariant restored, if a fix was authorized;
5. structural regression coverage;
6. every validation command and exact current result;
7. remaining oracle, performance and real-game uncertainty;
8. the next replay or benchmark that best tests the correction.

Never declare the engine issue resolved solely because one historical coordinate now
wins or because unit tests pass.

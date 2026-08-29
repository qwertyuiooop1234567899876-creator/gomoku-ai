# Native main-search ABI

`gn_main_search_v1` is the coarse boundary for the main PVS/Negamax migration.
It deliberately crosses Python/C++ once per root search rather than once per
node.

The request fixes these semantics before the implementation is enabled:

- board cells plus the complete ordered move history;
- side to move and ordered root candidates supplied by Python root policy;
- fixed depth and optional fixed node limit (`0` means unlimited);
- branch width, preselection, local radius, recent-history width and threat
  extension depth;
- PVS and TT feature flags.

The result reserves stable fields for completion state, stop reason, chosen
move, root scores, score, node count, PV, TT entry count and TT digest. Every
structure carries `struct_size` and `schema_version`; future fields must be
appended, not reordered.

V0.17.0 Phase 1 implements the fixed-depth C++ core behind this boundary. It
returns root scores in request order plus the selected move, score, PV, node
count, TT entry count and a deterministic 64-bit TT-content digest. A node
limit returns `STATUS_CUTOFF` without presenting the interrupted depth as
completed. Tests compare the C++ result with a Python oracle using the same
portable TT key; they lock score, PV, nodes, entry count and the complete
canonical TT digest rather than checking only the selected move.

The entry point remains a probe/benchmark capability. Nothing in
`SearchAI.choose_move` calls it in Phase 1, so production root policy, Proof,
VCF and review scheduling are unchanged. The same ABI already supports a
future full-window review caller by clearing the PVS flag and selecting a
larger `threat_extension_depth`; production integration requires a separate
phase and its own fallback/timeout gate.

Adding this symbol does not bump the existing kernel ABI: ABI 1 runtimes that
predate `gn_main_search_v1` remain usable for the proven VCF/profile kernels.
The Python wrapper reports `main_search_available=false` until a compiler has
produced a runtime containing the optional symbol. Older ABI-1 runtimes still
fall back cleanly because the symbol remains optional at load time.

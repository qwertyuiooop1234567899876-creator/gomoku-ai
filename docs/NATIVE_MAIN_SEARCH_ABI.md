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

The first implementation intentionally returns `GN_MAIN_SEARCH_UNSUPPORTED`
after validating the request and computing an input digest. This makes ABI
marshalling, history order and configuration part of the test gate without
changing production move selection. The C++ kernel may only replace that
status after it matches the Python oracle's score, PV, node semantics and TT
digest on fixed-depth/fixed-node cases.

Adding this symbol does not bump the existing kernel ABI: ABI 1 runtimes that
predate `gn_main_search_v1` remain usable for the proven VCF/profile kernels.
The Python wrapper reports `main_search_available=false` until a compiler has
produced a runtime containing the optional symbol.

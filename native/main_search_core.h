#pragma once

#include "gomoku_native.h"

namespace gomoku_native {

// Execute one isolated, fixed-depth search.  This Phase-1 entry point has no
// persistent state and is intentionally not wired into SearchAI::choose_move.
int run_main_search_v1(
    const GNMainSearchRequestV1& request,
    GNMainSearchResultV1& result
);

}  // namespace gomoku_native

#pragma once

#include <cstdint>

#if defined(_WIN32)
#define GN_API extern "C" __declspec(dllexport)
#else
#define GN_API extern "C" __attribute__((visibility("default")))
#endif

// Coarse-grained main-search ABI.  Every structure carries its byte size and
// schema version so fields may be appended without silently changing older
// callers.  Coordinates are encoded as row * board_size + column.
constexpr std::uint32_t GN_MAIN_SEARCH_SCHEMA_V1 = 1;
constexpr std::uint32_t GN_MAIN_SEARCH_FLAG_PVS = 1U << 0U;
constexpr std::uint32_t GN_MAIN_SEARCH_FLAG_TT = 1U << 1U;
constexpr int GN_MAIN_SEARCH_UNSUPPORTED = -2;

struct GNMainSearchRequestV1 {
    std::uint32_t struct_size;
    std::uint32_t schema_version;
    const std::uint8_t* cells;
    std::int32_t board_size;
    const std::int32_t* history_indices;
    const std::uint8_t* history_players;
    std::int32_t history_count;
    std::int32_t player;
    const std::int32_t* root_candidates;
    std::int32_t root_candidate_count;
    std::int32_t depth;
    std::int64_t node_limit;
    std::int32_t branch_candidate_limit;
    std::int32_t preselection_factor;
    std::int32_t candidate_radius;
    std::int32_t recent_move_count;
    std::int32_t threat_extension_depth;
    std::uint32_t flags;
};

struct GNMainSearchResultV1 {
    std::uint32_t struct_size;
    std::uint32_t schema_version;
    std::int32_t status;
    std::int32_t completed_depth;
    std::int32_t stop_reason;
    std::int32_t best_move;
    std::int32_t score;
    std::int64_t nodes;
    std::int64_t tt_entries;
    std::uint64_t input_digest;
    std::uint64_t tt_digest;
    std::int32_t* root_scores;
    std::int32_t root_score_capacity;
    std::int32_t root_score_count;
    std::int32_t* principal_variation;
    std::int32_t pv_capacity;
    std::int32_t pv_length;
};

GN_API int gn_main_search_v1(
    const GNMainSearchRequestV1* request,
    GNMainSearchResultV1* result
);

// Read-only defense-classification ABI.  The caller supplies exact attacker
// continuations; the native kernel only classifies every legal defender reply.
// It does not generate candidates or make a move-selection decision.
constexpr std::uint32_t GN_DEFENSE_CLASSIFICATION_SCHEMA_V1 = 1;
constexpr std::int32_t GN_DEFENSE_CLASSIFICATION_COMPLETE = 1;
constexpr std::int32_t GN_DEFENSE_CLASSIFICATION_CUTOFF = 2;
constexpr std::int32_t GN_DEFENSE_CUTOFF_NONE = 0;
constexpr std::int32_t GN_DEFENSE_CUTOFF_TIMEOUT = 1;
constexpr std::int32_t GN_DEFENSE_CUTOFF_REPLY_LIMIT = 2;
constexpr std::int32_t GN_DEFENSE_BUFFER_TOO_SMALL = -3;

struct GNDefenseContinuationV1 {
    std::int32_t move;
    std::int32_t immediate_win;
    const std::int32_t* winning_points;
    std::int32_t winning_point_count;
};

struct GNDefenseRefutationV1 {
    std::int32_t defense_move;
    std::int32_t continuation_move;
    std::int32_t continuation_is_immediate;
    std::int32_t winning_point_offset;
    std::int32_t winning_point_count;
};

struct GNDefenseClassificationRequestV1 {
    std::uint32_t struct_size;
    std::uint32_t schema_version;
    const std::uint8_t* cells;
    std::int32_t board_size;
    std::int32_t attacker;
    const GNDefenseContinuationV1* continuations;
    std::int32_t continuation_count;
    const std::int32_t* counter_wins;
    std::int32_t counter_win_count;
    // Negative values disable the corresponding cutoff.  Zero is an
    // immediate cutoff, which makes timeout/interruption tests deterministic.
    std::int32_t reply_limit;
    std::int32_t timeout_ms;
};

struct GNDefenseClassificationResultV1 {
    std::uint32_t struct_size;
    std::uint32_t schema_version;
    std::int32_t status;
    std::int32_t cutoff_reason;
    std::int32_t legal_reply_count;
    std::int32_t processed_reply_count;
    std::int32_t coverage_complete;
    std::int32_t analysis_completed;
    std::int32_t* required_defenses;
    std::int32_t required_capacity;
    std::int32_t required_count;
    GNDefenseRefutationV1* refutations;
    std::int32_t refutation_capacity;
    std::int32_t refutation_count;
    std::int32_t* unclassified_replies;
    std::int32_t unclassified_capacity;
    std::int32_t unclassified_count;
    std::int32_t* refutation_winning_points;
    std::int32_t winning_point_capacity;
    std::int32_t winning_point_count;
};

GN_API int gn_classify_defenses_v1(
    const GNDefenseClassificationRequestV1* request,
    GNDefenseClassificationResultV1* result
);

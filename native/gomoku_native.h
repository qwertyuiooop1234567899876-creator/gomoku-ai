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

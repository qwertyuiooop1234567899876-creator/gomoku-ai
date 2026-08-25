#include "gomoku_native.h"

#include <cstddef>
#include <cstdint>

namespace {

constexpr int BLACK = 1;
constexpr int WHITE = 2;
constexpr int STATUS_INVALID = -1;
constexpr std::uint64_t FNV_OFFSET = 1469598103934665603ULL;
constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;

void hash_u64(std::uint64_t& digest, std::uint64_t value) noexcept {
    for (int shift = 0; shift < 64; shift += 8) {
        digest ^= (value >> shift) & 0xFFU;
        digest *= FNV_PRIME;
    }
}

bool valid_request(const GNMainSearchRequestV1& request) noexcept {
    if (request.struct_size < sizeof(GNMainSearchRequestV1)
        || request.schema_version != GN_MAIN_SEARCH_SCHEMA_V1
        || request.cells == nullptr
        || request.board_size < 5 || request.board_size > 25
        || (request.player != BLACK && request.player != WHITE)
        || request.history_count < 0
        || request.root_candidate_count < 1
        || request.history_indices == nullptr
        || request.history_players == nullptr
        || request.root_candidates == nullptr
        || request.depth < 1
        || request.node_limit < 0
        || request.branch_candidate_limit < 1
        || request.preselection_factor < 1
        || request.candidate_radius < 1
        || request.recent_move_count < 1
        || request.threat_extension_depth < 0) {
        return false;
    }
    const int area = request.board_size * request.board_size;
    for (int index = 0; index < area; ++index) {
        if (request.cells[index] > WHITE) return false;
    }
    for (int offset = 0; offset < request.history_count; ++offset) {
        if (request.history_indices[offset] < 0
            || request.history_indices[offset] >= area
            || (request.history_players[offset] != BLACK
                && request.history_players[offset] != WHITE)) {
            return false;
        }
    }
    for (int offset = 0; offset < request.root_candidate_count; ++offset) {
        const int move = request.root_candidates[offset];
        if (move < 0 || move >= area || request.cells[move] != 0) return false;
    }
    return true;
}

std::uint64_t request_digest(const GNMainSearchRequestV1& request) noexcept {
    std::uint64_t digest = FNV_OFFSET;
    hash_u64(digest, request.schema_version);
    hash_u64(digest, static_cast<std::uint64_t>(request.board_size));
    const int area = request.board_size * request.board_size;
    for (int index = 0; index < area; ++index)
        hash_u64(digest, request.cells[index]);
    hash_u64(digest, static_cast<std::uint64_t>(request.history_count));
    for (int offset = 0; offset < request.history_count; ++offset) {
        hash_u64(digest, static_cast<std::uint64_t>(request.history_indices[offset]));
        hash_u64(digest, request.history_players[offset]);
    }
    hash_u64(digest, static_cast<std::uint64_t>(request.player));
    hash_u64(digest, static_cast<std::uint64_t>(request.root_candidate_count));
    for (int offset = 0; offset < request.root_candidate_count; ++offset)
        hash_u64(digest, static_cast<std::uint64_t>(request.root_candidates[offset]));
    hash_u64(digest, static_cast<std::uint64_t>(request.depth));
    hash_u64(digest, static_cast<std::uint64_t>(request.node_limit));
    hash_u64(digest, static_cast<std::uint64_t>(request.branch_candidate_limit));
    hash_u64(digest, static_cast<std::uint64_t>(request.preselection_factor));
    hash_u64(digest, static_cast<std::uint64_t>(request.candidate_radius));
    hash_u64(digest, static_cast<std::uint64_t>(request.recent_move_count));
    hash_u64(digest, static_cast<std::uint64_t>(request.threat_extension_depth));
    hash_u64(digest, request.flags);
    return digest;
}

}  // namespace

GN_API int gn_main_search_v1(
    const GNMainSearchRequestV1* request,
    GNMainSearchResultV1* result
) {
    if (request == nullptr || result == nullptr
        || result->struct_size < sizeof(GNMainSearchResultV1)
        || result->schema_version != GN_MAIN_SEARCH_SCHEMA_V1
        || !valid_request(*request)) {
        return STATUS_INVALID;
    }
    result->status = GN_MAIN_SEARCH_UNSUPPORTED;
    result->completed_depth = 0;
    result->stop_reason = 0;
    result->best_move = -1;
    result->score = 0;
    result->nodes = 0;
    result->tt_entries = 0;
    result->input_digest = request_digest(*request);
    result->tt_digest = 0;
    result->root_score_count = 0;
    result->pv_length = 0;
    return GN_MAIN_SEARCH_UNSUPPORTED;
}

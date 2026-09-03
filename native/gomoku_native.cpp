#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gomoku_native.h"

#if defined(_WIN32)
#define GN_EXPORT extern "C" __declspec(dllexport)
#else
#define GN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr int EMPTY = 0;
constexpr int BLACK = 1;
constexpr int WHITE = 2;
constexpr int ABI_VERSION = 1;
constexpr int STATUS_NOT_FOUND = 0;
constexpr int STATUS_FOUND = 1;
constexpr int STATUS_CUTOFF = 2;
constexpr int STATUS_INVALID = -1;
constexpr int DIRECTIONS[4][2] = {
    {0, 1}, {1, 0}, {1, 1}, {1, -1},
};

struct Move {
    int row = -1;
    int column = -1;

    [[nodiscard]] int index(int size) const noexcept {
        return row * size + column;
    }
};

struct Profile {
    int immediate_win = 0;
    int open_four_directions = 0;
    int four_directions = 0;
    int open_three_directions = 0;
    std::vector<Move> winning_moves;

    [[nodiscard]] int tactical_rank() const noexcept {
        if (immediate_win) return 100;
        if (four_directions >= 2) return 95;
        if (open_four_directions >= 1) return 90;
        if (four_directions >= 1 && open_three_directions >= 1) return 85;
        if (open_three_directions >= 2) return 80;
        if (four_directions >= 1) return 60;
        if (open_three_directions >= 1) return 40;
        return 0;
    }

    [[nodiscard]] bool is_vcf() const noexcept {
        return immediate_win || open_four_directions >= 1
            || four_directions >= 1;
    }
};

class Board {
public:
    Board(const std::uint8_t* cells, int size)
        : size_(size), cells_(cells, cells + size * size) {}

    [[nodiscard]] int size() const noexcept { return size_; }

    [[nodiscard]] bool inside(int row, int column) const noexcept {
        return row >= 0 && row < size_ && column >= 0 && column < size_;
    }

    [[nodiscard]] int at(int row, int column) const noexcept {
        return cells_[static_cast<std::size_t>(row * size_ + column)];
    }

    [[nodiscard]] bool empty(int row, int column) const noexcept {
        return inside(row, column) && at(row, column) == EMPTY;
    }

    void place(const Move move, int player) noexcept {
        const auto index = static_cast<std::size_t>(move.index(size_));
        cells_[index] = static_cast<std::uint8_t>(player);
        history_.push_back(move);
    }

    void undo() noexcept {
        const Move move = history_.back();
        history_.pop_back();
        cells_[static_cast<std::size_t>(move.index(size_))] = EMPTY;
    }

    [[nodiscard]] bool check_win(const Move move) const noexcept {
        const int player = at(move.row, move.column);
        if (player == EMPTY) return false;
        for (const auto& direction : DIRECTIONS) {
            int total = 1;
            for (const int sign : {-1, 1}) {
                int row = move.row + sign * direction[0];
                int column = move.column + sign * direction[1];
                while (inside(row, column) && at(row, column) == player) {
                    ++total;
                    row += sign * direction[0];
                    column += sign * direction[1];
                }
            }
            if (total >= 5) return true;
        }
        return false;
    }

    [[nodiscard]] bool is_winning_move(const Move move, int player) const noexcept {
        if (!empty(move.row, move.column)) return false;
        for (const auto& direction : DIRECTIONS) {
            int total = 1;
            for (const int sign : {-1, 1}) {
                int row = move.row + sign * direction[0];
                int column = move.column + sign * direction[1];
                while (inside(row, column) && at(row, column) == player) {
                    ++total;
                    row += sign * direction[0];
                    column += sign * direction[1];
                }
            }
            if (total >= 5) return true;
        }
        return false;
    }

    [[nodiscard]] std::vector<Move> legal_moves() const {
        std::vector<Move> result;
        result.reserve(cells_.size());
        for (int row = 0; row < size_; ++row) {
            for (int column = 0; column < size_; ++column) {
                if (empty(row, column)) result.push_back({row, column});
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<Move> winning_moves(int player) const {
        std::vector<Move> result;
        for (const Move move : legal_moves()) {
            if (is_winning_move(move, player)) result.push_back(move);
        }
        return result;
    }

    [[nodiscard]] std::uint64_t hash() const noexcept {
        // Local VCF cycle prevention only. A collision can at worst miss a
        // witness; Python independently validates every returned certificate.
        std::uint64_t value = 1469598103934665603ULL;
        for (std::size_t index = 0; index < cells_.size(); ++index) {
            const std::uint64_t token =
                (static_cast<std::uint64_t>(cells_[index]) << 32U)
                ^ static_cast<std::uint64_t>(index + 1U);
            value ^= token;
            value *= 1099511628211ULL;
        }
        return value;
    }

    [[nodiscard]] int quick_order_score(const Move move, int player) const noexcept {
        const int opponent = player == BLACK ? WHITE : BLACK;
        constexpr int weights[5] = {0, 24, 8, 3, 1};
        int score = 0;
        for (const auto& direction : DIRECTIONS) {
            for (const int sign : {-1, 1}) {
                for (int distance = 1; distance <= 4; ++distance) {
                    const int row = move.row + sign * distance * direction[0];
                    const int column = move.column + sign * distance * direction[1];
                    if (!inside(row, column)) break;
                    const int cell = at(row, column);
                    if (cell == player) score += weights[distance] * 3;
                    else if (cell == opponent) score += weights[distance] * 4;
                    else score += weights[distance];
                }
            }
        }
        const int doubled_row_distance = 2 * move.row - (size_ - 1);
        const int doubled_column_distance = 2 * move.column - (size_ - 1);
        score -= (doubled_row_distance * doubled_row_distance
                  + doubled_column_distance * doubled_column_distance) / 4;
        return score;
    }

    [[nodiscard]] bool has_local_support(
        const Move move,
        int player,
        int minimum
    ) const noexcept {
        for (const auto& direction : DIRECTIONS) {
            int friendly = 0;
            for (int offset = -4; offset <= 4; ++offset) {
                if (offset == 0) continue;
                const int row = move.row + offset * direction[0];
                const int column = move.column + offset * direction[1];
                if (inside(row, column) && at(row, column) == player) ++friendly;
            }
            if (friendly >= minimum) return true;
        }
        return false;
    }

    [[nodiscard]] Profile analyze_move(const Move move, int player) const {
        Profile profile;
        std::vector<int> unique_wins;
        for (const auto& direction : DIRECTIONS) {
            int line[9];
            for (int offset = -4; offset <= 4; ++offset) {
                const int index = offset + 4;
                const int row = move.row + offset * direction[0];
                const int column = move.column + offset * direction[1];
                if (offset == 0) line[index] = player;
                else if (inside(row, column)) line[index] = at(row, column);
                else line[index] = -1;
            }

            auto anchor_is_win = [&]() noexcept {
                int left = 3;
                while (left >= 0 && line[left] == player) --left;
                int right = 5;
                while (right < 9 && line[right] == player) ++right;
                return right - left - 1 >= 5;
            };
            profile.immediate_win = profile.immediate_win || anchor_is_win();

            auto line_move_wins = [&](int candidate) noexcept {
                if (line[candidate] != EMPTY) return false;
                int left = candidate - 1;
                while (left >= 0 && line[left] == player) --left;
                int right = candidate + 1;
                while (right < 9 && line[right] == player) ++right;
                return right - left - 1 >= 5 && left < 4 && 4 < right;
            };
            auto winning_indices = [&]() {
                std::vector<int> result;
                for (int index = 0; index < 9; ++index) {
                    if (line_move_wins(index)) result.push_back(index);
                }
                return result;
            };

            const auto wins = winning_indices();
            if (!wins.empty()) {
                ++profile.four_directions;
                if (wins.size() >= 2) ++profile.open_four_directions;
                for (const int index : wins) {
                    const int row = move.row + (index - 4) * direction[0];
                    const int column = move.column + (index - 4) * direction[1];
                    unique_wins.push_back(row * size_ + column);
                }
            } else {
                bool creates_open_three = false;
                for (int extension = 0; extension < 9 && !creates_open_three; ++extension) {
                    if (line[extension] != EMPTY) continue;
                    line[extension] = player;
                    int count = 0;
                    for (int index = 0; index < 9; ++index) {
                        if (line_move_wins(index)) ++count;
                    }
                    line[extension] = EMPTY;
                    creates_open_three = count >= 2;
                }
                if (creates_open_three) ++profile.open_three_directions;
            }
        }
        std::sort(unique_wins.begin(), unique_wins.end());
        unique_wins.erase(
            std::unique(unique_wins.begin(), unique_wins.end()),
            unique_wins.end()
        );
        for (const int index : unique_wins) {
            profile.winning_moves.push_back({index / size_, index % size_});
        }
        return profile;
    }

private:
    int size_;
    std::vector<std::uint8_t> cells_;
    std::vector<Move> history_;
};

struct Candidate {
    Move move;
    Profile profile;
    int quick_score = 0;
};

class VCFEngine {
public:
    VCFEngine(
        Board& board,
        int attacker,
        int max_nodes,
        int timeout_ms,
        int candidate_limit
    ) : board_(board),
        attacker_(attacker),
        max_nodes_(max_nodes <= 0 ? std::numeric_limits<int>::max() : max_nodes),
        candidate_limit_(candidate_limit <= 0 ? 16 : candidate_limit),
        deadline_(
            timeout_ms <= 0
                ? std::chrono::steady_clock::time_point::max()
                : std::chrono::steady_clock::now()
                    + std::chrono::milliseconds(timeout_ms)
        ) {}

    int find(int remaining_attacker_moves, std::vector<Move>& line) {
        const bool found = search(remaining_attacker_moves, line);
        if (found) return STATUS_FOUND;
        return cutoff_ ? STATUS_CUTOFF : STATUS_NOT_FOUND;
    }

    [[nodiscard]] int nodes() const noexcept { return nodes_; }

private:
    [[nodiscard]] bool stopped() noexcept {
        if (nodes_ >= max_nodes_) {
            cutoff_ = true;
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline_) {
            cutoff_ = true;
            return true;
        }
        return false;
    }

    [[nodiscard]] std::vector<Candidate> forcing_candidates() {
        std::vector<Candidate> candidates;
        for (const Move move : board_.legal_moves()) {
            if (stopped()) break;
            if (!board_.has_local_support(move, attacker_, 2)
                && !board_.is_winning_move(move, attacker_)) {
                continue;
            }
            Profile profile = board_.analyze_move(move, attacker_);
            if (!profile.is_vcf()) continue;
            candidates.push_back({
                move,
                std::move(profile),
                board_.quick_order_score(move, attacker_),
            });
        }
        const double center = (board_.size() - 1) / 2.0;
        std::sort(candidates.begin(), candidates.end(), [&](const Candidate& lhs, const Candidate& rhs) {
            const auto left_distance =
                (lhs.move.row - center) * (lhs.move.row - center)
                + (lhs.move.column - center) * (lhs.move.column - center);
            const auto right_distance =
                (rhs.move.row - center) * (rhs.move.row - center)
                + (rhs.move.column - center) * (rhs.move.column - center);
            if (lhs.profile.tactical_rank() != rhs.profile.tactical_rank())
                return lhs.profile.tactical_rank() > rhs.profile.tactical_rank();
            if (lhs.profile.winning_moves.size() != rhs.profile.winning_moves.size())
                return lhs.profile.winning_moves.size() > rhs.profile.winning_moves.size();
            if (lhs.quick_score != rhs.quick_score) return lhs.quick_score > rhs.quick_score;
            if (left_distance != right_distance) return left_distance < right_distance;
            return lhs.move.index(board_.size()) < rhs.move.index(board_.size());
        });
        if (static_cast<int>(candidates.size()) > candidate_limit_)
            candidates.resize(static_cast<std::size_t>(candidate_limit_));
        return candidates;
    }

    bool search(int remaining_attacker_moves, std::vector<Move>& line) {
        if (stopped()) return false;
        ++nodes_;

        const std::uint64_t key = board_.hash()
            ^ (static_cast<std::uint64_t>(remaining_attacker_moves) * 0x9E3779B97F4A7C15ULL);
        if (!visited_.insert(key).second) return false;

        const auto immediate = board_.winning_moves(attacker_);
        if (!immediate.empty()) {
            line = {immediate.front()};
            return true;
        }
        if (remaining_attacker_moves <= 0) return false;

        const int defender = attacker_ == BLACK ? WHITE : BLACK;
        const auto candidates = forcing_candidates();
        for (const Candidate& candidate : candidates) {
            if (stopped()) return false;
            const Move move = candidate.move;
            board_.place(move, attacker_);
            if (board_.check_win(move)) {
                board_.undo();
                line = {move};
                return true;
            }

            const auto attack_wins = board_.winning_moves(attacker_);
            if (attack_wins.size() >= 2) {
                board_.undo();
                line = {move};
                return true;
            }
            if (attack_wins.size() != 1 || !board_.winning_moves(defender).empty()) {
                board_.undo();
                continue;
            }

            const Move forced_block = attack_wins.front();
            board_.place(forced_block, defender);
            std::vector<Move> child;
            const bool found = !board_.check_win(forced_block)
                && search(remaining_attacker_moves - 1, child);
            board_.undo();
            board_.undo();
            if (found) {
                line.clear();
                line.reserve(child.size() + 2);
                line.push_back(move);
                line.push_back(forced_block);
                line.insert(line.end(), child.begin(), child.end());
                return true;
            }
        }
        return false;
    }

    Board& board_;
    int attacker_;
    int max_nodes_;
    int candidate_limit_;
    std::chrono::steady_clock::time_point deadline_;
    int nodes_ = 0;
    bool cutoff_ = false;
    std::unordered_set<std::uint64_t> visited_;
};

bool valid_input(const std::uint8_t* cells, int size, int player) {
    if (cells == nullptr || size < 5 || size > 25) return false;
    if (player != BLACK && player != WHITE) return false;
    for (int index = 0; index < size * size; ++index) {
        if (cells[index] > WHITE) return false;
    }
    return true;
}

}  // namespace

GN_EXPORT int gn_abi_version() { return ABI_VERSION; }

GN_EXPORT int gn_find_winning_moves(
    const std::uint8_t* cells,
    int size,
    int player,
    const int* candidate_indices,
    int candidate_count,
    int* output_indices,
    int output_capacity
) {
    if (!valid_input(cells, size, player) || output_indices == nullptr
        || output_capacity < 0 || candidate_count < 0) return STATUS_INVALID;
    Board board(cells, size);
    int written = 0;
    if (candidate_indices == nullptr) {
        for (const Move move : board.legal_moves()) {
            if (board.is_winning_move(move, player) && written < output_capacity)
                output_indices[written++] = move.index(size);
        }
    } else {
        for (int offset = 0; offset < candidate_count; ++offset) {
            const int index = candidate_indices[offset];
            if (index < 0 || index >= size * size) return STATUS_INVALID;
            const Move move{index / size, index % size};
            if (board.is_winning_move(move, player) && written < output_capacity)
                output_indices[written++] = index;
        }
    }
    return written;
}

GN_EXPORT int gn_analyze_move(
    const std::uint8_t* cells,
    int size,
    int row,
    int column,
    int player,
    int* output_values,
    int output_capacity
) {
    if (!valid_input(cells, size, player) || output_values == nullptr
        || output_capacity < 5 || row < 0 || row >= size
        || column < 0 || column >= size) return STATUS_INVALID;
    Board board(cells, size);
    if (!board.empty(row, column)) return STATUS_INVALID;
    const Profile profile = board.analyze_move({row, column}, player);
    const int needed = 5 + static_cast<int>(profile.winning_moves.size());
    if (output_capacity < needed) return STATUS_INVALID;
    output_values[0] = profile.immediate_win;
    output_values[1] = profile.open_four_directions;
    output_values[2] = profile.four_directions;
    output_values[3] = profile.open_three_directions;
    output_values[4] = static_cast<int>(profile.winning_moves.size());
    for (std::size_t index = 0; index < profile.winning_moves.size(); ++index)
        output_values[5 + index] = profile.winning_moves[index].index(size);
    return needed;
}

GN_EXPORT int gn_analyze_moves(
    const std::uint8_t* cells,
    int size,
    int player,
    const int* candidate_indices,
    int candidate_count,
    int* output_values,
    int output_stride
) {
    if (!valid_input(cells, size, player) || candidate_indices == nullptr
        || candidate_count < 0 || output_values == nullptr
        || output_stride < 13) return STATUS_INVALID;
    Board board(cells, size);
    for (int offset = 0; offset < candidate_count; ++offset) {
        const int encoded = candidate_indices[offset];
        if (encoded < 0 || encoded >= size * size) return STATUS_INVALID;
        const Move move{encoded / size, encoded % size};
        if (!board.empty(move.row, move.column)) return STATUS_INVALID;
        const Profile profile = board.analyze_move(move, player);
        if (5 + static_cast<int>(profile.winning_moves.size()) > output_stride)
            return STATUS_INVALID;
        int* output = output_values + offset * output_stride;
        output[0] = profile.immediate_win;
        output[1] = profile.open_four_directions;
        output[2] = profile.four_directions;
        output[3] = profile.open_three_directions;
        output[4] = static_cast<int>(profile.winning_moves.size());
        for (std::size_t index = 0; index < profile.winning_moves.size(); ++index)
            output[5 + index] = profile.winning_moves[index].index(size);
    }
    return candidate_count;
}

GN_EXPORT int gn_counter_support_mask(
    const std::uint8_t* cells,
    int size,
    int player,
    const int* candidate_indices,
    int candidate_count,
    int minimum,
    std::uint8_t* output_values
) {
    if (!valid_input(cells, size, player) || candidate_indices == nullptr
        || candidate_count < 0 || minimum < 1 || output_values == nullptr)
        return STATUS_INVALID;
    Board board(cells, size);
    for (int offset = 0; offset < candidate_count; ++offset) {
        const int encoded = candidate_indices[offset];
        if (encoded < 0 || encoded >= size * size) return STATUS_INVALID;
        const Move move{encoded / size, encoded % size};
        if (!board.empty(move.row, move.column)) return STATUS_INVALID;
        output_values[offset] = static_cast<std::uint8_t>(
            board.has_local_support(move, player, minimum)
        );
    }
    return candidate_count;
}

GN_EXPORT int gn_find_vcf(
    const std::uint8_t* cells,
    int size,
    int attacker,
    int remaining_attacker_moves,
    int max_nodes,
    int timeout_ms,
    int candidate_limit,
    int* output_indices,
    int output_capacity,
    int* output_nodes,
    int* output_line_length
) {
    if (!valid_input(cells, size, attacker) || remaining_attacker_moves < 0
        || output_indices == nullptr || output_capacity < 1
        || output_nodes == nullptr || output_line_length == nullptr) {
        return STATUS_INVALID;
    }
    Board board(cells, size);
    VCFEngine engine(board, attacker, max_nodes, timeout_ms, candidate_limit);
    std::vector<Move> line;
    const int status = engine.find(remaining_attacker_moves, line);
    *output_nodes = engine.nodes();
    *output_line_length = static_cast<int>(line.size());
    if (static_cast<int>(line.size()) > output_capacity) return STATUS_INVALID;
    for (std::size_t index = 0; index < line.size(); ++index)
        output_indices[index] = line[index].index(size);
    return status;
}

GN_EXPORT int gn_classify_defenses_v1(
    const GNDefenseClassificationRequestV1* request,
    GNDefenseClassificationResultV1* result
) {
    if (request == nullptr || result == nullptr
        || request->struct_size < sizeof(GNDefenseClassificationRequestV1)
        || result->struct_size < sizeof(GNDefenseClassificationResultV1)
        || request->schema_version != GN_DEFENSE_CLASSIFICATION_SCHEMA_V1
        || result->schema_version != GN_DEFENSE_CLASSIFICATION_SCHEMA_V1
        || !valid_input(request->cells, request->board_size, request->attacker)
        || request->continuation_count < 0 || request->counter_win_count < 0
        || (request->continuation_count > 0 && request->continuations == nullptr)
        || (request->counter_win_count > 0 && request->counter_wins == nullptr)) {
        return STATUS_INVALID;
    }

    const int size = request->board_size;
    const int cell_count = size * size;
    auto valid_index = [cell_count](int encoded) noexcept {
        return encoded >= 0 && encoded < cell_count;
    };
    Board board(request->cells, size);
    for (int index = 0; index < request->continuation_count; ++index) {
        const auto& continuation = request->continuations[index];
        if (!valid_index(continuation.move)
            || !board.empty(continuation.move / size, continuation.move % size)
            || continuation.winning_point_count < 0
            || (continuation.winning_point_count > 0
                && continuation.winning_points == nullptr)) {
            return STATUS_INVALID;
        }
        for (int point = 0; point < continuation.winning_point_count; ++point) {
            if (!valid_index(continuation.winning_points[point])) {
                return STATUS_INVALID;
            }
        }
    }
    std::unordered_set<int> counter_win_set;
    for (int index = 0; index < request->counter_win_count; ++index) {
        const int encoded = request->counter_wins[index];
        if (!valid_index(encoded)
            || !board.empty(encoded / size, encoded % size)) {
            return STATUS_INVALID;
        }
        counter_win_set.insert(encoded);
    }

    struct Witness {
        Move defense;
        Move continuation;
        bool immediate = false;
        std::vector<Move> winning_points;
    };
    std::vector<Move> required;
    std::vector<Witness> refutations;
    std::vector<Move> unclassified;
    const std::vector<Move> legal_replies = board.legal_moves();
    int processed = 0;
    int status = GN_DEFENSE_CLASSIFICATION_COMPLETE;
    int cutoff_reason = GN_DEFENSE_CUTOFF_NONE;

    if (request->continuation_count == 0) {
        unclassified = legal_replies;
    } else {
        const int defender = request->attacker == BLACK ? WHITE : BLACK;
        const auto deadline = std::chrono::steady_clock::now()
            + std::chrono::milliseconds(std::max(0, request->timeout_ms));
        for (std::size_t reply_index = 0;
             reply_index < legal_replies.size();
             ++reply_index) {
            if (request->timeout_ms >= 0
                && std::chrono::steady_clock::now() >= deadline) {
                status = GN_DEFENSE_CLASSIFICATION_CUTOFF;
                cutoff_reason = GN_DEFENSE_CUTOFF_TIMEOUT;
                unclassified.insert(
                    unclassified.end(),
                    legal_replies.begin() + static_cast<std::ptrdiff_t>(reply_index),
                    legal_replies.end()
                );
                break;
            }
            if (request->reply_limit >= 0
                && processed >= request->reply_limit) {
                status = GN_DEFENSE_CLASSIFICATION_CUTOFF;
                cutoff_reason = GN_DEFENSE_CUTOFF_REPLY_LIMIT;
                unclassified.insert(
                    unclassified.end(),
                    legal_replies.begin() + static_cast<std::ptrdiff_t>(reply_index),
                    legal_replies.end()
                );
                break;
            }

            const Move defense = legal_replies[reply_index];
            ++processed;
            if (counter_win_set.find(defense.index(size))
                != counter_win_set.end()) {
                continue;
            }

            bool found = false;
            Witness witness;
            if (counter_win_set.empty()
                && !board.has_local_support(defense, defender, 3)) {
                for (int index = 0; index < request->continuation_count; ++index) {
                    const auto& continuation = request->continuations[index];
                    bool blocked = defense.index(size) == continuation.move;
                    for (int point = 0;
                         !blocked && point < continuation.winning_point_count;
                         ++point) {
                        blocked = defense.index(size)
                            == continuation.winning_points[point];
                    }
                    if (blocked) continue;
                    witness.defense = defense;
                    witness.continuation = {
                        continuation.move / size,
                        continuation.move % size,
                    };
                    witness.immediate = continuation.immediate_win != 0;
                    witness.winning_points.reserve(
                        static_cast<std::size_t>(continuation.winning_point_count)
                    );
                    for (int point = 0;
                         point < continuation.winning_point_count;
                         ++point) {
                        const int encoded = continuation.winning_points[point];
                        witness.winning_points.push_back(
                            {encoded / size, encoded % size}
                        );
                    }
                    found = true;
                    break;
                }
            }
            if (found) {
                refutations.push_back(std::move(witness));
                continue;
            }

            board.place(defense, defender);
            if (!board.check_win(defense)) {
                for (int index = 0; index < request->continuation_count; ++index) {
                    const auto& continuation = request->continuations[index];
                    const Move continuation_move{
                        continuation.move / size,
                        continuation.move % size,
                    };
                    if (!board.empty(
                            continuation_move.row,
                            continuation_move.column)) {
                        continue;
                    }
                    board.place(continuation_move, request->attacker);
                    if (board.check_win(continuation_move)) {
                        witness = {
                            defense,
                            continuation_move,
                            true,
                            {continuation_move},
                        };
                        found = true;
                    } else if (board.winning_moves(defender).empty()) {
                        std::vector<Move> winning_points =
                            board.winning_moves(request->attacker);
                        if (winning_points.size() >= 2) {
                            witness = {
                                defense,
                                continuation_move,
                                false,
                                std::move(winning_points),
                            };
                            found = true;
                        }
                    }
                    board.undo();
                    if (found) break;
                }
            }
            board.undo();

            if (found) refutations.push_back(std::move(witness));
            else required.push_back(defense);
        }
    }

    int winning_point_count = 0;
    for (const Witness& witness : refutations) {
        winning_point_count += static_cast<int>(witness.winning_points.size());
    }
    result->status = status;
    result->cutoff_reason = cutoff_reason;
    result->legal_reply_count = static_cast<int>(legal_replies.size());
    result->processed_reply_count = processed;
    result->coverage_complete = (
        status == GN_DEFENSE_CLASSIFICATION_COMPLETE
        && request->continuation_count > 0
    );
    result->analysis_completed = status == GN_DEFENSE_CLASSIFICATION_COMPLETE;
    result->required_count = static_cast<int>(required.size());
    result->refutation_count = static_cast<int>(refutations.size());
    result->unclassified_count = static_cast<int>(unclassified.size());
    result->winning_point_count = winning_point_count;
    if ((result->required_count > 0
            && (result->required_defenses == nullptr
                || result->required_capacity < result->required_count))
        || (result->refutation_count > 0
            && (result->refutations == nullptr
                || result->refutation_capacity < result->refutation_count))
        || (result->unclassified_count > 0
            && (result->unclassified_replies == nullptr
                || result->unclassified_capacity < result->unclassified_count))
        || (winning_point_count > 0
            && (result->refutation_winning_points == nullptr
                || result->winning_point_capacity < winning_point_count))) {
        return GN_DEFENSE_BUFFER_TOO_SMALL;
    }

    for (std::size_t index = 0; index < required.size(); ++index) {
        result->required_defenses[index] = required[index].index(size);
    }
    int point_offset = 0;
    for (std::size_t index = 0; index < refutations.size(); ++index) {
        const Witness& witness = refutations[index];
        result->refutations[index] = {
            witness.defense.index(size),
            witness.continuation.index(size),
            witness.immediate ? 1 : 0,
            point_offset,
            static_cast<int>(witness.winning_points.size()),
        };
        for (const Move point : witness.winning_points) {
            result->refutation_winning_points[point_offset++] = point.index(size);
        }
    }
    for (std::size_t index = 0; index < unclassified.size(); ++index) {
        result->unclassified_replies[index] = unclassified[index].index(size);
    }
    return status;
}

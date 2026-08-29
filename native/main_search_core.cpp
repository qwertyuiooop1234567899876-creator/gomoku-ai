#include "main_search_core.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gomoku_native {
namespace {

constexpr int EMPTY = 0;
constexpr int BLACK = 1;
constexpr int WHITE = 2;
constexpr int MATE_SCORE = 1'000'000'000;
constexpr int HEURISTIC_SCORE_LIMIT = 100'000'000;
constexpr int INFINITY_SCORE = 2'000'000'000;
constexpr int STATUS_OK = 1;
constexpr int STATUS_CUTOFF = 2;
constexpr int STOP_COMPLETED = 0;
constexpr int STOP_NODE_LIMIT = 1;
constexpr std::uint64_t FNV_OFFSET = 1469598103934665603ULL;
constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;
constexpr int DIRECTIONS[4][2] = {
    {0, 1}, {1, 0}, {1, 1}, {1, -1},
};

int other_side(int player) noexcept {
    return player == BLACK ? WHITE : BLACK;
}

void hash_u64(std::uint64_t& digest, std::uint64_t value) noexcept {
    for (int shift = 0; shift < 64; shift += 8) {
        digest ^= (value >> shift) & 0xFFU;
        digest *= FNV_PRIME;
    }
}

struct Move {
    int row = -1;
    int column = -1;

    [[nodiscard]] int index(int size) const noexcept {
        return row * size + column;
    }

    friend bool operator==(const Move& lhs, const Move& rhs) noexcept {
        return lhs.row == rhs.row && lhs.column == rhs.column;
    }

    friend bool operator!=(const Move& lhs, const Move& rhs) noexcept {
        return !(lhs == rhs);
    }
};

struct HistoryMove {
    Move move;
    int player = EMPTY;
};

struct Profile {
    bool immediate_win = false;
    int open_four_directions = 0;
    int four_directions = 0;
    int open_three_directions = 0;
    std::vector<Move> winning_moves;

    [[nodiscard]] bool double_four() const noexcept {
        return four_directions >= 2;
    }

    [[nodiscard]] bool four_three() const noexcept {
        return four_directions >= 1 && open_three_directions >= 1;
    }

    [[nodiscard]] bool double_three() const noexcept {
        return open_three_directions >= 2;
    }

    [[nodiscard]] int tactical_rank() const noexcept {
        if (immediate_win) return 100;
        if (double_four()) return 95;
        if (open_four_directions >= 1) return 90;
        if (four_three()) return 85;
        if (double_three()) return 80;
        if (four_directions >= 1) return 60;
        if (open_three_directions >= 1) return 40;
        return 0;
    }
};

class Board {
public:
    explicit Board(const GNMainSearchRequestV1& request)
        : size_(request.board_size),
          cells_(request.cells, request.cells + request.board_size * request.board_size) {
        history_.reserve(static_cast<std::size_t>(size_ * size_));
        for (int offset = 0; offset < request.history_count; ++offset) {
            const int encoded = request.history_indices[offset];
            history_.push_back({
                {encoded / size_, encoded % size_},
                request.history_players[offset],
            });
        }
        empty_count_ = static_cast<int>(std::count(cells_.begin(), cells_.end(), 0));
    }

    [[nodiscard]] int size() const noexcept { return size_; }
    [[nodiscard]] int empty_count() const noexcept { return empty_count_; }
    [[nodiscard]] bool full() const noexcept { return empty_count_ == 0; }
    [[nodiscard]] const std::vector<HistoryMove>& history() const noexcept {
        return history_;
    }

    [[nodiscard]] bool inside(int row, int column) const noexcept {
        return row >= 0 && row < size_ && column >= 0 && column < size_;
    }

    [[nodiscard]] int at(int row, int column) const noexcept {
        return cells_[static_cast<std::size_t>(row * size_ + column)];
    }

    [[nodiscard]] bool empty(const Move move) const noexcept {
        return inside(move.row, move.column) && at(move.row, move.column) == EMPTY;
    }

    void place(const Move move, int player) {
        cells_[static_cast<std::size_t>(move.index(size_))] =
            static_cast<std::uint8_t>(player);
        history_.push_back({move, player});
        --empty_count_;
    }

    void undo() {
        const Move move = history_.back().move;
        history_.pop_back();
        cells_[static_cast<std::size_t>(move.index(size_))] = EMPTY;
        ++empty_count_;
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
        if (!empty(move)) return false;
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
        result.reserve(static_cast<std::size_t>(empty_count_));
        for (int row = 0; row < size_; ++row) {
            for (int column = 0; column < size_; ++column) {
                const Move move{row, column};
                if (empty(move)) result.push_back(move);
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<Move> winning_moves(
        int player,
        const std::vector<Move>& candidates
    ) const {
        std::vector<Move> result;
        for (const Move move : candidates) {
            if (is_winning_move(move, player)) result.push_back(move);
        }
        return result;
    }

    [[nodiscard]] int quick_score(const Move move, int player) const noexcept {
        constexpr int weights[5] = {0, 24, 8, 3, 1};
        const int opponent = other_side(player);
        int score = 0;
        for (const auto& direction : DIRECTIONS) {
            for (const int sign : {-1, 1}) {
                for (int distance = 1; distance <= 4; ++distance) {
                    const int row = move.row + sign * distance * direction[0];
                    const int column = move.column + sign * distance * direction[1];
                    if (!inside(row, column)) break;
                    const int cell = at(row, column);
                    const int weight = weights[distance];
                    if (cell == player) score += weight * 3;
                    else if (cell == opponent) score += weight * 4;
                    else score += weight;
                }
            }
        }
        const int doubled_row_distance = 2 * move.row - (size_ - 1);
        const int doubled_column_distance = 2 * move.column - (size_ - 1);
        score -= (
            doubled_row_distance * doubled_row_distance
            + doubled_column_distance * doubled_column_distance
        ) / 4;
        return score;
    }

    [[nodiscard]] Profile analyze_move(const Move move, int player) const {
        Profile profile;
        std::vector<int> unique_wins;
        for (const auto& direction : DIRECTIONS) {
            std::array<int, 9> line{};
            for (int offset = -4; offset <= 4; ++offset) {
                const int index = offset + 4;
                const int row = move.row + offset * direction[0];
                const int column = move.column + offset * direction[1];
                if (offset == 0) line[static_cast<std::size_t>(index)] = player;
                else if (inside(row, column))
                    line[static_cast<std::size_t>(index)] = at(row, column);
                else line[static_cast<std::size_t>(index)] = -1;
            }

            int left = 3;
            while (left >= 0 && line[static_cast<std::size_t>(left)] == player) --left;
            int right = 5;
            while (right < 9 && line[static_cast<std::size_t>(right)] == player) ++right;
            profile.immediate_win = profile.immediate_win || right - left - 1 >= 5;

            const auto line_move_wins = [&](int candidate) noexcept {
                if (line[static_cast<std::size_t>(candidate)] != EMPTY) return false;
                int candidate_left = candidate - 1;
                while (candidate_left >= 0
                       && line[static_cast<std::size_t>(candidate_left)] == player) {
                    --candidate_left;
                }
                int candidate_right = candidate + 1;
                while (candidate_right < 9
                       && line[static_cast<std::size_t>(candidate_right)] == player) {
                    ++candidate_right;
                }
                return candidate_right - candidate_left - 1 >= 5
                    && candidate_left < 4 && 4 < candidate_right;
            };

            std::vector<int> wins;
            for (int index = 0; index < 9; ++index) {
                if (line_move_wins(index)) wins.push_back(index);
            }
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
                    if (line[static_cast<std::size_t>(extension)] != EMPTY) continue;
                    line[static_cast<std::size_t>(extension)] = player;
                    int count = 0;
                    for (int index = 0; index < 9; ++index) {
                        if (line_move_wins(index)) ++count;
                    }
                    line[static_cast<std::size_t>(extension)] = EMPTY;
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
        for (const int encoded : unique_wins) {
            profile.winning_moves.push_back({encoded / size_, encoded % size_});
        }
        return profile;
    }

    [[nodiscard]] std::vector<int> line_through(
        const Move move,
        int row_step,
        int column_step,
        int& anchor
    ) const {
        int start_row = move.row;
        int start_column = move.column;
        anchor = 0;
        while (inside(start_row - row_step, start_column - column_step)) {
            start_row -= row_step;
            start_column -= column_step;
            ++anchor;
        }
        std::vector<int> line;
        while (inside(start_row, start_column)) {
            line.push_back(at(start_row, start_column));
            start_row += row_step;
            start_column += column_step;
        }
        return line;
    }

    [[nodiscard]] const std::vector<std::uint8_t>& cells() const noexcept {
        return cells_;
    }

private:
    int size_;
    std::vector<std::uint8_t> cells_;
    std::vector<HistoryMove> history_;
    int empty_count_ = 0;
};

struct PatternFeatures {
    int score = 0;
    int open_threes = 0;
    int jump_threes = 0;
    int forcing_patterns = 0;
};

const std::vector<std::pair<std::string, int>>& pattern_scores() {
    static const std::vector<std::pair<std::string, int>> patterns = [] {
        const std::pair<const char*, int> base[] = {
            {"XXXXX", 100'000'000}, {".XXXX.", 10'000'000},
            {"XXXX.", 1'000'000}, {"XXX.X", 1'000'000},
            {"XX.XX", 1'000'000}, {".XXX.", 100'000},
            {".XX.X.", 70'000}, {"XXX..", 10'000},
            {"XX.X.", 8'000}, {".XX.", 1'000},
            {".X.X.", 800}, {"XX...", 100},
        };
        std::map<std::string, int> unique;
        for (const auto& item : base) {
            const std::string pattern(item.first);
            const std::string mirrored(pattern.rbegin(), pattern.rend());
            unique[pattern] = std::max(unique[pattern], item.second);
            unique[mirrored] = std::max(unique[mirrored], item.second);
        }
        std::vector<std::pair<std::string, int>> result(unique.begin(), unique.end());
        std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.second != rhs.second) return lhs.second > rhs.second;
            if (lhs.first.size() != rhs.first.size()) return lhs.first.size() > rhs.first.size();
            return lhs.first < rhs.first;
        });
        return result;
    }();
    return patterns;
}

std::string line_text(const std::vector<int>& line, int player) {
    std::string text;
    text.reserve(line.size() + 2);
    text.push_back('O');
    for (const int cell : line) {
        text.push_back(cell == player ? 'X' : cell == EMPTY ? '.' : 'O');
    }
    text.push_back('O');
    return text;
}

PatternFeatures score_line_features(const std::string& text) {
    PatternFeatures features;
    std::vector<bool> occupied(text.size(), false);
    for (const auto& item : pattern_scores()) {
        std::size_t start = 0;
        while (true) {
            const std::size_t index = text.find(item.first, start);
            if (index == std::string::npos) break;
            const std::size_t end = index + item.first.size();
            bool overlaps = false;
            for (std::size_t position = index; position < end; ++position) {
                if (occupied[position]) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) {
                features.score += item.second;
                if (item.second >= 1'000'000) ++features.forcing_patterns;
                else if (item.first == ".XXX.") ++features.open_threes;
                else if (item.first == ".XX.X." || item.first == ".X.XX.")
                    ++features.jump_threes;
                for (std::size_t position = index; position < end; ++position)
                    occupied[position] = true;
            }
            start = index + 1;
        }
    }
    return features;
}

std::vector<std::vector<int>> all_lines(const Board& board) {
    std::vector<std::vector<int>> lines;
    const int size = board.size();
    for (int row = 0; row < size; ++row) {
        std::vector<int> line;
        for (int column = 0; column < size; ++column) line.push_back(board.at(row, column));
        lines.push_back(std::move(line));
    }
    for (int column = 0; column < size; ++column) {
        std::vector<int> line;
        for (int row = 0; row < size; ++row) line.push_back(board.at(row, column));
        lines.push_back(std::move(line));
    }
    const auto collect = [&](int start_row, int start_column, int dr, int dc) {
        std::vector<int> line;
        while (board.inside(start_row, start_column)) {
            line.push_back(board.at(start_row, start_column));
            start_row += dr;
            start_column += dc;
        }
        return line;
    };
    for (int column = 0; column < size; ++column) {
        auto line = collect(0, column, 1, 1);
        if (line.size() >= 5) lines.push_back(std::move(line));
    }
    for (int row = 1; row < size; ++row) {
        auto line = collect(row, 0, 1, 1);
        if (line.size() >= 5) lines.push_back(std::move(line));
    }
    for (int column = 0; column < size; ++column) {
        auto line = collect(0, column, 1, -1);
        if (line.size() >= 5) lines.push_back(std::move(line));
    }
    for (int row = 1; row < size; ++row) {
        auto line = collect(row, size - 1, 1, -1);
        if (line.size() >= 5) lines.push_back(std::move(line));
    }
    return lines;
}

PatternFeatures player_features(const Board& board, int player) {
    PatternFeatures total;
    for (const auto& line : all_lines(board)) {
        const PatternFeatures current = score_line_features(line_text(line, player));
        total.score += current.score;
        total.open_threes += current.open_threes;
        total.jump_threes += current.jump_threes;
        total.forcing_patterns += current.forcing_patterns;
    }
    return total;
}

int static_score(const Board& board, int perspective) {
    const PatternFeatures own = player_features(board, perspective);
    const PatternFeatures opponent = player_features(board, other_side(perspective));
    int score = own.score - opponent.score;
    if (own.forcing_patterns == 0 && opponent.forcing_patterns == 0) {
        const int initiative = std::min(
            10'000,
            own.open_threes * 8'000 + own.jump_threes * 5'000
        );
        score += initiative;
    }
    return std::max(-HEURISTIC_SCORE_LIMIT, std::min(HEURISTIC_SCORE_LIMIT, score));
}

int profile_bonus(const Profile& profile) noexcept {
    if (profile.immediate_win) return 100'000'000;
    if (profile.double_four()) return 50'000'000;
    if (profile.open_four_directions >= 1) return 49'000'000;
    if (profile.four_three()) return 48'000'000;
    if (profile.double_three()) return 12'000'000;
    return profile.four_directions * 1'000'000
        + profile.open_three_directions * 100'000;
}

int move_pattern_score(const std::vector<int>& line, int player) {
    return score_line_features(line_text(line, player)).score;
}

int evaluate_move(const Board& board, const Move move, int player) {
    const int opponent = other_side(player);
    int attack_gain = 0;
    int defense_gain = 0;
    for (const auto& direction : DIRECTIONS) {
        int anchor = 0;
        std::vector<int> line = board.line_through(
            move, direction[0], direction[1], anchor
        );
        const int player_before = move_pattern_score(line, player);
        const int opponent_before = move_pattern_score(line, opponent);
        line[static_cast<std::size_t>(anchor)] = player;
        const int player_after = move_pattern_score(line, player);
        line[static_cast<std::size_t>(anchor)] = opponent;
        const int opponent_after = move_pattern_score(line, opponent);
        attack_gain += player_after - player_before;
        defense_gain += opponent_after - opponent_before;
    }
    attack_gain = std::max(0, attack_gain);
    defense_gain = std::max(0, defense_gain);
    const Profile own = board.analyze_move(move, player);
    const Profile opposing = board.analyze_move(move, opponent);
    const int doubled_row_distance = 2 * move.row - (board.size() - 1);
    const int doubled_column_distance = 2 * move.column - (board.size() - 1);
    const int distance_squared_times_four =
        doubled_row_distance * doubled_row_distance
        + doubled_column_distance * doubled_column_distance;
    const int center_bonus = std::max(0, 100 - distance_squared_times_four / 2);
    return static_cast<int>(
        attack_gain + profile_bonus(own)
        + 1.15 * (defense_gain + profile_bonus(opposing))
        + center_bonus
    );
}

enum class Bound : int { Exact = 0, Lower = 1, Upper = 2 };

struct TTEntry {
    int depth = 0;
    int extension_depth = 0;
    int score = 0;
    Bound bound = Bound::Exact;
    Move best_move;
    bool has_best_move = false;
    std::vector<Move> pv;
};

struct SearchValue {
    int score = 0;
    std::vector<Move> pv;
};

struct RootValue {
    Move move;
    int score = 0;
    std::vector<Move> pv;
    int original_priority = 0;
};

struct NodeLimitReached {};

class SearchEngine {
public:
    explicit SearchEngine(const GNMainSearchRequestV1& request)
        : request_(request), board_(request),
          use_pvs_((request.flags & GN_MAIN_SEARCH_FLAG_PVS) != 0),
          use_tt_((request.flags & GN_MAIN_SEARCH_FLAG_TT) != 0),
          history_scores_(static_cast<std::size_t>(3 * request.board_size * request.board_size), 0),
          killers_(static_cast<std::size_t>(request.board_size * request.board_size + 2)) {}

    int run(GNMainSearchResultV1& output) {
        try {
            std::vector<Move> candidates;
            candidates.reserve(static_cast<std::size_t>(request_.root_candidate_count));
            for (int offset = 0; offset < request_.root_candidate_count; ++offset) {
                const int encoded = request_.root_candidates[offset];
                candidates.push_back({encoded / board_.size(), encoded % board_.size()});
            }
            const std::vector<RootValue> ranked = search_root(
                request_.player,
                request_.depth,
                candidates,
                -INFINITY_SCORE,
                INFINITY_SCORE
            );
            const RootValue& best = ranked.front();
            output.status = STATUS_OK;
            output.completed_depth = request_.depth;
            output.stop_reason = STOP_COMPLETED;
            output.best_move = best.move.index(board_.size());
            output.score = best.score;
            output.nodes = nodes_;
            output.tt_entries = static_cast<std::int64_t>(tt_.size());
            output.tt_digest = tt_digest();
            output.root_score_count = request_.root_candidate_count;
            for (int offset = 0; offset < request_.root_candidate_count; ++offset) {
                const int encoded = request_.root_candidates[offset];
                int score = 0;
                for (const RootValue& item : ranked) {
                    if (item.move.index(board_.size()) == encoded) {
                        score = item.score;
                        break;
                    }
                }
                output.root_scores[offset] = score;
            }
            output.pv_length = std::min(
                output.pv_capacity,
                static_cast<int>(best.pv.size())
            );
            for (int offset = 0; offset < output.pv_length; ++offset) {
                output.principal_variation[offset] =
                    best.pv[static_cast<std::size_t>(offset)].index(board_.size());
            }
            return STATUS_OK;
        } catch (const NodeLimitReached&) {
            output.status = STATUS_CUTOFF;
            output.completed_depth = 0;
            output.stop_reason = STOP_NODE_LIMIT;
            output.best_move = -1;
            output.score = 0;
            output.nodes = nodes_;
            output.tt_entries = static_cast<std::int64_t>(tt_.size());
            output.tt_digest = tt_digest();
            output.root_score_count = 0;
            output.pv_length = 0;
            return STATUS_CUTOFF;
        }
    }

private:
    [[nodiscard]] std::uint64_t position_key(int player) const noexcept {
        std::uint64_t digest = FNV_OFFSET;
        hash_u64(digest, static_cast<std::uint64_t>(board_.size()));
        for (std::size_t index = 0; index < board_.cells().size(); ++index) {
            hash_u64(digest, static_cast<std::uint64_t>(index + 1));
            hash_u64(digest, board_.cells()[index]);
        }
        hash_u64(digest, static_cast<std::uint64_t>(player));
        const auto& history = board_.history();
        const int count = std::min(request_.recent_move_count, static_cast<int>(history.size()));
        hash_u64(digest, static_cast<std::uint64_t>(count));
        const int begin = static_cast<int>(history.size()) - count;
        for (int offset = 0; offset < count; ++offset) {
            const HistoryMove& item = history[static_cast<std::size_t>(begin + offset)];
            hash_u64(digest, static_cast<std::uint64_t>(offset + 1));
            hash_u64(digest, static_cast<std::uint64_t>(item.move.index(board_.size())));
            hash_u64(digest, static_cast<std::uint64_t>(item.player));
        }
        return digest;
    }

    void check_node_limit() {
        if (request_.node_limit > 0 && nodes_ >= request_.node_limit)
            throw NodeLimitReached{};
    }

    [[nodiscard]] int history_score(int player, const Move move) const noexcept {
        return history_scores_[static_cast<std::size_t>(
            player * board_.size() * board_.size() + move.index(board_.size())
        )];
    }

    [[nodiscard]] int killer_priority(const Move move, int ply) const noexcept {
        if (ply < 0 || ply >= static_cast<int>(killers_.size())) return 0;
        const auto& moves = killers_[static_cast<std::size_t>(ply)];
        for (std::size_t index = 0; index < moves.size(); ++index) {
            if (moves[index] == move) return 2 - static_cast<int>(index);
        }
        return 0;
    }

    void record_cutoff(const Move move, int player, int depth, int ply) {
        auto& moves = killers_[static_cast<std::size_t>(ply)];
        if (std::find(moves.begin(), moves.end(), move) == moves.end()) {
            moves.insert(moves.begin(), move);
            if (moves.size() > 2) moves.resize(2);
        }
        int& value = history_scores_[static_cast<std::size_t>(
            player * board_.size() * board_.size() + move.index(board_.size())
        )];
        value = std::min(1'000'000, value + depth * depth * 16);
    }

    std::vector<Move> promote_move(
        const std::vector<Move>& moves,
        const Move preferred,
        bool has_preferred
    ) const {
        if (!has_preferred
            || std::find(moves.begin(), moves.end(), preferred) == moves.end()) {
            return moves;
        }
        std::vector<Move> promoted;
        promoted.reserve(moves.size());
        promoted.push_back(preferred);
        for (const Move move : moves) {
            if (move != preferred) promoted.push_back(move);
        }
        return promoted;
    }

    std::vector<Move> raw_candidates(const std::vector<Move>& legal) const {
        const int area = board_.size() * board_.size();
        std::vector<bool> included(static_cast<std::size_t>(area), false);
        const auto add = [&](int row, int column) {
            const Move move{row, column};
            if (board_.empty(move)) included[static_cast<std::size_t>(move.index(board_.size()))] = true;
        };
        const auto& history = board_.history();
        const int recent_count = std::min(
            request_.recent_move_count,
            static_cast<int>(history.size())
        );
        const int recent_begin = static_cast<int>(history.size()) - recent_count;
        for (int index = recent_begin; index < static_cast<int>(history.size()); ++index) {
            const Move anchor = history[static_cast<std::size_t>(index)].move;
            for (int dr = -request_.candidate_radius; dr <= request_.candidate_radius; ++dr) {
                for (int dc = -request_.candidate_radius; dc <= request_.candidate_radius; ++dc)
                    add(anchor.row + dr, anchor.column + dc);
            }
            for (const auto& direction : DIRECTIONS) {
                for (const int sign : {-1, 1}) {
                    for (int distance = 1; distance <= 4; ++distance) {
                        const int row = anchor.row + sign * distance * direction[0];
                        const int column = anchor.column + sign * distance * direction[1];
                        if (!board_.inside(row, column)) break;
                        add(row, column);
                    }
                }
            }
        }
        for (const HistoryMove& item : history) {
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc)
                    add(item.move.row + dr, item.move.column + dc);
            }
        }
        std::vector<Move> result;
        for (int index = 0; index < area; ++index) {
            if (included[static_cast<std::size_t>(index)])
                result.push_back({index / board_.size(), index % board_.size()});
        }
        return result.empty() ? legal : result;
    }

    std::vector<Move> order_specific(
        const std::vector<Move>& moves,
        int player,
        int ply,
        const Move tt_move,
        bool has_tt_move,
        bool full_evaluation,
        bool use_heuristics
    ) {
        struct Scored {
            Move move;
            int move_score;
            int killer;
            int history;
            int quick;
            int center_distance4;
        };
        std::vector<Scored> scored;
        std::vector<bool> seen(static_cast<std::size_t>(board_.size() * board_.size()), false);
        for (const Move move : moves) {
            const int encoded = move.index(board_.size());
            if (seen[static_cast<std::size_t>(encoded)]) continue;
            seen[static_cast<std::size_t>(encoded)] = true;
            const int quick = board_.quick_score(move, player);
            const int doubled_row = 2 * move.row - (board_.size() - 1);
            const int doubled_column = 2 * move.column - (board_.size() - 1);
            scored.push_back({
                move,
                full_evaluation ? evaluate_move(board_, move, player) : quick,
                use_heuristics ? killer_priority(move, ply) : 0,
                use_heuristics ? history_score(player, move) : 0,
                quick,
                doubled_row * doubled_row + doubled_column * doubled_column,
            });
        }
        std::sort(scored.begin(), scored.end(), [&](const Scored& lhs, const Scored& rhs) {
            const auto left_key = std::make_tuple(
                use_heuristics && has_tt_move && lhs.move == tt_move,
                lhs.killer, lhs.history, lhs.move_score, lhs.quick,
                -lhs.center_distance4, -lhs.move.row, -lhs.move.column
            );
            const auto right_key = std::make_tuple(
                use_heuristics && has_tt_move && rhs.move == tt_move,
                rhs.killer, rhs.history, rhs.move_score, rhs.quick,
                -rhs.center_distance4, -rhs.move.row, -rhs.move.column
            );
            return left_key > right_key;
        });
        std::vector<Move> result;
        result.reserve(scored.size());
        for (const Scored& item : scored) result.push_back(item.move);
        return result;
    }

    std::vector<Move> ordered_moves(
        int player,
        int ply,
        const Move tt_move,
        bool has_tt_move
    ) {
        const std::vector<Move> legal = board_.legal_moves();
        if (legal.empty()) return {};
        const std::vector<Move> own_wins = board_.winning_moves(player, legal);
        if (!own_wins.empty()) return promote_move(own_wins, tt_move, has_tt_move);
        const std::vector<Move> opponent_wins = board_.winning_moves(other_side(player), legal);
        if (opponent_wins.size() == 1) return opponent_wins;
        if (opponent_wins.size() >= 2) {
            return order_specific(
                opponent_wins, player, ply, tt_move, has_tt_move,
                ply <= 1, true
            );
        }
        std::vector<Move> raw = raw_candidates(legal);
        std::stable_sort(raw.begin(), raw.end(), [&](const Move lhs, const Move rhs) {
            return board_.quick_score(lhs, player) > board_.quick_score(rhs, player);
        });
        const int preselection = std::max(
            request_.branch_candidate_limit,
            request_.branch_candidate_limit * request_.preselection_factor
        );
        if (static_cast<int>(raw.size()) > preselection)
            raw.resize(static_cast<std::size_t>(preselection));
        std::vector<Move> ranked = order_specific(
            raw, player, ply, tt_move, has_tt_move, ply <= 1, false
        );
        if (static_cast<int>(ranked.size()) > request_.branch_candidate_limit)
            ranked.resize(static_cast<std::size_t>(request_.branch_candidate_limit));
        return order_specific(
            ranked, player, ply, tt_move, has_tt_move, true, true
        );
    }

    std::vector<std::pair<Move, bool>> forcing_options(int player) {
        const std::vector<Move> legal = board_.legal_moves();
        std::vector<Move> raw = raw_candidates(legal);
        std::stable_sort(raw.begin(), raw.end(), [&](const Move lhs, const Move rhs) {
            return board_.quick_score(lhs, player) > board_.quick_score(rhs, player);
        });
        const int shortlist_limit = std::max(4 * 3, 16);
        if (static_cast<int>(raw.size()) > shortlist_limit)
            raw.resize(static_cast<std::size_t>(shortlist_limit));
        struct Option {
            Move move;
            Profile profile;
            int quick;
            bool strict;
        };
        std::vector<Option> forcing;
        for (const Move move : raw) {
            const Profile profile = board_.analyze_move(move, player);
            const bool strict = profile.immediate_win
                || profile.open_four_directions >= 1
                || profile.four_directions >= 1
                || profile.double_four() || profile.four_three();
            const bool extension = strict || profile.double_three()
                || profile.open_three_directions >= 1;
            if (extension) forcing.push_back({move, profile, board_.quick_score(move, player), strict});
        }
        std::stable_sort(forcing.begin(), forcing.end(), [](const Option& lhs, const Option& rhs) {
            return std::make_tuple(
                lhs.profile.tactical_rank(),
                lhs.profile.winning_moves.size(),
                lhs.quick
            ) > std::make_tuple(
                rhs.profile.tactical_rank(),
                rhs.profile.winning_moves.size(),
                rhs.quick
            );
        });
        if (forcing.size() > 4) forcing.resize(4);
        std::vector<std::pair<Move, bool>> result;
        for (const Option& item : forcing) result.push_back({item.move, item.strict});
        return result;
    }

    SearchValue threat_extension(
        int player,
        int alpha,
        int beta,
        int ply,
        int extension_depth
    ) {
        const std::vector<Move> legal = board_.legal_moves();
        if (legal.empty()) return {0, {}};
        const std::vector<Move> own_wins = board_.winning_moves(player, legal);
        if (!own_wins.empty()) return {MATE_SCORE - ply, {own_wins.front()}};
        const int opponent = other_side(player);
        const std::vector<Move> opponent_wins = board_.winning_moves(opponent, legal);
        if (opponent_wins.size() >= 2) return {-MATE_SCORE + ply, {}};
        if (extension_depth <= 0) return {static_score(board_, player), {}};

        std::vector<std::pair<Move, bool>> options;
        if (opponent_wins.size() == 1) options.push_back({opponent_wins.front(), true});
        else options = forcing_options(player);
        if (options.empty()) return {static_score(board_, player), {}};

        int best_score = -INFINITY_SCORE;
        std::vector<Move> best_pv;
        for (const auto& option : options) {
            const Move move = option.first;
            board_.place(move, player);
            SearchValue child;
            int score = 0;
            if (board_.check_win(move)) score = MATE_SCORE - ply;
            else {
                child = threat_extension(
                    opponent, -beta, -alpha, ply + 1, extension_depth - 1
                );
                score = -child.score;
            }
            board_.undo();
            if (score > best_score) {
                best_score = score;
                best_pv.clear();
                best_pv.push_back(move);
                best_pv.insert(best_pv.end(), child.pv.begin(), child.pv.end());
            }
            alpha = std::max(alpha, score);
            if (alpha >= beta) break;
        }
        return {best_score, best_pv};
    }

    static int score_to_tt(int score, int ply) noexcept {
        if (score > HEURISTIC_SCORE_LIMIT) return score + ply;
        if (score < -HEURISTIC_SCORE_LIMIT) return score - ply;
        return score;
    }

    static int score_from_tt(int score, int ply) noexcept {
        if (score > HEURISTIC_SCORE_LIMIT) return score - ply;
        if (score < -HEURISTIC_SCORE_LIMIT) return score + ply;
        return score;
    }

    void store_tt(
        std::uint64_t key,
        int depth,
        int extension_depth,
        int score,
        int alpha_original,
        int beta_original,
        const std::vector<Move>& pv,
        const Move best_move,
        bool has_best_move,
        int ply
    ) {
        if (!use_tt_) return;
        const Bound bound = score <= alpha_original
            ? Bound::Upper
            : score >= beta_original ? Bound::Lower : Bound::Exact;
        const auto found = tt_.find(key);
        if (found != tt_.end()) {
            const TTEntry& previous = found->second;
            if (previous.depth > depth) return;
            if (previous.depth == depth && previous.extension_depth > extension_depth) return;
            if (previous.depth == depth && previous.extension_depth == extension_depth
                && previous.bound == Bound::Exact && bound != Bound::Exact) return;
        }
        tt_[key] = TTEntry{
            depth, extension_depth, score_to_tt(score, ply), bound,
            best_move, has_best_move, pv,
        };
    }

    SearchValue negamax(
        int player,
        int depth,
        int alpha,
        int beta,
        int ply,
        int extension_depth
    ) {
        check_node_limit();
        ++nodes_;
        if (board_.full()) return {0, {}};
        const int alpha_original = alpha;
        const int beta_original = beta;
        const std::uint64_t key = position_key(player);
        Move tt_move;
        bool has_tt_move = false;
        if (use_tt_) {
            const auto found = tt_.find(key);
            if (found != tt_.end()) {
                const TTEntry& entry = found->second;
                has_tt_move = entry.has_best_move && board_.empty(entry.best_move);
                if (has_tt_move) tt_move = entry.best_move;
                if (entry.depth >= depth && entry.extension_depth >= extension_depth) {
                    const int tt_score = score_from_tt(entry.score, ply);
                    if (entry.bound == Bound::Exact) return {tt_score, entry.pv};
                    if (entry.bound == Bound::Lower) alpha = std::max(alpha, tt_score);
                    else if (entry.bound == Bound::Upper) beta = std::min(beta, tt_score);
                    if (alpha >= beta) return {tt_score, entry.pv};
                }
            }
        }
        if (depth <= 0) {
            SearchValue value = threat_extension(player, alpha, beta, ply, extension_depth);
            const Move best = value.pv.empty() ? Move{} : value.pv.front();
            store_tt(
                key, depth, extension_depth, value.score,
                alpha_original, beta_original, value.pv,
                best, !value.pv.empty(), ply
            );
            return value;
        }

        const std::vector<Move> moves = ordered_moves(
            player, ply, tt_move, has_tt_move
        );
        if (moves.empty()) return {static_score(board_, player), {}};
        int best_score = -INFINITY_SCORE;
        Move best_move;
        bool has_best_move = false;
        std::vector<Move> best_pv;
        for (std::size_t index = 0; index < moves.size(); ++index) {
            const Move move = moves[index];
            board_.place(move, player);
            SearchValue child;
            int score = 0;
            try {
                if (board_.check_win(move)) score = MATE_SCORE - ply;
                else if (use_pvs_ && index > 0) {
                    child = negamax(
                        other_side(player), depth - 1,
                        -alpha - 1, -alpha, ply + 1, extension_depth
                    );
                    score = -child.score;
                    if (alpha < score && score < beta) {
                        child = negamax(
                            other_side(player), depth - 1,
                            -beta, -alpha, ply + 1, extension_depth
                        );
                        score = -child.score;
                    }
                } else {
                    child = negamax(
                        other_side(player), depth - 1,
                        -beta, -alpha, ply + 1, extension_depth
                    );
                    score = -child.score;
                }
            } catch (...) {
                board_.undo();
                throw;
            }
            board_.undo();
            if (score > best_score) {
                best_score = score;
                best_move = move;
                has_best_move = true;
                best_pv.clear();
                best_pv.push_back(move);
                best_pv.insert(best_pv.end(), child.pv.begin(), child.pv.end());
            }
            alpha = std::max(alpha, score);
            if (alpha >= beta) {
                record_cutoff(move, player, depth, ply);
                break;
            }
        }
        store_tt(
            key, depth, extension_depth, best_score,
            alpha_original, beta_original, best_pv,
            best_move, has_best_move, ply
        );
        return {best_score, best_pv};
    }

    std::vector<RootValue> search_root(
        int player,
        int depth,
        const std::vector<Move>& candidates,
        int alpha,
        int beta
    ) {
        const int alpha_original = alpha;
        std::vector<RootValue> ranked;
        ranked.reserve(candidates.size());
        const std::uint64_t root_key = position_key(player);
        Move tt_move;
        bool has_tt_move = false;
        if (use_tt_) {
            const auto found = tt_.find(root_key);
            if (found != tt_.end() && found->second.has_best_move
                && board_.empty(found->second.best_move)) {
                tt_move = found->second.best_move;
                has_tt_move = true;
            }
        }
        const std::vector<Move> moves = promote_move(candidates, tt_move, has_tt_move);
        for (std::size_t index = 0; index < moves.size(); ++index) {
            const Move move = moves[index];
            board_.place(move, player);
            SearchValue child;
            int score = 0;
            try {
                if (board_.check_win(move)) score = MATE_SCORE;
                else if (use_pvs_ && index > 0) {
                    child = negamax(
                        other_side(player), depth - 1,
                        -alpha - 1, -alpha, 1, request_.threat_extension_depth
                    );
                    score = -child.score;
                    if (alpha < score && score < beta) {
                        child = negamax(
                            other_side(player), depth - 1,
                            -beta, -alpha, 1, request_.threat_extension_depth
                        );
                        score = -child.score;
                    }
                } else {
                    child = negamax(
                        other_side(player), depth - 1,
                        -beta, -alpha, 1, request_.threat_extension_depth
                    );
                    score = -child.score;
                }
            } catch (...) {
                board_.undo();
                throw;
            }
            board_.undo();
            std::vector<Move> pv;
            pv.push_back(move);
            pv.insert(pv.end(), child.pv.begin(), child.pv.end());
            int priority = 0;
            for (std::size_t original = 0; original < candidates.size(); ++original) {
                if (candidates[original] == move) {
                    priority = static_cast<int>(candidates.size() - original);
                    break;
                }
            }
            ranked.push_back({move, score, std::move(pv), priority});
            alpha = std::max(alpha, score);
            if (alpha >= beta) break;
        }
        std::sort(ranked.begin(), ranked.end(), [](const RootValue& lhs, const RootValue& rhs) {
            return std::make_pair(lhs.score, lhs.original_priority)
                > std::make_pair(rhs.score, rhs.original_priority);
        });
        if (!ranked.empty() && ranked.front().score <= alpha_original)
            ranked.front().score = std::min(ranked.front().score, alpha_original);
        return ranked;
    }

    [[nodiscard]] std::uint64_t tt_digest() const {
        std::vector<std::pair<std::uint64_t, const TTEntry*>> entries;
        entries.reserve(tt_.size());
        for (const auto& item : tt_) entries.push_back({item.first, &item.second});
        std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });
        std::uint64_t digest = FNV_OFFSET;
        for (const auto& item : entries) {
            const TTEntry& entry = *item.second;
            hash_u64(digest, item.first);
            hash_u64(digest, static_cast<std::uint64_t>(entry.depth));
            hash_u64(digest, static_cast<std::uint64_t>(entry.extension_depth));
            hash_u64(digest, static_cast<std::uint64_t>(static_cast<std::int64_t>(entry.score)));
            hash_u64(digest, static_cast<std::uint64_t>(entry.bound));
            hash_u64(digest, entry.has_best_move
                ? static_cast<std::uint64_t>(entry.best_move.index(board_.size()) + 1)
                : 0U);
            hash_u64(digest, static_cast<std::uint64_t>(entry.pv.size()));
            for (const Move move : entry.pv)
                hash_u64(digest, static_cast<std::uint64_t>(move.index(board_.size())));
        }
        return digest;
    }

    const GNMainSearchRequestV1& request_;
    Board board_;
    bool use_pvs_ = false;
    bool use_tt_ = false;
    std::int64_t nodes_ = 0;
    std::unordered_map<std::uint64_t, TTEntry> tt_;
    std::vector<int> history_scores_;
    std::vector<std::vector<Move>> killers_;
};

}  // namespace

int run_main_search_v1(
    const GNMainSearchRequestV1& request,
    GNMainSearchResultV1& result
) {
    SearchEngine engine(request);
    return engine.run(result);
}

}  // namespace gomoku_native

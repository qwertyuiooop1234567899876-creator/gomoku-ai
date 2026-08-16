from __future__ import annotations

import sys

from engine.board import BLACK, WHITE, Board
from engine.yixin import (
    YixinEngine,
    YixinError,
    load_yixin_config,
    yixin_executable_sha256,
)


def main() -> int:
    try:
        config = load_yixin_config().with_time_limit(3.0)
        executable = config.resolve_executable()
        digest = yixin_executable_sha256(config)
        print(f"YiXin core: {executable}")
        print(f"SHA256: {digest or 'unavailable'}")
        print(
            "Settings: "
            f"threads={config.thread_num}, "
            f"split_depth={config.thread_split_depth}, "
            f"hash={config.hash_size}, "
            f"checkmate={config.checkmate}"
        )
        board = Board()
        board.place(7, 7, BLACK)
        engine = YixinEngine(player=WHITE, config=config)
        try:
            first_move = engine.choose_move(board)
            board.place(*first_move, WHITE)
            opponent_move = next(
                move
                for move in (
                    (6, 6),
                    (6, 8),
                    (8, 6),
                    (8, 8),
                )
                if board.is_empty(*move)
            )
            board.place(*opponent_move, BLACK)
            second_move = engine.choose_move(board)
            report = engine.last_report
        finally:
            engine.close()
    except (OSError, ValueError, YixinError) as error:
        print(f"YiXin smoke test failed: {error}", file=sys.stderr)
        return 2

    first_row, first_column = first_move
    opponent_row, opponent_column = opponent_move
    second_row, second_column = second_move
    print(
        "First move (white): "
        f"{chr(ord('A') + first_column)}{first_row + 1}"
    )
    print(
        "Simulated opponent move: "
        f"{chr(ord('A') + opponent_column)}{opponent_row + 1}"
    )
    print(
        "Second move (white): "
        f"{chr(ord('A') + second_column)}{second_row + 1}"
    )
    if report is not None:
        print(
            "Search: "
            f"depth={report.depth}-{report.selective_depth}, "
            f"evaluation={report.evaluation}, "
            f"time={report.elapsed_ms}ms"
        )
        print(
            "Bestline: "
            + (" -> ".join(report.bestline) or "?")
        )
    print("YiXin protocol smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

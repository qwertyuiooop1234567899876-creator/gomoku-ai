from __future__ import annotations

import sys

from engine.board import BLACK, Board
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
        engine = YixinEngine(player=BLACK, config=config)
        try:
            row, column = engine.choose_move(Board())
            report = engine.last_report
        finally:
            engine.close()
    except (OSError, ValueError, YixinError) as error:
        print(f"YiXin smoke test failed: {error}", file=sys.stderr)
        return 2

    print(f"Move: {chr(ord('A') + column)}{row + 1}")
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

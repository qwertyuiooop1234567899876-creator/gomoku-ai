# Gomoku AI

一个以纯 Python 搜索为主、可选 NativeCore 加速，并集成 YiXin 对弈与复盘的五子棋项目。
当前引擎版本由 `engine/version.py` 统一定义。

## 项目结构

```text
app/            用户入口的实际实现：竞技场、CLI、桌面 UI、Web UI
engine/         棋盘、评估、搜索、VCF/VCT、Proof、记录等核心库
tools/          构建、复盘、工作流、基准和维护命令的实际实现
tests/          单元测试和最小化结构回归局面
native/         NativeCore C++ 源码及发布产物
ui/             Web UI 静态资源
yixin/          外部 YiXin 引擎
release_notes/  统一版本更新日志
records/        本机运行生成的当前对局与分析，不作为源码或回归夹具
```

根目录只保留 BAT 启动器、配置与项目文档；Python 代码统一放在
`app.*`、`tools.*` 或 `engine.*` 包内。BAT 直接使用 `python -m`
运行正式模块，不再经过根目录转发壳。

## 常用入口

- `run_game.bat`：桌面棋盘 UI，Tk 不可用时自动转浏览器 UI。
- `run_game_web.bat`：浏览器棋盘 UI。
- `run_arena.bat`：交互式引擎对弈。
- `run_cvc_analysis.bat`：分析指定棋谱。
- `run_cvc_workflow.bat`：依次完成自对弈、分析、对 YiXin、分析。
- `run_search_benchmark.bat`：搜索性能基准。
- `build_native.bat`：构建 NativeCore。

也可以直接使用模块入口，例如：

```powershell
python -B -m app.arena
python -B -m app.cli
python -B -m app.web_ui
python -B -m tools.cvc_analysis --help
python -B -m tools.search_benchmark --help
python -B -m tools.native_search_baseline --help
```

## 验证

项目不依赖第三方 Python 包，测试使用标准库 `unittest`：

```powershell
python -B -m unittest discover -s tests -p "test_*.py" -v
python -B -m tools.search_benchmark --repeat 3
```

架构边界和继续拆分 `engine/search.py` 的原则见
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 数据边界

`records/` 是本机运行数据目录。清理或移动过的旧棋谱不会由 Git、测试或发布流程重新放回。需要长期保留的回归局面应压缩成最小夹具，放在 `tests/positions/`。

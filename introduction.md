# Gomoku AI — 项目介绍

> 一个 **15×15 五子棋 AI**：纯 Python 搜索为主，可选 NativeCore(C++) 加速，并集成外部 YiXin 引擎用于对弈与复盘。
> 引擎版本：`0.16.17`（单一定义于 `engine/version.py`）；棋谱格式版本：`1.2`。

---

## 目录

1. [项目简介](#1-项目简介)
2. [核心特性](#2-核心特性)
3. [架构分层](#3-架构分层)
4. [目录结构](#4-目录结构)
5. [快速开始](#5-快速开始)
6. [引擎家族](#6-引擎家族)
7. [推算流程（决策流水线）](#7-推算流程决策流水线)
8. [配置说明](#8-配置说明)
9. [测试与验证](#9-测试与验证)
10. [对局记录与数据边界](#10-对局记录与数据边界)
11. [NativeCore 原生加速](#11-nativecore-原生加速)
12. [Telegram 通知](#12-telegram-通知)
13. [版本历史](#13-版本历史)

---

## 1. 项目简介

本项目是一套完整可运行的五子棋程序与引擎研究平台，目标是在"自由规则"（无禁手）下提供**强战术、可审计、可对弈、可复盘**的 AI 体验。

核心亮点：

- **多引擎可选**：从随机基准到完整的搜索 AI，再到外部 YiXin 核心，支持相互对弈（Arena）。
- **多阶段推算**：单次落子走完"硬战术 → 根候选 → VCF 扫描 → 严格证明 → PVS 主搜索 → 根安全复核 → 最终证明审计"的完整流水线。
- **严格证明与启发式分离**：只有 AND/OR 证明搜索能给出"胜/败"三态结论；其余通道（VCF/VCT 探针、根安全、动态复核）只提供候选与排序证据，避免把"近将杀分"误当证明。
- **可选 C++ 加速**：NativeCore 提供批量一步胜扫描、威胁画像、防守反击支撑与 VCF 证书搜索；未编译时自动回退纯 Python 参考实现。
- **完整可复盘**：每手棋都录制思考耗时、落子前后评价与完整 AI 决策依据（`DecisionAnalysis`），可离线分析与对弈学习。

## 2. 核心特性

| 特性 | 说明 |
|---|---|
| 棋盘 | 15×15，增量维护 64 位 Zobrist 哈希、走子历史与撤销 |
| 静态评估 | 基础棋型分 + 先手方活三/跳三主动性加成（tempo-v1），可切换 static-v1 |
| 主搜索 | 迭代加深 PVS（Principal Variation Search）+ Negamax，aspiration window |
| 剪枝与排序 | 置换表(100k)、killer、history、TT best 优先、邻域+方向候选 |
| 威胁延伸 | 叶子处 VCF/VCT 强制链延伸，跨层保持强制着优先 |
| 战术搜索 | 独立 VCF 连续冲四搜索（含证书 Python 逐手重放验证） |
| 防御探针 | Defense-VCT / Mandatory-Defense 探针裁决"眼前都挡得住、但一边会放长链"的局面 |
| 严格证明 | AND/OR 三态证明（PROVEN_WIN / PROVEN_LOSS / UNKNOWN）+ 专属证明置换表 + VCF oracle |
| 根安全 | 等窗口独立比较近分候选、动态 leader/challenger 配对复核、边界平局双通道升级 |
| 最终审计 | 对实际返回着做最终证明复核；被证败则从败证主线生成通用证书拦截点 |
| 时间管理 | 软/硬截止 + 各阶段子预算 + Proof/复核预留，防止主搜索吞噬审计时间 |
| 原生加速 | NativeCore（C ABI，ctypes 加载），六类内核 |
| 外部引擎 | 集成 YiXin（2017 Kernel）对弈与位置评估 |
| 界面 | Tk 桌面 UI、本地浏览器 Web UI、命令行 CLI |
| 记录 | 每手决策依据（txt + json 双格式），支持自动保存与复盘 |

## 3. 架构分层

依赖方向严格单向：

```text
       根目录 BAT（启动器）
          │
          ▼
    app/         tools/
       \         /
        ▼      ▼
         engine/   （核心库，不依赖 app/tools）
```

- `engine/` **不导入** `app/` 或 `tools/`，可独立测试与复用。
- `app/` 只负责用户交互与对局编排，**不承载搜索算法**。
- `tools/` 只做离线分析、构建、基准与自动流程，不承载 UI 状态。
- 根目录不保留 Python 转发壳；BAT 直接通过 `python -m` 运行包内模块。

## 4. 目录结构

```text
gomoku-ai/
├─ *.bat                    根目录启动器（用户可双击）
├─ introduction.md          本文档
├─ README.md                快速说明
├─ docs/                    架构文档、Native ABI/基线文档
├─ app/                     用户入口实现
│   ├─ arena.py             竞技场：引擎工厂与对局编排
│   ├─ cli.py               命令行对弈
│   ├─ desktop_ui.py        Tk 桌面棋盘 UI
│   ├─ web_ui.py            本地浏览器棋盘 UI（HTTP 服务）
│   └─ ui_common.py         共享 UI 工具
├─ engine/                  核心库
│   ├─ search.py            主搜索协调器 + PVS/Negamax 热路径
│   ├─ proof_search.py      严格 AND/OR 证明搜索
│   ├─ threats.py           威胁图/前沿精确描述
│   ├─ evaluator.py         静态棋型评分与威胁画像
│   ├─ ai.py                Random/Tactical/Scoring AI 与决策数据类
│   ├─ vcf.py               连续冲四(VCF) 搜索 + 证书验证
│   ├─ root_candidates.py   根候选来源与结构分类
│   ├─ root_policy.py       根结果仲裁（唯一合并点）
│   ├─ root_review.py       有界根复核策略
│   ├─ root_safety.py       根安全策略
│   ├─ search_types.py      共享配置/结果/分数契约
│   ├─ search_diagnostics.py 不可变决策诊断
│   ├─ time_manager.py      软/硬截止与子预算
│   ├─ records.py           对局录制
│   ├─ yixin.py             外部 YiXin 引擎封装
│   ├─ native_core.py       NativeCore C ABI 封装
│   ├─ board.py / zobrist.py / game.py / settings.py
│   └─ version.py           版本单源
├─ native/                  NativeCore C++ 源码与发布产物
│   ├─ gomoku_native.cpp    六类原生内核
│   ├─ main_search.cpp      主搜索下沉契约 stub
│   └─ bin/                 编译产物（dll/lib/obj）
├─ tools/                   离线命令
│   ├─ build_native.py      NativeCore 唯一构建入口
│   ├─ cvc_analysis.py      分析指定棋谱
│   ├─ cvc_workflow.py      自对弈→分析→对 YiXin→分析 全流程
│   ├─ search_benchmark.py  搜索性能基准
│   ├─ native_search_baseline.py  原生整体搜索基线
│   ├─ yixin_smoke_test.py  YiXin 冒烟测试
│   ├─ telegram_notify.py   Telegram 通知
│   └─ _local/              本机临时复盘脚本（git 忽略）
├─ tests/                   单元测试（标准库 unittest）
│   └─ positions/           最小化结构回归局面
├─ ui/gomoku.html           Web UI 静态资源
├─ yixin/engine.exe         外部 YiXin 引擎
├─ records/                 本机对局与分析（git 忽略）
├─ release_notes/           版本更新日志
├─ arena_settings.json      竞技场配置
├─ yixin_settings.json      YiXin 配置
└─ search_settings.json     玩家搜索设置（可选生成）
```

## 5. 快速开始

### 环境要求

- Python 3.10+（推荐 3.12+）；项目**不依赖任何第三方 Python 包**（测试用标准库 `unittest`）。
- 可选：C++ 编译器（MSVC / MinGW）用于构建 NativeCore。
- 可选：`yixin/engine.exe`（随项目提供）用于 YiXin 对弈。

### 启动器一览

| 启动器 | 作用 |
|---|---|
| `run_game.bat` | 桌面棋盘 UI（Tk），Tk 不可用时自动转浏览器 UI |
| `run_game_web.bat` | 浏览器棋盘 UI |
| `run_arena.bat` | 交互式引擎对弈（Random / Tactical / Scoring / Search / YiXin） |
| `run_cvc_analysis.bat` | 分析指定棋谱 |
| `run_cvc_workflow.bat` | 依次完成：自对弈 → 分析 → 对 YiXin → 分析 |
| `run_search_benchmark.bat` | 搜索性能基准（写 `search-benchmark-results.json`） |
| `run_yixin_smoke_test.bat` | YiXin 引擎冒烟测试 |
| `build_native.bat` | 构建并验证 NativeCore |

### 直接模块入口

```powershell
python -B -m app.cli              # 命令行对弈
python -B -m app.arena            # 竞技场
python -B -m app.desktop_ui       # 桌面 UI
python -B -m app.web_ui           # 浏览器 UI
python -B -m tools.cvc_analysis --help
python -B -m tools.cvc_workflow
python -B -m tools.search_benchmark --repeat 3
python -B -m tools.build_native   # NativeCore 唯一构建入口
```

> 说明：`-B` 避免生成 `__pycache__`；`-X utf8`（BAT 内已设置）保证 Windows 下中文正常。

### 快速对弈示例

```powershell
# 人机对战（命令行）
python -B -m app.cli

# SearchAI 与 YiXin 各下一盘
python -B -m app.arena

# 跑一轮完整自对弈 + 复盘 + 对 YiXin + 复盘
python -B -m tools.cvc_workflow
```

## 6. 引擎家族

| 引擎 | 说明 |
|---|---|
| `RandomAI` | 随机落子，作为基准/热身 |
| `TacticalAI` | 立即五连、唯一封堵、对手多胜点取舍，并优先靠近棋局落子 |
| `ScoringAI` | 一步战术 + 复合威胁画像 + 静态棋型评分排序 |
| `SearchAI` | **主引擎**：PVS 迭代加深、VCF/VCT、AND/OR Proof、根安全/动态复核、最终证明审计 |
| `YixinEngine` | 外部 YiXin 2017 核心，标准行协议通信，支持位置评估 |

SearchAI 默认搜索参数（`engine/search_types.py::SearchConfig`）：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `max_depth` | 3 | 迭代深度上限 |
| `time_limit_seconds` | 2.0 | 单步时限（秒） |
| `root_candidate_limit` | 12 | 根候选宽度 |
| `branch_candidate_limit` | 8 | 分支候选宽度 |
| `threat_extension_depth` | 2 | 叶子威胁延伸层数 |
| `use_pvs` / `use_aspiration` | True / True | PVS 与 aspiration 窗口 |
| `transposition_max_entries` | 100_000 | 每实例置换表容量 |

> 竞技场默认把黑白双方都设为 `SearchAI(d=8, t=60s)`（见 `arena_settings.json`），适合深度自对弈与棋力验证。

## 7. 推算流程（决策流水线）

单次 `SearchAI.choose_move(board)` 的完整流程：

```text
choose_move
 ├─ 0. 每步状态重置（世代+1、清 per-move 缓存、TT 剪枝、history 衰减）
 ├─ 1. 快速回退点 _quick_fallback
 ├─ 2. 硬战术捷径 _try_tactical_shortcut   ──► 命中则直接返回
 │     ├─ 立即五连 → MATE
 │     ├─ 唯一封堵（对手单胜点）
 │     ├─ 对手多胜点（强制败势，选反击价值最高点）
 │     ├─ 空棋盘 → 天元
 │     └─ VCF 快速通道（己方≥3子，小预算找连续冲四）
 ├─ 3. 根候选生成 _prepare_root_candidate_plan
 │     ├─ 相关点池 + 批量威胁画像（Native 加速）
 │     ├─ 对手多重威胁前沿检测
 │     ├─ 模式分类：合并强制 / 己方强制 / 必防 / 前沿防守 / 普通
 │     └─ 各模式组装候选（15 种来源标记）+ 可选防御探针
 ├─ 4. 根对手 VCF 扫描  淘汰"落子后立即陷入对手连续冲四"的候选
 ├─ 5. 初始 Proof 仲裁   ──► 证明己方强制胜则直接返回
 ├─ 6. 迭代 PVS 主搜索（1..max_depth）
 │     ├─ 软截止检查 + 验证预留（Proof/复核时间片）
 │     ├─ aspiration / 全窗口根搜索
 │     ├─ 近败势根扩展 / 未证实优势根扩展（根集合最多翻倍）
 │     ├─ Mate 分隔离（未证明高分拉回启发式量纲）
 │     ├─ Proof/防御探针 tiebreak + 威胁风险纠正
 │     └─ 根安全探针 / 动态 leader-challenger 配对复核
 ├─ 7. 最终 Proof 审计（对实际返回着严格复核）
 │     ├─ 已证败 → 拒绝 + 证书拦截点补入重查
 │     ├─ UNKNOWN 保留 + 紧急 VCF 门禁
 │     └─ 选择被改变则生成新 RootResult
 └─ 8. 组装 DecisionAnalysis → 返回 best_move
```

关键设计原则：

1. **先证后选**：初始 Proof 在 PVS 前跑，最终 Proof 在 PVS 后对实际返回着复核；二者共用同一总预算。
2. **候选完整性优先**：必防/前沿/压力预防类候选必须进入根搜索，避免启发式在搜索前排除唯一活路。
3. **启发式不冒充证明**：VCF/VCT 探针、根安全、动态复核都是启发式证据；只有 AND/OR Proof 的三态可定胜败。
4. **恢复基于性质**：被证败的候选从败证主线生成通用证书拦截点，而不是绑定单一坐标。
5. **Native 只加速不决断**：所有 C++ 结果经 Python 重放/校验后才进入决策。
6. **选择性延伸不越权**：边界第二通道中的冲四与唯一必防可以继续递归；活三、双三和安静前沿落下一层后，只比较有界续攻封堵并回到静态评价，不把未枚举的防守应法解释成 Mate。主 PVS 保持既有语义，避免仲裁补丁改变整棵主搜索树。

## 8. 配置说明

### `search_settings.json`（可选，玩家级搜索设置）

由 `engine/settings.py` 管理，字段：

```json
{ "max_depth": 3, "time_limit_seconds": 2.0 }
```

- `max_depth`：1～8；`time_limit_seconds`：0.1～60.0。文件缺失或损坏时回退默认值；以原子替换方式保存。

### `arena_settings.json`（竞技场配置）

```json
{
  "black": { "engine_name": "search", "max_depth": 8, "time_limit_seconds": 60.0 },
  "white": { "engine_name": "search", "max_depth": 8, "time_limit_seconds": 60.0 },
  "watch": false,
  "show_evaluation": false,
  "delay_seconds": 0.0,
  "save_record": true
}
```

- `engine_name` 可选：`random` / `tactical` / `scoring` / `search` / `yixin`。
- `watch`：是否自动落子；`show_evaluation`：是否展示局面评价；`save_record`：是否保存棋谱。

### `yixin_settings.json`（YiXin 引擎配置）

```json
{
  "executable_path": "yixin/engine.exe",
  "thread_num": 2,
  "thread_split_depth": 6,
  "hash_size": 24,
  "caution_factor": 2,
  "checkmate": 0,
  "rule": 0,
  "pondering": false,
  "timeout_turn_seconds": 10.0,
  "evaluation_time_seconds": 2.0
}
```

- 当前仅支持 15×15 对局；`checkmate`/`rule` 均取自由规则（0）。
- `max_depth` / `max_node` 为 `null` 表示不限制。

## 9. 测试与验证

项目不依赖第三方包，测试全部使用标准库 `unittest`：

```powershell
# 全量测试
python -B -m unittest discover -s tests -p "test_*.py" -v

# 搜索性能基准（重复 3 次）
python -B -m tools.search_benchmark --repeat 3

# 原生搜索基线（固定局面，跨 Python/C++ 第一层真值）
python -B -m tools.native_search_baseline --mode full-window --depths 1-8 --threat-extension-depth 2 --branch-candidate-limit 8
```

测试目录按功能与版本组织：`test_board / test_evaluator / test_search / test_ai / test_arena / test_records / test_v0xxx_*`（每个版本的正确性/回归/架构回归），以及 `test_v0140_native_core`（NativeCore 构建验证）等。

回归数据策略：`tests/positions/` 只保存最小化、可审查的结构回归局面；`records/` 中的历史对局不进入 Git、不参与回归。

## 10. 对局记录与数据边界

- **录制**：`engine/records.py::GameRecorder` 每手记录 `number / player / 坐标 / actor / think_seconds / 落子前后评价 / 完整 analysis 决策依据`，以及悔棋/重开等事件。
- **输出**：每个对局同时产出 `.txt`（人读）与 `.json`（机器分析）双格式，命名形如 `search-d8-t60-vs-yixin-t10-<version>-<timestamp>.{txt,json}`。
- **分析**：`tools.cvc_analysis` 可对任意棋谱输出 YiXin 视角的复盘（含评价条、关键着判断）。
- **边界**：`records/` 整个目录被 `.gitignore` 排除，属于本机运行数据；需要长期保留的回归局面应压缩为最小夹具放入 `tests/positions/`。

## 11. NativeCore 原生加速

- **源码**：`native/gomoku_native.cpp`（六类内核）+ `native/main_search.cpp`（主搜索下沉契约 stub）。
- **内核**：
  - `gn_abi_version` — ABI 版本握手；
  - `gn_find_winning_moves` — 批量一步胜扫描；
  - `gn_analyze_move` / `gn_analyze_moves` — 单点/批量威胁画像；
  - `gn_counter_support_mask` — 防守反击支撑掩码；
  - `gn_find_vcf` — VCF 证书搜索；
  - `gn_main_search_v1` — 整体主搜索下沉契约（当前返回 UNSUPPORTED，验证 ABI 与 digest 机制）。
- **加载**：`engine/native_core.py` 通过 ctypes 加载 `native/bin/gomoku_native.{dll,dylib,so}`；可用环境变量 `GOMOKU_NATIVE_DISABLE=1` 禁用。
- **回退**：未编译/加载失败时自动回退纯 Python 参考实现（V0.13.0 语义），功能不缺失。
- **构建**：唯一入口 `python -m tools.build_native`（`build_native.bat` 负责调用并在成功后原子替换运行库），随后运行 `test_v0140_native_core` 验证。

## 12. Telegram 通知

项目内置无第三方依赖的 Telegram Bot 通知入口，适合长时自对弈/基准完成后的通知。

```powershell
# 先设置环境变量
$env:TELEGRAM_BOT_TOKEN="<your-bot-token>"

# 获取 chat id（先向机器人发送过 /start）
python -B -m tools.telegram_notify --get-chat-id

# 测试发送
python -B -m tools.telegram_notify --test

# 固定接收者后发送自定义消息
$env:TELEGRAM_CHAT_ID="<chat-id>"
python -B -m tools.telegram_notify "自对弈已完成"
```

Token 只从环境变量读取，不写入项目文件，也不会进入 Git。

## 13. 版本历史

`release_notes/` 下按版本号组织更新日志，重要里程碑示例：

| 版本 | 要点 |
|---|---|
| V0.8.x | 前沿搜索、停止策略、Defense-VCT、候选完整性、回归夹具 |
| V0.9.x | Proof 搜索 / 多阶段义务合并 / 威胁图 |
| V0.12.x | 根候选、根策略、根安全、VCF 独立成模块；多必防端独立仲裁 |
| V0.13.x | 严格 AND/OR 证明 + VCF 证书优先收口；最终首选必须通过独立 Proof 复核 |
| V0.14.x | 迁移 NativeCore（一步胜/威胁画像/VCF）；Mate 分隔离；最终 Proof 区分严格生存与 UNKNOWN |
| V0.15.x | 动态复核、记录驱动修复、强制主动反击 |
| V0.16.x | 根完整性、候选证据来源分离、边界双通道、宽安静进攻枢纽、动态 pair 先于全池结构预评分 |

每个版本记录均包含"问题来源 / 候选完整性 / 复核预算 / 结构回归 / 非目标"五部分，可追溯每次改动的原因与边界。

---

*本文档基于当前代码（引擎 0.16.17）编写，如与代码有出入以代码与 `release_notes/` 为准。*

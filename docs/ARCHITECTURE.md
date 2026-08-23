# 项目架构

## 分层

依赖方向保持为：

```text
        根目录 BAT
          |
          v
       app/        tools/
          \        /
           v      v
            engine/
```

- `engine/` 不导入 `app/` 或 `tools/`，因此可以独立测试和复用。
- `app/` 负责用户交互与对局编排，不承载搜索算法。
- `tools/` 负责离线分析、构建、基准和自动流程，不承载 UI 状态。
- 根目录不保留 Python 转发壳；BAT 直接通过 `python -m`
  运行包内模块。

## 搜索模块边界

`engine/search.py` 仍是有状态的搜索协调器和 PVS/Negamax 热路径。它较大，但不应按行数机械切分：递归搜索、TT、killer/history、候选排序和时间检查共享大量热状态，强行拆成 mixin 会隐藏耦合并增加调试成本。

已经独立的职责包括：

- `search_types.py`：配置、结果、计数器和跨模块数据契约。
- `root_candidates.py`：根候选来源、合并与结构分类。
- `root_policy.py`：根结果仲裁与证明状态策略。
- `root_review.py`、`root_safety.py`：根节点独立复核与安全规则。
- `search_diagnostics.py`：决策说明和不可变诊断记录构建。
- `proof_search.py`、`vcf.py`、`threats.py`：独立战术搜索与威胁图。
- `time_manager.py`：软硬截止时间和子预算。

后续拆分必须满足至少一个条件：

1. 输入和输出能用明确的数据类或 Protocol 表达，而不是传入整个 `SearchAI`。
2. 模块可以独立单测，不依赖隐式修改十几个 `_root_*` 字段。
3. 拆分后只有一个实现来源，不复制搜索规则。
4. 固定局面节点数、着法结果和性能基准没有退化。

因此，下一批适合抽离的是纯根规划数据构建或可冻结的根阶段上下文；不适合抽离的是仅把现有方法搬进 `SearchMixin`。

## 入口管理

根目录只保留用户可双击的 BAT。正式 Python 入口为：

- `python -m app.cli`、`python -m app.arena`
- `python -m app.desktop_ui`、`python -m app.web_ui`
- `python -m tools.cvc_analysis`、`python -m tools.cvc_workflow`
- `python -m tools.build_native`、`python -m tools.search_benchmark`

NativeCore 的唯一构建入口是 `python -m tools.build_native`（根目录
`build_native.bat` 只负责调用该模块）。构建器按确定顺序编译
`native/*.cpp` 的全部翻译单元，并在成功后原子替换运行库；项目不再
保留另一套未被启动器调用的构建定义。

Native 整体搜索下沉前的固定局面基线入口为：

```powershell
python -B -m tools.native_search_baseline --mode full-window --depths 1-8
python -B -m tools.native_search_baseline --mode iterative --depths 8 --node-limit 9000
```

基线夹具必须保存有序走子历史。TT 等价门禁比较完整内容摘要，而不只
比较最终着法；限时棋谱无法保证重建同一份热 TT，因此冷启动固定节点
基线是跨 Python/C++ 实现的第一层真值。

测试与内部导入也只使用这些正式包路径，避免两套模块身份。

## 运行数据与回归数据

- `records/`：仅保存用户当前运行产生的数据，整个目录由根 `.gitignore` 排除。
- `tests/positions/`：只保存最小化、可审查的结构回归局面。
- `release_notes/`：只保存版本更新日志。
- `tools/_local/`：本机临时复盘脚本，由 Git 忽略，不作为正式工具发布。

测试、Git 操作和发布脚本不得从历史提交恢复 `records/` 中已经删除或移动的棋谱。

# VCT / ProofSearch 后续计划

更新日期：2026-09-02  
当前基线：`main` / `16f0275` / engine `0.17.5`

## 1. 文档用途

本文件是 VCT / ProofSearch 性能与正确性工作的长期上下文入口。后续开始
相关任务时，应先阅读本文件，再检查当前 Git 状态、版本、测试和实际代码；
不得仅凭旧对话继续修改。

所有历史回归局面只允许保存在 `tests/positions/` 的最小夹具中。不得恢复、
重建、复制或重新加入 `records/` 中已删除或移走的文件，也不得改写现有记录。

## 2. 当前目标

在不损害三态证明语义的前提下，让独立 VCT 参考验证器在固定节点预算内：

1. 更少地重复展开无价值分支；
2. 更合理地把节点分配给可能改变结论的 AND / OR 前沿；
3. 对完成的 `PROVEN_WIN` / `PROVEN_LOSS` 提供可独立复放的证据；
4. 对预算耗尽、候选不完整或防守覆盖不完整继续返回 `UNKNOWN`；
5. 只有经过离线等价性与性能门禁后，才讨论生产接入。

当前主要问题不是单个函数的执行速度，而是硬局面中的节点组合爆炸和
未解决前沿调度。

## 3. 已确认事实

### 3.1 正确性边界

- `ProofState` 始终相对固定 attacker。
- `UNKNOWN` 或未完成结果不得写入 `ProofTable` 作为精确结果。
- 安静前沿和非 VCF 计划只有在防守集合完整时才可成为严格证明。
- Native 只能作为候选证据或证书查找加速器；未经 Python 独立复放的
  Native 证明不得采信。
- VCF 查找失败没有失败证明含义。现有 VCF 路径也是 Native 找线、Python
  复放验证，入口位于 `engine/proof_search.py::_find_vcf_witness`，验证器位于
  `engine/vcf.py::validate_vcf_certificate`。

### 3.2 当前固定夹具

必须持续覆盖以下三组局面及其候选：

| 夹具 | 候选 |
|---|---|
| `tests/positions/v0175_reverse_move10_vct.json` | G10 / K6 |
| `tests/positions/v0175_selfplay_move24_vct.json` | J10 / J6 |
| `tests/positions/v0175_yixin_move21_vct.json` | I8 / H11 |

这些夹具用于验证搜索语义、节点使用、UNKNOWN 保守性和证书，不直接断言
某个候选一定守和或获胜，除非严格证明已经完成。

### 3.3 Phase 1B-1E 的结论

- 增加 attacker 深度、扩大节点预算和关闭安静前沿都没有让三个硬局面完成。
- 单次冷启动中存在重复 UNKNOWN key，但重复极度分散：最高频 key 仅出现
  约 3-6 次，Top 100 对重复查询的贡献很小。
- 共享现有 `ProofTable` 的暖重入收益不稳定，整体没有形成可持续加速。
- 因此不得缓存 deadline / node-limit 截断的 UNKNOWN，也不应设计小型热点
  LRU 或扩大现有 ProofTable 来赌收益。

### 3.4 窄 Native 防守分类实验是负结果

当前工作区存在一个**未接生产**的 Native defense-classification V1 只读
实验：

- ABI：`native/gomoku_native.h::GNDefenseClassificationRequestV1`
- 实现：`native/gomoku_native.cpp::gn_classify_defenses_v1`
- Python 封装：`engine/native_core.py::classify_defenses`
- 基准：`tools/native_defense_baseline.py`
- 等价测试：`tests/test_native_defense_classification.py`

它在三个固定夹具、随机局面、部分中断、零秒 timeout 和棋盘恢复上与
Python `DefenseSet.signature` 完全一致。聚焦回归通过；全量结果为
`504 passed, 157 subtests passed`。

但是 500 次空闲本机基准未达到 3x 门槛：

| 非平凡候选 | Native / Python 速度比 |
|---|---:|
| J10 | 0.70x |
| J6 | 0.65x |
| H11 | 0.62x |

这里的 `0.62x` 表示 Native 更慢，而不是加速 62%。主要原因是一次完整
分类本来只有约 1ms，ctypes、输入封送、输出数组和 Python refutation 对象
重建超过了节省的循环成本。

**结论：不得把这一窄 ABI 接入 `ThreatAnalyzer` 或生产搜索。** 在后续提交前
必须明确决定：保留在实验分支，还是撤销该实验代码及编译产物。不能因为
等价测试通过就跳过性能门禁。

复现命令：

```powershell
.\.venv\Scripts\python.exe -m tools.native_defense_baseline --repeats 500 --minimum-speedup 3.0
```

## 4. 明确非目标

后续阶段不得顺手实施以下内容：

- 不缓存 UNKNOWN 或未完成证明；
- 不减少完整防守集合，不把选择性防守枚举包装成完整覆盖；
- 不扩大或缩小生产 root candidate 池；
- 不修改 PVS、root review、Final Proof 或实际落子；
- 不把 Native miss 解释为 `PROVEN_LOSS`；
- 不删除 Python 独立证书复放；
- 不继续微调当前窄 Native defense ABI 以追逐小幅性能数字；
- 不使用 GPU 执行当前不规则 AND / OR 树；
- 不在没有新证据时全局增加深度、宽度或 Proof 预算。

## 5. 路线与执行顺序

### Phase 1F：ThreatAnalyzer 与 AND 节点审计

目标：先回答节点浪费在哪里，不修改证明语义。

计划：

1. 在 `tools/vct_reference.py` 输出现有 `ThreatAnalyzer.stats()` 的完整增量：
   `cache_queries`、`cache_hits`、`cache_stores`、`cache_skips`。
2. 只通过工具层诊断，统计：
   - 每个候选的 exact threat description 次数；
   - 完整 DefenseSet 生成次数；
   - AND 节点展开的防守数量；
   - 第几个防守首次得到 `PROVEN_LOSS`；
   - `_replay_linear_plan` 的尝试、成功和失败原因；
   - 节点预算耗尽时尚未检查的防守数量。
3. 诊断字段使用工具内部 dataclass，并通过显式 `--json <path>` 选择性保存：
   - 顶层使用独立的 `diagnostic_schema_version`；
   - 默认只打印、不写文件；
   - 不修改正式 record format；
   - 不默认写入 `records/`；
   - schema 字段优先只追加，不随意改名。
4. 不改候选成员、不改遍历顺序、不接生产。

验收：

- 三个固定夹具六个候选均产出上述统计；
- 固定节点数、三态、cutoff reason 和棋盘恢复与当前参考器一致；
- 工具测试覆盖序列化和计数不变量；
- 根据数据能明确选择“缓存 / 排序 / 新算法 / 停止优化”中的至少一条，
  不能只得到新的模糊百分比。

当前进度（2026-09-03）：

- 已在工具层完成 Phase 1F 旁路审计；生产 ProofSearch、ThreatAnalyzer 和
  record schema 均未修改。
- 三个夹具六个候选按每候选 50,000 节点运行，全部保持
  `UNKNOWN/node_limit`；17,619 次 DefenseSet 生成中 17,605 次完整，
  AND 节点实际检查 277,503 / 280,603 个防守。
- Threat cache 为 552,428 / 1,009,713 次命中；15,058 次线性计划复放中
  7,655 次成功。84 次首次 `PROVEN_LOSS` 中 82 次出现在前 5 个防守，
  预算耗尽时未检查防守合计仅 2,571 个。
- 这批固定节点证据不支持继续扩大缓存或把防守重排当作主线；Phase 2 仅在
  后续出现更集中的排序浪费证据时执行，主线继续进入 Phase 3 DFPN / PNS。
- 工具聚焦测试 10 项通过，相关子系统 46 项通过，全量测试 502 项、
  135 个 subtests 通过。上述墙钟包含诊断开销，不作为生产性能数字。

停止条件：如果统计显示 Threat cache 已高命中、AND 失败位置无集中规律且
plan replay 复用也很低，停止继续微调现有 DFS。

### Phase 2：仅排序的 A/B 实验

仅当 Phase 1F 显示 AND 防守顺序或 OR 候选顺序有明显集中浪费时才执行。

允许：

- 使用已有 TT / hint / history 或静态、可重复的结构信息调整遍历顺序；
- 优先检查更可能推翻攻击证明的防守；
- 优先学习更可能跨防守复用的线性计划。

禁止：

- 删除候选或防守；
- 改变 coverage-complete 判定；
- 把排序分数当证明；
- 让预算不足升级成完成。

A/B 门禁：

- 相同局面、候选、节点预算和 attacker 深度；
- 已完成样本的三态与证书必须一致；
- UNKNOWN 仍保持 UNKNOWN；
- 至少满足以下一项才保留：
  - 已知完成证明的节点数降低 20% 以上；
  - 固定 200k 节点下至少一个硬候选从 UNKNOWN 变为可验证完成；
  - `_replay_linear_plan` 有稳定、跨样本的明显增益。

未达门禁则撤销排序实验，不继续堆规则。

### Phase 3：独立 DFPN / PNS 参考原型

这是当前最值得投入的主线。建议新建独立工具，例如
`tools/vct_dfpn_reference.py`，不替换现有 `ProofSearch`。

核心原则：

- 使用 proof number / disproof number 或等价的 best-first AND / OR 前沿分配；
- unresolved 节点只存在于本次搜索的有界内存树中，不写入现有 ProofTable
  作为精确结论；
- 候选生成和完整 DefenseSet 仍由现有 Python 权威语义提供；
- cutoff 始终返回 UNKNOWN；
- 完成证明必须导出可独立复放的分支证书；
- 工具退出或预算耗尽后不得污染棋盘或生产缓存。

实验梯形：

| 维度 | 记录内容 |
|---|---|
| 节点预算 | 50k -> 200k -> 1m |
| 夹具 | 三组固定局面、六个候选 |
| 对照 | 当前 DFS ProofSearch |
| 时间 | 总墙钟、每节点时间 |
| 内存 | 峰值 RSS / working set、最大活跃节点数、估算单位节点字节 |
| 前沿 | 最大未解决前沿宽度、预算结束时仍存活的前沿数 |

峰值内存是与节点数、墙钟同等级的第一公民指标，不能只在达到硬上限时
记录一次。每档实验都必须报告峰值内存和最大活跃节点数，用于判断瓶颈是否
从计算量转移成显式树驻留；不同实现之间只比较使用同一测量口径的数据。

保留门禁：

- 不出现任何假 completed；
- complete / incomplete / timeout / node-limit / board-restore 测试通过；
- 至少一个硬候选在相同节点预算下完成，或相同证明节点数减少 3x；
- 内存必须有显式上限，超过上限返回 UNKNOWN，不静默丢分支。

如果 1m 节点仍没有任何夹具改善，应先复查候选生成和 VCT 表示能力，不能
直接把 DFPN 移植到生产或 C++。

### Phase 4：候选级 CPU 多进程（可独立执行）

目标仅是缩短离线 workflow，不宣称提升单候选证明能力或棋力。

优先级已确定低于 Phase 3。它不得阻塞 DFPN / PNS 主线；只有批量实验墙钟
已经明显拖慢主线验证时，才作为独立体验优化实施。

当前 `tools/vct_reference.py::run_reference` 已为每个候选创建独立
`ThreatAnalyzer` 和 `ProofTable`，适合以进程隔离并行。

门禁：

- 串行与并行输出逐字段一致；
- 每个候选预算仍独立；
- 任一 worker 失败不能把其他候选包装成完成；
- 三个以上候选时本机墙钟至少改善 1.5x，否则不引入多进程复杂度；
- 不接生产 `choose_move`。

### Phase 5：粗粒度 Native VCT（条件阶段）

只有当 DFPN / PNS 已证明算法方向有效、且新的 profile 仍显示 Python
每节点成本是主要瓶颈时，才启动这一阶段。

边界必须是一候选一次 ABI 调用，而不是再次拆成 defense-classification
这类毫秒级小调用。Native 在内部运行完整证明搜索；成功时返回分支证明
证书，Python 独立复放后才能采信。

初始能力仅允许：

- `PROVEN_WIN` + 可复放证书；
- 预算耗尽或找不到证书 -> `UNKNOWN`。

在独立的 loss/disproof 证书格式和验证器完成前，Native 不得独立产生可信
`PROVEN_LOSS`。

当前三个夹具在候选落子后，把固定 attacker 设置为候选方的对手。因此，
对手视角的 `PROVEN_WIN` 能直接证明“该候选会输”，Phase 5 的 win-only
第一版可以推动这部分原始问题。它不能证明候选安全：对手视角的
`PROVEN_LOSS` 才能排除对手强制获胜；在该证书和验证器完成前，所有未找到
对手胜证的候选仍必须保持 `UNKNOWN`，不能据此守和、排序或改选。

接入门禁：

- Python / Native 对已完成证书逐项等价；
- cutoff 与棋盘恢复完全一致；
- 端到端每候选至少 3x，而不是只测 C++ 内核；
- 工具只读阶段通过后，另立生产接入方案并重新获得批准。

## 6. 已确认执行决策

推荐顺序：

1. 不接入窄 Native defense ABI；把完整负结果隔离保留在独立实验分支，
   再从 `main` 工作区精确撤销该实验，不影响其他未提交工具改动；
2. 先做 Phase 1F 只读审计；
3. 若有集中排序收益，做 Phase 2 A/B；
4. 主线推进 Phase 3 DFPN / PNS 离线原型；
5. Phase 1F 诊断采用 opt-in、独立版本号 JSON，不进入正式 record schema；
6. Phase 4 多进程低于 Phase 3，只作为独立体验优化；
7. 只有 Phase 3 有效后才评估 Phase 5 粗粒度 Native。

如果团队当前更重视实战棋力而不是 VCT 工具吞吐，也可以在第 1 步后暂停
这条路线，回到新的对局记录进行只读诊断。不得把“VCT 性能线暂停”等同于
“现有候选和搜索语义已解决”。

## 7. 每阶段统一验证

聚焦测试先行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_vct_reference.py -q
```

涉及 Native 时重新构建并运行 Native 相关测试：

```powershell
.\.venv\Scripts\python.exe tools/build_native.py
.\.venv\Scripts\python.exe -m pytest tests/test_v0140_native_core.py tests/test_native_main_search_contract.py -q
```

完整门禁：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

报告必须列出：命令、测试数、subtests、失败/跳过、节点数、墙钟、三态、
cutoff reason、证书验证结果和棋盘恢复。单元测试通过不能宣称棋力提升；
涉及实际决策的改动仍需新自对弈或 YiXin 对局验证。

## 8. 新上下文接手清单

每次上下文压缩、换模型或换任务后，按以下顺序恢复：

1. 阅读本文件；
2. 运行 `git status --short`，区分用户文件、旧实验和本轮改动；
3. 确认当前 branch、commit、engine version；
4. 保持 `records/` 当前状态，不恢复或清理；
5. 确认正在执行哪个 Phase，不能跳阶段；
6. 检查该 Phase 是否已有明确用户批准；
7. 先复现聚焦基线，再编辑；
8. 技术结论引用当前代码行，不能沿用旧行号；
9. 不提交、不推送，除非用户明确要求；
10. 若新证据改变根因或范围，停止修改并更新本计划后重新确认。

## 9. 已决事项

- **窄 Native 实验：隔离保留。** 保存到独立实验分支后，从 `main` 精确
  撤销这组负结果代码和编译产物；不得清理其他未提交改动。
- **Phase 1F 输出：折中独立保存。** 使用工具 dataclass、opt-in JSON 和
  独立 diagnostic schema version；默认不写文件，不进入正式 record schema。
- **优先级：Phase 3 高于 Phase 4。** DFPN / PNS 是单候选证明能力主线；
  多进程仅在批量 workflow 墙钟妨碍实验时实施。

下一项操作应先隔离窄 Native 实验，再开始 Phase 1F。创建分支、提交或推送
仍需用户明确指令；上述选择本身不授权 Git 提交或生产搜索变更。

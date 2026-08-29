# Native 搜索下沉前基线

## 目标

这份基线用于区分“执行速度不足”和“当前搜索语义无法区分候选”。
固定局面来自 V0.16.8 对 YiXin 第 13 手之前，黑方待走，候选为 F7、
J11。最小夹具保存 12 手有序历史和 Zobrist 哈希，不依赖或恢复
`records/` 中的棋谱。

复现实验：

```powershell
python -B -m tools.native_search_baseline --mode full-window --depths 1-9 --threat-extension-depth 2 --branch-candidate-limit 8
python -B -m tools.native_search_baseline --mode full-window --depths 8 --threat-extension-depth 2 --branch-candidate-limit 8 --candidate-trace-limit 12 --candidate-sample-limit 8 --leaf-trace-limit 12
python -B -m tools.native_search_baseline --mode iterative --depths 8 --node-limit 1000
python -B -m tools.native_search_baseline --mode iterative --depths 8 --node-limit 3000
python -B -m tools.native_search_baseline --mode iterative --depths 8 --node-limit 6000
python -B -m tools.native_search_baseline --mode iterative --depths 8 --node-limit 9000
python -B -m tools.native_search_baseline --mode iterative --depths 8 --node-limit 10000
python -B -m tools.native_search_baseline --mode native --depths 1-8
```

用于参数语义矩阵时，固定深度和两个候选不变，只改变主 PVS 的
`--threat-extension-depth` 与 `--branch-candidate-limit`：

```powershell
foreach ($extension in 0, 2, 4) {
    foreach ($branch in 8, 12) {
        python -B -m tools.native_search_baseline --mode full-window --depths 8 --threat-extension-depth $extension --branch-candidate-limit $branch --candidate-trace-limit 12 --candidate-sample-limit 8 --leaf-trace-limit 12
    }
}
```

`full-window` 只运行主 PVS。默认不启用 trace，且直接使用普通
`SearchAI`，因此历史 Native 性能基线的默认计时不加入追踪器开销。
`--candidate-trace-limit` 和 `--leaf-trace-limit` 都是显式 opt-in；前者
按 `(ply, 剩余深度, 延伸深度)` 保留有界候选集合样本，后者只报告有界的
叶面**返回分类**与分数，不声称提供精确终止原因或完整评价分解。

安静前沿绝不能作为 `full-window` 的开关。它只能通过独立动态 pair
复核路径测试：

```powershell
python -B -m tools.native_search_baseline --mode dynamic-pair --depths 8 --threat-extension-depth 2 --branch-candidate-limit 12 --quiet-frontier --review-budget 10
```

Defense VCT 也有独立模式，不能伪装为主 PVS 参数：

```powershell
python -B -m tools.native_search_baseline --mode defense-vct --depths 3 --time-limit 10
```

工具只重建 `tests/positions/` 中的最小夹具；除用户明确传入 `--json`
外不写入文件，也不会读取、恢复或生成 `records/` 内容。

## V0.16.8 参数观察矩阵（本机）

以下 d8 的 branch=8 运行启用了 4 层候选摘要和 4 条叶面返回分类，故墙钟
只用于同组观察，不作为无 trace 的性能门禁。所有格都是独立、冷启动、单
候选全窗口主 PVS；分数不是严格 Proof。

| d8 branch | 延伸 | F7：分数 / 节点 / 延伸 / 秒 | J11：分数 / 节点 / 延伸 / 秒 |
|---:|---:|:---|:---|
| 8 | 0 | -909,000 / 8,106 / 0 / 14.449 | -1,007,100 / 8,204 / 0 / 14.932 |
| 8 | 2 | -979,000 / 7,495 / 8,357 / 23.333 | -999,000 / 6,591 / 7,092 / 20.377 |
| 8 | 4 | -899,300 / 8,381 / 18,434 / 45.152 | -999,999,988 / 6,893 / 14,995 / 36.855 |
| 12 | 2 | -899,100 / 30,370 / 29,744 / 95.823 | -970,100 / 30,063 / 32,645 / 107.042 |

branch=12、延伸=0 的 d8 成对进程在 60 秒没有产生完成结果，按外部运行
上限中止；延伸=4 的 branch=12 未运行。这个超时本身说明宽度的成本显著，
不能被描述成一个完成深度的比较结果。

为分离宽度敏感性，又在无 trace 的 d6 做了快速对照：

| d6 延伸 / branch | F7：分数 / 节点 / 秒 | J11：分数 / 节点 / 秒 |
|:---|:---|:---|
| 0 / 8 | -987,300 / 1,187 / 1.778 | -999,000 / 1,116 / 1.707 |
| 0 / 12 | -909,400 / 3,232 / 5.113 | -987,500 / 2,910 / 4.684 |
| 2 / 8 | -892,000 / 1,232 / 4.577 | -990,000 / 1,294 / 4.587 |
| 2 / 12 | -908,900 / 3,184 / 10.935 | -908,200 / 3,702 / 13.663 |

候选摘要显示 branch=8 的内部层保持 8 个有序候选上限；延伸为 0 的 sampled
叶面返回为 `extension_limit_static`，延伸为 2/4 的 sampled 返回多为
`forcing_extension`。这些是有界工具 trace，不是完整叶面覆盖或评价分解。
d6 的延伸=2、branch=12 中 J11 仅高 700 分，表明宽度敏感性，但不能单独
证明候选成员、Proof 三态或 root 选择不变量被违反。

因此目前还不能写出“结构条件 → 组件违反不变量 → 错误选着”的通用根因；
本轮没有修改任何搜索选择语义、评估权重或全局参数。下一步应利用新增的
候选/叶面 trace 与 Final-Proof/root-review provenance，寻找可结构化复现的
不变量违例，而不是把这一局面的结果编码成特判。

## 冷启动全窗口结果

每个候选使用独立 SearchAI 和完整窗口，避免候选之间共享 TT 或 PVS
窄窗。以下是 2026-08-23 在 Python 3.14.7 上的一次基线；墙钟只作本机
参考，节点、分数和 TT 摘要才是等价门禁。表中是未经根 Proof 隔离的
原始 PVS 分数，深度 2/4 的 Mate 带夹值不代表严格证明。

| 深度 | F7 分数 | F7 节点 | J11 分数 | J11 节点 | 相对结果 |
|---:|---:|---:|---:|---:|:---|
| 1 | 3,700 | 1 | -8,200 | 1 | F7 |
| 2 | -999,999,996 | 11 | -999,999,996 | 10 | 同夹值 |
| 3 | 3,500 | 37 | 800 | 46 | F7 |
| 4 | -999,999,994 | 111 | -999,999,994 | 108 | 同夹值 |
| 5 | 11,900 | 595 | -8,000 | 518 | F7 |
| 6 | -892,000 | 1,232 | -990,000 | 1,294 | F7 |
| 7 | 11,000 | 3,533 | -7,100 | 3,381 | F7 |
| 8 | -979,000 | 7,495 | -999,000 | 6,591 | F7 |
| 9 | 3,900 | 22,420 | 100 | 18,036 | F7 |

深度 9 仍未出现 J11 反超。因此当前证据不支持“Native 只要多完成一层
就会自然修正第 13 手”。Native 仍值得做，但初版目标必须先是同语义
等价和深度提升；若更深层仍偏向 F7，需要单独检查评价、选择性候选和
威胁延伸。

## 固定节点结果

双候选迭代 PVS 使用同一棵冷启动搜索树：

| 节点上限 | 完成深度 | 停止原因 | 结果 |
|---:|---:|:---|:---|
| 1,000 | 5 | `node_limit` | F7 11,900；J11 10,900 |
| 3,000 | 6 | `node_limit` | F7 -892,000；J11 -901,100 |
| 6,000 | 7 | `node_limit` | F7 11,000；J11 10,900 |
| 9,000 | 7 | `node_limit` | F7 11,000；J11 10,900 |
| 10,000 | 8 | `requested_depth_completed` | 9,253 节点；F7 -979,000，J11 -987,500 |

节点上限只约束主 PVS；到达上限时保留最后一个完整深度，不把中断层
升级为完成结果。Proof、VCF 和后续复核仍保留各自原有预算语义。

## 热状态限制

仅按棋谱重建棋盘不能重建实战的 TT、history、killer 和 generation。
严格逐手预热在第 3 手已经出现当前引擎 G8 与记录 G7 的分歧，因此工具
主动中止，没有把不同内部状态描述为同一份实战 TT。后续跨实现等价以
冷启动固定节点为第一层；热状态等价需要可序列化状态快照或完全确定的
固定节点整局重放。

## Native 验收顺序

1. 固定深度下保持走法、候选分数、bound、PV、节点语义和完整 TT 摘要一致。
2. 截止继续表示未完成，不能把 UNKNOWN 或中断层转成安全结论。
3. 在不收窄候选与延伸的前提下，同等时间至少多完成一层。
4. 完成更深搜索后重新检查 F7/J11；J11 反超属于棋力验收，不是初版移植的正确性断言。

## V0.17.0 Phase 1 结果

`gn_main_search_v1` 已从 ABI 占位符变为隔离的 C++ 固定深度核心，但未
接入生产 `choose_move`。第 13 手 F7/J11 在深度 1–3 的分数、PV、节点、
TT 条目数与完整规范化 TT 摘要逐项等于 Python；关闭 PVS 的全窗口模式
也通过同一门禁，给后续 root-safety/dynamic-review 接入保留了路径。

V0.16.18 对 YiXin 第 21 手另存为不依赖 `records/` 的最小夹具。C++ 与
Python 都复现同一个固定窗口振荡：d6 选 K8、d7 选 H7、d8 再选 K8；
因此 Phase 1 没有借移植偷偷改变算法或把 K8 写成坐标特判。

本机冷启动双候选 d8 对照：Python 32.47 秒，Native 5.30 秒（同为
10,716 个主搜索节点；墙钟只作本机参考）。这证明整体下沉具备吞吐
杠杆，但不能据此宣布棋力提升；生产接入和复核通道接入仍属于后续阶段。
